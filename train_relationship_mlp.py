"""Train an MLP to predict advisor/advisee relationships from OpenAlex works."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "researchers_advisors_works_found.csv"
DEFAULT_DATASET_PATH = SCRIPT_DIRECTORY / "relationship_training_dataset.csv"
DEFAULT_PREDICTIONS_PATH = SCRIPT_DIRECTORY / "relationship_predictions.csv"
DEFAULT_METRICS_PATH = SCRIPT_DIRECTORY / "relationship_test_metrics.json"
DEFAULT_MODEL_PATH = SCRIPT_DIRECTORY / "relationship_mlp.pt"
FEATURE_NAMES = [
    "coauthored_work_count",
    "collaboration_year_span",
    "first_collaboration_year",
    "last_collaboration_year",
    "title_token_overlap",
    "researcher_observed_work_count",
    "advisor_observed_work_count",
    "name_token_similarity",
]
REQUIRED_COLUMNS = {
    "researcher_id",
    "researcher_name",
    "advisor_id",
    "advisor_name",
    "researcher_open_alex_id",
    "advisor_open_alex_id",
    "work_id",
    "work_title",
    "publication_year",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and test an MLP for advisor/advisee prediction."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_tokens(value: str) -> set[str]:
    """Normalize a text value into comparable tokens."""
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def name_similarity(left: str, right: str) -> float:
    """Calculate token overlap between two names."""
    left_tokens = normalize_tokens(left)
    right_tokens = normalize_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load the OpenAlex works CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        return list(reader)


def years_from_rows(rows: Iterable[dict[str, str]]) -> list[int]:
    """Extract valid publication years."""
    return [
        int(row["publication_year"])
        for row in rows
        if row.get("publication_year", "").isdigit()
    ]


def title_overlap(rows: Iterable[dict[str, str]]) -> float:
    """Calculate average pairwise Jaccard overlap among work titles."""
    title_tokens = [
        normalize_tokens(row.get("work_title", ""))
        for row in rows
        if normalize_tokens(row.get("work_title", ""))
    ]
    overlaps = [
        len(left & right) / len(left | right)
        for left, right in combinations(title_tokens, 2)
        if left | right
    ]
    return float(np.mean(overlaps)) if overlaps else 0.0


def aggregate_positive_pairs(
    rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[str, dict[str, str]]]:
    """Aggregate works into positive researcher/advisor pair records."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    people: dict[str, dict[str, str]] = {}
    for row in rows:
        researcher_id = row["researcher_open_alex_id"]
        advisor_id = row["advisor_open_alex_id"]
        grouped[(researcher_id, advisor_id)].append(row)
        people.setdefault(
            researcher_id,
            {
                "researcher_id": row["researcher_id"],
                "name": row["researcher_name"],
                "openalex_id": researcher_id,
                "role": "researcher",
            },
        )
        people.setdefault(
            advisor_id,
            {
                "researcher_id": row["advisor_id"],
                "name": row["advisor_name"],
                "openalex_id": advisor_id,
                "role": "advisor",
            },
        )

    positive_pairs: dict[tuple[str, str], dict[str, object]] = {}
    for (researcher_id, advisor_id), pair_rows in grouped.items():
        years = years_from_rows(pair_rows)
        researcher = pair_rows[0]
        positive_pairs[(researcher_id, advisor_id)] = {
            "researcher_open_alex_id": researcher_id,
            "advisor_open_alex_id": advisor_id,
            "researcher_id": researcher["researcher_id"],
            "researcher_name": researcher["researcher_name"],
            "advisor_id": researcher["advisor_id"],
            "advisor_name": researcher["advisor_name"],
            "works": pair_rows,
            "label": 1,
            "features": build_features(
                researcher["researcher_name"],
                researcher["advisor_name"],
                pair_rows,
                len({row["work_id"] for row in rows if row["researcher_open_alex_id"] == researcher_id}),
                len({row["work_id"] for row in rows if row["advisor_open_alex_id"] == advisor_id}),
            ),
        }
    return positive_pairs, people


def build_features(
    researcher_name: str,
    advisor_name: str,
    pair_rows: list[dict[str, str]],
    researcher_work_count: int,
    advisor_work_count: int,
) -> list[float]:
    """Build the bibliometric feature vector for a pair."""
    years = years_from_rows(pair_rows)
    first_year = min(years) if years else 0
    last_year = max(years) if years else 0
    return [
        float(len({row["work_id"] for row in pair_rows})),
        float(last_year - first_year) if years else 0.0,
        float(first_year),
        float(last_year),
        title_overlap(pair_rows),
        float(researcher_work_count),
        float(advisor_work_count),
        name_similarity(researcher_name, advisor_name),
    ]


def build_dataset(
    rows: list[dict[str, str]],
    negative_ratio: float,
    seed: int,
) -> list[dict[str, object]]:
    """Create positive and sampled negative pair examples."""
    positive_pairs, people = aggregate_positive_pairs(rows)
    researcher_people = [
        person for person in people.values() if person["role"] == "researcher"
    ]
    advisor_people = [
        person for person in people.values() if person["role"] == "advisor"
    ]
    candidates = [
        (researcher, advisor)
        for researcher in researcher_people
        for advisor in advisor_people
        if (researcher["openalex_id"], advisor["openalex_id"]) not in positive_pairs
    ]
    random.Random(seed).shuffle(candidates)
    negative_count = min(
        len(candidates),
        max(1, int(round(len(positive_pairs) * negative_ratio))),
    )

    dataset = list(positive_pairs.values())
    researcher_work_ids: defaultdict[str, set[str]] = defaultdict(set)
    advisor_work_ids: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        researcher_work_ids[row["researcher_open_alex_id"]].add(row["work_id"])
        advisor_work_ids[row["advisor_open_alex_id"]].add(row["work_id"])

    for researcher, advisor in candidates[:negative_count]:
        pair = {
            "researcher_open_alex_id": researcher["openalex_id"],
            "advisor_open_alex_id": advisor["openalex_id"],
            "researcher_id": researcher["researcher_id"],
            "researcher_name": researcher["name"],
            "advisor_id": advisor["researcher_id"],
            "advisor_name": advisor["name"],
            "label": 0,
        }
        pair["features"] = build_features(
            researcher["name"],
            advisor["name"],
            [],
            len(researcher_work_ids[researcher["openalex_id"]]),
            len(advisor_work_ids[advisor["openalex_id"]]),
        )
        dataset.append(pair)
    return dataset


def split_dataset(
    dataset: list[dict[str, object]], seed: int
) -> tuple[list[int], list[int]]:
    """Split positive and negative examples into stratified train/validation sets."""
    random_generator = random.Random(seed)
    train: list[int] = []
    validation: list[int] = []
    for label in (0, 1):
        indices = [index for index, row in enumerate(dataset) if row["label"] == label]
        random_generator.shuffle(indices)
        split_at = max(1, int(len(indices) * 0.8))
        train.extend(indices[:split_at])
        validation.extend(indices[split_at:])
    random_generator.shuffle(train)
    random_generator.shuffle(validation)
    return train, validation


class RelationshipMLP(nn.Module):
    """Small multilayer perceptron for pair classification."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Compute logits."""
        return self.network(values).squeeze(1)


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate threshold and ranking metrics for binary classification."""
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    total = max(len(labels), 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "specificity": true_negative / max(true_negative + false_positive, 1),
        "negative_predictive_value": true_negative
        / max(true_negative + false_negative, 1),
    }


def write_dataset(path: Path, dataset: list[dict[str, object]]) -> None:
    """Write the feature dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "researcher_open_alex_id",
        "advisor_open_alex_id",
        "researcher_name",
        "advisor_name",
        *FEATURE_NAMES,
        "label",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in dataset:
            writer.writerow(
                {
                    "researcher_open_alex_id": row["researcher_open_alex_id"],
                    "advisor_open_alex_id": row["advisor_open_alex_id"],
                    "researcher_name": row["researcher_name"],
                    "advisor_name": row["advisor_name"],
                    **dict(zip(FEATURE_NAMES, row["features"])),
                    "label": row["label"],
                }
            )


def main() -> None:
    """Train, validate, and export relationship predictions."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.input)
    dataset = build_dataset(rows, args.negative_ratio, args.seed)
    if len({row["label"] for row in dataset}) < 2:
        raise ValueError("Training requires both positive and negative examples")

    train_indices, validation_indices = split_dataset(dataset, args.seed)
    features = np.asarray([row["features"] for row in dataset], dtype=np.float32)
    labels = np.asarray([row["label"] for row in dataset], dtype=np.float32)
    mean = features[train_indices].mean(axis=0)
    standard_deviation = features[train_indices].std(axis=0)
    standard_deviation[standard_deviation == 0] = 1.0
    normalized_features = (features - mean) / standard_deviation

    model = RelationshipMLP(len(FEATURE_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    positive_count = max(int(labels[train_indices].sum()), 1)
    negative_count = max(len(train_indices) - positive_count, 1)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative_count / positive_count])
    )
    train_values = torch.tensor(normalized_features[train_indices])
    train_labels = torch.tensor(labels[train_indices])

    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = loss_function(model(train_values), train_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        all_probabilities = torch.sigmoid(
            model(torch.tensor(normalized_features))
        ).numpy()
    validation_probabilities = all_probabilities[validation_indices]
    validation_labels = labels[validation_indices].astype(int)
    validation_predictions = (validation_probabilities >= 0.5).astype(int)
    metrics = calculate_metrics(
        validation_labels, validation_predictions, validation_probabilities
    )
    train_features = normalized_features[train_indices]
    test_features = normalized_features[validation_indices]
    train_labels = labels[train_indices].astype(int)
    test_labels = validation_labels

    baseline_probabilities = (
        features[validation_indices, 0] > 0
    ).astype(float)
    baseline_predictions = baseline_probabilities.astype(int)
    baselines = {
        "coauthorship_rule": calculate_metrics(
            test_labels, baseline_predictions, baseline_probabilities
        )
    }

    logistic_regression = LogisticRegression(
        class_weight="balanced", random_state=args.seed, max_iter=1000
    )
    logistic_regression.fit(train_features, train_labels)
    logistic_probabilities = logistic_regression.predict_proba(test_features)[:, 1]
    baselines["logistic_regression"] = calculate_metrics(
        test_labels,
        (logistic_probabilities >= 0.5).astype(int),
        logistic_probabilities,
    )

    gradient_boosting = GradientBoostingClassifier(random_state=args.seed)
    gradient_boosting.fit(
        train_features,
        train_labels,
        sample_weight=np.where(train_labels == 1, negative_count / positive_count, 1.0),
    )
    gradient_probabilities = gradient_boosting.predict_proba(test_features)[:, 1]
    baselines["gradient_boosting"] = calculate_metrics(
        test_labels,
        (gradient_probabilities >= 0.5).astype(int),
        gradient_probabilities,
    )
    metrics.update(
        {
            "training_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "positive_examples": int(labels.sum()),
            "negative_examples": int((labels == 0).sum()),
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
            "features": FEATURE_NAMES,
            "negative_sampling": "random pairs not present in the input positives",
            "class_weighting": "positive loss weight and balanced logistic regression",
            "baselines": baselines,
        }
    )

    write_dataset(args.dataset, dataset)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "researcher_open_alex_id",
                "advisor_open_alex_id",
                "researcher_name",
                "advisor_name",
                "label",
                "predicted_probability",
                "predicted_label",
                "split",
            ],
        )
        writer.writeheader()
        train_set = set(train_indices)
        for index, row in enumerate(dataset):
            writer.writerow(
                {
                    "researcher_open_alex_id": row["researcher_open_alex_id"],
                    "advisor_open_alex_id": row["advisor_open_alex_id"],
                    "researcher_name": row["researcher_name"],
                    "advisor_name": row["advisor_name"],
                    "label": row["label"],
                    "predicted_probability": f"{all_probabilities[index]:.6f}",
                    "predicted_label": int(all_probabilities[index] >= 0.5),
                    "split": "train" if index in train_set else "validation",
                }
            )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "mean": mean.tolist(),
            "standard_deviation": standard_deviation.tolist(),
            "seed": args.seed,
        },
        args.model,
    )

    print(f"Pares no conjunto: {len(dataset)}")
    print(f"Treinamento: {len(train_indices)} (80%)")
    print(f"Teste: {len(validation_indices)} (20%)")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Métricas: {args.metrics}")
    print(f"Previsões: {args.predictions}")


if __name__ == "__main__":
    main()
