"""Synchronize relationship OpenAlex IDs from the researcher cache."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CACHE_PATH = SCRIPT_DIRECTORY / "researchers_with_open_alex_id.csv"
DEFAULT_SOURCE_PATH = (
    SCRIPT_DIRECTORY / "researchers_with_advisors_and_open_alex_id.csv"
)
DEFAULT_TARGET_PATH = (
    SCRIPT_DIRECTORY / "researchers_with_advisors_and_open_alex_id_complete.csv"
)
ID_FIELDS = ("researcher_open_alex_id", "advisor_open_alex_id")
KEY_FIELDS = ("researcher_id", "advisor_id")
REQUIRED_SOURCE_FIELDS = {
    "researcher_id",
    "advisor_id",
    *ID_FIELDS,
}
REQUIRED_CACHE_FIELDS = {"researcher_id", "open_alex_id"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fill missing relationship OpenAlex IDs from the local cache."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH)
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, str]:
    """Load non-empty OpenAlex IDs indexed by the MGP researcher ID."""
    if not path.exists():
        raise FileNotFoundError(f"Cache CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_CACHE_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Cache CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        return {
            row["researcher_id"]: row["open_alex_id"]
            for row in reader
            if row.get("researcher_id") and row.get("open_alex_id")
        }


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load a CSV and preserve its column order."""
    if not path.exists():
        return [], []

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), reader.fieldnames or []


def relationship_key(row: dict[str, str]) -> tuple[str, str]:
    """Return the unique relationship key."""
    return row["researcher_id"], row["advisor_id"]


def synchronize(
    cache: dict[str, str],
    source_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int, int]:
    """Fill missing IDs and append source relationships not in the target."""
    rows_by_key = {
        relationship_key(row): dict(row)
        for row in target_rows
        if row.get("researcher_id") and row.get("advisor_id")
    }
    existing_keys = set(rows_by_key)
    new_keys: set[tuple[str, str]] = set()
    filled_values = 0

    for source_row in source_rows:
        key = relationship_key(source_row)
        row = rows_by_key.setdefault(key, dict(source_row))
        if key not in existing_keys:
            new_keys.add(key)
            existing_keys.add(key)

        for id_field, person_field in (
            ("researcher_open_alex_id", "researcher_id"),
            ("advisor_open_alex_id", "advisor_id"),
        ):
            if row.get(id_field):
                continue
            replacement = source_row.get(id_field) or cache.get(
                source_row[person_field], ""
            )
            if replacement:
                row[id_field] = replacement
                filled_values += 1

    complete_rows = [
        row
        for row in rows_by_key.values()
        if row.get("researcher_open_alex_id")
        and row.get("advisor_open_alex_id")
    ]
    appended_rows = sum(
        relationship_key(row) in new_keys for row in complete_rows
    )
    return complete_rows, filled_values, appended_rows


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write the synchronized relationship CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the cache-to-relationship synchronization."""
    args = parse_args()
    cache = load_cache(args.cache)
    source_rows, source_fields = load_csv(args.source)
    target_rows, target_fields = load_csv(args.target)

    missing_source_fields = REQUIRED_SOURCE_FIELDS - set(source_fields)
    if missing_source_fields:
        raise ValueError(
            "Source CSV is missing required columns: "
            + ", ".join(sorted(missing_source_fields))
        )

    fieldnames = target_fields or source_fields
    missing_target_fields = set(source_fields) - set(fieldnames)
    if missing_target_fields:
        raise ValueError(
            "Target CSV is missing required columns: "
            + ", ".join(sorted(missing_target_fields))
        )

    rows, filled_values, appended_rows = synchronize(
        cache, source_rows, target_rows
    )
    write_csv(args.target, rows, fieldnames)

    print(f"Linhas sincronizadas: {len(rows)}")
    print(f"IDs preenchidos: {filled_values}")
    print(f"Novas relações adicionadas: {appended_rows}")
    print(f"Arquivo atualizado: {args.target}")


if __name__ == "__main__":
    main()
