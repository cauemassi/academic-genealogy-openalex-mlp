"""Extract complete advisor/advisee OpenAlex relationships incrementally."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    SCRIPT_DIRECTORY / "researchers_with_advisors_and_open_alex_id.csv"
)
DEFAULT_OUTPUT_PATH = (
    SCRIPT_DIRECTORY / "researchers_with_advisors_and_open_alex_id_complete.csv"
)
REQUIRED_COLUMNS = {
    "researcher_id",
    "advisor_id",
    "researcher_open_alex_id",
    "advisor_open_alex_id",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Append only advisor/advisee rows that contain both OpenAlex IDs."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load source rows and preserve their column order."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        return list(reader), fieldnames


def relationship_key(row: dict[str, str]) -> tuple[str, str]:
    """Build the unique researcher/advisor relationship key."""
    return row["researcher_id"], row["advisor_id"]


def append_complete_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    output_path: Path,
) -> int:
    """Append complete relationships that are not already in the output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[tuple[str, str]] = set()
    output_exists = output_path.exists() and output_path.stat().st_size > 0

    if output_exists:
        with output_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = set(fieldnames) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    "Output CSV is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            existing_keys = {
                relationship_key(row)
                for row in reader
                if row.get("researcher_id") and row.get("advisor_id")
            }

    new_rows: list[dict[str, str]] = []
    for row in rows:
        if not row.get("researcher_open_alex_id") or not row.get(
            "advisor_open_alex_id"
        ):
            continue

        key = relationship_key(row)
        if key in existing_keys:
            continue
        new_rows.append(row)
        existing_keys.add(key)

    if not new_rows:
        return 0

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not output_exists:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(new_rows)


def main() -> None:
    """Run the incremental complete-relationship extraction."""
    args = parse_args()
    rows, fieldnames = load_rows(args.input)
    inserted = append_complete_rows(rows, fieldnames, args.output)

    complete_rows = sum(
        bool(row.get("researcher_open_alex_id"))
        and bool(row.get("advisor_open_alex_id"))
        for row in rows
    )
    print(f"Linhas completas encontradas: {complete_rows}")
    print(f"Novas tuplas inseridas: {inserted}")
    print(f"Arquivo de saída: {args.output}")


if __name__ == "__main__":
    main()
