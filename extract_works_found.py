"""Extract rows that contain an OpenAlex work."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "researchers_advisors_works.csv"
DEFAULT_OUTPUT_PATH = SCRIPT_DIRECTORY / "researchers_advisors_works_found.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract only rows with a work ID."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def extract_rows(input_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read rows with a non-empty work ID."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "work_id" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain a 'work_id' column")
        rows = [
            row
            for row in reader
            if row.get("work_id", "").strip()
        ]
        return rows, reader.fieldnames or []


def write_csv(
    output_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write extracted rows preserving the input columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the work-row extraction."""
    args = parse_args()
    rows, fieldnames = extract_rows(args.input)
    write_csv(args.output, rows, fieldnames)
    print(f"Linhas com trabalho: {len(rows)}")
    print(f"Arquivo gerado: {args.output}")


if __name__ == "__main__":
    main()
