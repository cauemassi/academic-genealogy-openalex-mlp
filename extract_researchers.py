"""Extract a subset of researchers with advisors from the MGP SQLite database."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Sequence


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "mgp.sqlite"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name("researchers_with_advisors.csv")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract a deterministic subset of researchers that have at least one "
            "advisor, starting from a seed researcher_id, and export researcher/"
            "advisor pairs and their institutions to CSV."
        )
    )
    parser.add_argument(
        "seed",
        type=int,
        help="Researcher ID to start from.",
    )
    parser.add_argument(
        "limit",
        type=int,
        help="Number of researchers to extract from the seed onward.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to the SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def ensure_valid_parameters(seed: int, limit: int) -> None:
    """Validate the requested extraction window."""
    if seed <= 0:
        raise ValueError("seed must be greater than zero")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")


def fetch_researcher_advisor_rows(
    connection: sqlite3.Connection, seed: int, limit: int
) -> list[sqlite3.Row]:
    """Fetch researcher/advisor pairs starting at ``seed``."""
    connection.row_factory = sqlite3.Row
    query = """
        WITH selected_researcher_ids AS (
            SELECT
                at.researcher_id
            FROM academic_titles AS at
            JOIN advisors_academic_titles AS aa
                ON aa.academic_title_id = at.academic_title_id
            WHERE at.researcher_id >= ?
            GROUP BY at.researcher_id
            ORDER BY at.researcher_id
            LIMIT ?
        )
        SELECT
            r.researcher_id,
            r.name AS researcher_name,
            adv.researcher_id AS advisor_id,
            adv.name AS advisor_name,
            COALESCE(
                (
                    SELECT REPLACE(
                        GROUP_CONCAT(
                            DISTINCT NULLIF(TRIM(researcher_title.institution), '')
                        ),
                        ',',
                        ', '
                    )
                    FROM academic_titles AS researcher_title
                    WHERE researcher_title.researcher_id = r.researcher_id
                ),
                ''
            ) AS researcher_universities,
            COALESCE(
                (
                    SELECT REPLACE(
                        GROUP_CONCAT(
                            DISTINCT NULLIF(TRIM(advisor_title.institution), '')
                        ),
                        ',',
                        ', '
                    )
                    FROM academic_titles AS advisor_title
                    WHERE advisor_title.researcher_id = adv.researcher_id
                ),
                ''
            ) AS advisor_universities
        FROM selected_researcher_ids AS sri
        JOIN researchers AS r
            ON r.researcher_id = sri.researcher_id
        JOIN academic_titles AS at
            ON at.researcher_id = r.researcher_id
        JOIN advisors_academic_titles AS aa
            ON aa.academic_title_id = at.academic_title_id
        JOIN researchers AS adv
            ON adv.researcher_id = aa.advisor_id
        GROUP BY
            r.researcher_id,
            r.name,
            adv.researcher_id,
            adv.name
        ORDER BY r.researcher_id, adv.researcher_id
    """
    return list(connection.execute(query, (seed, limit)))


def write_csv(rows: Sequence[sqlite3.Row], output_path: Path) -> None:
    """Write the extracted rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "researcher_id",
                "researcher_name",
                "advisor_id",
                "advisor_name",
                "researcher_universities",
                "advisor_universities",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def main() -> None:
    """Run the extraction pipeline."""
    args = parse_args()
    ensure_valid_parameters(args.seed, args.limit)

    if not args.db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.db_path}")

    with sqlite3.connect(args.db_path) as connection:
        rows = fetch_researcher_advisor_rows(connection, args.seed, args.limit)

    write_csv(rows, args.output)
    selected_researchers = len({row["researcher_id"] for row in rows})

    print(
        f"Exported {selected_researchers} researchers and {len(rows)} researcher-advisor "
        f"pairs to {args.output}"
    )


if __name__ == "__main__":
    main()
