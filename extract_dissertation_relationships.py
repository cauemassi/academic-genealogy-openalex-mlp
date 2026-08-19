"""Extract works co-authored by advisor/advisee pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    SCRIPT_DIRECTORY / "researchers_with_advisors_and_open_alex_id_complete.csv"
)
DEFAULT_OUTPUT_PATH = (
    SCRIPT_DIRECTORY / "researchers_advisors_works.csv"
)
OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
OPENALEX_API_KEY_ENV = "OPENALEX_API_KEY"
OUTPUT_FIELDS = [
    "researcher_id",
    "researcher_name",
    "advisor_id",
    "advisor_name",
    "researcher_open_alex_id",
    "advisor_open_alex_id",
    "work_id",
    "work_title",
    "publication_year",
    "doi",
    "work_url",
    "query_status",
]
REQUIRED_INPUT_FIELDS = {
    "researcher_id",
    "researcher_name",
    "advisor_id",
    "advisor_name",
    "researcher_open_alex_id",
    "advisor_open_alex_id",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Find works containing both the advisee and advisor "
            "OpenAlex authors."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between OpenAlex requests in seconds (default: 0.2).",
    )
    return parser.parse_args()


def get_api_key() -> str:
    """Read the OpenAlex API key from the environment."""
    api_key = os.environ.get(OPENALEX_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set {OPENALEX_API_KEY_ENV} before querying the OpenAlex API"
        )
    return api_key


def load_pairs(path: Path) -> list[dict[str, str]]:
    """Load unique complete researcher/advisor pairs from the input."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_INPUT_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        pairs: dict[tuple[str, str], dict[str, str]] = {}
        for row in reader:
            researcher_open_alex_id = row["researcher_open_alex_id"].strip()
            advisor_open_alex_id = row["advisor_open_alex_id"].strip()
            if not researcher_open_alex_id or not advisor_open_alex_id:
                continue
            key = (researcher_open_alex_id, advisor_open_alex_id)
            pairs.setdefault(key, row)
        return list(pairs.values())


def load_processed_pairs(path: Path) -> set[tuple[str, str]]:
    """Load relationship pairs already represented in the output."""
    if not path.exists() or path.stat().st_size == 0:
        return set()

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = {
            "researcher_open_alex_id",
            "advisor_open_alex_id",
        } - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Output CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        return {
            (row["researcher_open_alex_id"], row["advisor_open_alex_id"])
            for row in reader
            if row.get("researcher_open_alex_id")
            and row.get("advisor_open_alex_id")
        }


def build_works_url(researcher_open_alex_id: str, cursor: str) -> str:
    """Build a works query for one researcher."""
    author_id = researcher_open_alex_id.rsplit("/", 1)[-1]
    params = {
        "filter": f"authorships.author.id:{author_id}",
        "per-page": "200",
        "cursor": cursor,
        "select": "id,display_name,publication_year,doi,authorships",
        "api_key": get_api_key(),
    }
    return f"{OPENALEX_WORKS_ENDPOINT}?{urlencode(params, quote_via=quote)}"


def fetch_works(
    researcher_open_alex_id: str,
    advisor_open_alex_id: str,
) -> list[dict[str, object]]:
    """Fetch works that contain both OpenAlex author IDs."""
    works: list[dict[str, object]] = []
    cursor = "*"

    while cursor:
        request = Request(
            build_works_url(researcher_open_alex_id, cursor),
            headers={"User-Agent": "data-organizer-openalex/1.0"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(
                f"OpenAlex request failed with HTTP {exc.code}: {body or exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAlex request failed: {exc.reason}") from exc

        if "error" in payload:
            raise RuntimeError(f"OpenAlex error: {payload['error']}")

        for work in payload.get("results", []):
            author_ids = {
                str(authorship.get("author", {}).get("id", "")).lower()
                for authorship in work.get("authorships", [])
                if isinstance(authorship, dict)
                and isinstance(authorship.get("author"), dict)
            }
            if advisor_open_alex_id.lower() in author_ids:
                works.append(work)

        cursor = payload.get("meta", {}).get("next_cursor")
        if not payload.get("results"):
            break

    return works


def work_row(pair: dict[str, str], work: dict[str, object]) -> dict[str, str]:
    """Convert an OpenAlex work to an output row."""
    work_id = str(work.get("id", ""))
    return {
        "researcher_id": pair["researcher_id"],
        "researcher_name": pair["researcher_name"],
        "advisor_id": pair["advisor_id"],
        "advisor_name": pair["advisor_name"],
        "researcher_open_alex_id": pair["researcher_open_alex_id"],
        "advisor_open_alex_id": pair["advisor_open_alex_id"],
        "work_id": work_id,
        "work_title": str(work.get("display_name", "")),
        "publication_year": str(work.get("publication_year", "")),
        "doi": str(work.get("doi", "")),
        "work_url": work_id,
        "query_status": "found",
    }


def no_result_row(pair: dict[str, str]) -> dict[str, str]:
    """Create a marker row so a pair without results is not queried again."""
    return {
        "researcher_id": pair["researcher_id"],
        "researcher_name": pair["researcher_name"],
        "advisor_id": pair["advisor_id"],
        "advisor_name": pair["advisor_name"],
        "researcher_open_alex_id": pair["researcher_open_alex_id"],
        "advisor_open_alex_id": pair["advisor_open_alex_id"],
        "work_id": "",
        "work_title": "",
        "publication_year": "",
        "doi": "",
        "work_url": "",
        "query_status": "no_results",
    }


def append_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Append result rows to the output file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    output_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        if not output_exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the incremental works extraction."""
    args = parse_args()
    pairs = load_pairs(args.input)
    processed_pairs = load_processed_pairs(args.output)
    pending_pairs = [
        pair
        for pair in pairs
        if (
            pair["researcher_open_alex_id"],
            pair["advisor_open_alex_id"],
        )
        not in processed_pairs
    ]

    inserted = 0
    errors = 0
    for pair in pending_pairs:
        print(
            "Consultando dissertações: "
            f"{pair['researcher_name']} + {pair['advisor_name']}",
            flush=True,
        )
        try:
            works = fetch_works(
                pair["researcher_open_alex_id"],
                pair["advisor_open_alex_id"],
            )
        except RuntimeError as exc:
            errors += 1
            print(f"Erro: {exc}", flush=True)
            continue

        rows = [work_row(pair, work) for work in works] or [no_result_row(pair)]
        append_rows(args.output, rows)
        inserted += len(rows)
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Pares completos encontrados: {len(pairs)}")
    print(f"Pares já processados: {len(pairs) - len(pending_pairs)}")
    print(f"Novos registros inseridos: {inserted}")
    print(f"Erros: {errors}")
    print(f"Arquivo de saída: {args.output}")


if __name__ == "__main__":
    main()
