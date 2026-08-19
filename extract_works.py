"""Query OpenAlex for advisor/advisee pairs from a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().with_name("researchers_with_advisors.csv")
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name(
    "researchers_openalex_works.csv"
)
OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
OPENALEX_WORKS_BROWSER_URL = "https://openalex.org/works"
OPENALEX_API_KEY_ENV = "OPENALEX_API_KEY"


@dataclass(frozen=True)
class ResearcherPair:
    """A researcher/advisor pair loaded from the input CSV."""

    advisor_name: str
    researcher_name: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Read researcher/advisor pairs from a CSV and search OpenAlex works "
            "for each pair."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Delay between requests in seconds (default: 0).",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    """Normalize a person name loaded from CSV."""
    return " ".join(value.split())


def load_pairs(input_path: Path) -> list[ResearcherPair]:
    """Load unique advisor/researcher pairs from the input CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    pairs: list[ResearcherPair] = []
    seen: set[tuple[str, str]] = set()

    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "advisor_name" not in reader.fieldnames or "researcher_name" not in reader.fieldnames:
            raise ValueError(
                "Input CSV must contain 'advisor_name' and 'researcher_name' columns"
            )

        for row in reader:
            advisor_name = normalize_name(row["advisor_name"])
            researcher_name = normalize_name(row["researcher_name"])
            if not advisor_name or not researcher_name:
                continue

            key = (advisor_name, researcher_name)
            if key in seen:
                continue

            seen.add(key)
            pairs.append(
                ResearcherPair(
                    advisor_name=advisor_name,
                    researcher_name=researcher_name,
                )
            )

    return pairs


def get_openalex_api_key() -> str:
    """Read the OpenAlex API key from the environment."""
    api_key = os.environ.get(OPENALEX_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set {OPENALEX_API_KEY_ENV} before querying the OpenAlex API"
        )
    return api_key


def build_search_url(advisor_name: str, researcher_name: str) -> str:
    """Build the OpenAlex API works search URL for a pair."""
    query = f"{advisor_name} {researcher_name}"
    params: dict[str, str] = {
        "search": query,
        "per_page": "200",
        "cursor": "*",
        "select": "id",
        "api_key": get_openalex_api_key(),
    }
    return f"{OPENALEX_WORKS_ENDPOINT}?{urlencode(params, quote_via=quote)}"


def build_browser_search_url(advisor_name: str, researcher_name: str) -> str:
    """Build the browser-facing OpenAlex works search URL for a pair."""
    query = f"{advisor_name} {researcher_name}"
    return (
        f"{OPENALEX_WORKS_BROWSER_URL}?"
        f"{urlencode({'search': query}, quote_via=quote)}"
    )


def fetch_work_results(search_url: str) -> tuple[int, list[str]]:
    """Fetch the OpenAlex result count and links for a search URL."""
    current_url = search_url
    work_links: list[str] = []
    works_found: int | None = None
    try:
        while current_url:
            request = Request(current_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "error" in payload:
                raise RuntimeError(f"OpenAlex error: {payload['error']}")

            if works_found is None:
                works_found = int(payload["meta"]["count"])
            work_links.extend(
                work["id"]
                for work in payload.get("results", [])
                if isinstance(work.get("id"), str) and work["id"]
            )

            next_cursor = payload.get("meta", {}).get("next_cursor")
            if not next_cursor or not payload.get("results"):
                break
            split_url = urlsplit(search_url)
            params = dict(parse_qsl(split_url.query, keep_blank_values=True))
            params["cursor"] = next_cursor
            current_url = urlunsplit(
                (
                    split_url.scheme,
                    split_url.netloc,
                    split_url.path,
                    urlencode(params, quote_via=quote),
                    split_url.fragment,
                )
            )

        return works_found or 0, work_links
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}
        message = payload.get("message") or payload.get("error") or exc.reason
        retry_after = payload.get("retryAfter") or exc.headers.get("Retry-After")
        raise RuntimeError(
            f"OpenAlex request failed with HTTP {exc.code}: {message}"
            + (f" (retry after {retry_after}s)" if retry_after else "")
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAlex request failed: {exc.reason}") from exc


def process_pairs(
    pairs: Iterable[ResearcherPair],
    sleep_seconds: float,
) -> list[dict[str, object]]:
    """Query OpenAlex for each researcher/advisor pair."""
    pair_list = list(pairs)
    total_pairs = len(pair_list)
    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pair_list, start=1):
        print(
            f"Processing OpenAlex search {index}/{total_pairs}: "
            f"{pair.advisor_name} + {pair.researcher_name}",
            flush=True,
        )
        api_search_url = build_search_url(
            pair.advisor_name, pair.researcher_name
        )
        works_found, work_links = fetch_work_results(api_search_url)
        rows.append(
            {
                "advisor_name": pair.advisor_name,
                "researcher_name": pair.researcher_name,
                "works_found": works_found,
                "work_links": " | ".join(work_links),
                "search_url": build_browser_search_url(
                    pair.advisor_name, pair.researcher_name
                ),
            }
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write the OpenAlex results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "advisor_name",
                "researcher_name",
                "works_found",
                "work_links",
                "search_url",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the OpenAlex search pipeline."""
    args = parse_args()
    pairs = load_pairs(args.input)
    rows = process_pairs(pairs, args.sleep)
    write_csv(rows, args.output)

    print(f"Exported {len(rows)} OpenAlex search rows to {args.output}")


if __name__ == "__main__":
    main()
