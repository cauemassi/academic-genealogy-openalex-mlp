"""Identify researchers from the advisor/advisee CSV in OpenAlex."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "researchers_with_advisors.csv"
DEFAULT_CACHE_PATH = SCRIPT_DIRECTORY / "researchers_with_open_alex_id.csv"
DEFAULT_OUTPUT_PATH = SCRIPT_DIRECTORY / "researchers_with_advisors_and_open_alex_id.csv"
OPENALEX_AUTHORS_ENDPOINT = "https://api.openalex.org/authors"
OPENALEX_API_KEY_ENV = "OPENALEX_API_KEY"
CACHE_FIELDS = [
    "researcher_id",
    "open_alex_id",
    "name",
    "universities",
    "status",
    "last_error",
]
OUTPUT_ID_FIELDS = ["researcher_open_alex_id", "advisor_open_alex_id"]


@dataclass
class Person:
    """A unique researcher or advisor from the input file."""

    researcher_id: str
    name: str
    universities: set[str]


@dataclass(frozen=True)
class MatchResult:
    """The result of an OpenAlex author identification attempt."""

    open_alex_id: str
    status: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Identify advisor and advisee IDs in OpenAlex incrementally."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between OpenAlex requests in seconds (default: 0.2).",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    """Normalize accents, case, punctuation, and whitespace."""
    without_accents = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        char for char in without_accents if not unicodedata.combining(char)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_accents.casefold()))


def normalize_name(value: str) -> str:
    """Normalize a person name for comparisons."""
    return normalize_text(value)


def name_similarity(left: str, right: str) -> float:
    """Compare names while tolerating initials and parenthetical names."""
    left_tokens = normalize_name(left).split()
    right_tokens = normalize_name(right).split()
    left_initials = "".join(token[0] for token in left_tokens if token)
    right_initials = "".join(token[0] for token in right_tokens if token)
    if left_initials == right_initials:
        return 1.0
    left_long = {token for token in left_tokens if len(token) > 1}
    right_long = {token for token in right_tokens if len(token) > 1}
    if left_long and right_long:
        overlap = len(left_long & right_long) / max(len(left_long), len(right_long))
    else:
        overlap = 0.0
    ratio = SequenceMatcher(
        None, " ".join(left_tokens), " ".join(right_tokens)
    ).ratio()
    return max(overlap, ratio)


def names_are_equivalent(left: str, right: str) -> bool:
    """Return whether names are equal after ignoring initials."""
    left_tokens = normalize_name(left).split()
    right_tokens = normalize_name(right).split()
    left_long = {token for token in left_tokens if len(token) > 1}
    right_long = {token for token in right_tokens if len(token) > 1}
    if left_long and left_long == right_long:
        return True
    left_initials = "".join(token[0] for token in left_tokens if token)
    right_initials = "".join(token[0] for token in right_tokens if token)
    return bool(left_initials and left_initials == right_initials)


def split_universities(value: str) -> set[str]:
    """Split the institution list stored in the input CSV."""
    return {item.strip() for item in value.split(",") if item.strip()}


def load_input(path: Path) -> tuple[list[dict[str, str]], dict[str, Person]]:
    """Load relationship rows and collect unique people."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "researcher_id",
            "researcher_name",
            "advisor_id",
            "advisor_name",
            "researcher_universities",
            "advisor_universities",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    people: dict[str, Person] = {}
    for row in rows:
        researcher_universities = split_universities(
            row["researcher_universities"]
        )
        advisor_universities = split_universities(row["advisor_universities"])
        pair_universities = researcher_universities | advisor_universities

        researcher = people.setdefault(
            row["researcher_id"],
            Person(row["researcher_id"], row["researcher_name"].strip(), set()),
        )
        researcher.universities.update(pair_universities)

        advisor = people.setdefault(
            row["advisor_id"],
            Person(row["advisor_id"], row["advisor_name"].strip(), set()),
        )
        advisor.universities.update(pair_universities)
    return rows, people


def load_cache(path: Path) -> dict[str, dict[str, str]]:
    """Load the local researcher cache, if it exists."""
    if not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(CACHE_FIELDS[:3]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Cache CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        return {
            row["researcher_id"]: {
                field: row.get(field, "")
                for field in CACHE_FIELDS
            }
            for row in reader
            if row.get("researcher_id")
        }


def write_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    """Persist the researcher cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        writer.writerows(cache[key] for key in sorted(cache, key=str))


def get_openalex_api_key() -> str:
    """Read the OpenAlex API key from the environment."""
    api_key = os.environ.get(OPENALEX_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set {OPENALEX_API_KEY_ENV} before querying the OpenAlex API"
        )
    return api_key


def build_author_search_url(name: str) -> str:
    """Build an OpenAlex author search URL."""
    params = {
        "search": name,
        "per-page": "25",
        "select": (
            "id,display_name,last_known_institutions,affiliations,"
            "x_concepts,topics"
        ),
        "api_key": get_openalex_api_key(),
    }
    return f"{OPENALEX_AUTHORS_ENDPOINT}?{urlencode(params, quote_via=quote)}"


def fetch_author_candidates(name: str) -> list[dict[str, object]]:
    """Fetch OpenAlex author candidates by name."""
    request = Request(
        build_author_search_url(name),
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
    return [candidate for candidate in payload.get("results", []) if candidate.get("id")]


def candidate_institutions(candidate: dict[str, object]) -> list[str]:
    """Extract institution names from an OpenAlex author candidate."""
    institutions: list[str] = []
    for key in ("last_known_institutions", "affiliations"):
        values = candidate.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            institution = ""
            if isinstance(value, dict):
                institution = value.get("display_name", "")
                nested_institution = value.get("institution")
                if isinstance(nested_institution, dict):
                    institution = nested_institution.get("display_name", institution)
            if isinstance(institution, str) and institution:
                institutions.append(institution)
    return institutions


def institution_matches(
    person_universities: Iterable[str], candidate: dict[str, object]
) -> bool:
    """Return whether any input institution matches a candidate institution."""
    candidate_names = [normalize_text(item) for item in candidate_institutions(candidate)]
    for university in person_universities:
        normalized = normalize_text(university)
        if not normalized:
            continue
        if any(
            normalized == candidate_name
            or normalized in candidate_name
            or candidate_name in normalized
            for candidate_name in candidate_names
        ):
            return True
    return False


def is_mathematics_candidate(candidate: dict[str, object]) -> bool:
    """Return whether an author has a mathematics-related OpenAlex area."""
    area_names: list[str] = []
    for key in ("x_concepts", "topics"):
        values = candidate.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            display_name = value.get("display_name")
            if isinstance(display_name, str):
                area_names.append(display_name)
            for nested_key in ("domain", "field", "subfield"):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    nested_name = nested.get("display_name")
                    if isinstance(nested_name, str):
                        area_names.append(nested_name)
    return any("mathemat" in normalize_text(name) for name in area_names)


def identify_person(person: Person) -> MatchResult:
    """Identify one person, requiring institution agreement when available."""
    candidates = fetch_author_candidates(person.name)
    if not candidates:
        return MatchResult("", "not_found", "no OpenAlex author candidates")

    normalized_person_name = normalize_name(person.name)
    scored: list[tuple[float, bool, dict[str, object]]] = []
    for candidate in candidates:
        candidate_name = str(candidate.get("display_name", ""))
        name_score = name_similarity(normalized_person_name, candidate_name)
        scored.append(
            (name_score, institution_matches(person.universities, candidate), candidate)
        )

    institution_scored = [item for item in scored if item[1]]
    eligible = institution_scored if person.universities else scored
    if not eligible:
        mathematics_candidates = [
            item for item in scored if is_mathematics_candidate(item[2])
        ]
        if len(mathematics_candidates) == 1:
            candidate = mathematics_candidates[0]
            candidate_name = str(candidate[2].get("display_name", ""))
            if names_are_equivalent(normalized_person_name, candidate_name):
                return MatchResult(
                    str(candidate[2]["id"]),
                    "found",
                )
        return MatchResult("", "ambiguous", "no candidate matches the institution")

    eligible.sort(key=lambda item: item[0], reverse=True)

    best = eligible[0]
    second_score = eligible[1][0] if len(eligible) > 1 else 0.0
    exact_name = names_are_equivalent(
        normalized_person_name, str(best[2].get("display_name", ""))
    )
    if not exact_name and best[0] < 0.82:
        return MatchResult("", "ambiguous", "candidate name is not sufficiently similar")
    if len(eligible) > 1 and not exact_name and best[0] - second_score < 0.08:
        return MatchResult("", "ambiguous", "multiple similarly ranked candidates")

    return MatchResult(str(best[2]["id"]), "found")


def update_cache(
    cache: dict[str, dict[str, str]],
    people: dict[str, Person],
    cache_path: Path,
    sleep_seconds: float,
) -> tuple[int, int, int, int]:
    """Identify missing people and return cache/query/found/not-found counts."""
    recovered_from_cache = sum(
        bool(cache.get(researcher_id, {}).get("open_alex_id"))
        for researcher_id in people
    )
    new_queries = 0
    found = 0
    not_found = 0

    for researcher_id, person in people.items():
        cached = cache.get(researcher_id, {})
        if cached.get("open_alex_id"):
            continue

        new_queries += 1
        try:
            result = identify_person(person)
        except RuntimeError as exc:
            result = MatchResult("", "error", str(exc))
        if result.open_alex_id:
            found += 1
        else:
            not_found += 1

        cache[researcher_id] = {
            "researcher_id": researcher_id,
            "open_alex_id": result.open_alex_id,
            "name": person.name,
            "universities": ", ".join(sorted(person.universities)),
            "status": result.status,
            "last_error": result.error,
        }
        write_cache(cache_path, cache)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return recovered_from_cache, new_queries, found, not_found


def append_output(
    rows: list[dict[str, str]],
    cache: dict[str, dict[str, str]],
    path: Path,
) -> int:
    """Append only new input relationship rows with their OpenAlex IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return 0

    fieldnames = list(rows[0]) + OUTPUT_ID_FIELDS
    existing_keys: set[tuple[str, str]] = set()
    output_exists = path.exists() and path.stat().st_size > 0

    if output_exists:
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = set(fieldnames) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Output CSV is missing required columns: {', '.join(sorted(missing))}"
                )
            existing_keys = {
                (row["researcher_id"], row["advisor_id"])
                for row in reader
            }

    new_rows: list[dict[str, str]] = []
    for row in rows:
        key = (row["researcher_id"], row["advisor_id"])
        if key in existing_keys:
            continue
        output_row = dict(row)
        output_row["researcher_open_alex_id"] = cache.get(
            row["researcher_id"], {}
        ).get("open_alex_id", "")
        output_row["advisor_open_alex_id"] = cache.get(
            row["advisor_id"], {}
        ).get("open_alex_id", "")
        new_rows.append(output_row)
        existing_keys.add(key)

    if not new_rows:
        return 0

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not output_exists:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(new_rows)


def main() -> None:
    """Run the incremental OpenAlex identification pipeline."""
    args = parse_args()
    rows, people = load_input(args.input)
    cache = load_cache(args.cache)

    recovered, queries, found, not_found = update_cache(
        cache, people, args.cache, args.sleep
    )
    write_cache(args.cache, cache)
    new_records = append_output(rows, cache, args.output)

    print(f"Pesquisadores processados: {len(people)}")
    print(f"Recuperados do cache: {recovered}")
    print(f"Novas consultas ao OpenAlex: {queries}")
    print(f"Pesquisadores encontrados: {found}")
    print(f"Pesquisadores não encontrados: {not_found}")
    print(f"Novos registros adicionados: {new_records}")
    print(f"Cache: {args.cache}")
    print(f"Saída: {args.output}")


if __name__ == "__main__":
    main()
