"""Data and annotation contracts for hallucination-localization pilots.

Correctness is available to the deterministic sampler and private lineage, but
is deliberately absent from model-visible annotation items.  A model cites an
exact response substring; a separate tokenizer-backed mapper binds that span
to the frozen output-token sequence.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CLAIM_STATUSES = {
    "supported",
    "contradicted",
    "unsupported",
    "non_claim",
    "uncertain",
}
PROBLEM_STATUSES = {"contradicted", "unsupported"}
PATH_STATUSES = {"hallucinated", "clean", "uncertain"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
ANNOTATION_KEYS = {
    "item_id",
    "claim_reviews",
    "path_status",
    "earliest_problem_claim_index",
    "confidence",
    "summary",
}
CLAIM_KEYS = {"claim_text", "occurrence", "status", "reason"}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain one JSON object")
            rows.append(value)
    return rows


def atomic_write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def atomic_write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(output)


def _require_source_row(row: Mapping[str, Any]) -> None:
    required = {
        "id",
        "query_id",
        "candidate_index",
        "correctness",
        "question",
        "response",
        "prompt_token_ids",
        "output_token_ids",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Source row is missing required fields: {sorted(missing)}")
    if isinstance(row["correctness"], bool) or row["correctness"] not in (0, 1):
        raise ValueError("Source correctness must be numeric 0 or 1")
    if not isinstance(row["response"], str) or not row["response"].strip():
        raise ValueError("Source response must be a non-empty string")
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError("Source question must be a non-empty string")
    if not isinstance(row["output_token_ids"], list) or not row["output_token_ids"]:
        raise ValueError("Source output_token_ids must be a non-empty list")


def _length_bins(rows: Sequence[Mapping[str, Any]], bins: int) -> dict[str, int]:
    if bins <= 0:
        raise ValueError("Length-bin count must be positive")
    ordered = sorted(
        rows,
        key=lambda row: (len(row["output_token_ids"]), str(row["id"])),
    )
    return {
        str(row["id"]): min(bins - 1, index * bins // len(ordered))
        for index, row in enumerate(ordered)
    }


def _priority(seed: int, namespace: str, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{row_id}".encode("utf-8")).hexdigest()


def select_stratified_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_query_ids: set[str],
    seed: int,
    per_correctness: int = 32,
    length_bins: int = 4,
) -> list[dict[str, Any]]:
    """Select one row per query with equal correctness and length-bin counts."""

    if per_correctness <= 0 or per_correctness % length_bins:
        raise ValueError("per_correctness must be positive and divisible by length_bins")
    eligible: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        _require_source_row(row)
        row_id = str(row["id"])
        if row_id in seen_ids:
            raise ValueError(f"Duplicate source row id: {row_id}")
        seen_ids.add(row_id)
        if str(row["query_id"]) not in excluded_query_ids:
            eligible.append(row)

    selected: list[dict[str, Any]] = []
    selected_queries: set[str] = set()
    target_per_bin = per_correctness // length_bins
    # The minority incorrect stratum goes first so query uniqueness cannot make
    # its deterministic quota harder to satisfy.
    for correctness in (0, 1):
        stratum = [row for row in eligible if int(row["correctness"]) == correctness]
        if len(stratum) < per_correctness:
            raise ValueError(f"Correctness={correctness} has only {len(stratum)} eligible rows")
        bins_by_id = _length_bins(stratum, length_bins)
        for length_bin in range(length_bins):
            candidates = [
                row
                for row in stratum
                if bins_by_id[str(row["id"])] == length_bin
            ]
            candidates.sort(
                key=lambda row: _priority(seed, f"select-{correctness}-{length_bin}", str(row["id"]))
            )
            chosen = 0
            for row in candidates:
                query_id = str(row["query_id"])
                if query_id in selected_queries:
                    continue
                record = dict(row)
                record["selection_correctness"] = correctness
                record["selection_length_bin"] = length_bin
                record["selection_priority"] = _priority(
                    seed,
                    f"select-{correctness}-{length_bin}",
                    str(row["id"]),
                )
                selected.append(record)
                selected_queries.add(query_id)
                chosen += 1
                if chosen == target_per_bin:
                    break
            if chosen != target_per_bin:
                raise ValueError(
                    f"Could select only {chosen}/{target_per_bin} rows for "
                    f"correctness={correctness}, length_bin={length_bin}"
                )

    expected = 2 * per_correctness
    if len(selected) != expected or len(selected_queries) != expected:
        raise ValueError("Stratified selection count/query uniqueness drifted")
    selected.sort(key=lambda row: _priority(seed, "blind-order", str(row["id"])))
    return selected


def make_item_id(source_row_id: str, *, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{source_row_id}".encode("utf-8")).hexdigest()
    return f"HLA-{digest[:16]}"


def build_annotation_records(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    source_manifest: str,
    source_manifest_sha256: str,
    protocol_sha256: str,
    item_salt: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return private selection, blind items, and identity/token lineage."""

    selection: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    for order, row in enumerate(selected_rows):
        _require_source_row(row)
        item_id = make_item_id(str(row["id"]), salt=item_salt)
        if item_id in seen_item_ids:
            raise ValueError("Hallucination annotation item-id collision")
        seen_item_ids.add(item_id)
        output_token_ids = list(row["output_token_ids"])
        prompt_token_ids = list(row["prompt_token_ids"])
        response = str(row["response"])
        question = str(row["question"])
        common = {
            "schema_version": "clir-hallucination-selection-row-v1",
            "item_id": item_id,
            "selection_order": order,
            "source_id": str(row["id"]),
            "query_id": str(row["query_id"]),
            "candidate_index": int(row["candidate_index"]),
            "correctness_stratum": int(row["selection_correctness"]),
            "length_bin": int(row["selection_length_bin"]),
            "output_token_count": len(output_token_ids),
            "selection_priority": str(row["selection_priority"]),
        }
        selection.append(common)
        items.append(
            {
                "schema_version": "clir-hallucination-annotation-item-v1",
                "item_id": item_id,
                "problem": question,
                "trajectory": response,
            }
        )
        lineage.append(
            {
                "schema_version": "clir-hallucination-annotation-lineage-v1",
                "item_id": item_id,
                "source_id": str(row["id"]),
                "query_id": str(row["query_id"]),
                "candidate_index": int(row["candidate_index"]),
                "correctness": int(row["correctness"]),
                "problem_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "trajectory_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "prompt_token_ids_sha256": canonical_sha256(prompt_token_ids),
                "output_token_ids_sha256": canonical_sha256(output_token_ids),
                "output_token_count": len(output_token_ids),
                "source_manifest": source_manifest,
                "source_manifest_sha256": source_manifest_sha256,
                "protocol_sha256": protocol_sha256,
            }
        )
    return selection, items, lineage


def locate_occurrence(text: str, quote: str, occurrence: int) -> tuple[int, int]:
    if not isinstance(quote, str) or not quote:
        raise ValueError("claim_text must be a non-empty exact substring")
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 0:
        raise ValueError("claim occurrence must be a zero-based non-negative integer")
    starts: list[int] = []
    cursor = 0
    while True:
        position = text.find(quote, cursor)
        if position < 0:
            break
        starts.append(position)
        cursor = position + 1
    if occurrence >= len(starts):
        raise ValueError(
            f"claim_text occurrence {occurrence} is absent; found {len(starts)} occurrence(s)"
        )
    start = starts[occurrence]
    return start, start + len(quote)


def validate_annotation(
    annotation: Mapping[str, Any],
    item: Mapping[str, Any],
) -> list[tuple[int, int]]:
    """Validate one raw annotation and return exact claim character spans."""

    if set(annotation) != ANNOTATION_KEYS:
        raise ValueError(
            f"Annotation keys differ from schema: {sorted(set(annotation) ^ ANNOTATION_KEYS)}"
        )
    if annotation["item_id"] != item["item_id"]:
        raise ValueError("Annotation item_id drifted")
    path_status = annotation["path_status"]
    if path_status not in PATH_STATUSES:
        raise ValueError("Invalid path_status")
    if annotation["confidence"] not in CONFIDENCE_LEVELS:
        raise ValueError("Invalid annotation confidence")
    if not isinstance(annotation["summary"], str) or len(annotation["summary"].strip()) < 12:
        raise ValueError("Annotation summary must be evidence-specific")
    claims = annotation["claim_reviews"]
    if not isinstance(claims, list) or not claims:
        raise ValueError("claim_reviews must be a non-empty list")
    trajectory = item["trajectory"]
    if not isinstance(trajectory, str):
        raise ValueError("Annotation item trajectory must be a string")

    spans: list[tuple[int, int]] = []
    seen: set[tuple[str, int]] = set()
    statuses: list[str] = []
    for claim_index, claim in enumerate(claims):
        if not isinstance(claim, Mapping) or set(claim) != CLAIM_KEYS:
            raise ValueError(f"claim_reviews[{claim_index}] keys differ from schema")
        status = claim["status"]
        if status not in CLAIM_STATUSES:
            raise ValueError(f"claim_reviews[{claim_index}] has invalid status")
        if not isinstance(claim["reason"], str) or len(claim["reason"].strip()) < 8:
            raise ValueError(f"claim_reviews[{claim_index}] reason is too short")
        key = (claim["claim_text"], claim["occurrence"])
        if key in seen:
            raise ValueError("Duplicate claim_text/occurrence entry")
        seen.add(key)
        span = locate_occurrence(trajectory, claim["claim_text"], claim["occurrence"])
        if spans and span[0] < spans[-1][0]:
            raise ValueError("claim_reviews must be ordered by first response character")
        spans.append(span)
        statuses.append(status)

    earliest = annotation["earliest_problem_claim_index"]
    problem_indices = [
        index for index, status in enumerate(statuses) if status in PROBLEM_STATUSES
    ]
    uncertain_indices = [
        index for index, status in enumerate(statuses) if status == "uncertain"
    ]
    if path_status == "hallucinated":
        if isinstance(earliest, bool) or not isinstance(earliest, int):
            raise ValueError("hallucinated path requires an integer earliest index")
        if not 0 <= earliest < len(claims):
            raise ValueError("earliest_problem_claim_index is out of range")
        if statuses[earliest] not in PROBLEM_STATUSES:
            raise ValueError("earliest problem must be contradicted or unsupported")
        if not problem_indices or earliest != problem_indices[0]:
            raise ValueError("earliest index must name the first reviewed problem claim")
        if any(index < earliest for index in uncertain_indices):
            raise ValueError("An earlier uncertain claim makes the onset uncertain")
    elif path_status == "clean":
        if earliest is not None:
            raise ValueError("clean path requires null earliest index")
        if problem_indices or uncertain_indices:
            raise ValueError("clean path cannot contain problem or uncertain claim reviews")
    else:
        if earliest is not None:
            if isinstance(earliest, bool) or not isinstance(earliest, int):
                raise ValueError("uncertain earliest index must be null or an integer")
            if not 0 <= earliest < len(claims) or statuses[earliest] != "uncertain":
                raise ValueError("uncertain earliest index must reference an uncertain claim")
        if not uncertain_indices and not problem_indices:
            raise ValueError("uncertain path needs a problem or uncertain reviewed claim")
    return spans


def content_token_offsets(
    *,
    response: str,
    output_token_ids: Sequence[int],
    encoded_token_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    trailing_token_decodes_to_empty: Sequence[bool],
) -> list[tuple[int, int]]:
    """Validate response re-tokenization against frozen IDs and return offsets.

    Generated manifests include a terminal control token after the visible
    response.  Only an exact encoded prefix followed exclusively by invisible
    terminal tokens is accepted.
    """

    encoded = [int(value) for value in encoded_token_ids]
    output = [int(value) for value in output_token_ids]
    if output[: len(encoded)] != encoded:
        raise ValueError("Response re-tokenization differs from frozen output token IDs")
    trailing = output[len(encoded) :]
    if len(trailing) != len(trailing_token_decodes_to_empty):
        raise ValueError("Trailing-token audit length drifted")
    if not trailing or not all(trailing_token_decodes_to_empty):
        raise ValueError("Frozen output must end in at least one invisible terminal token")
    normalized = [(int(start), int(end)) for start, end in offsets]
    if len(normalized) != len(encoded):
        raise ValueError("Tokenizer offset count differs from encoded token count")
    if not normalized or normalized[0][0] != 0 or normalized[-1][1] != len(response):
        raise ValueError("Tokenizer offsets do not cover the complete visible response")
    return normalized


def char_span_to_token_span(
    char_span: tuple[int, int],
    offsets: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    char_start, char_end = char_span
    if not 0 <= char_start < char_end:
        raise ValueError("Invalid positive-width character span")
    overlapping = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > char_start and token_start < char_end
    ]
    if not overlapping:
        raise ValueError("Character span does not overlap a visible output token")
    return overlapping[0], overlapping[-1] + 1


def map_annotation(
    annotation: Mapping[str, Any],
    item: Mapping[str, Any],
    lineage: Mapping[str, Any],
    *,
    offsets: Sequence[tuple[int, int]],
    annotator: Mapping[str, Any],
) -> dict[str, Any]:
    spans = validate_annotation(annotation, item)
    claims = []
    for claim_index, (claim, char_span) in enumerate(
        zip(annotation["claim_reviews"], spans)
    ):
        token_span = char_span_to_token_span(char_span, offsets)
        claims.append(
            {
                **dict(claim),
                "claim_index": claim_index,
                "char_start": char_span[0],
                "char_end": char_span[1],
                "token_start": token_span[0],
                "token_end_exclusive": token_span[1],
            }
        )
    path_status = annotation["path_status"]
    label: dict[str, Any] = {
        "schema_version": "clir-hallucination-mapped-label-v1",
        "item_id": item["item_id"],
        "id": lineage["source_id"],
        "query_id": lineage["query_id"],
        "output_token_ids_sha256": lineage["output_token_ids_sha256"],
        "path_status": path_status,
        "claim_reviews": claims,
        "confidence": annotation["confidence"],
        "summary": annotation["summary"],
        "annotator": dict(annotator),
    }
    if path_status == "hallucinated":
        earliest = int(annotation["earliest_problem_claim_index"])
        label["path_hallucinated"] = 1
        label["hallucination_onset"] = claims[earliest]["token_start"]
        label["earliest_problem_claim_index"] = earliest
    elif path_status == "clean":
        label["path_hallucinated"] = 0
        label["hallucination_onset"] = -1
        label["earliest_problem_claim_index"] = None
    else:
        label["earliest_problem_claim_index"] = annotation[
            "earliest_problem_claim_index"
        ]
    return label


def annotation_counts(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path_counts = Counter(str(label["path_status"]) for label in labels)
    correctness_cross_tab: Counter[str] = Counter()
    for label in labels:
        if "correctness" in label:
            correctness_cross_tab[
                f"correctness_{label['correctness']}__{label['path_status']}"
            ] += 1
    return {
        "path_status": dict(sorted(path_counts.items())),
        "correctness_cross_tab": dict(sorted(correctness_cross_tab.items())),
    }


__all__ = [
    "ANNOTATION_KEYS",
    "CLAIM_KEYS",
    "CLAIM_STATUSES",
    "CONFIDENCE_LEVELS",
    "PATH_STATUSES",
    "atomic_write_json",
    "atomic_write_jsonl",
    "build_annotation_records",
    "canonical_json",
    "canonical_sha256",
    "char_span_to_token_span",
    "content_token_offsets",
    "file_sha256",
    "locate_occurrence",
    "make_item_id",
    "map_annotation",
    "read_jsonl",
    "select_stratified_rows",
    "validate_annotation",
]
