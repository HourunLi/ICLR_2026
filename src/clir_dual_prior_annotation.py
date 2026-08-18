"""Contracts for the CLIR key/complete evidence annotation pilot."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


ANNOTATION_SCHEMA = "clir-dual-prior-evidence-annotation-v1"
ITEM_SCHEMA = "clir-dual-prior-evidence-item-v1"
ELIGIBILITY_VALUES = {
    "usable",
    "insufficient_unitization",
    "no_auditable_reasoning",
}
CONFIDENCE_VALUES = {"high", "medium", "low"}
ANNOTATION_KEYS = {
    "schema_version",
    "item_id",
    "eligibility",
    "key_unit_indices",
    "complete_unit_indices",
    "confidence",
    "rationale",
}
ITEM_KEYS = {"schema_version", "item_id", "problem", "trajectory", "units"}
UNIT_KEYS = {"unit_index", "text"}
_HEADER = re.compile(
    r"^(?P<header>(?:#{1,6}\s*)?(?:step\s+\d+|final\s+answer|answer|solution)\s*:)(?P<gap>\s*)(?P<body>.*)$",
    flags=re.IGNORECASE,
)
_SENTENCE_GAP = re.compile(r"(?<=[.!?])\s+")


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split one non-empty line without changing any visible characters."""

    output: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_GAP.finditer(text, start, end):
        if match.end() > end:
            break
        span = _trimmed_span(text, cursor, match.start())
        if span is not None:
            output.append(span)
        cursor = match.end()
    span = _trimmed_span(text, cursor, end)
    if span is not None:
        output.append(span)
    return output


def segment_reasoning_units(trajectory: str) -> list[dict[str, Any]]:
    """Return deterministic, non-overlapping sentence/line units with exact spans.

    The segmenter is deliberately domain-agnostic. Newlines and sentence-final
    punctuation create boundaries; common ``Step N:``/``Answer:`` prefixes are
    separated so formatting tokens can remain negative when their body is evidence.
    """

    if not isinstance(trajectory, str) or not trajectory.strip():
        raise ValueError("trajectory must be a non-empty string")
    spans: list[tuple[int, int]] = []
    for line_match in re.finditer(r"[^\r\n]+", trajectory):
        trimmed = _trimmed_span(trajectory, line_match.start(), line_match.end())
        if trimmed is None:
            continue
        line_start, line_end = trimmed
        line_text = trajectory[line_start:line_end]
        header = _HEADER.match(line_text)
        if header is None:
            spans.extend(_sentence_spans(trajectory, line_start, line_end))
            continue
        header_end = line_start + header.end("header")
        spans.append((line_start, header_end))
        body_start = line_start + header.start("body")
        if body_start < line_end:
            spans.extend(_sentence_spans(trajectory, body_start, line_end))

    if not spans:
        raise ValueError("unitization produced no units")
    previous_end = -1
    for start, end in spans:
        if start < previous_end or not 0 <= start < end <= len(trajectory):
            raise ValueError("unitization produced overlapping or invalid spans")
        previous_end = end
    covered = [False] * len(trajectory)
    for start, end in spans:
        for index in range(start, end):
            covered[index] = True
    if any(not known for index, known in enumerate(covered) if not trajectory[index].isspace()):
        raise ValueError("unitization failed to cover a visible trajectory character")
    return [
        {
            "unit_index": index,
            "text": trajectory[start:end],
            "char_start": start,
            "char_end": end,
        }
        for index, (start, end) in enumerate(spans)
    ]


def public_units(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"unit_index": int(unit["unit_index"]), "text": str(unit["text"])}
        for unit in units
    ]


def validate_item(item: Mapping[str, Any]) -> None:
    if set(item) != ITEM_KEYS:
        raise ValueError(f"item keys differ: {sorted(set(item) ^ ITEM_KEYS)}")
    if item.get("schema_version") != ITEM_SCHEMA:
        raise ValueError(f"item must use {ITEM_SCHEMA}")
    for field in ("item_id", "problem", "trajectory"):
        if not isinstance(item.get(field), str) or not str(item[field]).strip():
            raise ValueError(f"item {field} must be a non-empty string")
    units = item.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("item units must be a non-empty list")
    for expected_index, unit in enumerate(units):
        if not isinstance(unit, Mapping) or set(unit) != UNIT_KEYS:
            raise ValueError("every public unit must contain only unit_index and text")
        if unit.get("unit_index") != expected_index:
            raise ValueError("unit indices must be contiguous and zero based")
        if not isinstance(unit.get("text"), str) or not unit["text"].strip():
            raise ValueError("unit text must be non-empty")


def _index_list(value: Any, *, field: str, unit_count: int) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    output: list[int] = []
    for element in value:
        if isinstance(element, bool) or not isinstance(element, int):
            raise ValueError(f"{field} must contain only integers")
        if not 0 <= element < unit_count:
            raise ValueError(f"{field} contains an out-of-range unit index")
        output.append(int(element))
    if output != sorted(set(output)):
        raise ValueError(f"{field} must be sorted and contain no duplicates")
    return output


def validate_annotation(
    annotation: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one blind key/complete annotation and return normalized fields."""

    validate_item(item)
    if not isinstance(annotation, Mapping) or set(annotation) != ANNOTATION_KEYS:
        observed = set(annotation) if isinstance(annotation, Mapping) else set()
        raise ValueError(
            f"annotation keys differ: {sorted(observed ^ ANNOTATION_KEYS)}"
        )
    if annotation.get("schema_version") != ANNOTATION_SCHEMA:
        raise ValueError(f"annotation must use {ANNOTATION_SCHEMA}")
    if annotation.get("item_id") != item["item_id"]:
        raise ValueError("annotation item_id does not match item")
    eligibility = annotation.get("eligibility")
    if eligibility not in ELIGIBILITY_VALUES:
        raise ValueError(f"eligibility must be one of {sorted(ELIGIBILITY_VALUES)}")
    confidence = annotation.get("confidence")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"confidence must be one of {sorted(CONFIDENCE_VALUES)}")
    rationale = annotation.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    unit_count = len(item["units"])
    key = _index_list(
        annotation.get("key_unit_indices"),
        field="key_unit_indices",
        unit_count=unit_count,
    )
    complete = _index_list(
        annotation.get("complete_unit_indices"),
        field="complete_unit_indices",
        unit_count=unit_count,
    )
    if eligibility == "usable":
        if not key or not complete:
            raise ValueError("usable annotations require non-empty key and complete sets")
        if not set(key).issubset(complete):
            raise ValueError("key_unit_indices must be a subset of complete_unit_indices")
    elif key or complete:
        raise ValueError("ineligible annotations must leave both evidence sets empty")
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "item_id": str(annotation["item_id"]),
        "eligibility": str(eligibility),
        "key_unit_indices": key,
        "complete_unit_indices": complete,
        "confidence": str(confidence),
        "rationale": rationale.strip(),
    }


__all__ = [
    "ANNOTATION_SCHEMA",
    "CONFIDENCE_VALUES",
    "ELIGIBILITY_VALUES",
    "ITEM_SCHEMA",
    "public_units",
    "segment_reasoning_units",
    "validate_annotation",
    "validate_item",
]
