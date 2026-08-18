"""Contracts for the repaired Route A on-policy equivalence pilot.

The v1 Silver decisions are deliberately not inputs to this module.  They may
identify the 31-pair roster that was previously trained, but every pair must
receive a fresh bidirectional claim certificate before it can contribute to a
new consistency manifest.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .clir_on_policy_pairs import canonical_json
from .clir_reasoning_rewrite import (
    CHECKER_SCHEMA,
    build_soft_risk_probe,
    derive_acceptance_status,
    validate_domain_checker_result,
    validate_verifier_report,
)


ITEM_SCHEMA = "clir-on-policy-reannotation-item-v1"
LINEAGE_SCHEMA = "clir-on-policy-reannotation-lineage-v1"
ANNOTATION_SCHEMA = "clir-on-policy-claim-annotation-v1"
TARGET_STYLE = "native_compact_to_expanded"
COMPACT_STYLE_ID = "native_compact"
EXPANDED_STYLE_ID = "native_expanded"

_ITEM_KEYS = {
    "schema_version",
    "item_id",
    "problem",
    "source_role",
    "rewrite_role",
    "source_trajectory",
    "rewrite_trajectory",
    "source_output_tokens",
    "rewrite_output_tokens",
    "target_style",
    "risk_probe",
}
_LINEAGE_KEYS = {
    "schema_version",
    "item_id",
    "query_id",
    "candidate_sha256",
    "source_row_id",
    "source_row_sha256",
    "rewrite_row_id",
    "rewrite_row_sha256",
    "source_original_role",
    "rewrite_original_role",
    "checker",
}
_ANNOTATION_KEYS = {"schema_version", "item_id", "report"}


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{path} keys differ: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _orient_candidate(candidate: Mapping[str, Any]) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Orient a native pair by exact output-token length, never candidate index."""

    sides: list[tuple[str, dict[str, Any]]] = []
    for role in ("anchor", "alternate"):
        side = candidate.get(role)
        if not isinstance(side, Mapping):
            raise ValueError(f"Candidate {candidate.get('pair_id')!r} lacks {role}")
        normalized = dict(side)
        if not normalized.get("response") or not normalized.get("output_token_ids"):
            raise ValueError(f"Candidate {candidate.get('pair_id')!r}/{role} is empty")
        sides.append((role, normalized))
    sides.sort(
        key=lambda item: (
            len(item[1]["output_token_ids"]),
            len(item[1]["response"]),
            str(item[1]["id"]),
        )
    )
    compact_role, compact = sides[0]
    expanded_role, expanded = sides[1]
    if len(compact["output_token_ids"]) >= len(expanded["output_token_ids"]):
        raise ValueError(
            f"Candidate {candidate.get('pair_id')!r} has no strict output-token length contrast"
        )
    return compact_role, compact, expanded_role, expanded


def build_reannotation_records(
    candidate: Mapping[str, Any],
    *,
    source_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one blind item and its private identity/checker lineage."""

    pair_id = str(candidate.get("pair_id", ""))
    if not pair_id:
        raise ValueError("Candidate pair_id is required")
    compact_role, compact, expanded_role, expanded = _orient_candidate(candidate)
    if compact["id"] not in source_rows or expanded["id"] not in source_rows:
        raise ValueError(f"Candidate {pair_id!r} references a missing source row")
    compact_source = dict(source_rows[compact["id"]])
    expanded_source = dict(source_rows[expanded["id"]])
    if _object_sha256(compact_source) != compact["row_sha256"]:
        raise ValueError(f"Candidate {pair_id!r} compact source hash drifted")
    if _object_sha256(expanded_source) != expanded["row_sha256"]:
        raise ValueError(f"Candidate {pair_id!r} expanded source hash drifted")
    if compact_source["query_id"] != expanded_source["query_id"]:
        raise ValueError(f"Candidate {pair_id!r} crosses query IDs")

    compact_outcome = str(compact_source["normalized_candidate_answer"])
    expanded_outcome = str(expanded_source["normalized_candidate_answer"])
    compact_correctness = int(compact_source["correctness"])
    expanded_correctness = int(expanded_source["correctness"])
    compact_checker = str(compact_source.get("checker_version", ""))
    expanded_checker = str(expanded_source.get("checker_version", ""))
    if not compact_checker or compact_checker != expanded_checker:
        raise ValueError(f"Candidate {pair_id!r} checker identity drifted")
    compact_revision = str(
        compact_source.get("label_provenance", {}).get("label_protocol_sha256", "")
    )
    expanded_revision = str(
        expanded_source.get("label_provenance", {}).get("label_protocol_sha256", "")
    )
    if not compact_revision or compact_revision != expanded_revision:
        raise ValueError(f"Candidate {pair_id!r} checker revision drifted")

    checker = {
        "schema_version": CHECKER_SCHEMA,
        "available": True,
        "checker_id": compact_checker,
        "checker_revision": compact_revision,
        "source_outcome": compact_outcome,
        "rewrite_outcome": expanded_outcome,
        "same_outcome": compact_outcome == expanded_outcome,
        "source_correctness": compact_correctness,
        "rewrite_correctness": expanded_correctness,
    }
    validate_domain_checker_result(checker)
    risk_probe = build_soft_risk_probe(compact["response"], expanded["response"])
    item = {
        "schema_version": ITEM_SCHEMA,
        "item_id": pair_id,
        "problem": candidate["problem"],
        "source_role": COMPACT_STYLE_ID,
        "rewrite_role": EXPANDED_STYLE_ID,
        "source_trajectory": compact["response"],
        "rewrite_trajectory": expanded["response"],
        "source_output_tokens": len(compact["output_token_ids"]),
        "rewrite_output_tokens": len(expanded["output_token_ids"]),
        "target_style": TARGET_STYLE,
        "risk_probe": risk_probe,
    }
    lineage = {
        "schema_version": LINEAGE_SCHEMA,
        "item_id": pair_id,
        "query_id": compact_source["query_id"],
        "candidate_sha256": _object_sha256(candidate),
        "source_row_id": compact["id"],
        "source_row_sha256": compact["row_sha256"],
        "rewrite_row_id": expanded["id"],
        "rewrite_row_sha256": expanded["row_sha256"],
        "source_original_role": f"native_{compact_role}",
        "rewrite_original_role": f"native_{expanded_role}",
        "checker": checker,
    }
    validate_item(item)
    validate_lineage(lineage)
    return item, lineage


def validate_item(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Reannotation item must be an object")
    item = dict(value)
    _require_exact_keys(item, _ITEM_KEYS, "item")
    if item["schema_version"] != ITEM_SCHEMA:
        raise ValueError("Item schema_version drifted")
    if not isinstance(item["item_id"], str) or not item["item_id"].strip():
        raise ValueError("Item item_id is required")
    for key in ("problem", "source_trajectory", "rewrite_trajectory"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise ValueError(f"Item {key} must be non-empty")
    if item["source_role"] != COMPACT_STYLE_ID or item["rewrite_role"] != EXPANDED_STYLE_ID:
        raise ValueError("Item compact/expanded roles drifted")
    if item["target_style"] != TARGET_STYLE:
        raise ValueError("Item target_style drifted")
    if type(item["source_output_tokens"]) is not int or type(item["rewrite_output_tokens"]) is not int:
        raise ValueError("Item output-token lengths must be integers")
    if item["source_output_tokens"] >= item["rewrite_output_tokens"]:
        raise ValueError("Item is not oriented compact-to-expanded")
    probe = item["risk_probe"]
    if not isinstance(probe, Mapping) or not isinstance(probe.get("expected_risk_ids"), list):
        raise ValueError("Item risk_probe is invalid")
    risk_ids = [risk.get("risk_id") for risk in probe.get("risks", [])]
    if risk_ids != probe["expected_risk_ids"] or len(risk_ids) != len(set(risk_ids)):
        raise ValueError("Item risk IDs drifted")
    return item


def validate_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Reannotation lineage must be an object")
    lineage = dict(value)
    _require_exact_keys(lineage, _LINEAGE_KEYS, "lineage")
    if lineage["schema_version"] != LINEAGE_SCHEMA:
        raise ValueError("Lineage schema_version drifted")
    for key in (
        "item_id",
        "query_id",
        "candidate_sha256",
        "source_row_id",
        "source_row_sha256",
        "rewrite_row_id",
        "rewrite_row_sha256",
        "source_original_role",
        "rewrite_original_role",
    ):
        if not isinstance(lineage[key], str) or not lineage[key].strip():
            raise ValueError(f"Lineage {key} must be non-empty")
    validate_domain_checker_result(lineage["checker"])
    return lineage


def validate_annotation_row(
    value: Mapping[str, Any],
    *,
    item: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one independent label and derive its programmatic status."""

    if not isinstance(value, Mapping):
        raise ValueError("Annotation row must be an object")
    annotation = dict(value)
    _require_exact_keys(annotation, _ANNOTATION_KEYS, "annotation")
    if annotation["schema_version"] != ANNOTATION_SCHEMA:
        raise ValueError("Annotation schema_version drifted")
    normalized_item = validate_item(item)
    normalized_lineage = validate_lineage(lineage)
    if annotation["item_id"] != normalized_item["item_id"]:
        raise ValueError("Annotation item_id drifted")
    if normalized_lineage["item_id"] != normalized_item["item_id"]:
        raise ValueError("Item/lineage identity drifted")
    report = validate_verifier_report(annotation["report"])
    if report["style_assessment"]["target_style"] != normalized_item["target_style"]:
        raise ValueError("Annotation target style drifted")
    expected_risk_ids = normalized_item["risk_probe"]["expected_risk_ids"]
    result = derive_acceptance_status(
        report,
        checker=normalized_lineage["checker"],
        expected_risk_ids=expected_risk_ids,
    )
    return {**annotation, "report": report}, result


def validate_annotation_set(
    annotations: Sequence[Mapping[str, Any]],
    *,
    items: Sequence[Mapping[str, Any]],
    lineages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(annotations) != len(items) or len(items) != len(lineages):
        raise ValueError("Annotation, item, and lineage counts differ")
    normalized: list[dict[str, Any]] = []
    observed: set[str] = set()
    for index, (annotation, item, lineage) in enumerate(
        zip(annotations, items, lineages), start=1
    ):
        row, result = validate_annotation_row(annotation, item=item, lineage=lineage)
        item_id = row["item_id"]
        if item_id in observed:
            raise ValueError(f"Annotation repeats item_id {item_id!r}")
        observed.add(item_id)
        if item_id != item["item_id"] or item_id != lineage["item_id"]:
            raise ValueError(f"Annotation row {index} order drifted")
        normalized.append({"annotation": row, "acceptance": result})
    return normalized


__all__ = [
    "ANNOTATION_SCHEMA",
    "COMPACT_STYLE_ID",
    "EXPANDED_STYLE_ID",
    "ITEM_SCHEMA",
    "LINEAGE_SCHEMA",
    "TARGET_STYLE",
    "build_reannotation_records",
    "validate_annotation_row",
    "validate_annotation_set",
    "validate_item",
    "validate_lineage",
]
