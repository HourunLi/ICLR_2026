"""Validation contracts for role-blinded CLIR dual-prior adjudication."""

from __future__ import annotations

from typing import Any, Mapping

from src.clir_dual_prior_annotation import (
    ANNOTATION_SCHEMA,
    validate_annotation,
    validate_item,
)


ADJUDICATION_SCHEMA = "clir-dual-prior-evidence-adjudication-v1"
ADJUDICATION_ITEM_SCHEMA = "clir-dual-prior-evidence-adjudication-item-v1"
RESOLUTION_VALUES = {"adopt_a", "adopt_b", "synthesize"}
ADJUDICATION_KEYS = {
    "schema_version",
    "item_id",
    "resolution",
    "eligibility",
    "key_unit_indices",
    "complete_unit_indices",
    "confidence",
    "rationale",
}
ADJUDICATION_ITEM_KEYS = {
    "schema_version",
    "item_id",
    "problem",
    "trajectory",
    "units",
    "annotation_a",
    "annotation_b",
}
CANDIDATE_KEYS = {
    "eligibility",
    "key_unit_indices",
    "complete_unit_indices",
    "confidence",
    "rationale",
}


def annotation_item(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return the embedded public annotation item after validating packet shape."""

    if not isinstance(packet, Mapping) or set(packet) != ADJUDICATION_ITEM_KEYS:
        observed = set(packet) if isinstance(packet, Mapping) else set()
        raise ValueError(
            f"adjudication item keys differ: "
            f"{sorted(observed ^ ADJUDICATION_ITEM_KEYS)}"
        )
    if packet.get("schema_version") != ADJUDICATION_ITEM_SCHEMA:
        raise ValueError(f"adjudication item must use {ADJUDICATION_ITEM_SCHEMA}")
    item = {
        "schema_version": "clir-dual-prior-evidence-item-v1",
        "item_id": packet["item_id"],
        "problem": packet["problem"],
        "trajectory": packet["trajectory"],
        "units": packet["units"],
    }
    validate_item(item)
    for name in ("annotation_a", "annotation_b"):
        candidate = packet.get(name)
        if not isinstance(candidate, Mapping) or set(candidate) != CANDIDATE_KEYS:
            raise ValueError(f"{name} has an invalid public candidate shape")
        validate_annotation(
            {
                "schema_version": ANNOTATION_SCHEMA,
                "item_id": packet["item_id"],
                **dict(candidate),
            },
            item,
        )
    return item


def _target_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        annotation["eligibility"],
        tuple(annotation["key_unit_indices"]),
        tuple(annotation["complete_unit_indices"]),
    )


def validate_adjudication(
    adjudication: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one adjudication and enforce its declared A/B/synthesis resolution."""

    item = annotation_item(packet)
    if not isinstance(adjudication, Mapping) or set(adjudication) != ADJUDICATION_KEYS:
        observed = set(adjudication) if isinstance(adjudication, Mapping) else set()
        raise ValueError(
            f"adjudication keys differ: {sorted(observed ^ ADJUDICATION_KEYS)}"
        )
    if adjudication.get("schema_version") != ADJUDICATION_SCHEMA:
        raise ValueError(f"adjudication must use {ADJUDICATION_SCHEMA}")
    if adjudication.get("item_id") != packet["item_id"]:
        raise ValueError("adjudication item_id does not match packet")
    resolution = adjudication.get("resolution")
    if resolution not in RESOLUTION_VALUES:
        raise ValueError(f"resolution must be one of {sorted(RESOLUTION_VALUES)}")
    normalized_target = validate_annotation(
        {
            "schema_version": ANNOTATION_SCHEMA,
            "item_id": adjudication["item_id"],
            "eligibility": adjudication["eligibility"],
            "key_unit_indices": adjudication["key_unit_indices"],
            "complete_unit_indices": adjudication["complete_unit_indices"],
            "confidence": adjudication["confidence"],
            "rationale": adjudication["rationale"],
        },
        item,
    )
    candidates: dict[str, dict[str, Any]] = {}
    for suffix in ("a", "b"):
        candidates[suffix] = validate_annotation(
            {
                "schema_version": ANNOTATION_SCHEMA,
                "item_id": packet["item_id"],
                **dict(packet[f"annotation_{suffix}"]),
            },
            item,
        )
    target_signature = _target_signature(normalized_target)
    a_signature = _target_signature(candidates["a"])
    b_signature = _target_signature(candidates["b"])
    if a_signature == b_signature:
        raise ValueError("adjudication packet must contain a target disagreement")
    if resolution == "adopt_a" and target_signature != a_signature:
        raise ValueError("adopt_a must exactly reproduce annotation_a target sets")
    if resolution == "adopt_b" and target_signature != b_signature:
        raise ValueError("adopt_b must exactly reproduce annotation_b target sets")
    if resolution == "synthesize" and target_signature in {a_signature, b_signature}:
        raise ValueError("synthesize must differ from both proposed target sets")
    return {
        "schema_version": ADJUDICATION_SCHEMA,
        "item_id": normalized_target["item_id"],
        "resolution": str(resolution),
        "eligibility": normalized_target["eligibility"],
        "key_unit_indices": normalized_target["key_unit_indices"],
        "complete_unit_indices": normalized_target["complete_unit_indices"],
        "confidence": normalized_target["confidence"],
        "rationale": normalized_target["rationale"],
    }


__all__ = [
    "ADJUDICATION_ITEM_SCHEMA",
    "ADJUDICATION_SCHEMA",
    "RESOLUTION_VALUES",
    "annotation_item",
    "validate_adjudication",
]
