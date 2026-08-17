#!/usr/bin/env python3
"""Validate v1 annotations and compare independent labels when available."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "configs" / "verifier_selection_v1"
DECISIONS = {"accept", "reject", "review"}
STYLES = {"terse_formal", "explanatory_conversational"}
LABEL_KEYS = {
    "item_id", "decision", "same_task_and_goal", "same_core_premises",
    "same_reasoning_method", "same_key_inferences", "same_intermediate_conclusions",
    "same_final_conclusion", "entailed_elaboration_present",
    "entailed_elaboration_valid", "introduced_new_premise_or_evidence",
    "omitted_essential_claim", "contradicted_source_claim",
    "replaced_with_different_solution", "introduced_new_error",
    "error_alignment_applicable", "same_error_mechanism",
    "same_semantic_error_location", "same_downstream_effect", "target_style",
    "style_satisfied", "confidence", "reason",
}
PRIMARY_EXTRA_KEYS = {"primary_annotation_provenance", "relation_failure_expected"}
BOOL_KEYS = {
    "same_task_and_goal", "same_core_premises", "same_reasoning_method",
    "same_key_inferences", "same_intermediate_conclusions", "same_final_conclusion",
    "entailed_elaboration_present", "introduced_new_premise_or_evidence",
    "omitted_essential_claim", "contradicted_source_claim",
    "replaced_with_different_solution", "introduced_new_error",
    "error_alignment_applicable", "style_satisfied",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{index} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{index} must be an object")
        rows.append(value)
    return rows


def validate_labels(
    items: list[dict[str, Any]], labels: list[dict[str, Any]], *, primary: bool
) -> None:
    if len(labels) != len(items):
        raise ValueError("Label count differs from item count")
    expected_keys = LABEL_KEYS | (PRIMARY_EXTRA_KEYS if primary else set())
    for index, (item, label) in enumerate(zip(items, labels), start=1):
        if set(label) != expected_keys:
            raise ValueError(f"label {index} keys differ from the required schema")
        if label["item_id"] != item["item_id"]:
            raise ValueError(f"label {index} item_id/order drifted")
        if label["target_style"] != item["target_style"] or label["target_style"] not in STYLES:
            raise ValueError(f"label {index} target_style drifted")
        if label["decision"] not in DECISIONS:
            raise ValueError(f"label {index} has an invalid decision")
        if label["confidence"] not in {"high", "medium", "low"}:
            raise ValueError(f"label {index} has invalid confidence")
        if not isinstance(label["reason"], str) or len(label["reason"].strip()) < 20:
            raise ValueError(f"label {index} reason is not evidence-specific")
        for key in BOOL_KEYS:
            if type(label[key]) is not bool:
                raise ValueError(f"label {index}/{key} must be Boolean")
        if label["entailed_elaboration_present"]:
            if type(label["entailed_elaboration_valid"]) is not bool:
                raise ValueError(f"label {index} needs entailed_elaboration_valid")
        elif label["entailed_elaboration_valid"] is not None:
            raise ValueError(f"label {index} must null unused elaboration validity")
        error_values = [
            label["same_error_mechanism"], label["same_semantic_error_location"],
            label["same_downstream_effect"],
        ]
        if label["error_alignment_applicable"]:
            if any(type(value) is not bool for value in error_values):
                raise ValueError(f"label {index} needs complete incorrect-source alignment")
        elif any(value is not None for value in error_values):
            raise ValueError(f"label {index} must null inapplicable error alignment")
        if primary and (
            not isinstance(label["primary_annotation_provenance"], str)
            or type(label["relation_failure_expected"]) is not bool
        ):
            raise ValueError(f"primary label {index} provenance fields are invalid")


def kappa(primary: list[str], secondary: list[str]) -> float:
    observed = sum(a == b for a, b in zip(primary, secondary)) / len(primary)
    pa, pb = Counter(primary), Counter(secondary)
    expected = sum(pa[key] * pb[key] for key in DECISIONS) / (len(primary) ** 2)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--secondary", type=Path)
    args = parser.parse_args()
    items = load_jsonl(args.root / "verifier_selection_items_v1.jsonl")
    primary = load_jsonl(args.root / "verifier_selection_labels_primary_v1.jsonl")
    if len(items) != 64 or len({row["item_id"] for row in items}) != 64:
        raise ValueError("Items must contain 64 unique IDs")
    forbidden_item_keys = LABEL_KEYS - {"item_id", "target_style"}
    if any(set(item) & forbidden_item_keys for item in items):
        raise ValueError("Anonymous items leak annotation fields")
    by_base: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_base.setdefault(item["base_source_id"], []).append(item)
    if len(by_base) != 16 or any(
        len(rows) != 4 or Counter(row["target_style"] for row in rows) != Counter({style: 2 for style in STYLES})
        for rows in by_base.values()
    ):
        raise ValueError("Base-source grouping/style balance drifted")
    validate_labels(items, primary, primary=True)
    result: dict[str, Any] = {"items_valid": True, "primary_valid": True, "items": len(items)}
    if args.secondary:
        secondary = load_jsonl(args.secondary)
        validate_labels(items, secondary, primary=False)
        disagreements = []
        for item, first, second in zip(items, primary, secondary):
            if first["decision"] != second["decision"]:
                disagreements.append({
                    "item_id": item["item_id"], "base_source_id": item["base_source_id"],
                    "domain": item["domain"], "primary_decision": first["decision"],
                    "secondary_decision": second["decision"],
                })
        output = args.root / "verifier_selection_disagreements_v1.jsonl"
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in disagreements), encoding="utf-8")
        first_values = [row["decision"] for row in primary]
        second_values = [row["decision"] for row in secondary]
        result.update({
            "secondary_valid": True,
            "decision_agreement": 1.0 - len(disagreements) / len(items),
            "cohen_kappa": kappa(first_values, second_values),
            "decision_disagreements": len(disagreements),
            "disagreement_path": str(output),
        })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
