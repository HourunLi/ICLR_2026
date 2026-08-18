#!/usr/bin/env python3
"""Apply the frozen dual-prior agreement gate and build a blind adjudication packet."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_dual_prior_annotation import validate_annotation  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/dual_prior_evidence_v1/agreement_adjudication_protocol_v1.json"
)
ANNOTATION_FIELDS = (
    "schema_version",
    "item_id",
    "eligibility",
    "key_unit_indices",
    "complete_unit_indices",
    "confidence",
    "rationale",
)
BLINDING_PREFIX = "clir-dual-prior-adjudication-v1:"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def annotation_view(row: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate the frozen seven-field annotation from an enriched row."""

    missing = [field for field in ANNOTATION_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Annotation row is missing fields: {missing}")
    return validate_annotation({field: row[field] for field in ANNOTATION_FIELDS}, item)


def unit_set_f1(left: Sequence[int], right: Sequence[int]) -> float:
    """Symmetric set F1 with a defined score for two empty sets."""

    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    return 2.0 * len(left_set & right_set) / (len(left_set) + len(right_set))


def unit_set_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def primary_is_annotation_a(item_id: str) -> bool:
    digest = hashlib.sha256(f"{BLINDING_PREFIX}{item_id}".encode("utf-8")).digest()
    return digest[0] % 2 == 0


def public_candidate(annotation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eligibility": annotation["eligibility"],
        "key_unit_indices": list(annotation["key_unit_indices"]),
        "complete_unit_indices": list(annotation["complete_unit_indices"]),
        "confidence": annotation["confidence"],
        "rationale": annotation["rationale"],
    }


def current_git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status)}


def assert_input_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(spec["path"])
        digest = file_sha256(path)
        if digest != spec["sha256"]:
            raise ValueError(
                f"Frozen agreement input hash drifted for {name}: "
                f"expected {spec['sha256']}, got {digest}"
            )
        observed[name] = digest
    return observed


def assert_inherited_gate(
    protocol: Mapping[str, Any], parent: Mapping[str, Any]
) -> None:
    inherited = dict(protocol["selection_gate_inherited_without_changes"])
    inherited.pop("source")
    expected_keys = {
        "structural_validity_per_annotator",
        "minimum_usable_rows_per_annotator",
        "minimum_usable_overlap_rows",
        "minimum_eligibility_agreement",
        "minimum_macro_unit_f1_on_usable_overlap",
        "required_nested_fraction",
    }
    parent_gate = {key: parent["selection_gate"][key] for key in expected_keys}
    if inherited != parent_gate:
        raise ValueError("Agreement protocol no longer exactly inherits parent gates")


def summarize(
    *,
    items: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not (len(items) == len(primary_rows) == len(secondary_rows)):
        raise ValueError("Dual-prior agreement inputs have different row counts")

    eligibility_match = 0
    primary_usable = 0
    secondary_usable = 0
    nested_annotations = 0
    usable_annotations = 0
    key_f1_values: list[float] = []
    complete_f1_values: list[float] = []
    key_jaccard_values: list[float] = []
    complete_jaccard_values: list[float] = []
    exact_key_rows = 0
    exact_complete_rows = 0
    exact_target_rows = 0
    disagreement_types: Counter[str] = Counter()
    row_diagnostics: list[dict[str, Any]] = []
    public_items: list[dict[str, Any]] = []
    private_lineage: list[dict[str, Any]] = []

    for row_number, (item, primary_raw, secondary_raw) in enumerate(
        zip(items, primary_rows, secondary_rows), start=1
    ):
        expected_id = item["item_id"]
        if not (
            primary_raw.get("item_id") == expected_id
            and secondary_raw.get("item_id") == expected_id
        ):
            raise ValueError(f"Agreement item order/id drift at row {row_number}")
        primary = annotation_view(primary_raw, item)
        secondary = annotation_view(secondary_raw, item)
        eligibility_equal = primary["eligibility"] == secondary["eligibility"]
        eligibility_match += int(eligibility_equal)
        primary_is_usable = primary["eligibility"] == "usable"
        secondary_is_usable = secondary["eligibility"] == "usable"
        primary_usable += int(primary_is_usable)
        secondary_usable += int(secondary_is_usable)
        for annotation in (primary, secondary):
            if annotation["eligibility"] == "usable":
                usable_annotations += 1
                nested_annotations += int(
                    set(annotation["key_unit_indices"]).issubset(
                        annotation["complete_unit_indices"]
                    )
                )

        usable_overlap = primary_is_usable and secondary_is_usable
        key_exact = primary["key_unit_indices"] == secondary["key_unit_indices"]
        complete_exact = (
            primary["complete_unit_indices"]
            == secondary["complete_unit_indices"]
        )
        exact_key_rows += int(key_exact)
        exact_complete_rows += int(complete_exact)
        exact_target = eligibility_equal and key_exact and complete_exact
        exact_target_rows += int(exact_target)

        key_f1 = complete_f1 = key_jaccard = complete_jaccard = None
        if usable_overlap:
            key_f1 = unit_set_f1(
                primary["key_unit_indices"], secondary["key_unit_indices"]
            )
            complete_f1 = unit_set_f1(
                primary["complete_unit_indices"],
                secondary["complete_unit_indices"],
            )
            key_jaccard = unit_set_jaccard(
                primary["key_unit_indices"], secondary["key_unit_indices"]
            )
            complete_jaccard = unit_set_jaccard(
                primary["complete_unit_indices"],
                secondary["complete_unit_indices"],
            )
            key_f1_values.append(key_f1)
            complete_f1_values.append(complete_f1)
            key_jaccard_values.append(key_jaccard)
            complete_jaccard_values.append(complete_jaccard)

        changed: list[str] = []
        if not eligibility_equal:
            changed.append("eligibility")
        if not key_exact:
            changed.append("key")
        if not complete_exact:
            changed.append("complete")
        if changed:
            disagreement_types["+".join(changed)] += 1

        row_diagnostics.append(
            {
                "row_number_1_based": row_number,
                "item_id": expected_id,
                "primary_eligibility": primary["eligibility"],
                "secondary_eligibility": secondary["eligibility"],
                "usable_overlap": usable_overlap,
                "key_unit_f1": key_f1,
                "complete_unit_f1": complete_f1,
                "key_unit_jaccard": key_jaccard,
                "complete_unit_jaccard": complete_jaccard,
                "key_exact": key_exact,
                "complete_exact": complete_exact,
                "exact_target_agreement": exact_target,
                "changed_targets": changed,
            }
        )

        if exact_target:
            continue
        primary_a = primary_is_annotation_a(expected_id)
        annotation_a = primary if primary_a else secondary
        annotation_b = secondary if primary_a else primary
        public_items.append(
            {
                "schema_version": "clir-dual-prior-evidence-adjudication-item-v1",
                "item_id": expected_id,
                "problem": item["problem"],
                "trajectory": item["trajectory"],
                "units": item["units"],
                "annotation_a": public_candidate(annotation_a),
                "annotation_b": public_candidate(annotation_b),
            }
        )
        private_lineage.append(
            {
                "schema_version": "clir-dual-prior-evidence-adjudication-lineage-v1",
                "item_id": expected_id,
                "annotation_a_source": "primary" if primary_a else "secondary",
                "annotation_b_source": "secondary" if primary_a else "primary",
                "annotation_a_sha256": canonical_sha256(annotation_a),
                "annotation_b_sha256": canonical_sha256(annotation_b),
            }
        )

    rows = len(items)
    usable_overlap_rows = len(key_f1_values)
    eligibility_agreement = eligibility_match / rows if rows else 0.0
    nested_fraction = (
        nested_annotations / usable_annotations if usable_annotations else 0.0
    )
    key_macro = sum(key_f1_values) / usable_overlap_rows if usable_overlap_rows else None
    complete_macro = (
        sum(complete_f1_values) / usable_overlap_rows
        if usable_overlap_rows
        else None
    )
    key_jaccard_macro = (
        sum(key_jaccard_values) / usable_overlap_rows
        if usable_overlap_rows
        else None
    )
    complete_jaccard_macro = (
        sum(complete_jaccard_values) / usable_overlap_rows
        if usable_overlap_rows
        else None
    )
    minimum_f1 = thresholds["minimum_macro_unit_f1_on_usable_overlap"]
    gates = {
        "structural_validity_primary": True,
        "structural_validity_secondary": True,
        "primary_usable_yield": primary_usable
        >= int(thresholds["minimum_usable_rows_per_annotator"]),
        "secondary_usable_yield": secondary_usable
        >= int(thresholds["minimum_usable_rows_per_annotator"]),
        "usable_overlap": usable_overlap_rows
        >= int(thresholds["minimum_usable_overlap_rows"]),
        "eligibility_agreement": eligibility_agreement
        >= float(thresholds["minimum_eligibility_agreement"]),
        "key_macro_unit_f1": key_macro is not None
        and key_macro >= float(minimum_f1["key"]),
        "complete_macro_unit_f1": complete_macro is not None
        and complete_macro >= float(minimum_f1["complete"]),
        "nested_fraction": nested_fraction
        >= float(thresholds["required_nested_fraction"]),
    }
    agreement_gate_passed = all(gates.values())
    report = {
        "rows": rows,
        "structural_validity": {"primary": 1.0, "secondary": 1.0},
        "usable_rows": {"primary": primary_usable, "secondary": secondary_usable},
        "usable_overlap_rows": usable_overlap_rows,
        "eligibility_agreement": eligibility_agreement,
        "macro_unit_set_f1_on_usable_overlap": {
            "key": key_macro,
            "complete": complete_macro,
        },
        "macro_unit_set_jaccard_on_usable_overlap": {
            "key": key_jaccard_macro,
            "complete": complete_jaccard_macro,
        },
        "nested_fraction_across_usable_annotations": nested_fraction,
        "exact_set_agreement_rows": {
            "key": exact_key_rows,
            "complete": exact_complete_rows,
            "both_targets_and_eligibility": exact_target_rows,
        },
        "disagreement_rows": rows - exact_target_rows,
        "disagreement_type_counts": dict(sorted(disagreement_types.items())),
        "gate_components": gates,
        "agreement_gate_passed": agreement_gate_passed,
        "adjudication_authorized": agreement_gate_passed
        and exact_target_rows < rows,
        "guide_or_unitization_repair_required": not agreement_gate_passed,
        "training_gold_authorized": agreement_gate_passed
        and exact_target_rows == rows,
        "row_diagnostics": row_diagnostics,
    }
    return report, public_items, private_lineage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-dual-prior-agreement-adjudication-protocol-v1"
    ):
        raise ValueError("Unexpected dual-prior agreement protocol schema")
    git = current_git_state()
    if git["dirty"]:
        raise ValueError("Agreement must run from a clean committed worktree")
    observed_hashes = assert_input_hashes(protocol)
    parent_path = resolve(protocol["inputs"]["parent_protocol"]["path"])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    assert_inherited_gate(protocol, parent)

    items = read_jsonl(resolve(protocol["inputs"]["annotation_items"]["path"]))
    primary = read_jsonl(resolve(protocol["inputs"]["primary_labels"]["path"]))
    secondary = read_jsonl(resolve(protocol["inputs"]["secondary_labels"]["path"]))
    expected_rows = int(protocol["inputs"]["annotation_items"]["rows"])
    if not (len(items) == len(primary) == len(secondary) == expected_rows):
        raise ValueError("Frozen dual-prior agreement row count drifted")

    summary, public_items, private_lineage = summarize(
        items=items,
        primary_rows=primary,
        secondary_rows=secondary,
        thresholds=protocol["selection_gate_inherited_without_changes"],
    )
    outputs = protocol["outputs"]
    items_output = resolve(outputs["adjudication_items"])
    lineage_output = resolve(outputs["adjudication_lineage"])
    atomic_write_jsonl(items_output, public_items)
    atomic_write_jsonl(lineage_output, private_lineage)

    report = {
        "schema_version": "clir-dual-prior-agreement-report-v1",
        "status": (
            "agreement_gate_passed_adjudication_required"
            if summary["adjudication_authorized"]
            else (
                "agreement_gate_passed_no_disagreements"
                if summary["agreement_gate_passed"]
                else "agreement_gate_failed_repair_required"
            )
        ),
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "execution": git,
        "input_hashes": observed_hashes,
        "secondary_independence_limit": (
            "The blind workflow and user attestation establish the independent role; "
            "the frozen seven-field label schema did not record secondary model identity."
        ),
        **summary,
        "outputs": {
            "adjudication_items": str(items_output.relative_to(ROOT)),
            "adjudication_items_sha256": file_sha256(items_output),
            "adjudication_lineage": str(lineage_output.relative_to(ROOT)),
            "adjudication_lineage_sha256": file_sha256(lineage_output),
        },
        "allowed_conclusion": (
            "The preregistered annotator-agreement gate passed, but every non-identical "
            "target row must be semantically adjudicated before gold materialization."
            if summary["agreement_gate_passed"]
            else "The preregistered annotator-agreement gate failed; repair the guide or "
            "unitization under a new version and do not use adjudication to rescue v1."
        ),
        "pilot_or_final_test_accessed": False,
        "formal_claim_allowed": False,
    }
    report_path = resolve(outputs["agreement_report"])
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
