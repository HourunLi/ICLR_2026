#!/usr/bin/env python3
"""Expand the manually reviewed Route A v1a material-claim specs to strict labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.clir_data import read_jsonl
from src.clir_on_policy_pairs import canonical_json
from src.clir_on_policy_reaudit import (
    ANNOTATION_SCHEMA,
    TARGET_STYLE,
    validate_annotation_set,
)
from src.clir_reasoning_rewrite import VERIFIER_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "configs/on_policy_pilot0_v1a"
GOOD_SOURCE_RELATIONS = {"equivalent", "preserved_implicitly"}
GOOD_REWRITE_RELATIONS = {"equivalent", "entailed_elaboration"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claim(
    *,
    claim_id: str,
    claim_text_key: str,
    claim_text: str,
    applicability_key: str,
    evidence_key: str,
    evidence: str,
    relation: str,
    explanation: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        claim_text_key: claim_text,
        applicability_key: True,
        evidence_key: evidence,
        "relation": relation,
        "explanation": explanation,
    }


def _risk_review(item: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    unresolved = spec.get("unresolved_risks", {})
    notes = spec.get("risk_notes", {})
    expected = item["risk_probe"]["expected_risk_ids"]
    unknown = (set(unresolved) | set(notes)) - set(expected)
    if unknown:
        raise ValueError(f"{item['item_id']} has notes for unknown risks: {sorted(unknown)}")
    result = []
    by_id = {row["risk_id"]: row for row in item["risk_probe"]["risks"]}
    for risk_id in expected:
        if risk_id in unresolved:
            resolved = False
            explanation = unresolved[risk_id]
        else:
            resolved = True
            explanation = notes.get(
                risk_id,
                "Locally checked the signal "
                f"{json.dumps(by_id[risk_id]['details'], ensure_ascii=False, sort_keys=True)}; "
                "it reflects repetition or equivalent re-expression and does not change a material claim.",
            )
        result.append(
            {"risk_id": risk_id, "resolved": resolved, "explanation": explanation}
        )
    return result


def _build_report(item: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    source_claims: list[dict[str, Any]] = []
    rewrite_claims: list[dict[str, Any]] = []
    for index, aligned in enumerate(spec["aligned_claims"], start=1):
        source_relation = aligned.get("source_relation", "equivalent")
        rewrite_relation = aligned.get("rewrite_relation", "equivalent")
        source_claims.append(
            _claim(
                claim_id=f"S{index}",
                claim_text_key="source_claim",
                claim_text=aligned["source_claim"],
                applicability_key="essential",
                evidence_key="rewrite_evidence",
                evidence=aligned["rewrite_claim"],
                relation=source_relation,
                explanation=aligned["explanation"],
            )
        )
        rewrite_claims.append(
            _claim(
                claim_id=f"R{index}",
                claim_text_key="rewrite_claim",
                claim_text=aligned["rewrite_claim"],
                applicability_key="substantive",
                evidence_key="source_evidence",
                evidence=aligned["source_claim"],
                relation=rewrite_relation,
                explanation=aligned["explanation"],
            )
        )
    for extra in spec.get("rewrite_only_claims", []):
        rewrite_claims.append(
            _claim(
                claim_id=f"R{len(rewrite_claims) + 1}",
                claim_text_key="rewrite_claim",
                claim_text=extra["rewrite_claim"],
                applicability_key="substantive",
                evidence_key="source_evidence",
                evidence=extra.get("source_evidence", ""),
                relation=extra["relation"],
                explanation=extra["explanation"],
            )
        )

    global_relation = {
        "same_task_and_goal": True,
        "same_core_premises": True,
        "same_reasoning_method": True,
        "same_key_inferences": True,
        "same_intermediate_conclusions": True,
        "same_final_conclusion": True,
        "introduced_external_fact": False,
        "omitted_essential_claim": False,
        "contradicted_source_claim": False,
        "replaced_with_different_solution": False,
        "introduced_new_error": False,
    }
    global_relation.update(spec.get("global_overrides", {}))
    error_alignment = spec.get(
        "error_alignment",
        {
            "applicable": False,
            "source_error": None,
            "rewrite_error": None,
            "same_error_mechanism": None,
            "same_semantic_error_location": None,
            "same_downstream_effect": None,
        },
    )
    return {
        "schema_version": VERIFIER_SCHEMA,
        "source_to_rewrite": {
            "claims": source_claims,
            "all_essential_claims_preserved": all(
                row["relation"] in GOOD_SOURCE_RELATIONS for row in source_claims
            ),
        },
        "rewrite_to_source": {
            "claims": rewrite_claims,
            "all_substantive_claims_supported": all(
                row["relation"] in GOOD_REWRITE_RELATIONS for row in rewrite_claims
            ),
        },
        "global_relation": global_relation,
        "error_alignment": error_alignment,
        "style_assessment": {
            "target_style": TARGET_STYLE,
            "satisfied": spec.get("style_satisfied", True),
            "evidence": spec["style_evidence"],
        },
        "risk_review": _risk_review(item, spec),
        "confidence": spec.get("confidence", "high"),
        "decision": spec.get("decision", "accept"),
        "issues": spec.get("issues", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    items = read_jsonl(args.root / "annotation_items_v1.jsonl")
    lineages = read_jsonl(args.root / "annotation_lineage_v1.jsonl")
    specs_doc = json.loads(
        (args.root / "primary_material_claims_v1.json").read_text(encoding="utf-8")
    )
    specs = specs_doc["items"]
    if [row["item_id"] for row in specs] != [row["item_id"] for row in items]:
        raise ValueError("Primary material-claim specs do not match item order")
    annotations = [
        {
            "schema_version": ANNOTATION_SCHEMA,
            "item_id": item["item_id"],
            "report": _build_report(item, spec),
        }
        for item, spec in zip(items, specs)
    ]
    validated = validate_annotation_set(
        annotations, items=items, lineages=lineages
    )
    output = args.root / "labels_primary_v1.jsonl"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in annotations),
        encoding="utf-8",
    )
    temporary.replace(output)
    counts: dict[str, int] = {}
    for row in validated:
        status = row["acceptance"]["status"]
        counts[status] = counts.get(status, 0) + 1
    report = {
        "schema_version": "clir-on-policy-primary-annotation-report-v1",
        "evidence_tier": "pipeline_pilot",
        "annotation_policy": specs_doc["policy"],
        "annotator": specs_doc["annotator"],
        "annotation_date": specs_doc["annotation_date"],
        "rows": len(annotations),
        "structural_checks_passed": True,
        "derived_status_counts": dict(sorted(counts.items())),
        "guide_sha256": _sha256(args.root / "annotation_guide_v1.md"),
        "items_sha256": _sha256(args.root / "annotation_items_v1.jsonl"),
        "lineage_sha256": _sha256(args.root / "annotation_lineage_v1.jsonl"),
        "material_claim_specs_sha256": _sha256(
            args.root / "primary_material_claims_v1.json"
        ),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": _sha256(output),
        "training_ready": False,
        "blocking_gate": "blind_secondary_annotation_and_adjudication",
    }
    report_path = args.root / "primary_report_v1.json"
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_report.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
