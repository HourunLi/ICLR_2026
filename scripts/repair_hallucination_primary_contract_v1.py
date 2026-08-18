#!/usr/bin/env python3
"""Deterministically repair only primary annotation span/index contract drift."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hallucination_primary_v1 import parse_annotation  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    annotation_decision_signature,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    repair_annotation_contract,
    validate_annotation,
)
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/hallucination_localization_v1/protocol_v1.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Primary contract repair requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    outputs = protocol["outputs"]
    items_path = resolve(outputs["annotation_items"])
    run_root = resolve(outputs["primary_run_root"])
    raw_results_path = run_root / "all_results.jsonl"
    run_path = run_root / "run.json"
    resolved_results_path = run_root / "all_results_resolved_v1.jsonl"
    report_path = run_root / "contract_repair_report_v1.json"

    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    if run_record.get("status") != "complete":
        raise ValueError("Primary annotation run is incomplete")
    if run_record["frozen"]["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError("Primary run protocol hash drifted")
    if run_record["results_sha256"] != file_sha256(raw_results_path):
        raise ValueError("Primary raw-result hash drifted")

    items = read_jsonl(items_path)
    raw_results = read_jsonl(raw_results_path)
    if len(items) != len(raw_results) or len(items) != int(protocol["selection"]["rows"]):
        raise ValueError("Primary contract-repair row count drifted")

    resolved_results: list[dict[str, Any]] = []
    repaired_items: list[dict[str, Any]] = []
    operation_counts: Counter[str] = Counter()
    for row_number, (item, raw) in enumerate(zip(items, raw_results), start=1):
        if item["item_id"] != raw["item_id"]:
            raise ValueError(f"Primary row {row_number} identity/order drifted")
        if raw["parse_valid"]:
            validate_annotation(raw["annotation"], item)
            resolved_results.append(raw)
            continue

        parsed, original_parse_mode = parse_annotation(raw["completion_text"])
        before_signature = annotation_decision_signature(parsed)
        repaired, operations = repair_annotation_contract(parsed, item)
        if before_signature != annotation_decision_signature(repaired):
            raise AssertionError("Semantic decision signature changed during repair")
        for operation in operations:
            operation_counts[str(operation["operation"])] += 1
        repair_record = {
            "schema_version": "clir-hallucination-contract-repair-row-v1",
            "policy": "whitespace_quote_alignment_and_derived_first_problem_index_only",
            "raw_parse_error": raw["parse_error"],
            "raw_completion_sha256": raw["completion_sha256"],
            "semantic_decision_signature_sha256": canonical_sha256(before_signature),
            "operations": operations,
        }
        resolved = {
            **raw,
            "parse_valid": True,
            "parse_mode": f"{original_parse_mode}+deterministic_contract_repair_v1",
            "parse_error": None,
            "annotation": repaired,
            "contract_repair": repair_record,
        }
        resolved_results.append(resolved)
        repaired_items.append(
            {
                "item_id": item["item_id"],
                "row_number": row_number,
                "raw_parse_error": raw["parse_error"],
                "semantic_decision_signature_sha256": canonical_sha256(before_signature),
                "operations": operations,
            }
        )

    atomic_write_jsonl(resolved_results_path, resolved_results)
    reparsed = read_jsonl(resolved_results_path)
    if len(reparsed) != len(items):
        raise AssertionError("Resolved primary-result row count drifted after write")
    for item, result in zip(items, reparsed):
        if not result["parse_valid"] or result["annotation"] is None:
            raise AssertionError("Resolved primary result remains invalid")
        validate_annotation(result["annotation"], item)

    report = {
        "schema_version": "clir-hallucination-contract-repair-report-v1",
        "evidence_tier": "pipeline_pilot",
        "code": code,
        "protocol_sha256": file_sha256(protocol_path),
        "primary_run": str(run_path.relative_to(ROOT)),
        "primary_run_sha256": file_sha256(run_path),
        "raw_results": str(raw_results_path.relative_to(ROOT)),
        "raw_results_sha256": file_sha256(raw_results_path),
        "resolved_results": str(resolved_results_path.relative_to(ROOT)),
        "resolved_results_sha256": file_sha256(resolved_results_path),
        "rows": len(resolved_results),
        "raw_parse_valid": sum(int(row["parse_valid"]) for row in raw_results),
        "repaired_rows": len(repaired_items),
        "resolved_parse_valid": len(resolved_results),
        "operation_counts": dict(sorted(operation_counts.items())),
        "repaired_items": repaired_items,
        "semantic_judgments_changed": False,
        "policy_limitations": [
            "no path_status changes",
            "no claim status or reason changes",
            "no confidence or summary changes",
            "no fuzzy non-whitespace quote matching",
        ],
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
