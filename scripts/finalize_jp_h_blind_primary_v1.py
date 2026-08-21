#!/usr/bin/env python3
"""Publish the validated primary labels while secondary labels/scores stay sealed."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    read_jsonl,
    validate_annotation,
)
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/jp_h_blind_validation_v1/primary_annotation_protocol_v1.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def build_primary_labels(
    items: Sequence[Mapping[str, Any]],
    resolved_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(items) != 96 or len(resolved_results) != 96:
        raise ValueError("Primary finalization requires exactly 96 aligned rows")
    labels: list[dict[str, Any]] = []
    for row_number, (item, result) in enumerate(
        zip(items, resolved_results), start=1
    ):
        if item.get("item_id") != result.get("item_id"):
            raise ValueError(f"Primary finalization identity drifted at row {row_number}")
        annotation = result.get("annotation")
        if result.get("parse_valid") is not True or not isinstance(annotation, Mapping):
            raise ValueError(f"Primary result remains invalid at row {row_number}")
        validate_annotation(annotation, item)
        labels.append(dict(annotation))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Primary finalization requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    items_path = resolve(protocol["outputs"]["annotation_items"])
    run_root = resolve(protocol["outputs"]["primary_run_root"])
    run_path = run_root / "run.json"
    raw_path = run_root / "all_results.jsonl"
    resolved_path = run_root / "all_results_resolved_v1.jsonl"
    repair_report_path = run_root / "contract_repair_report_v1.json"
    labels_path = resolve(protocol["outputs"]["primary_labels_raw"])
    report_path = labels_path.with_name("primary_annotation_report_v1.json")
    if labels_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite finalized blind primary labels")

    run = json.loads(run_path.read_text(encoding="utf-8"))
    repair = json.loads(repair_report_path.read_text(encoding="utf-8"))
    if (
        run.get("status") != "complete"
        or run.get("items") != 96
        or run.get("results_sha256") != file_sha256(raw_path)
        or run.get("frozen", {}).get("protocol_sha256")
        != file_sha256(protocol_path)
    ):
        raise ValueError("Primary run provenance failed finalization")
    if (
        repair.get("status") != "passed"
        or repair.get("rows") != 96
        or repair.get("resolved_parse_valid") != 96
        or repair.get("resolved_results_sha256") != file_sha256(resolved_path)
        or repair.get("raw_results_sha256") != file_sha256(raw_path)
        or repair.get("secondary_labels_accessed") is not False
        or repair.get("model_predictions_accessed") is not False
        or repair.get("top_level_semantic_judgments_changed") is not False
        or repair.get("claim_status_or_reason_changed") is not False
    ):
        raise ValueError("Primary contract-repair provenance failed finalization")

    items = read_jsonl(items_path)
    resolved_results = read_jsonl(resolved_path)
    labels = build_primary_labels(items, resolved_results)
    atomic_write_jsonl(labels_path, labels)
    persisted = read_jsonl(labels_path)
    build_primary_labels(items, [{"item_id": row["item_id"], "parse_valid": True, "annotation": row} for row in persisted])

    path_counts = Counter(str(row["path_status"]) for row in labels)
    confidence_counts = Counter(str(row["confidence"]) for row in labels)
    claim_counts = Counter(
        str(claim["status"])
        for row in labels
        for claim in row["claim_reviews"]
    )
    report = {
        "schema_version": "clir-jp-h-blind-primary-annotation-report-v1",
        "status": "primary_complete_secondary_and_predictions_still_sealed",
        "evidence_tier": "blind_validation_candidate_primary",
        "code": code,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "items_sha256": file_sha256(items_path),
        "model_id": protocol["primary_annotator"]["model_id"],
        "model_revision": protocol["primary_annotator"]["model_revision"],
        "decoding": protocol["primary_annotator"]["decoding"],
        "raw_run": str(run_path.relative_to(ROOT)),
        "raw_run_sha256": file_sha256(run_path),
        "raw_results": str(raw_path.relative_to(ROOT)),
        "raw_results_sha256": file_sha256(raw_path),
        "raw_parse_valid": int(run["parse_valid"]),
        "resolved_results": str(resolved_path.relative_to(ROOT)),
        "resolved_results_sha256": file_sha256(resolved_path),
        "contract_repair_report": str(repair_report_path.relative_to(ROOT)),
        "contract_repair_report_sha256": file_sha256(repair_report_path),
        "contract_repaired_rows": int(repair["repaired_rows"]),
        "labels": str(labels_path.relative_to(ROOT)),
        "labels_sha256": file_sha256(labels_path),
        "rows": len(labels),
        "path_status_counts": dict(sorted(path_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "claim_status_counts": dict(sorted(claim_counts.items())),
        "primary_is_candidate_not_gold": True,
        "secondary_labels_accessed": False,
        "model_predictions_accessed": False,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
