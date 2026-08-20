#!/usr/bin/env python3
"""Freeze the row-id-only packing pool for the seed-42 JPH packing pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import (  # noqa: E402
    CLIRTrajectoryDataset,
    SemanticGroupBatchSampler,
    load_batch_packing_pools,
    read_jsonl,
)
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state  # noqa: E402


PARENT_PROTOCOL = ROOT / "configs/joint_training_drop_one_v1/training_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "configs/joint_training_packing_v1/mechanism_packing_pool_v1.jsonl"
DEFAULT_REPORT = ROOT / "configs/joint_training_packing_v1/packing_report_v1.json"
POOL_ID = "mechanism_explicit_hallucination_and_original_prior"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def is_mechanism_row(row: Mapping[str, Any]) -> bool:
    required = {
        "token_hallucination_target",
        "token_hallucination_mask",
        "key_prior_target",
        "complete_prior_target",
    }
    return required.issubset(row)


def semantic_pair_counts(rows: list[Mapping[str, Any]], batches: list[list[int]]) -> tuple[int, int]:
    positive = 0
    negative = 0
    for batch in batches:
        for left_position, left_index in enumerate(batch):
            left = rows[left_index]
            if "semantic_id" not in left or "style_id" not in left:
                continue
            for right_index in batch[left_position + 1 :]:
                right = rows[right_index]
                if "semantic_id" not in right or "style_id" not in right:
                    continue
                same_semantic = left["semantic_id"] == right["semantic_id"]
                same_style = left["style_id"] == right["style_id"]
                positive += int(same_semantic and not same_style)
                negative += int(not same_semantic and same_style)
    return positive, negative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, default=PARENT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    parent_path = args.parent_protocol.resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    train_spec = parent["manifests"]["train"]
    train_path = resolve(train_spec["path"])
    if file_sha256(train_path) != train_spec["sha256"]:
        raise ValueError("Frozen parent train manifest hash drifted")
    rows = read_jsonl(train_path)
    mechanism_indices = [index for index, row in enumerate(rows) if is_mechanism_row(row)]
    if len(rows) != 3968 or len(mechanism_indices) != 48:
        raise ValueError("Expected the frozen 3968-row stream with exactly 48 mechanism rows")
    if any(
        "semantic_id" in rows[index] or "style_id" in rows[index]
        for index in mechanism_indices
    ):
        raise ValueError("Mechanism packing rows must remain disjoint from semantic metadata")
    if len({str(rows[index]["id"]) for index in mechanism_indices}) != 48:
        raise ValueError("Mechanism row ids must be unique")

    output = args.output.resolve()
    report_path = args.report.resolve()
    if not args.force and (output.exists() or report_path.exists()):
        raise FileExistsError("Refusing to overwrite frozen packing artifacts")
    sidecar_rows = [
        {"id": str(rows[index]["id"]), "packing_pool_id": POOL_ID}
        for index in mechanism_indices
    ]
    atomic_write_jsonl(output, sidecar_rows)

    dataset = CLIRTrajectoryDataset(
        train_path,
        check_finite=False,
        require_correctness=True,
        load_condition=False,
        hidden_state_source="precomputed",
    )
    pools = load_batch_packing_pools(output, dataset)
    epoch_reports: list[dict[str, Any]] = []
    for epoch in range(1, 6):
        sampler = SemanticGroupBatchSampler(
            dataset,
            batch_size=4,
            shuffle=True,
            drop_last=False,
            seed=42,
            packing_pools=pools,
        )
        sampler.epoch = epoch - 1
        batches = list(sampler)
        flattened = [index for batch in batches for index in batch]
        if len(batches) != 992 or sorted(flattened) != list(range(3968)):
            raise ValueError(f"Epoch {epoch} does not cover the frozen stream exactly once")
        mechanism_counts = [
            sum(index in set(mechanism_indices) for index in batch)
            for batch in batches
        ]
        if set(mechanism_counts) - {0, 4} or mechanism_counts.count(4) != 12:
            raise ValueError(f"Epoch {epoch} failed exclusive mechanism packing")
        positive, negative = semantic_pair_counts(rows, batches)
        if positive != 27 or negative != 26:
            raise ValueError(f"Epoch {epoch} changed consistency pair coverage")
        epoch_reports.append(
            {
                "epoch": epoch,
                "batches": len(batches),
                "mechanism_active_batches": mechanism_counts.count(4),
                "mechanism_batch_row_counts": sorted(
                    count for count in mechanism_counts if count
                ),
                "consistency_positive_pairs": positive,
                "consistency_negative_pairs": negative,
            }
        )

    report = {
        "schema_version": "clir-joint-training-packing-report-v1",
        "status": "passed",
        "evidence_tier": "small-scale real integration pilot preparation",
        "parent_protocol": str(parent_path.relative_to(ROOT)),
        "parent_protocol_sha256": file_sha256(parent_path),
        "train_manifest": str(train_path.relative_to(ROOT)),
        "train_manifest_sha256": file_sha256(train_path),
        "sidecar": str(output.relative_to(ROOT)),
        "sidecar_sha256": file_sha256(output),
        "sidecar_schema": "id_and_packing_pool_id_jsonl",
        "pool_id": POOL_ID,
        "rows": len(sidecar_rows),
        "semantic_metadata_changed": False,
        "loss_targets_changed": False,
        "epoch_reports": epoch_reports,
        "code": git_state(ROOT),
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
