#!/usr/bin/env python
"""Validate and merge query-atomic Stage A shards in frozen membership order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl
from src.clir_real_data import (
    file_sha256,
    load_protocol,
    protocol_hashes,
    validate_protocol_reference,
    validate_rollout_row,
)
from src.clir_stage_a import (
    atomic_write_json,
    atomic_write_jsonl,
    load_split_manifest,
    membership_entries,
    query_shard_dir,
    validate_completion_marker,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phi35_gsm8k_pilot_v4.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--stage", choices=("rollout", "extraction"), required=True)
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _summary(values: list[int]) -> Dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": statistics.fmean(values), "max": max(values)}


def validate_candidate_index_policy(
    rows: list[Mapping[str, Any]],
    expected_policy: str | None,
    query_id: str,
) -> None:
    if expected_policy is None:
        return
    actual_policies = {
        row.get("generation", {}).get("candidate_index_policy")
        for row in rows
    }
    if actual_policies != {expected_policy}:
        raise ValueError(
            f"Candidate index policy mismatch in {query_id}: expected {expected_policy!r}, "
            f"got {sorted(map(str, actual_policies))}"
        )


def merge_query_shards(
    *,
    protocol_path: str | Path,
    split_manifest_path: str | Path,
    membership: str,
    shard_root: str | Path,
    stage: str,
    expected_candidates: int,
    max_queries: int | None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    if expected_candidates <= 0:
        raise ValueError("expected_candidates must be positive")
    protocol = load_protocol(protocol_path)
    hashes = protocol_hashes(protocol)
    protocol_hash = hashes["protocol_sha256"]
    acquisition_hash = hashes["acquisition_protocol_sha256"]
    split_manifest = load_split_manifest(split_manifest_path)
    split_hash = split_manifest["manifest_sha256"]
    validate_protocol_reference(split_manifest, protocol)
    expected_candidate_policy = protocol.get("generation", {}).get("candidate_index_policy")
    entries = membership_entries(split_manifest, membership, max_queries=max_queries)
    if not entries:
        raise ValueError("Selected membership is empty")

    marker_name = "_ROLLOUT_SUCCESS.json" if stage == "rollout" else "_EXTRACTION_SUCCESS.json"
    all_rows: list[Dict[str, Any]] = []
    feature_bytes = 0
    per_query: list[Dict[str, Any]] = []
    for entry in entries:
        query_id = entry["query_id"]
        query_dir = query_shard_dir(shard_root, query_id)
        marker = validate_completion_marker(
            query_dir,
            marker_name,
            stage=stage,
            query_id=query_id,
            protocol_sha256=protocol_hash,
            acquisition_protocol_sha256=acquisition_hash,
            label_protocol_sha256=hashes["label_protocol_sha256"],
            split_manifest_sha256=split_hash,
            expected_candidate_count=expected_candidates,
            rows_loader=read_jsonl,
            row_validator=validate_rollout_row,
        )
        rows = [dict(row) for row in read_jsonl(marker["_rows_path"])]
        rows.sort(key=lambda row: int(row["candidate_index"]))
        validate_candidate_index_policy(rows, expected_candidate_policy, query_id)
        labels = [int(row["correctness"]) for row in rows]
        if any(label not in (0, 1) for label in labels):
            raise ValueError(f"Non-binary correctness label in {query_id}")
        per_query.append({
            "query_id": query_id,
            "source_split": entry["source_split"],
            "source_index": entry["source_index"],
            "candidates": len(rows),
            "correct": sum(labels),
            "incorrect": len(labels) - sum(labels),
            "pool_type": (
                "all_correct" if all(labels) else "all_wrong" if not any(labels) else "mixed"
            ),
        })
        feature_bytes += int(marker.get("stats", {}).get("feature_bytes", 0))
        all_rows.extend(rows)

    output_lengths = [len(row["output_token_ids"]) for row in all_rows]
    prompt_lengths = [len(row["prompt_token_ids"]) for row in all_rows]
    labels = [int(row["correctness"]) for row in all_rows]
    pool_counts = {
        pool_type: sum(item["pool_type"] == pool_type for item in per_query)
        for pool_type in ("mixed", "all_correct", "all_wrong")
    }
    report: Dict[str, Any] = {
        "schema_version": "clir-stage-a-merge-report-v1",
        "stage": stage,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_hash,
        "acquisition_protocol_sha256": acquisition_hash,
        "label_protocol_sha256": hashes["label_protocol_sha256"],
        "split_manifest_sha256": split_hash,
        "membership": membership,
        "membership_sha256": split_manifest["membership_sha256"][membership],
        "max_queries": max_queries,
        "candidate_count_per_query": expected_candidates,
        "candidate_index_policy": expected_candidate_policy,
        "queries": len(entries),
        "rows": len(all_rows),
        "correct": sum(labels),
        "incorrect": len(labels) - sum(labels),
        "accuracy": sum(labels) / len(labels),
        "pool_counts": pool_counts,
        "prompt_tokens": _summary(prompt_lengths),
        "output_tokens": _summary(output_lengths),
        "feature_bytes": feature_bytes if stage == "extraction" else None,
        "uniform_feature_metadata": None,
        "per_query": per_query,
    }
    if stage == "extraction":
        metadata = {
            (
                row.get("feature_metadata", {}).get("layer_count"),
                row.get("feature_metadata", {}).get("per_layer_hidden_size"),
                row.get("feature_metadata", {}).get("feature_dim"),
                row.get("feature_metadata", {}).get("storage_dtype"),
            )
            for row in all_rows
        }
        if len(metadata) != 1 or None in next(iter(metadata)):
            raise ValueError(f"Extraction rows have non-uniform feature metadata: {metadata}")
        layer_count, per_layer_dim, feature_dim, dtype = next(iter(metadata))
        report["uniform_feature_metadata"] = {
            "layer_count": layer_count,
            "per_layer_hidden_size": per_layer_dim,
            "feature_dim": feature_dim,
            "storage_dtype": dtype,
        }
    return all_rows, report


def main() -> None:
    args = parse_args()
    output = Path(args.output_jsonl).resolve()
    report_path = (
        Path(args.output_report).resolve()
        if args.output_report
        else output.with_suffix(output.suffix + ".report.json")
    )
    for path in (output, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    rows, report = merge_query_shards(
        protocol_path=args.protocol_config,
        split_manifest_path=args.split_manifest,
        membership=args.membership,
        shard_root=args.shard_root,
        stage=args.stage,
        expected_candidates=args.expected_candidates,
        max_queries=args.max_queries,
    )
    atomic_write_jsonl(output, rows)
    report["output_jsonl"] = str(output)
    report["output_sha256"] = file_sha256(output)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
