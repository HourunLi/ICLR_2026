#!/usr/bin/env python3
"""Materialize a query-disjoint engineering train/validation split from development_32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl
from src.clir_real_data import file_sha256
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, load_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--train-queries", type=int, default=24)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [Path(args.train_output).resolve(), Path(args.validation_output).resolve(), Path(args.report).resolve()]
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite: {existing}")
    manifest = load_split_manifest(args.split_manifest)
    development_ids = list(manifest["memberships"]["development_32"])
    if not 0 < args.train_queries < len(development_ids):
        raise ValueError("train-queries must leave non-empty train and validation query sets")
    train_ids = development_ids[: args.train_queries]
    validation_ids = development_ids[args.train_queries :]
    rows = read_jsonl(args.input_jsonl)
    grouped = {query_id: [] for query_id in development_ids}
    for row in rows:
        query_id = str(row.get("query_id"))
        if query_id not in grouped:
            raise ValueError(f"Input contains query outside development_32: {query_id}")
        grouped[query_id].append(row)
    if any(not grouped[query_id] for query_id in development_ids):
        raise ValueError("Input is missing one or more development_32 queries")
    candidate_counts = {len(query_rows) for query_rows in grouped.values()}
    if len(candidate_counts) != 1:
        raise ValueError(f"Non-uniform candidate counts: {candidate_counts}")
    train_rows = [row for query_id in train_ids for row in grouped[query_id]]
    validation_rows = [row for query_id in validation_ids for row in grouped[query_id]]
    atomic_write_jsonl(outputs[0], train_rows)
    atomic_write_jsonl(outputs[1], validation_rows)
    report = {
        "schema_version": "clir-development-split-v1",
        "purpose": "engineering_only_not_formal_evaluation",
        "parent_membership": "development_32",
        "split_manifest_sha256": manifest["manifest_sha256"],
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "input_sha256": file_sha256(args.input_jsonl),
        "candidate_count_per_query": next(iter(candidate_counts)),
        "train": {
            "query_ids": train_ids,
            "queries": len(train_ids),
            "rows": len(train_rows),
            "output": str(outputs[0]),
            "sha256": file_sha256(outputs[0]),
        },
        "validation": {
            "query_ids": validation_ids,
            "queries": len(validation_ids),
            "rows": len(validation_rows),
            "output": str(outputs[1]),
            "sha256": file_sha256(outputs[1]),
        },
    }
    atomic_write_json(outputs[2], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
