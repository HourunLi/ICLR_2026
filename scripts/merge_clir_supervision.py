#!/usr/bin/env python
"""Merge versioned external CLIR supervision into an immutable trajectory manifest."""

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
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state
from src.clir_supervision import audit_supervision_coverage, merge_supervision_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--annotations-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--expected-input-sha256", default=None)
    parser.add_argument("--expected-annotations-sha256", default=None)
    parser.add_argument("--expected-reconstruction-dim", type=int, default=None)
    parser.add_argument("--require-all-rows-annotated", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    annotations_path = Path(args.annotations_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    report_path = Path(args.output_report).resolve()
    named_paths = {
        "input": input_path,
        "annotations": annotations_path,
        "output": output_path,
        "report": report_path,
    }
    if len(set(named_paths.values())) != len(named_paths):
        raise ValueError(f"Input and output paths must be distinct: {named_paths}")
    if args.expected_reconstruction_dim is not None and args.expected_reconstruction_dim <= 0:
        raise ValueError("expected_reconstruction_dim must be positive")
    for path in (output_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    input_sha256 = file_sha256(input_path)
    annotations_sha256 = file_sha256(annotations_path)
    if args.expected_input_sha256 and input_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"Input SHA256 mismatch: expected {args.expected_input_sha256}, got {input_sha256}"
        )
    if args.expected_annotations_sha256 and annotations_sha256 != args.expected_annotations_sha256:
        raise ValueError(
            "Annotations SHA256 mismatch: expected "
            f"{args.expected_annotations_sha256}, got {annotations_sha256}"
        )

    rows = read_jsonl(input_path)
    annotations = read_jsonl(annotations_path)
    code = git_state(PROJECT_ROOT)
    annotation_artifact = {
        "annotations_jsonl": str(annotations_path),
        "annotations_sha256": annotations_sha256,
        "merge_code": code,
    }
    merged = merge_supervision_annotations(
        rows,
        annotations,
        annotation_artifact=annotation_artifact,
        expected_reconstruction_dim=args.expected_reconstruction_dim,
        require_all_rows_annotated=args.require_all_rows_annotated,
    )
    coverage = audit_supervision_coverage(
        merged,
        expected_reconstruction_dim=args.expected_reconstruction_dim,
        require_provenance=True,
    )
    atomic_write_jsonl(output_path, merged)
    report = {
        "schema_version": "clir-supervision-merge-report-v1",
        "input_jsonl": str(input_path),
        "input_sha256": input_sha256,
        "annotations_jsonl": str(annotations_path),
        "annotations_sha256": annotations_sha256,
        "annotation_rows": len(annotations),
        "output_jsonl": str(output_path),
        "output_sha256": file_sha256(output_path),
        "output_rows": len(merged),
        "require_all_rows_annotated": args.require_all_rows_annotated,
        "expected_reconstruction_dim": args.expected_reconstruction_dim,
        "coverage": coverage,
        "code": code,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
