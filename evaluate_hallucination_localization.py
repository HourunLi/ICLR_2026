#!/usr/bin/env python3
"""Evaluate CLIR hallucination path, tail-token, onset, and value predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.clir_data import read_jsonl
from src.clir_localization_evaluation import evaluate_localization_rows
from src.clir_real_data import file_sha256
from src.clir_stage_a import atomic_write_json, git_state


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-input-sha256", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--path-log-threshold", type=float, default=None)
    parser.add_argument("--token-threshold", type=float, default=None)
    parser.add_argument("--onset-threshold", type=float, default=None)
    parser.add_argument("--negative-tail-margin", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.negative_tail_margin < 0.0:
        raise ValueError("negative-tail-margin must be non-negative")
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_json).resolve()
    if input_path == output_path:
        raise ValueError("Localization evaluation output must differ from scored input")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite localization evaluation: {output_path}")
    observed_sha256 = file_sha256(input_path)
    if args.expected_input_sha256 and observed_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"Scored input SHA256 mismatch: expected {args.expected_input_sha256}, "
            f"got {observed_sha256}"
        )
    report = {
        "schema_version": "clir-hallucination-localization-evaluation-v1",
        "evidence_tier": "pipeline_pilot",
        "input_jsonl": str(input_path),
        "input_sha256": observed_sha256,
        "metrics": evaluate_localization_rows(
            read_jsonl(input_path),
            threshold=args.threshold,
            negative_tail_margin=args.negative_tail_margin,
            path_log_threshold=args.path_log_threshold,
            token_threshold=args.token_threshold,
            onset_threshold=args.onset_threshold,
        ),
        "code": git_state(ROOT),
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
