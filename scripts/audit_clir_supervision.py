#!/usr/bin/env python
"""Audit real CLIR auxiliary-target coverage without loading feature payloads."""

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
from src.clir_stage_a import atomic_write_json, git_state
from src.clir_supervision import audit_supervision_coverage


COMPONENTS = (
    "consistency",
    "path_hallucination",
    "onset_localization",
    "token_advantage",
    "progress",
    "dual_prior",
    "reconstruction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--expected-input-sha256", default=None)
    parser.add_argument("--expected-reconstruction-dim", type=int, default=None)
    parser.add_argument("--require", nargs="*", choices=COMPONENTS, default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_json).resolve() if args.output_json else None
    if output_path == input_path:
        raise ValueError("Audit output must differ from the input JSONL")
    if output_path is not None and output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing audit: {output_path}")
    input_sha256 = file_sha256(input_path)
    if args.expected_input_sha256 and input_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"Input SHA256 mismatch: expected {args.expected_input_sha256}, got {input_sha256}"
        )
    coverage = audit_supervision_coverage(
        read_jsonl(input_path),
        expected_reconstruction_dim=args.expected_reconstruction_dim,
        require_provenance=True,
    )
    report = {
        "schema_version": "clir-supervision-audit-v1",
        "input_jsonl": str(input_path),
        "input_sha256": input_sha256,
        "required_components": list(args.require),
        "expected_reconstruction_dim": args.expected_reconstruction_dim,
        "coverage": coverage,
        "code": git_state(PROJECT_ROOT),
    }
    failed = [
        component
        for component in args.require
        if not coverage["eligible_components"][component]
    ]
    report["failed_required_components"] = failed
    if output_path is not None:
        atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise RuntimeError(f"Missing required CLIR supervision components: {failed}")


if __name__ == "__main__":
    main()
