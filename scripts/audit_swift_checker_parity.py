#!/usr/bin/env python
"""Compare CLIR GSM8K labels with SWIFT's checker at a local pinned checkout."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl
from src.clir_real_data import check_gsm8k_response, extract_gsm8k_reference
from src.clir_stage_a import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--swift-repo", required=True)
    parser.add_argument("--expected-swift-commit", default=None)
    parser.add_argument(
        "--checker-version",
        default="clir_gsm8k_numeric_v4",
        choices=(
            "clir_gsm8k_numeric_v2",
            "clir_gsm8k_numeric_v3",
            "clir_gsm8k_numeric_v4",
        ),
    )
    parser.add_argument("--max-disagreements", type=int, default=20)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    swift_repo = Path(args.swift_repo).resolve()
    if not (swift_repo / "generate" / "generate_utils" / "__init__.py").exists():
        raise FileNotFoundError(f"Not a SWIFT checkout: {swift_repo}")
    if args.expected_swift_commit:
        import subprocess

        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=swift_repo, text=True
        ).strip()
        if actual != args.expected_swift_commit:
            raise ValueError(f"SWIFT commit mismatch: expected {args.expected_swift_commit}, got {actual}")

    sys.path.insert(0, str(swift_repo))
    evaluate_math = importlib.import_module("generate.generate_utils").evaluate_math

    rows = [row for path in args.input_jsonl for row in read_jsonl(path)]
    confusion: Counter[tuple[int, int]] = Counter()
    disagreements = []
    clir_statuses: Counter[str] = Counter()
    for row in rows:
        clir = check_gsm8k_response(
            row["response"],
            row["raw_reference"],
            checker_version=args.checker_version,
        )
        _, swift_correct, swift_answer = evaluate_math(
            row["response"], extract_gsm8k_reference(row["raw_reference"])
        )
        pair = (int(clir["correctness"]), int(bool(swift_correct)))
        confusion[pair] += 1
        clir_statuses[clir["checker_status"]] += 1
        if pair[0] != pair[1]:
            disagreements.append(
                {
                    "id": row["id"],
                    "clir_correctness": pair[0],
                    "swift_correctness": pair[1],
                    "parsed_answer": clir["parsed_answer"],
                    "normalized_candidate_answer": clir["normalized_candidate_answer"],
                    "swift_parsed_answer": swift_answer,
                    "reference_answer": clir["reference_answer"],
                }
            )

    total = len(rows)
    report = {
        "checker_version": args.checker_version,
        "rows": total,
        "agreement_count": total - len(disagreements),
        "agreement_rate": (total - len(disagreements)) / max(total, 1),
        "confusion_clir_swift": {
            f"{clir_label},{swift_label}": count
            for (clir_label, swift_label), count in sorted(confusion.items())
        },
        "clir_statuses": dict(sorted(clir_statuses.items())),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements[: args.max_disagreements],
    }
    if args.output_json:
        output = Path(args.output_json).resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
        atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
