#!/usr/bin/env python
"""Write a versioned GSM8K correctness manifest without changing feature payloads."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl
from src.clir_real_data import (
    check_gsm8k_response,
    file_sha256,
    load_protocol,
    protocol_hashes,
    validate_rollout_row,
)
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--label-protocol-config", default=None)
    parser.add_argument(
        "--checker-version",
        default="clir_gsm8k_numeric_v5",
        choices=(
            "clir_gsm8k_numeric_v2",
            "clir_gsm8k_numeric_v3",
            "clir_gsm8k_numeric_v4",
            "clir_gsm8k_numeric_v5",
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def relabel_rows(
    rows: list[Dict[str, Any]],
    *,
    checker_version: str,
    label_provenance: Dict[str, Any] | None = None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    output_rows: list[Dict[str, Any]] = []
    transitions: Counter[tuple[int, int]] = Counter()
    statuses: Counter[str] = Counter()
    changed: list[Dict[str, Any]] = []
    for row in rows:
        validate_rollout_row(row)
        if "raw_reference" not in row:
            raise ValueError(f"Row {row.get('id')!r} is missing raw_reference")
        previous = int(row["correctness"])
        checker = check_gsm8k_response(
            row["response"],
            row["raw_reference"],
            checker_version=checker_version,
        )
        current = int(checker["correctness"])
        transitions[(previous, current)] += 1
        statuses[checker["checker_status"]] += 1
        if previous != current:
            changed.append(
                {
                    "id": row["id"],
                    "query_id": row["query_id"],
                    "candidate_index": row.get("candidate_index"),
                    "previous_checker_version": row.get("checker_version"),
                    "previous_correctness": previous,
                    "correctness": current,
                    "parsed_answer": checker["parsed_answer"],
                    "normalized_candidate_answer": checker["normalized_candidate_answer"],
                    "reference_answer": checker["reference_answer"],
                }
            )
        updated = dict(row)
        updated.update(checker)
        if label_provenance is not None:
            updated["label_provenance"] = dict(label_provenance)
        output_rows.append(updated)

    report = {
        "schema_version": "clir-gsm8k-relabel-report-v1",
        "checker_version": checker_version,
        "rows": len(output_rows),
        "correct": sum(int(row["correctness"]) for row in output_rows),
        "incorrect": sum(not int(row["correctness"]) for row in output_rows),
        "transitions_previous_to_new": {
            f"{previous}->{current}": count
            for (previous, current), count in sorted(transitions.items())
        },
        "checker_statuses": dict(sorted(statuses.items())),
        "changed_count": len(changed),
        "changed_rows": changed,
        "code": git_state(PROJECT_ROOT),
    }
    return output_rows, report


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    report_path = (
        Path(args.output_report).resolve()
        if args.output_report
        else output_path.with_suffix(output_path.suffix + ".relabel.json")
    )
    for path in (output_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    label_provenance = None
    if args.label_protocol_config:
        protocol_path = Path(args.label_protocol_config).resolve()
        protocol = load_protocol(protocol_path)
        protocol_checker = protocol.get("correctness", {}).get("checker")
        if protocol_checker != args.checker_version:
            raise ValueError(
                f"Label protocol checker is {protocol_checker!r}, not {args.checker_version!r}"
            )
        label_provenance = {
            "schema_version": "clir-label-provenance-v1",
            "protocol_config": str(protocol_path),
            "protocol_version": protocol.get("protocol_version"),
            "label_protocol_sha256": protocol_hashes(protocol)["label_protocol_sha256"],
            "checker_version": args.checker_version,
            "code": git_state(PROJECT_ROOT),
        }

    rows, report = relabel_rows(
        read_jsonl(input_path),
        checker_version=args.checker_version,
        label_provenance=label_provenance,
    )
    atomic_write_jsonl(output_path, rows)
    report.update(
        {
            "input_jsonl": str(input_path),
            "input_sha256": file_sha256(input_path),
            "output_jsonl": str(output_path),
            "output_sha256": file_sha256(output_path),
            "label_provenance": label_provenance,
        }
    )
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
