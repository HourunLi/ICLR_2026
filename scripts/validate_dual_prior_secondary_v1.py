#!/usr/bin/env python3
"""Validate and normalize the independent secondary dual-prior annotations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_dual_prior_annotation import validate_annotation  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    items_path = resolve(protocol["outputs"]["annotation_items"])
    input_path = (
        args.input.resolve()
        if args.input is not None
        else resolve(protocol["annotation"]["secondary_output"])
    )
    output_path = input_path.with_name("labels_secondary_v1.jsonl")
    report_path = input_path.with_name("secondary_report_v1.json")
    items = read_jsonl(items_path)
    raw = read_jsonl(input_path)
    if len(raw) != len(items):
        raise ValueError(f"Secondary row count mismatch: expected {len(items)}, got {len(raw)}")
    labels = []
    eligibility: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    for row_number, (item, annotation) in enumerate(zip(items, raw), start=1):
        if annotation.get("item_id") != item["item_id"]:
            raise ValueError(f"Secondary item order/id drift at row {row_number}")
        normalized = validate_annotation(annotation, item)
        labels.append(normalized)
        eligibility[normalized["eligibility"]] += 1
        confidence[normalized["confidence"]] += 1
    atomic_write_jsonl(output_path, labels)
    minimum = int(protocol["selection_gate"]["minimum_usable_rows_per_annotator"])
    report = {
        "schema_version": "clir-dual-prior-evidence-secondary-report-v1",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "rows": len(labels),
        "structural_validation": "passed",
        "eligibility_counts": dict(sorted(eligibility.items())),
        "confidence_counts": dict(sorted(confidence.items())),
        "usable_yield_gate_passed": eligibility["usable"] >= minimum,
        "minimum_usable_rows": minimum,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
