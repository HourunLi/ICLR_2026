#!/usr/bin/env python3
"""Validate and normalize the complete role-blind dual-prior adjudication output."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_dual_prior_adjudication import validate_adjudication  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/dual_prior_evidence_v1/agreement_adjudication_protocol_v1.json"
)
DEFAULT_INPUT = ROOT / "configs/dual_prior_evidence_v1/adjudications_raw_v1.jsonl"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    agreement_path = resolve(protocol["outputs"]["agreement_report"])
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    if not agreement.get("agreement_gate_passed") or not agreement.get(
        "adjudication_authorized"
    ):
        raise ValueError("Agreement report does not authorize adjudication")
    packet_path = resolve(protocol["outputs"]["adjudication_items"])
    if file_sha256(packet_path) != agreement["outputs"][
        "adjudication_items_sha256"
    ]:
        raise ValueError("Adjudication packet hash differs from agreement report")
    packets = read_jsonl(packet_path)
    input_path = args.input.resolve()
    raw = read_jsonl(input_path)
    if len(raw) != len(packets):
        raise ValueError(
            f"Adjudication row count mismatch: expected {len(packets)}, got {len(raw)}"
        )

    normalized = []
    resolutions: Counter[str] = Counter()
    eligibility: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    for row_number, (packet, row) in enumerate(zip(packets, raw), start=1):
        if row.get("item_id") != packet["item_id"]:
            raise ValueError(f"Adjudication item order/id drift at row {row_number}")
        value = validate_adjudication(row, packet)
        normalized.append(value)
        resolutions[value["resolution"]] += 1
        eligibility[value["eligibility"]] += 1
        confidence[value["confidence"]] += 1

    output_path = input_path.with_name("adjudications_v1.jsonl")
    report_path = input_path.with_name("adjudication_report_v1.json")
    atomic_write_jsonl(output_path, normalized)
    report = {
        "schema_version": "clir-dual-prior-evidence-adjudication-report-v1",
        "status": "complete_and_structurally_valid",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "agreement_report": str(agreement_path.relative_to(ROOT)),
        "agreement_report_sha256": file_sha256(agreement_path),
        "adjudication_packet_sha256": file_sha256(packet_path),
        "input": str(input_path.relative_to(ROOT)),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path.relative_to(ROOT)),
        "output_sha256": file_sha256(output_path),
        "rows": len(normalized),
        "resolution_counts": dict(sorted(resolutions.items())),
        "eligibility_counts": dict(sorted(eligibility.items())),
        "confidence_counts": dict(sorted(confidence.items())),
        "structural_validation": "passed",
        "all_disagreements_resolved": len(normalized) == int(
            agreement["disagreement_rows"]
        ),
        "ready_for_gold_materialization": True,
        "allowed_conclusion": "Every v1 target-set disagreement now has one valid role-blind semantic adjudication; deterministic gold materialization may begin, but no learnability result exists yet.",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
