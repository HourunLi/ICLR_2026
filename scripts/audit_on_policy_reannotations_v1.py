#!/usr/bin/env python3
"""Validate one or two independent Route A v1a claim-level label files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from src.clir_data import read_jsonl
from src.clir_on_policy_reaudit import validate_annotation_set


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "configs/on_policy_pilot0_v1a"


def _kappa(first: list[str], second: list[str]) -> float:
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    values = {"accepted", "rejected", "review_required"}
    expected = sum(first_counts[key] * second_counts[key] for key in values) / len(first) ** 2
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def _statuses(rows: list[dict[str, Any]]) -> list[str]:
    return [row["acceptance"]["status"] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--secondary", type=Path)
    args = parser.parse_args()
    items = read_jsonl(args.root / "annotation_items_v1.jsonl")
    lineages = read_jsonl(args.root / "annotation_lineage_v1.jsonl")
    result: dict[str, Any] = {
        "schema_version": "clir-on-policy-reannotation-audit-v1",
        "items": len(items),
        "items_and_lineage_valid": True,
    }
    primary_rows = None
    if args.primary:
        primary_rows = validate_annotation_set(
            read_jsonl(args.primary), items=items, lineages=lineages
        )
        result.update(
            {
                "primary_valid": True,
                "primary_status_counts": dict(sorted(Counter(_statuses(primary_rows)).items())),
            }
        )
    if args.secondary:
        if primary_rows is None:
            raise ValueError("--secondary requires --primary for blind comparison")
        secondary_rows = validate_annotation_set(
            read_jsonl(args.secondary), items=items, lineages=lineages
        )
        first = _statuses(primary_rows)
        second = _statuses(secondary_rows)
        disagreements = [
            {
                "item_id": item["item_id"],
                "primary_status": left,
                "secondary_status": right,
            }
            for item, left, right in zip(items, first, second)
            if left != right
        ]
        result.update(
            {
                "secondary_valid": True,
                "secondary_status_counts": dict(sorted(Counter(second).items())),
                "status_agreement": 1.0 - len(disagreements) / len(items),
                "cohen_kappa": _kappa(first, second),
                "status_disagreements": len(disagreements),
                "disagreements": disagreements,
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
