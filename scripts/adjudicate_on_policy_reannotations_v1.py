#!/usr/bin/env python3
"""Resolve only Route A v1a status disagreements and freeze final relations."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from src.clir_data import read_jsonl
from src.clir_on_policy_pairs import canonical_json
from src.clir_on_policy_reaudit import validate_annotation_set


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "configs/on_policy_pilot0_v1a"
ADJUDICATION_SCHEMA = "clir-on-policy-reannotation-adjudication-v1"
FINAL_SCHEMA = "clir-on-policy-adjudicated-relation-v1"
VALID_STATUSES = {"accepted", "rejected", "review_required"}
ADJUDICATION_KEYS = {
    "schema_version",
    "item_id",
    "primary_status",
    "secondary_status",
    "final_status",
    "adjudicator",
    "rationale",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _validate_adjudication(row: dict[str, Any]) -> None:
    if set(row) != ADJUDICATION_KEYS:
        raise ValueError("Adjudication keys differ from the required schema")
    if row["schema_version"] != ADJUDICATION_SCHEMA:
        raise ValueError("Adjudication schema_version drifted")
    for key in ("item_id", "adjudicator", "rationale"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"Adjudication {key} must be non-empty")
    for key in ("primary_status", "secondary_status", "final_status"):
        if row[key] not in VALID_STATUSES:
            raise ValueError(f"Adjudication {key} is invalid")
    if row["primary_status"] == row["secondary_status"]:
        raise ValueError("Consensus items must not receive an adjudication row")


def _kappa(first: list[str], second: list[str]) -> float:
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        first_counts[key] * second_counts[key] for key in VALID_STATUSES
    ) / len(first) ** 2
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    items_path = args.root / "annotation_items_v1.jsonl"
    lineage_path = args.root / "annotation_lineage_v1.jsonl"
    primary_path = args.root / "labels_primary_v1.jsonl"
    secondary_path = args.root / "labels_secondary_v1.jsonl"
    adjudication_path = args.root / "adjudications_v1.jsonl"
    items = read_jsonl(items_path)
    lineages = read_jsonl(lineage_path)
    primary = validate_annotation_set(
        read_jsonl(primary_path), items=items, lineages=lineages
    )
    secondary = validate_annotation_set(
        read_jsonl(secondary_path), items=items, lineages=lineages
    )
    primary_statuses = [row["acceptance"]["status"] for row in primary]
    secondary_statuses = [row["acceptance"]["status"] for row in secondary]
    disagreements = {
        item["item_id"]: (left, right)
        for item, left, right in zip(items, primary_statuses, secondary_statuses)
        if left != right
    }
    adjudications = read_jsonl(adjudication_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in adjudications:
        _validate_adjudication(row)
        if row["item_id"] in by_id:
            raise ValueError(f"Duplicate adjudication for {row['item_id']!r}")
        by_id[row["item_id"]] = row
    if set(by_id) != set(disagreements):
        raise ValueError(
            "Adjudication IDs must exactly equal status disagreements: "
            f"missing={sorted(set(disagreements) - set(by_id))}, "
            f"unexpected={sorted(set(by_id) - set(disagreements))}"
        )
    for item_id, (left, right) in disagreements.items():
        row = by_id[item_id]
        if row["primary_status"] != left or row["secondary_status"] != right:
            raise ValueError(f"Adjudication source statuses drifted for {item_id}")

    final_rows = []
    for item, primary_status, secondary_status in zip(
        items, primary_statuses, secondary_statuses
    ):
        item_id = item["item_id"]
        adjudication = by_id.get(item_id)
        final_status = (
            adjudication["final_status"] if adjudication else primary_status
        )
        final_rows.append(
            {
                "schema_version": FINAL_SCHEMA,
                "item_id": item_id,
                "primary_status": primary_status,
                "secondary_status": secondary_status,
                "final_status": final_status,
                "resolution": "adjudicated" if adjudication else "consensus",
                "adjudication_sha256": (
                    hashlib.sha256(canonical_json(adjudication).encode("utf-8")).hexdigest()
                    if adjudication
                    else None
                ),
            }
        )
    output = args.root / "adjudicated_relations_v1.jsonl"
    write_atomic(output, "".join(canonical_json(row) + "\n" for row in final_rows))
    final_counts = Counter(row["final_status"] for row in final_rows)
    report = {
        "schema_version": "clir-on-policy-reannotation-adjudication-report-v1",
        "evidence_tier": "pipeline_pilot",
        "items": len(items),
        "primary_status_counts": dict(sorted(Counter(primary_statuses).items())),
        "secondary_status_counts": dict(sorted(Counter(secondary_statuses).items())),
        "status_agreement": 1.0 - len(disagreements) / len(items),
        "cohen_kappa": _kappa(primary_statuses, secondary_statuses),
        "status_disagreements": len(disagreements),
        "adjudications": len(adjudications),
        "final_status_counts": dict(sorted(final_counts.items())),
        "items_sha256": sha256(items_path),
        "lineage_sha256": sha256(lineage_path),
        "primary_labels_sha256": sha256(primary_path),
        "secondary_labels_sha256": sha256(secondary_path),
        "adjudications_sha256": sha256(adjudication_path),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256(output),
        "training_relation_gate_passed": final_counts.get("accepted", 0) >= 2
        and final_counts.get("review_required", 0) == 0,
    }
    report_path = args.root / "adjudication_report_v1.json"
    write_atomic(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
