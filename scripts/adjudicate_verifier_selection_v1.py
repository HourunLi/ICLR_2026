#!/usr/bin/env python3
"""Freeze v1 gold from independent decisions and explicit adjudications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_verifier_selection_v1 import load_jsonl, validate_labels


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs" / "verifier_selection_v1"

ADJUDICATIONS = {
    "VS002": ("accept", "The target is explanatory_conversational, not terse; the rewrite clearly satisfies it and preserves the multiplication reasoning."),
    "VS004": ("accept", "The equal-group explanation is natural and sufficiently conversational while preserving the method and conclusion."),
    "VS012": ("reject", "The rewrite repairs 5×7, invents missing pens, and reaches 30 through a different error mechanism."),
    "VS018": ("accept", "The target is explanatory_conversational; the filter-then-map explanation satisfies it and preserves execution order."),
    "VS020": ("accept", "Removing 1 is directly entailed by the filter and the explanatory wording is sufficient."),
    "VS028": ("reject", "Simultaneous updates and a 14–20 range replace both the reversed-order error and the definite conclusion."),
    "VS034": ("accept", "The target is explanatory_conversational; the closure-to-no-entry bridge is directly entailed and style-compliant."),
    "VS036": ("accept", "The rewrite gives the same causal reason and conclusion in natural explanatory wording."),
    "VS044": ("reject", "An outdoor-time mediator replaces correlation-as-causation and weakens the definite conclusion to possibility."),
    "VS050": ("accept", "The target is explanatory_conversational; explaining that baking would skip a requirement is directly entailed."),
    "VS052": ("accept", "The concise wording still naturally explains the unmet chilling prerequisite and satisfies the conversational target."),
    "VS060": ("reject", "Alarm reliability replaces affirming the consequent and changes must-opened to probably-opened."),
}


def line(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    items = load_jsonl(DATA / "verifier_selection_items_v1.jsonl")
    primary = load_jsonl(DATA / "verifier_selection_labels_primary_v1.jsonl")
    secondary = load_jsonl(DATA / "verifier_selection_labels_secondary_v1.jsonl")
    validate_labels(items, primary, primary=True)
    validate_labels(items, secondary, primary=False)
    disagreements = {
        first["item_id"]
        for first, second in zip(primary, secondary)
        if first["decision"] != second["decision"]
    }
    if disagreements != set(ADJUDICATIONS):
        raise ValueError("Decision disagreement set drifted from frozen adjudications")

    gold = []
    records = []
    for item, first, second in zip(items, primary, secondary):
        item_id = item["item_id"]
        if item_id in ADJUDICATIONS:
            decision, reason = ADJUDICATIONS[item_id]
            if decision != second["decision"]:
                raise ValueError(f"Adjudication {item_id} no longer matches its evidence")
            records.append({
                "schema_version": "clir-verifier-selection-adjudication-v1",
                "item_id": item_id,
                "primary_decision": first["decision"],
                "secondary_decision": second["decision"],
                "gold_decision": decision,
                "field_source": "secondary_after_manual_evidence_review",
                "reason": reason,
            })
        elif first["decision"] != second["decision"]:
            raise AssertionError("Unadjudicated decision disagreement")
        # Secondary structured fields are internally consistent and manually
        # grounded. Primary fields remain immutable as construction provenance.
        gold.append(dict(second))

    gold_path = DATA / "verifier_selection_gold_v1.jsonl"
    adjudication_path = DATA / "verifier_selection_adjudications_v1.jsonl"
    gold_path.write_text("".join(line(row) + "\n" for row in gold), encoding="utf-8")
    adjudication_path.write_text(
        "".join(line(row) + "\n" for row in records), encoding="utf-8"
    )
    report = {
        "schema_version": "clir-verifier-selection-gold-report-v1",
        "items": len(items),
        "decision_agreements": len(items) - len(records),
        "decision_adjudications": len(records),
        "gold_field_source": "secondary_after_schema_validation_and_manual_decision_adjudication",
        "primary_preserved": True,
        "secondary_preserved": True,
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "adjudications_sha256": hashlib.sha256(adjudication_path.read_bytes()).hexdigest(),
    }
    (DATA / "verifier_selection_gold_report_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
