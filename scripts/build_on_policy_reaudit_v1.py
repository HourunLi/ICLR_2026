#!/usr/bin/env python3
"""Build the blind 31-pair Route A v1a reannotation package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.clir_data import read_jsonl
from src.clir_on_policy_pairs import canonical_json
from src.clir_on_policy_reaudit import build_reannotation_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/on_policy_pilot0_v1a/repair_protocol_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _resolve(config: dict[str, Any], key: str) -> Path:
    return ROOT / config["inputs"][key]["path"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("candidate_manifest", "published_v1_manifest", "source_manifest"):
        path = _resolve(config, key)
        observed = sha256(path)
        expected = config["inputs"][key]["sha256"]
        if observed != expected:
            raise ValueError(f"{key} SHA256 drifted: expected {expected}, got {observed}")

    candidates = {row["pair_id"]: row for row in read_jsonl(_resolve(config, "candidate_manifest"))}
    published = read_jsonl(_resolve(config, "published_v1_manifest"))
    source_rows = {row["id"]: row for row in read_jsonl(_resolve(config, "source_manifest"))}
    selected_pair_ids: list[str] = []
    seen: set[str] = set()
    for row in published:
        pair_id = row["semantic_id"]
        if pair_id not in seen:
            selected_pair_ids.append(pair_id)
            seen.add(pair_id)
    if len(selected_pair_ids) != config["selection_roster"]["expected_pairs"]:
        raise ValueError("Selected v1 pair roster count drifted")
    if any(pair_id not in candidates for pair_id in selected_pair_ids):
        raise ValueError("Published v1 roster references an unknown candidate")

    items: list[dict[str, Any]] = []
    lineages: list[dict[str, Any]] = []
    for pair_id in selected_pair_ids:
        item, lineage = build_reannotation_records(
            candidates[pair_id], source_rows=source_rows
        )
        items.append(item)
        lineages.append(lineage)
    if len({row["item_id"] for row in items}) != len(items):
        raise ValueError("Reannotation item IDs are not unique")
    if len({row["query_id"] for row in lineages}) != len(lineages):
        raise ValueError("Reannotation roster must contain at most one pair per query")

    output_root = ROOT / config["output_root"]
    items_path = output_root / "annotation_items_v1.jsonl"
    lineage_path = output_root / "annotation_lineage_v1.jsonl"
    write_atomic(items_path, "".join(canonical_json(row) + "\n" for row in items))
    write_atomic(lineage_path, "".join(canonical_json(row) + "\n" for row in lineages))
    report = {
        "schema_version": "clir-on-policy-reannotation-package-report-v1",
        "evidence_tier": "pipeline_pilot",
        "repair_config": str(args.config.relative_to(ROOT)),
        "repair_config_sha256": sha256(args.config),
        "items": len(items),
        "queries": len({row["query_id"] for row in lineages}),
        "source_role": "native_compact",
        "rewrite_role": "native_expanded",
        "style_target": "native_compact_to_expanded",
        "claim_scope": "material_claims_only",
        "old_v1_decisions_used_as_labels": False,
        "items_path": str(items_path.relative_to(ROOT)),
        "items_sha256": sha256(items_path),
        "lineage_path": str(lineage_path.relative_to(ROOT)),
        "lineage_sha256": sha256(lineage_path),
        "training_ready": False,
        "blocking_gate": "independent_primary_and_secondary_annotations_then_adjudication",
    }
    write_atomic(
        output_root / "package_report_v1.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
