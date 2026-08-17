#!/usr/bin/env python3
"""Build and audit the frozen Route A Pilot-0 candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.clir_data import read_jsonl
from src.clir_on_policy_pairs import audit_candidates, canonical_json, mine_candidate_pairs


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/on_policy_pilot0_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = ROOT / config["source_manifest"]
    split_path = ROOT / config["split_manifest"]
    if sha256(source) != config["source_manifest_sha256"]:
        raise ValueError("Source manifest SHA256 drifted")
    if sha256(split_path) != config["split_manifest_sha256"]:
        raise ValueError("Split manifest SHA256 drifted")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    membership = split["memberships"][config["membership"]]
    candidates, report = mine_candidate_pairs(
        read_jsonl(source), allowed_query_ids=membership,
        correctness=config["correctness"], max_pairs_per_query=config["max_pairs_per_query"],
    )
    audit_candidates(candidates, report)
    output_root = ROOT / config["output_root"]
    candidate_path = output_root / "candidates.jsonl"
    report.update({
        "schema_version": report["schema_version"],
        "config_sha256": sha256(args.config),
        "source_manifest": config["source_manifest"],
        "source_manifest_sha256": config["source_manifest_sha256"],
        "split_manifest": config["split_manifest"],
        "split_manifest_sha256": config["split_manifest_sha256"],
        "membership": config["membership"],
        "candidate_manifest": str(candidate_path.relative_to(ROOT)),
    })
    payload = "".join(canonical_json(row) + "\n" for row in candidates)
    write_atomic(candidate_path, payload)
    report["candidate_manifest_sha256"] = sha256(candidate_path)
    write_atomic(output_root / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
