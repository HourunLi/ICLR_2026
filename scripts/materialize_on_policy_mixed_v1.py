#!/usr/bin/env python3
"""Publish the repaired Route A v1a mixed train and prefix validation manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from src.clir_data import read_jsonl
from src.clir_on_policy_pairs import canonical_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/on_policy_pilot0_v1a/mixed_manifest_protocol_v1.json"
ADDED_FIELDS = {"semantic_id", "style_id", "consistency_provenance"}
AUXILIARY_FIELDS = {
    "hallucination_onset",
    "path_hallucinated",
    "token_advantage",
    "token_advantages",
    "progress_targets",
    "key_prior_target",
    "complete_prior_target",
    "complete_reconstruction_target",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _checked_input(config: dict[str, Any], key: str) -> Path:
    entry = config["inputs"][key]
    path = ROOT / entry["path"]
    observed = sha256(path)
    if observed != entry["sha256"]:
        raise ValueError(
            f"{key} SHA256 drifted: expected {entry['sha256']}, got {observed}"
        )
    return path


def _query_id(row: dict[str, Any]) -> str:
    value = row.get("query_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Every row requires a non-empty query_id")
    return value


def _validate_feature_references(rows: list[dict[str, Any]]) -> dict[str, int]:
    trajectory_paths = set()
    condition_paths = set()
    for row in rows:
        trajectory = Path(row["hidden_states_path"])
        condition = Path(row["condition_states_path"])
        if not trajectory.is_absolute() or not condition.is_absolute():
            raise ValueError("v1a manifests must preserve absolute feature paths")
        if not trajectory.is_file() or not condition.is_file():
            raise FileNotFoundError(f"Missing feature reference for {row['id']}")
        trajectory_paths.add(str(trajectory))
        condition_paths.add(str(condition))
    return {
        "unique_trajectory_features": len(trajectory_paths),
        "unique_condition_features": len(condition_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_path = _checked_input(config, "train_base")
    validation_path = _checked_input(config, "validation_base")
    lineage_path = _checked_input(config, "lineage")
    relation_path = _checked_input(config, "adjudicated_relations")
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    lineages = read_jsonl(lineage_path)
    relations = read_jsonl(relation_path)
    relation_by_id = {row["item_id"]: row for row in relations}
    if len(relation_by_id) != len(relations):
        raise ValueError("Adjudicated relations repeat item IDs")
    accepted = {
        item_id: row
        for item_id, row in relation_by_id.items()
        if row["final_status"] == "accepted"
    }
    expected_pairs = config["train_materialization"]["expected_accepted_pairs"]
    if len(accepted) != expected_pairs:
        raise ValueError(
            f"Accepted pair count drifted: expected {expected_pairs}, got {len(accepted)}"
        )
    lineage_by_id = {row["item_id"]: row for row in lineages}
    if set(accepted) - set(lineage_by_id):
        raise ValueError("An accepted relation lacks lineage")

    annotations_by_source: dict[str, dict[str, Any]] = {}
    for item_id, relation in accepted.items():
        lineage = lineage_by_id[item_id]
        for row_id_key, row_hash_key, style_id in (
            ("source_row_id", "source_row_sha256", "native_compact"),
            ("rewrite_row_id", "rewrite_row_sha256", "native_expanded"),
        ):
            row_id = lineage[row_id_key]
            if row_id in annotations_by_source:
                raise ValueError(f"Source row {row_id!r} belongs to multiple accepted pairs")
            annotations_by_source[row_id] = {
                "item_id": item_id,
                "style_id": style_id,
                "source_row_sha256": lineage[row_hash_key],
                "relation_row_sha256": object_sha256(relation),
                "resolution": relation["resolution"],
            }

    if any(ADDED_FIELDS & set(row) for row in train_rows):
        raise ValueError("Base train manifest already contains v1a metadata fields")
    train_by_id = {row["id"]: row for row in train_rows}
    if len(train_by_id) != len(train_rows):
        raise ValueError("Base train manifest repeats row IDs")
    if set(annotations_by_source) - set(train_by_id):
        raise ValueError("Accepted lineage references a row outside the base train manifest")

    mixed_rows: list[dict[str, Any]] = []
    for source in train_rows:
        annotation = annotations_by_source.get(source["id"])
        if annotation is None:
            mixed_rows.append(dict(source))
            continue
        if object_sha256(source) != annotation["source_row_sha256"]:
            raise ValueError(f"Source identity drifted for {source['id']}")
        row = dict(source)
        row.update(
            {
                "semantic_id": annotation["item_id"],
                "style_id": annotation["style_id"],
                "consistency_provenance": {
                    "schema_version": "clir-on-policy-consistency-provenance-v1",
                    "route": "A_on_policy_equivalence_mining_repaired_v1a",
                    "item_id": annotation["item_id"],
                    "final_relation_status": "accepted",
                    "resolution": annotation["resolution"],
                    "source_row_sha256": annotation["source_row_sha256"],
                    "relation_row_sha256": annotation["relation_row_sha256"],
                    "adjudicated_relations_sha256": config["inputs"][
                        "adjudicated_relations"
                    ]["sha256"],
                },
            }
        )
        mixed_rows.append(row)

    train_protocol = config["train_materialization"]
    if len(mixed_rows) != train_protocol["expected_base_rows"]:
        raise ValueError("Mixed train row count drifted")
    positive_rows = sum(int(row["correctness"]) for row in mixed_rows)
    if positive_rows != train_protocol["expected_correct_rows"]:
        raise ValueError("Mixed train correct-row count drifted")
    if len(mixed_rows) - positive_rows != train_protocol["expected_incorrect_rows"]:
        raise ValueError("Mixed train incorrect-row count drifted")
    annotated_rows = [row for row in mixed_rows if "semantic_id" in row]
    if len(annotated_rows) != train_protocol["expected_consistency_rows"]:
        raise ValueError("Mixed train consistency-row count drifted")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated_rows:
        groups[row["semantic_id"]].append(row)
        original = {key: value for key, value in row.items() if key not in ADDED_FIELDS}
        if canonical_json(original) != canonical_json(train_by_id[row["id"]]):
            raise ValueError(f"Non-consistency source fields drifted for {row['id']}")
    if len(groups) != expected_pairs or any(
        len(rows) != 2
        or {row["style_id"] for row in rows}
        != {"native_compact", "native_expanded"}
        or len({_query_id(row) for row in rows}) != 1
        for rows in groups.values()
    ):
        raise ValueError("Accepted semantic groups are incomplete or cross queries")
    if any(int(row["correctness"]) != 1 for row in annotated_rows):
        raise ValueError("v1a correct-only consistency metadata reached an incorrect row")

    selected_query_ids: list[str] = []
    seen_queries: set[str] = set()
    for row in validation_rows:
        query_id = _query_id(row)
        if query_id not in seen_queries:
            if len(selected_query_ids) == config["validation_materialization"][
                "expected_queries"
            ]:
                break
            seen_queries.add(query_id)
            selected_query_ids.append(query_id)
    selected_set = set(selected_query_ids)
    pilot_validation = [
        dict(row) for row in validation_rows if _query_id(row) in selected_set
    ]
    validation_counts = Counter(_query_id(row) for row in pilot_validation)
    expected_candidates = config["validation_materialization"][
        "expected_candidates_per_query"
    ]
    if len(validation_counts) != len(selected_query_ids) or any(
        count != expected_candidates for count in validation_counts.values()
    ):
        raise ValueError("Prefix validation candidate counts drifted")
    if any(ADDED_FIELDS & set(row) for row in pilot_validation):
        raise ValueError("Validation rows unexpectedly contain consistency metadata")
    train_queries = {_query_id(row) for row in mixed_rows}
    if train_queries & selected_set:
        raise ValueError("Train/validation query leakage detected")

    feature_counts = _validate_feature_references(mixed_rows + pilot_validation)
    output_root = ROOT / config["output_root"]
    mixed_path = output_root / "train_mixed_precomputed_v1.jsonl"
    pilot_validation_path = output_root / "validation_prefix32_precomputed_v1.jsonl"
    write_atomic(mixed_path, "".join(canonical_json(row) + "\n" for row in mixed_rows))
    write_atomic(
        pilot_validation_path,
        "".join(canonical_json(row) + "\n" for row in pilot_validation),
    )
    validation_positive = sum(int(row["correctness"]) for row in pilot_validation)
    report = {
        "schema_version": "clir-on-policy-mixed-manifest-report-v1",
        "evidence_tier": "pipeline_pilot",
        "protocol": str(args.config.relative_to(ROOT)),
        "protocol_sha256": sha256(args.config),
        "train_base_sha256": sha256(train_path),
        "adjudicated_relations_sha256": sha256(relation_path),
        "train_rows": len(mixed_rows),
        "train_queries": len(train_queries),
        "train_correct": positive_rows,
        "train_incorrect": len(mixed_rows) - positive_rows,
        "consistency_rows": len(annotated_rows),
        "semantic_groups": len(groups),
        "style_counts": dict(sorted(Counter(row["style_id"] for row in annotated_rows).items())),
        "potential_positive_pairs": len(groups),
        "potential_negative_pairs": 2 * (len(groups) * (len(groups) - 1) // 2),
        "injected_auxiliary_fields": sorted(
            set().union(*(AUXILIARY_FIELDS & set(row) for row in mixed_rows))
            - set().union(*(AUXILIARY_FIELDS & set(row) for row in train_rows))
        ),
        "validation_selection": config["validation_materialization"]["selection"],
        "validation_query_ids_sha256": hashlib.sha256(
            canonical_json(selected_query_ids).encode("utf-8")
        ).hexdigest(),
        "validation_queries": len(selected_query_ids),
        "validation_rows": len(pilot_validation),
        "validation_correct": validation_positive,
        "validation_incorrect": len(pilot_validation) - validation_positive,
        "train_validation_query_overlap": 0,
        **feature_counts,
        "feature_integrity_basis": config["feature_integrity_basis"],
        "train_manifest": str(mixed_path.relative_to(ROOT)),
        "train_manifest_sha256": sha256(mixed_path),
        "validation_manifest": str(pilot_validation_path.relative_to(ROOT)),
        "validation_manifest_sha256": sha256(pilot_validation_path),
        "training_ready": True,
        "formal_mechanism_claim_allowed": False,
    }
    write_atomic(
        output_root / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
