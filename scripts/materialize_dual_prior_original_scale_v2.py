#!/usr/bin/env python3
"""Materialize the query-disjoint mixed-supervision dual-prior scale train set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    read_jsonl,
)
from src.clir_supervision import (  # noqa: E402
    audit_supervision_coverage,
    merge_supervision_annotations,
)


DEFAULT_PROTOCOL = (
    ROOT / "configs/dual_prior_original_scale_v2/data_protocol_v2.json"
)
PROTECTED_IDENTITY_FIELDS = (
    "id",
    "query_id",
    "output_token_ids",
    "feature_sha256",
    "condition_sha256",
    "correctness",
)
FORBIDDEN_EXISTING_FIELDS = {
    "key_prior_target",
    "complete_prior_target",
    "complete_reconstruction_target",
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def indexed(
    rows: Sequence[Mapping[str, Any]], key: str, *, name: str
) -> dict[str, dict[str, Any]]:
    output = {str(row[key]): dict(row) for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{name} contains duplicate {key} values")
    return output


def candidate_group_report(
    rows: Sequence[Mapping[str, Any]], *, expected_width: int, name: str
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["query_id"])].append(row)
    for query_id, candidates in groups.items():
        indices = {int(row["candidate_index"]) for row in candidates}
        if len(candidates) != expected_width or indices != set(range(expected_width)):
            raise ValueError(
                f"{name}/{query_id}: expected contiguous 0..{expected_width - 1} "
                f"candidate group, got {sorted(indices)}"
            )
    correct_counts = [sum(int(row["correctness"]) for row in group) for group in groups.values()]
    return {
        "rows": len(rows),
        "queries": len(groups),
        "candidates_per_query": expected_width,
        "correct": int(sum(correct_counts)),
        "incorrect": int(len(rows) - sum(correct_counts)),
        "mixed_queries": int(sum(0 < count < expected_width for count in correct_counts)),
        "all_correct_queries": int(sum(count == expected_width for count in correct_counts)),
        "all_wrong_queries": int(sum(count == 0 for count in correct_counts)),
    }


def verify_inputs(protocol: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(spec["path"])
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"Frozen input hash drifted: {name}")
        paths[name] = path
    integrity = json.loads(
        paths["feature_integrity_report"].read_text(encoding="utf-8")
    )
    if int(integrity.get("failure_count", -1)) != int(
        protocol["inputs"]["feature_integrity_report"]["required_failure_count"]
    ):
        raise ValueError("Frozen feature-integrity report contains failures")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-dual-prior-original-scale-data-protocol-v2"
    ):
        raise ValueError("Unexpected dual-prior scale data protocol schema")
    paths = verify_inputs(protocol)
    split = protocol["split_policy"]

    outcome_train = read_jsonl(paths["outcome_train"])
    ranking_validation = read_jsonl(paths["ranking_validation"])
    prior_train = read_jsonl(paths["prior_train_annotations"])
    prior_dev = read_jsonl(paths["prior_dev_annotations"])
    expected = protocol["inputs"]
    for name, rows in (
        ("outcome_train", outcome_train),
        ("ranking_validation", ranking_validation),
        ("prior_train_annotations", prior_train),
        ("prior_dev_annotations", prior_dev),
    ):
        if len(rows) != int(expected[name]["rows"]):
            raise ValueError(f"{name}: row count drifted")

    if any(FORBIDDEN_EXISTING_FIELDS.intersection(row) for row in outcome_train):
        raise ValueError("Outcome train already contains dual-prior/reconstruction targets")
    outcome_by_id = indexed(outcome_train, "id", name="outcome train")
    train_annotation_by_id = indexed(
        prior_train, "id", name="prior train annotations"
    )
    dev_annotation_by_id = indexed(prior_dev, "id", name="prior dev annotations")
    for membership, annotations in (
        ("train", train_annotation_by_id),
        ("dev", dev_annotation_by_id),
    ):
        for row_id, annotation in annotations.items():
            source = outcome_by_id.get(row_id)
            if source is None:
                raise ValueError(f"Prior {membership} row is absent from outcome train: {row_id}")
            for field in PROTECTED_IDENTITY_FIELDS:
                if field in annotation and annotation[field] != source.get(field):
                    raise ValueError(
                        f"{row_id}: prior {membership} annotation changed protected {field}"
                    )

    train_prior_queries = {str(row["query_id"]) for row in prior_train}
    dev_prior_queries = {str(row["query_id"]) for row in prior_dev}
    ranking_queries = {str(row["query_id"]) for row in ranking_validation}
    if train_prior_queries & dev_prior_queries:
        raise ValueError("Prior train/dev query leakage detected")
    all_prior_queries = train_prior_queries | dev_prior_queries
    if all_prior_queries & ranking_queries:
        raise ValueError("Prior train/dev queries overlap ranking validation")

    filtered = [
        dict(row)
        for row in outcome_train
        if str(row["query_id"]) not in dev_prior_queries
    ]
    merged = merge_supervision_annotations(
        filtered,
        prior_train,
        annotation_artifact={
            "annotations_jsonl": str(paths["prior_train_annotations"].relative_to(ROOT)),
            "annotations_sha256": file_sha256(paths["prior_train_annotations"]),
            "data_protocol": str(protocol_path.relative_to(ROOT)),
            "data_protocol_sha256": file_sha256(protocol_path),
        },
        require_all_rows_annotated=False,
    )

    train_group_report = candidate_group_report(
        merged,
        expected_width=int(expected["outcome_train"]["candidates_per_query"]),
        name="mixed train",
    )
    validation_group_report = candidate_group_report(
        ranking_validation,
        expected_width=int(expected["ranking_validation"]["candidates_per_query"]),
        name="ranking validation",
    )
    if train_group_report["rows"] != int(split["expected_train_rows"]):
        raise ValueError("Mixed train row count differs from the frozen contract")
    if train_group_report["queries"] != int(split["expected_train_queries"]):
        raise ValueError("Mixed train query count differs from the frozen contract")
    if len(dev_prior_queries) != int(split["expected_excluded_queries"]):
        raise ValueError("Excluded prior-dev query count differs from the frozen contract")

    coverage = audit_supervision_coverage(merged, require_provenance=True)
    expected_supervised = int(split["expected_prior_supervised_rows"])
    if (
        int(coverage["field_rows"]["key_prior_target"]) != expected_supervised
        or int(coverage["field_rows"]["complete_prior_target"])
        != expected_supervised
        or int(coverage["joint_prior_rows"]) != expected_supervised
    ):
        raise ValueError("Mixed train prior coverage differs from the frozen contract")
    supervised_ids = {
        str(row["id"])
        for row in merged
        if "key_prior_target" in row or "complete_prior_target" in row
    }
    if supervised_ids != set(train_annotation_by_id):
        raise ValueError("Mixed train supervised identities differ from prior Gold")

    output_path = resolve(protocol["outputs"]["train"])
    atomic_write_jsonl(output_path, merged)
    report = {
        "schema_version": "clir-dual-prior-original-scale-data-report-v2",
        "status": "mixed_supervision_data_ready",
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "inputs": {
            name: {"path": spec["path"], "sha256": spec["sha256"]}
            for name, spec in protocol["inputs"].items()
        },
        "train": {
            "path": str(output_path.relative_to(ROOT)),
            "sha256": file_sha256(output_path),
            **train_group_report,
            "supervision": coverage,
        },
        "ranking_validation": {
            "path": str(paths["ranking_validation"].relative_to(ROOT)),
            "sha256": file_sha256(paths["ranking_validation"]),
            **validation_group_report,
        },
        "prior_membership": {
            "train_queries": len(train_prior_queries),
            "dev_queries": len(dev_prior_queries),
            "train_dev_overlap": 0,
            "prior_ranking_validation_overlap": 0,
            "excluded_dev_query_rows": len(outcome_train) - len(filtered),
        },
        "protected_identity_fields": list(PROTECTED_IDENTITY_FIELDS),
        "allowed_conclusion": (
            "The mixed train manifest preserves complete query groups, keeps the "
            "localization-dev and ranking-validation queries out of training, and adds "
            "external key/complete targets to exactly the 48 adjudicated rows. It does "
            "not establish reward-ranking efficacy."
        ),
    }
    report_path = resolve(protocol["outputs"]["report"])
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
