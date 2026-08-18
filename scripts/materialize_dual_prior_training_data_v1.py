#!/usr/bin/env python3
"""Build matched D0-D3 train/dev manifests from adjudicated dual-prior gold."""

from __future__ import annotations

import argparse
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
    MERGED_FIELDS,
    audit_supervision_coverage,
    merge_supervision_annotations,
)


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/training_data_protocol_v1.json"
FORBIDDEN_AUXILIARY_FIELDS = {
    "path_hallucinated",
    "hallucination_onset",
    "token_hallucination_target",
    "token_hallucination_mask",
    "token_advantage",
    "progress_targets",
    "semantic_id",
    "style_id",
    "domain_id",
    "complete_reconstruction_target",
}
ANNOTATION_METADATA_FIELDS = {
    "schema_version",
    "id",
    "query_id",
    "output_token_ids_sha256",
    "annotation_provenance",
    "annotation_metadata",
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_inputs(protocol: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(spec["path"])
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"Frozen training-data input hash drifted: {name}")
        paths[name] = path
    integrity = json.loads(paths["feature_integrity_report"].read_text(encoding="utf-8"))
    failure_count = integrity.get("failure_count")
    if failure_count is None:
        failure_count = integrity.get("summary", {}).get("failure_count")
    if int(failure_count) != int(
        protocol["inputs"]["feature_integrity_report"]["required_failure_count"]
    ):
        raise ValueError("Frozen feature integrity gate contains failures")
    gold = json.loads(paths["gold_report"].read_text(encoding="utf-8"))
    if not gold.get("ready_for_standalone_D0_D3_protocol"):
        raise ValueError("Gold report does not authorize standalone D0-D3 preparation")
    return paths


def index(rows: Sequence[Mapping[str, Any]], key: str, *, name: str) -> dict[str, dict[str, Any]]:
    output = {str(row[key]): dict(row) for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{name} contains duplicate {key} values")
    return output


def filter_annotations(
    annotations: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    selected = set(fields)
    if not selected.issubset({"key_prior_target", "complete_prior_target"}):
        raise ValueError(f"Unsupported dual-prior fields: {sorted(selected)}")
    output: list[dict[str, Any]] = []
    for annotation in annotations:
        missing = selected - set(annotation)
        if missing:
            raise ValueError(f"Gold annotation lacks requested fields: {sorted(missing)}")
        output.append(
            {
                key: value
                for key, value in annotation.items()
                if key in ANNOTATION_METADATA_FIELDS or key in selected
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-dual-prior-training-data-protocol-v1":
        raise ValueError("Unexpected dual-prior training-data protocol schema")
    paths = verify_inputs(protocol)
    protocol_sha = file_sha256(protocol_path)

    source_rows = read_jsonl(paths["trajectory_manifest"])
    source_by_id = index(source_rows, "id", name="trajectory manifest")
    split_rows = read_jsonl(paths["split_manifest"])
    if len(split_rows) != 64:
        raise ValueError("Dual-prior split manifest must contain 64 rows")
    split_members: dict[str, list[dict[str, Any]]] = {
        "dense_train": [],
        "localization_dev": [],
    }
    for split in split_rows:
        membership = str(split["membership"])
        if membership not in split_members:
            raise ValueError(f"Unexpected split membership: {membership}")
        source_id = str(split["source_id"])
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"Split references unknown source row {source_id}")
        if str(source["query_id"]) != str(split["query_id"]):
            raise ValueError(f"{source_id}: split query identity drift")
        for feature_key in ("hidden_states_path", "condition_states_path"):
            feature_path = Path(str(source[feature_key]))
            if not feature_path.is_file():
                raise ValueError(f"{source_id}: missing frozen feature {feature_key}")
        split_members[membership].append(dict(source))

    expected = protocol["split"]
    if len(split_members["dense_train"]) != int(expected["dense_train_rows"]):
        raise ValueError("dense_train row count drifted")
    if len(split_members["localization_dev"]) != int(
        expected["localization_dev_rows"]
    ):
        raise ValueError("localization_dev row count drifted")
    train_queries = {str(row["query_id"]) for row in split_members["dense_train"]}
    dev_queries = {str(row["query_id"]) for row in split_members["localization_dev"]}
    if train_queries & dev_queries:
        raise ValueError("Dual-prior train/dev query leakage detected")

    supervision_by_membership = {
        "dense_train": read_jsonl(paths["supervision_dense_train"]),
        "localization_dev": read_jsonl(paths["supervision_localization_dev"]),
    }
    for membership, annotations in supervision_by_membership.items():
        source_ids = {str(row["id"]) for row in split_members[membership]}
        annotation_ids = {str(row["id"]) for row in annotations}
        if source_ids != annotation_ids or len(annotation_ids) != len(annotations):
            raise ValueError(f"{membership}: gold supervision identity coverage drifted")

    output_root = resolve(protocol["output_root"])
    report_cells: dict[str, Any] = {}
    reference_orders: dict[str, list[str]] | None = None
    for cell_name, cell in protocol["cells"].items():
        fields = list(cell["supervision_fields"])
        cell_root = output_root / cell_name
        cell_report: dict[str, Any] = {"supervision_fields": fields, "splits": {}}
        observed_orders: dict[str, list[str]] = {}
        for membership, filename in (
            ("dense_train", "train48.jsonl"),
            ("localization_dev", "dev16.jsonl"),
        ):
            base = split_members[membership]
            if fields:
                source_annotations = supervision_by_membership[membership]
                filtered = filter_annotations(source_annotations, fields)
                annotation_path = paths[
                    "supervision_dense_train"
                    if membership == "dense_train"
                    else "supervision_localization_dev"
                ]
                merged = merge_supervision_annotations(
                    base,
                    filtered,
                    annotation_artifact={
                        "annotations_jsonl": str(annotation_path.relative_to(ROOT)),
                        "annotations_sha256": file_sha256(annotation_path),
                        "training_data_protocol_sha256": protocol_sha,
                        "cell": cell_name,
                    },
                    require_all_rows_annotated=True,
                )
            else:
                merged = [dict(row) for row in base]
            for row in merged:
                present = set(row).intersection(MERGED_FIELDS)
                if present != set(fields):
                    raise ValueError(
                        f"{cell_name}/{membership}/{row['id']}: fields {sorted(present)} "
                        f"differ from frozen {sorted(fields)}"
                    )
                forbidden = set(row).intersection(FORBIDDEN_AUXILIARY_FIELDS)
                if forbidden:
                    raise ValueError(
                        f"{cell_name}/{membership}: forbidden auxiliary fields {sorted(forbidden)}"
                    )
            coverage = audit_supervision_coverage(
                merged,
                require_provenance=bool(fields),
            )
            output_path = cell_root / filename
            atomic_write_jsonl(output_path, merged)
            order = [str(row["id"]) for row in merged]
            observed_orders[membership] = order
            cell_report["splits"][membership] = {
                "path": str(output_path.relative_to(ROOT)),
                "sha256": file_sha256(output_path),
                "rows": len(merged),
                "queries": len({str(row["query_id"]) for row in merged}),
                "coverage": coverage,
            }
        if reference_orders is None:
            reference_orders = observed_orders
        elif observed_orders != reference_orders:
            raise ValueError(f"{cell_name}: row identities or order differ across cells")
        report_cells[cell_name] = cell_report

    report = {
        "schema_version": "clir-dual-prior-training-data-report-v1",
        "status": "matched_D0_D3_data_ready",
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "input_hashes": {
            name: spec["sha256"] for name, spec in protocol["inputs"].items()
        },
        "membership_counts": {
            membership: len(rows) for membership, rows in split_members.items()
        },
        "query_disjoint": not bool(train_queries & dev_queries),
        "identical_id_order_across_cells": True,
        "cells": report_cells,
        "allowed_conclusion": (
            "The four data cells differ only in the frozen direct-target fields named by "
            "the protocol; training effectiveness has not yet been evaluated."
        ),
    }
    report_path = resolve(protocol["report"])
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
