#!/usr/bin/env python3
"""Build the frozen 64-row key/complete evidence annotation package."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_dual_prior_annotation import (  # noqa: E402
    ITEM_SCHEMA,
    public_units,
    segment_reasoning_units,
    validate_item,
)
from src.clir_hallucination_annotation import (  # noqa: E402
    canonical_json,
    canonical_sha256,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_input(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"Frozen input drifted: {path}: expected {expected_sha256}, got {observed}")


def publish_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite drifted artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def publish_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite drifted artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def item_id_for(source_id: str) -> str:
    digest = hashlib.sha256(f"clir-dual-prior-evidence-v1:{source_id}".encode()).hexdigest()
    return f"DPA-{digest[:16]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-dual-prior-evidence-protocol-v1":
        raise ValueError("Unexpected dual-prior protocol schema")
    protocol_sha256 = file_sha256(protocol_path)
    source_cfg = protocol["source"]
    input_specs = {
        "trajectory_manifest": "trajectory_manifest_sha256",
        "hallucination_items": "hallucination_items_sha256",
        "hallucination_lineage": "hallucination_lineage_sha256",
        "hallucination_adjudicated_labels": "hallucination_adjudicated_labels_sha256",
        "training_split_manifest": "training_split_manifest_sha256",
    }
    paths: dict[str, Path] = {}
    for path_key, sha_key in input_specs.items():
        path = resolve(source_cfg[path_key])
        verify_input(path, source_cfg[sha_key])
        paths[path_key] = path

    old_items = read_jsonl(paths["hallucination_items"])
    old_lineage = read_jsonl(paths["hallucination_lineage"])
    old_labels = read_jsonl(paths["hallucination_adjudicated_labels"])
    split_rows = read_jsonl(paths["training_split_manifest"])
    source_rows = read_jsonl(paths["trajectory_manifest"])
    expected_rows = int(source_cfg["rows"])
    if not (len(old_items) == len(old_lineage) == len(old_labels) == len(split_rows) == expected_rows):
        raise ValueError("The four reused 64-row artifacts differ in size")

    source_by_id = {str(row["id"]): row for row in source_rows}
    split_by_item = {str(row["item_id"]): row for row in split_rows}
    label_by_item = {str(row["item_id"]): row for row in old_labels}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Source trajectory manifest contains duplicate ids")
    if len(split_by_item) != expected_rows or len(label_by_item) != expected_rows:
        raise ValueError("Reused split or adjudicated label artifact contains duplicate item ids")

    selection: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    unit_counts: list[int] = []
    memberships: Counter[str] = Counter()
    path_statuses: Counter[str] = Counter()
    seen_new_ids: set[str] = set()
    for order, (old_item, old_identity) in enumerate(zip(old_items, old_lineage)):
        old_item_id = str(old_item["item_id"])
        if old_identity["item_id"] != old_item_id:
            raise ValueError(f"Reused item/lineage order drift at row {order}")
        source_id = str(old_identity["source_id"])
        source = source_by_id[source_id]
        split = split_by_item[old_item_id]
        prior_label = label_by_item[old_item_id]
        if split["source_id"] != source_id or prior_label["id"] != source_id:
            raise ValueError(f"Reused identity drift for {old_item_id}")
        trajectory = str(old_item["trajectory"])
        problem = str(old_item["problem"])
        if source["response"] != trajectory or source["question"] != problem:
            raise ValueError(f"Blind item text differs from frozen trajectory source: {old_item_id}")
        token_ids = [int(value) for value in source["output_token_ids"]]
        if canonical_sha256(token_ids) != old_identity["output_token_ids_sha256"]:
            raise ValueError(f"Frozen output token identity drift for {old_item_id}")

        private_units = segment_reasoning_units(trajectory)
        new_item_id = item_id_for(source_id)
        if new_item_id in seen_new_ids:
            raise ValueError("Dual-prior annotation item-id collision")
        seen_new_ids.add(new_item_id)
        item = {
            "schema_version": ITEM_SCHEMA,
            "item_id": new_item_id,
            "problem": problem,
            "trajectory": trajectory,
            "units": public_units(private_units),
        }
        validate_item(item)
        membership = str(split["membership"])
        path_status = str(prior_label["path_status"])
        unit_counts.append(len(private_units))
        memberships[membership] += 1
        path_statuses[path_status] += 1
        selection.append(
            {
                "schema_version": "clir-dual-prior-evidence-selection-row-v1",
                "selection_order": order,
                "item_id": new_item_id,
                "source_item_id": old_item_id,
                "source_id": source_id,
                "query_id": str(old_identity["query_id"]),
                "membership": membership,
                "correctness": int(old_identity["correctness"]),
                "prior_path_status": path_status,
                "output_token_count": len(token_ids),
                "unit_count": len(private_units),
            }
        )
        items.append(item)
        lineage.append(
            {
                "schema_version": "clir-dual-prior-evidence-lineage-v1",
                "item_id": new_item_id,
                "source_item_id": old_item_id,
                "source_id": source_id,
                "query_id": str(old_identity["query_id"]),
                "candidate_index": int(old_identity["candidate_index"]),
                "membership": membership,
                "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
                "trajectory_sha256": hashlib.sha256(trajectory.encode()).hexdigest(),
                "output_token_ids_sha256": canonical_sha256(token_ids),
                "output_token_count": len(token_ids),
                "units": private_units,
                "protocol_sha256": protocol_sha256,
                "source_manifest": source_cfg["trajectory_manifest"],
                "source_manifest_sha256": source_cfg["trajectory_manifest_sha256"],
            }
        )

    if memberships != Counter({"dense_train": 48, "localization_dev": 16}):
        raise ValueError(f"Reused train/dev membership drifted: {memberships}")
    outputs = protocol["outputs"]
    selection_path = resolve(outputs["selection_manifest"])
    items_path = resolve(outputs["annotation_items"])
    lineage_path = resolve(outputs["annotation_lineage"])
    report_path = resolve(outputs["package_report"])
    publish_jsonl(selection_path, selection)
    publish_jsonl(items_path, items)
    publish_jsonl(lineage_path, lineage)
    report = {
        "schema_version": "clir-dual-prior-evidence-package-report-v1",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256,
        "rows": len(items),
        "queries": len({row["query_id"] for row in selection}),
        "membership_counts": dict(sorted(memberships.items())),
        "prior_path_status_counts_for_private_audit_only": dict(sorted(path_statuses.items())),
        "unit_count": {
            "total": sum(unit_counts),
            "min": min(unit_counts),
            "median": statistics.median(unit_counts),
            "max": max(unit_counts),
        },
        "blind_item_fields": sorted(items[0]),
        "forbidden_fields_absent_from_blind_items": all(
            not {"correctness", "path_status", "membership", "query_id", "source_id"}.intersection(item)
            for item in items
        ),
        "non_whitespace_coverage_verified": True,
        "selection_manifest_sha256": file_sha256(selection_path),
        "annotation_items_sha256": file_sha256(items_path),
        "annotation_lineage_sha256": file_sha256(lineage_path),
        "annotation_guide_sha256": file_sha256(resolve(protocol["annotation"]["guide"])),
    }
    publish_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
