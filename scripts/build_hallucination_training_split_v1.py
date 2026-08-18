#!/usr/bin/env python3
"""Build deterministic query-disjoint train/dev manifests for localization Pilot-0."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    canonical_json,
    file_sha256,
    read_jsonl,
)
from src.clir_stage_a import git_state  # noqa: E402
from src.clir_supervision import audit_supervision_coverage  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/hallucination_localization_v1/training_split_protocol_v1.json"
)
PRIORITY_SALT = "clir-hallucination-localization-dev-v1"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify_input(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen input hash drifted for {path}: expected {expected_sha256}, got {observed}"
        )


def priority(item_id: str) -> str:
    return hashlib.sha256(f"{PRIORITY_SALT}:{item_id}".encode("utf-8")).hexdigest()


def jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def publish_or_verify(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite drifted split artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def cross_tab(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        f"correctness_{int(row['correctness'])}__{row['path_status']}"
        for row in rows
    )
    return dict(sorted(counts.items()))


def path_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["path_status"]) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Training split materialization requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-hallucination-training-split-protocol-v1":
        raise ValueError("Unknown hallucination training split protocol schema")
    paths = {
        key: resolve(value)
        for key, value in protocol["inputs"].items()
        if not key.endswith("_sha256")
    }
    for key, path in paths.items():
        verify_input(path, protocol["inputs"][f"{key}_sha256"])

    merged = read_jsonl(paths["merged_manifest"])
    labels = read_jsonl(paths["adjudicated_labels"])
    lineage = read_jsonl(paths["annotation_lineage"])
    if len(labels) != len(lineage) or len(labels) != 64:
        raise ValueError("Localization labels and lineage must contain exactly 64 rows")
    merged_by_id: dict[str, Mapping[str, Any]] = {}
    for row in merged:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in merged_by_id:
            raise ValueError("Merged manifest requires unique non-empty row IDs")
        merged_by_id[row_id] = row

    joined: list[dict[str, Any]] = []
    for label, identity in zip(labels, lineage):
        if label["item_id"] != identity["item_id"]:
            raise ValueError("Localization label/lineage order drifted")
        source_id = str(identity["source_id"])
        source = merged_by_id[source_id]
        if source["query_id"] != identity["query_id"]:
            raise ValueError("Localization query identity drifted")
        if source.get("path_hallucinated") != label.get("path_hallucinated"):
            raise ValueError("Merged path supervision differs from adjudicated label")
        if source.get("hallucination_onset") != label.get("hallucination_onset"):
            raise ValueError("Merged onset supervision differs from adjudicated label")
        joined.append(
            {
                "item_id": label["item_id"],
                "source_id": source_id,
                "query_id": str(identity["query_id"]),
                "correctness": int(identity["correctness"]),
                "path_status": str(label["path_status"]),
                "priority": priority(str(label["item_id"])),
            }
        )

    quotas = {
        key: int(value) for key, value in protocol["selection"]["dev_quotas"].items()
    }
    dev_ids: set[str] = set()
    for stratum, quota in quotas.items():
        correctness_text, path_status = stratum.split("__", 1)
        correctness = int(correctness_text.removeprefix("correctness_"))
        candidates = sorted(
            (
                row
                for row in joined
                if row["correctness"] == correctness
                and row["path_status"] == path_status
            ),
            key=lambda row: (row["priority"], row["item_id"]),
        )
        if len(candidates) < quota:
            raise ValueError(f"Split stratum {stratum} has only {len(candidates)}/{quota} rows")
        dev_ids.update(str(row["source_id"]) for row in candidates[:quota])
    if len(dev_ids) != sum(quotas.values()) or len(dev_ids) != 16:
        raise ValueError("Development selection must contain exactly 16 unique rows")

    split_rows: list[dict[str, Any]] = []
    for row in joined:
        membership = "localization_dev" if row["source_id"] in dev_ids else "dense_train"
        split_rows.append(
            {
                "schema_version": "clir-hallucination-training-split-row-v1",
                **row,
                "membership": membership,
            }
        )

    train_ids = {
        str(row["source_id"])
        for row in split_rows
        if row["membership"] == "dense_train"
    }
    dev_queries = {
        str(row["query_id"])
        for row in split_rows
        if row["membership"] == "localization_dev"
    }
    train_queries = {
        str(row["query_id"])
        for row in split_rows
        if row["membership"] == "dense_train"
    }
    if train_queries & dev_queries:
        raise AssertionError("Dense train and localization dev queries overlap")

    dense_train = [dict(row) for row in merged if str(row["id"]) in train_ids]
    localization_dev = [dict(row) for row in merged if str(row["id"]) in dev_ids]
    mixed_train = [dict(row) for row in merged if str(row["query_id"]) not in dev_queries]
    if len(dense_train) != 48 or len(localization_dev) != 16:
        raise AssertionError("Dense train/dev row count drifted")
    if any("path_hallucinated" not in row for row in dense_train + localization_dev):
        raise AssertionError("Dense train/dev contains an unlabeled row")
    if any(str(row["query_id"]) in dev_queries for row in mixed_train):
        raise AssertionError("Mixed train retained a development query")

    outputs = {key: resolve(value) for key, value in protocol["outputs"].items()}
    publish_or_verify(outputs["split_manifest"], jsonl_payload(split_rows))
    publish_or_verify(outputs["dense_train"], jsonl_payload(dense_train))
    publish_or_verify(outputs["mixed_train"], jsonl_payload(mixed_train))
    publish_or_verify(outputs["localization_dev"], jsonl_payload(localization_dev))

    joined_by_id = {str(row["source_id"]): row for row in joined}
    train_summary_rows = [joined_by_id[str(row["id"])] for row in dense_train]
    dev_summary_rows = [joined_by_id[str(row["id"])] for row in localization_dev]
    mixed_coverage = audit_supervision_coverage(
        mixed_train,
        require_provenance=True,
    )
    report = {
        "schema_version": "clir-hallucination-training-split-report-v1",
        "evidence_tier": "pipeline_pilot",
        "code": code,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "selection_priority_salt": PRIORITY_SALT,
        "dev_quotas": quotas,
        "dense_train": {
            "rows": len(dense_train),
            "queries": len(train_queries),
            "path_status_counts": path_counts(train_summary_rows),
            "correctness_path_cross_tab": cross_tab(train_summary_rows),
            "path": str(outputs["dense_train"].relative_to(ROOT)),
            "sha256": file_sha256(outputs["dense_train"]),
        },
        "localization_dev": {
            "rows": len(localization_dev),
            "queries": len(dev_queries),
            "path_status_counts": path_counts(dev_summary_rows),
            "correctness_path_cross_tab": cross_tab(dev_summary_rows),
            "path": str(outputs["localization_dev"].relative_to(ROOT)),
            "sha256": file_sha256(outputs["localization_dev"]),
        },
        "mixed_train": {
            "rows": len(mixed_train),
            "queries": len({str(row["query_id"]) for row in mixed_train}),
            "excluded_dev_queries": len(dev_queries),
            "coverage": mixed_coverage,
            "path": str(outputs["mixed_train"].relative_to(ROOT)),
            "sha256": file_sha256(outputs["mixed_train"]),
        },
        "query_overlap": {
            "dense_train_vs_localization_dev": len(train_queries & dev_queries),
            "mixed_train_vs_localization_dev": len(
                {str(row["query_id"]) for row in mixed_train} & dev_queries
            ),
        },
        "split_manifest": str(outputs["split_manifest"].relative_to(ROOT)),
        "split_manifest_sha256": file_sha256(outputs["split_manifest"]),
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    publish_or_verify(
        outputs["split_report"],
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
