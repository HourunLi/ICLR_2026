#!/usr/bin/env python3
"""Derive exact-token sparse hallucination targets from adjudicated claim spans."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import read_jsonl  # noqa: E402
from src.clir_localization_evaluation import (  # noqa: E402
    binary_average_precision,
    binary_roc_auc,
)
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state  # noqa: E402
from src.clir_supervision import (  # noqa: E402
    ANNOTATION_SCHEMA,
    PROVENANCE_SCHEMA,
    audit_supervision_coverage,
    merge_supervision_annotations,
    output_token_ids_sha256,
)


DEFAULT_PROTOCOL = (
    ROOT / "configs/hallucination_localization_v2/span_target_protocol_v2.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen input hash drifted for {path}: expected {expected_sha256}, got {observed}"
        )


def unique_by_id(rows: Sequence[Mapping[str, Any]], *, source: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in indexed:
            raise ValueError(f"{source} requires unique non-empty ids")
        indexed[identifier] = row
    return indexed


def derive_annotation(
    row: Mapping[str, Any],
    label: Mapping[str, Any],
    *,
    protocol_path: Path,
    protocol_sha256: str,
    labels_path: Path,
    labels_sha256: str,
    mapping: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    identifier = str(row["id"])
    token_ids = row.get("output_token_ids")
    token_hash = output_token_ids_sha256(token_ids)
    if token_hash != label.get("output_token_ids_sha256"):
        raise ValueError(f"{identifier}/adjudicated output token identity drifted")
    if str(row.get("query_id")) != str(label.get("query_id")):
        raise ValueError(f"{identifier}/query_id drifted")

    token_count = len(token_ids)
    target = [0] * token_count
    mask = [0] * token_count
    assigned: list[int | None] = [None] * token_count
    positive_statuses = set(mapping["positive_statuses"])
    negative_statuses = set(mapping["negative_statuses"])
    masked_statuses = set(mapping["masked_statuses"])
    known_statuses = positive_statuses | negative_statuses | masked_statuses
    claim_spans: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    reviews = label.get("claim_reviews")
    if not isinstance(reviews, list) or not reviews:
        raise ValueError(f"{identifier}/claim_reviews must be a non-empty list")
    for review in reviews:
        if not isinstance(review, Mapping):
            raise ValueError(f"{identifier}/claim review must be an object")
        status = str(review.get("status", ""))
        if status not in known_statuses:
            raise ValueError(f"{identifier}/unknown claim status {status!r}")
        start = review.get("token_start")
        end = review.get("token_end_exclusive")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end <= token_count
        ):
            raise ValueError(f"{identifier}/claim span is outside output tokens")
        counts[f"claim_{status}"] += 1
        counts[f"span_tokens_{status}"] += end - start
        if status in masked_statuses:
            continue
        label_value = int(status in positive_statuses)
        for position in range(start, end):
            prior = assigned[position]
            if prior is not None and prior != label_value:
                raise ValueError(f"{identifier}/claim spans assign conflicting token targets")
            assigned[position] = label_value
            target[position] = label_value
            mask[position] = 1
        claim_spans.append(
            {
                "claim_index": int(review["claim_index"]),
                "token_start": start,
                "token_end_exclusive": end,
                "status": status,
                "target": label_value,
            }
        )

    positive_positions = [position for position, value in enumerate(target) if value]
    path_hallucinated = int(label["path_hallucinated"])
    onset = int(label["hallucination_onset"])
    if bool(positive_positions) != bool(path_hallucinated):
        raise ValueError(f"{identifier}/explicit token positives disagree with path label")
    if positive_positions and min(positive_positions) != onset:
        raise ValueError(f"{identifier}/first explicit positive token disagrees with onset")
    if not positive_positions and onset != -1:
        raise ValueError(f"{identifier}/clean explicit targets require onset=-1")
    if not any(mask):
        raise ValueError(f"{identifier}/no claim tokens are supervised")

    counts["trajectory_tokens"] += token_count
    counts["supervised_tokens"] += sum(mask)
    counts["positive_tokens"] += sum(target)
    counts["negative_tokens"] += sum(mask) - sum(target)
    annotation = {
        "schema_version": ANNOTATION_SCHEMA,
        "id": identifier,
        "query_id": str(row["query_id"]),
        "output_token_ids_sha256": token_hash,
        "annotation_provenance": {
            "schema_version": PROVENANCE_SCHEMA,
            "protocol_sha256": protocol_sha256,
            "annotation_source": str(labels_path.relative_to(ROOT)),
            "annotation_source_sha256": labels_sha256,
            "derivation_protocol": str(protocol_path.relative_to(ROOT)),
            "claim_level_gold": False,
            "evidence_tier": "pipeline_pilot",
        },
        "annotation_metadata": {
            "derivation": "claim_status_to_sparse_exact_token_target",
            "reviewed_claims": len(reviews),
            "supervised_tokens": sum(mask),
            "positive_tokens": sum(target),
        },
        "token_hallucination_target": target,
        "token_hallucination_mask": mask,
    }
    return annotation, claim_spans, counts


def position_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels: list[int] = []
    absolute: list[float] = []
    normalized: list[float] = []
    claim_labels: list[int] = []
    claim_absolute: list[float] = []
    claim_normalized: list[float] = []
    for row in rows:
        target = row["token_hallucination_target"]
        mask = row["token_hallucination_mask"]
        length = len(target)
        for position, (value, known) in enumerate(zip(target, mask)):
            if not known:
                continue
            labels.append(int(value))
            absolute.append(float(position))
            normalized.append(position / max(length - 1, 1))
        for span in row["hallucination_claim_spans"]:
            start = int(span["token_start"])
            end = int(span["token_end_exclusive"])
            mean_position = (start + end - 1) / 2.0
            claim_labels.append(int(span["target"]))
            claim_absolute.append(mean_position)
            claim_normalized.append(mean_position / max(length - 1, 1))
    return {
        "token_absolute": {
            "roc_auc": binary_roc_auc(labels, absolute),
            "average_precision": binary_average_precision(labels, absolute),
        },
        "token_normalized": {
            "roc_auc": binary_roc_auc(labels, normalized),
            "average_precision": binary_average_precision(labels, normalized),
        },
        "claim_absolute": {
            "roc_auc": binary_roc_auc(claim_labels, claim_absolute),
            "average_precision": binary_average_precision(claim_labels, claim_absolute),
        },
        "claim_normalized": {
            "roc_auc": binary_roc_auc(claim_labels, claim_normalized),
            "average_precision": binary_average_precision(claim_labels, claim_normalized),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-hallucination-span-target-protocol-v2":
        raise ValueError("Unknown hallucination span-target protocol")
    protocol_sha256 = file_sha256(protocol_path)
    sources = protocol["sources"]
    for source in sources.values():
        verify(resolve(source["path"]), source["sha256"])

    labels_path = resolve(sources["adjudicated_labels"]["path"])
    labels_sha256 = sources["adjudicated_labels"]["sha256"]
    labels = unique_by_id(read_jsonl(labels_path), source="adjudicated labels")
    split_rows = {
        "dense_train": read_jsonl(resolve(sources["dense_train"]["path"])),
        "localization_dev": read_jsonl(resolve(sources["localization_dev"]["path"])),
    }
    all_rows = split_rows["dense_train"] + split_rows["localization_dev"]
    indexed_rows = unique_by_id(all_rows, source="dense train/dev rows")
    if set(indexed_rows) != set(labels):
        raise ValueError("Dense train/dev ids do not exactly cover adjudicated labels")

    outputs = {key: resolve(value) for key, value in protocol["outputs"].items()}
    if not args.overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite span-target artifacts: {existing}")

    annotations: list[dict[str, Any]] = []
    spans_by_id: dict[str, list[dict[str, Any]]] = {}
    raw_counts_by_id: dict[str, Counter[str]] = {}
    for row in all_rows:
        annotation, spans, counts = derive_annotation(
            row,
            labels[str(row["id"])],
            protocol_path=protocol_path,
            protocol_sha256=protocol_sha256,
            labels_path=labels_path,
            labels_sha256=labels_sha256,
            mapping=protocol["mapping"],
        )
        annotations.append(annotation)
        spans_by_id[str(row["id"])] = spans
        raw_counts_by_id[str(row["id"])] = counts

    atomic_write_jsonl(outputs["annotations"], annotations)
    annotations_sha256 = file_sha256(outputs["annotations"])
    merged_splits: dict[str, list[dict[str, Any]]] = {}
    split_reports: dict[str, Any] = {}
    for split_name, rows in split_rows.items():
        split_ids = {str(row["id"]) for row in rows}
        split_annotations = [
            annotation for annotation in annotations if annotation["id"] in split_ids
        ]
        merged = merge_supervision_annotations(
            rows,
            split_annotations,
            annotation_artifact={
                "annotations_sha256": annotations_sha256,
                "annotations_path": str(outputs["annotations"].relative_to(ROOT)),
                "derivation": "adjudicated_claim_spans_v2",
            },
            require_all_rows_annotated=True,
        )
        counts: Counter[str] = Counter()
        for row in merged:
            identifier = str(row["id"])
            row["hallucination_claim_spans"] = spans_by_id[identifier]
            row["hallucination_span_provenance"] = {
                "protocol_sha256": protocol_sha256,
                "annotations_sha256": annotations_sha256,
                "adjudicated_labels_sha256": labels_sha256,
            }
            counts.update(raw_counts_by_id[identifier])
        coverage = audit_supervision_coverage(merged, require_provenance=True)
        expected = protocol["expected"][split_name]
        observed = {
            "rows": len(merged),
            "trajectory_tokens": counts["trajectory_tokens"],
            "supervised_tokens": counts["supervised_tokens"],
            "positive_tokens": counts["positive_tokens"],
            "negative_tokens": counts["negative_tokens"],
        }
        for key, value in observed.items():
            if value != expected[key]:
                raise ValueError(
                    f"{split_name}/{key} drifted: expected {expected[key]}, got {value}"
                )
        merged_splits[split_name] = merged
        split_reports[split_name] = {
            **observed,
            "claim_counts": {
                key.removeprefix("claim_"): int(value)
                for key, value in sorted(counts.items())
                if key.startswith("claim_")
            },
            "coverage": coverage,
            "position_shortcuts": position_metrics(merged),
        }

    atomic_write_jsonl(outputs["dense_train"], merged_splits["dense_train"])
    atomic_write_jsonl(outputs["localization_dev"], merged_splits["localization_dev"])
    train_queries = {str(row["query_id"]) for row in merged_splits["dense_train"]}
    dev_queries = {str(row["query_id"]) for row in merged_splits["localization_dev"]}
    if train_queries & dev_queries:
        raise ValueError("Span-target train/dev splits overlap by query")

    report = {
        "schema_version": "clir-hallucination-span-target-audit-v2",
        "evidence_tier": "pipeline_pilot",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256,
        "source_labels_sha256": labels_sha256,
        "annotations": {
            "path": str(outputs["annotations"].relative_to(ROOT)),
            "sha256": annotations_sha256,
            "rows": len(annotations),
        },
        "splits": {
            "dense_train": {
                **split_reports["dense_train"],
                "path": str(outputs["dense_train"]),
                "sha256": file_sha256(outputs["dense_train"]),
            },
            "localization_dev": {
                **split_reports["localization_dev"],
                "path": str(outputs["localization_dev"]),
                "sha256": file_sha256(outputs["localization_dev"]),
            },
        },
        "query_overlap": 0,
        "code": git_state(ROOT),
        "claim_level_gold": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(outputs["audit_report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
