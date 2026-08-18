#!/usr/bin/env python3
"""Stratify final dual-prior gold position bias and prior error-localization overlap."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    file_sha256,
    read_jsonl,
)


def indexed(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    output = {str(row[key]): dict(row) for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{path} contains duplicate {key} values")
    return output


def position_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positions: list[float] = []
    all_last = 0
    all_first = 0
    unit_counts: Counter[int] = Counter()
    for row in rows:
        denominator = max(1, len(row["mapped_units"]) - 1)
        row_positions = [int(index) / denominator for index in row["key_unit_indices"]]
        positions.extend(row_positions)
        all_last += int(all(value >= 0.75 for value in row_positions))
        all_first += int(all(value <= 0.25 for value in row_positions))
        unit_counts[len(row["key_unit_indices"])] += 1
    return {
        "rows": len(rows),
        "key_unit_count_rows": {str(key): value for key, value in sorted(unit_counts.items())},
        "selected_key_unit_relative_position_mean": sum(positions) / len(positions),
        "selected_key_unit_relative_position_median": statistics.median(positions),
        "rows_with_every_key_unit_in_last_quarter": all_last,
        "rows_with_every_key_unit_in_first_quarter": all_first,
    }


def overlap_summary(
    rows: Sequence[Mapping[str, Any]],
    path_by_id: Mapping[str, Mapping[str, Any]],
    sparse_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    hallucinated = 0
    key_onset = 0
    complete_onset = 0
    key_sparse = 0
    complete_sparse = 0
    for row in rows:
        source_id = str(row["id"])
        path = path_by_id[source_id]
        if int(path["path_hallucinated"]) != 1:
            continue
        hallucinated += 1
        onset = int(path["hallucination_onset"])
        key = row["key_prior_target"]
        complete = row["complete_prior_target"]
        sparse = sparse_by_id[source_id]
        target = sparse["token_hallucination_target"]
        mask = sparse["token_hallucination_mask"]
        if not (len(key) == len(complete) == len(target) == len(mask)):
            raise ValueError(f"{source_id}: diagnostic token axes differ")
        key_onset += int(bool(key[onset]))
        complete_onset += int(bool(complete[onset]))
        key_sparse += int(
            any(bool(k) and bool(t) and bool(m) for k, t, m in zip(key, target, mask))
        )
        complete_sparse += int(
            any(bool(c) and bool(t) and bool(m) for c, t, m in zip(complete, target, mask))
        )
    return {
        "previously_adjudicated_hallucinated_rows": hallucinated,
        "key_contains_exact_hallucination_onset": key_onset,
        "complete_contains_exact_hallucination_onset": complete_onset,
        "key_overlaps_any_reviewed_unsupported_or_contradicted_span": key_sparse,
        "complete_overlaps_any_reviewed_unsupported_or_contradicted_span": complete_sparse,
    }


def main() -> None:
    config = ROOT / "configs/dual_prior_evidence_v1"
    gold_path = config / "labels_gold_v1.jsonl"
    primary_path = config / "labels_primary_v1.jsonl"
    path_path = ROOT / "configs/hallucination_localization_v1/labels_adjudicated_v1.jsonl"
    sparse_path = (
        ROOT / "configs/hallucination_localization_v2/span_supervision_annotations_v2.jsonl"
    )
    gold = read_jsonl(gold_path)
    primary = read_jsonl(primary_path)
    path_by_id = indexed(path_path, "id")
    sparse_by_id = indexed(sparse_path, "id")
    if not (len(gold) == len(primary) == len(path_by_id) == len(sparse_by_id) == 64):
        raise ValueError("Gold semantic audit requires the aligned 64-row artifacts")
    primary_by_id = {str(row["id"]): row for row in primary}
    if len(primary_by_id) != 64:
        raise ValueError("Primary labels contain duplicate source identities")
    if {str(row["id"]) for row in gold} != set(path_by_id):
        raise ValueError("Gold and prior localization identities differ")

    hallucinated_gold = [
        row for row in gold if int(path_by_id[str(row["id"])]["path_hallucinated"]) == 1
    ]
    clean_gold = [
        row for row in gold if int(path_by_id[str(row["id"])]["path_hallucinated"]) == 0
    ]
    changed_key = sum(
        int(row["key_unit_indices"] != primary_by_id[str(row["id"])]["key_unit_indices"])
        for row in gold
    )
    changed_complete = sum(
        int(
            row["complete_unit_indices"]
            != primary_by_id[str(row["id"])]["complete_unit_indices"]
        )
        for row in gold
    )
    changed_hallucinated_key = sum(
        int(row["key_unit_indices"] != primary_by_id[str(row["id"])]["key_unit_indices"])
        for row in hallucinated_gold
    )

    report = {
        "schema_version": "clir-dual-prior-evidence-gold-semantic-audit-v1",
        "scope": (
            "diagnostic only; prior hallucination onset/span labels are not substituted for "
            "dual-prior evidence gold"
        ),
        "inputs": {
            "gold_labels_sha256": file_sha256(gold_path),
            "primary_labels_sha256": file_sha256(primary_path),
            "hallucination_adjudicated_labels_sha256": file_sha256(path_path),
            "hallucination_sparse_span_labels_sha256": file_sha256(sparse_path),
        },
        "rows": len(gold),
        "path_strata": {
            "previously_adjudicated_hallucinated": len(hallucinated_gold),
            "previously_adjudicated_clean": len(clean_gold),
        },
        "semantic_change_from_unilateral_primary": {
            "key_target_changed_rows": changed_key,
            "complete_target_changed_rows": changed_complete,
            "hallucinated_rows_with_key_target_changed": changed_hallucinated_key,
        },
        "position_diagnostic": {
            "all": position_summary(gold),
            "previously_hallucinated": position_summary(hallucinated_gold),
            "previously_clean": position_summary(clean_gold),
            "interpretation": (
                "Late position remains a strong dataset shortcut, especially on sound paths. "
                "Every learnability result must be compared with a token-position-only baseline."
            ),
        },
        "wrong_path_overlap_diagnostic": {
            "primary": overlap_summary(primary, path_by_id, sparse_by_id),
            "adjudicated_gold": overlap_summary(gold, path_by_id, sparse_by_id),
            "interpretation": (
                "Overlap is a guide-alignment diagnostic only: evidence key, exact onset, and "
                "unsupported-span supervision answer different questions."
            ),
        },
        "allowed_conclusion": (
            "Adjudication changed the unilateral primary semantics and the final targets remain "
            "position-skewed; proceed only with a position baseline and held-out direct-target metrics."
        ),
    }
    output = config / "gold_semantic_audit_v1.json"
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
