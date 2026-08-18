#!/usr/bin/env python3
"""Audit primary dual-prior target position bias and wrong-path guide alignment."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import atomic_write_json, file_sha256, read_jsonl  # noqa: E402


def main() -> None:
    config = ROOT / "configs/dual_prior_evidence_v1"
    items_path = config / "annotation_items_v1.jsonl"
    labels_path = config / "labels_primary_v1.jsonl"
    selection_path = config / "selection_manifest_v1.jsonl"
    adjudicated_path = ROOT / "configs/hallucination_localization_v1/labels_adjudicated_v1.jsonl"
    span_path = ROOT / "configs/hallucination_localization_v2/span_supervision_annotations_v2.jsonl"
    items = {row["item_id"]: row for row in read_jsonl(items_path)}
    labels = read_jsonl(labels_path)
    selection = {row["item_id"]: row for row in read_jsonl(selection_path)}
    adjudicated = {row["id"]: row for row in read_jsonl(adjudicated_path)}
    spans = {row["id"]: row for row in read_jsonl(span_path)}
    if not (len(items) == len(labels) == len(selection) == len(adjudicated) == len(spans) == 64):
        raise ValueError("Primary semantic audit requires the aligned 64-row artifacts")

    key_unit_counts: Counter[int] = Counter()
    relative_positions: list[float] = []
    rows_key_only_last_quarter = 0
    rows_key_only_first_quarter = 0
    hallucinated_rows = 0
    key_contains_exact_onset = 0
    complete_contains_exact_onset = 0
    key_overlaps_reviewed_problem_span = 0
    for label in labels:
        item = items[label["item_id"]]
        private = selection[label["item_id"]]
        key_units = label["key_unit_indices"]
        unit_denominator = max(1, len(item["units"]) - 1)
        row_positions = [index / unit_denominator for index in key_units]
        relative_positions.extend(row_positions)
        key_unit_counts[len(key_units)] += 1
        rows_key_only_last_quarter += int(all(position >= 0.75 for position in row_positions))
        rows_key_only_first_quarter += int(all(position <= 0.25 for position in row_positions))

        source_id = label["id"]
        prior_path = adjudicated[source_id]
        if int(prior_path["path_hallucinated"]) != 1:
            continue
        hallucinated_rows += 1
        onset = int(prior_path["hallucination_onset"])
        key_target = label["key_prior_target"]
        complete_target = label["complete_prior_target"]
        sparse = spans[source_id]
        key_contains_exact_onset += int(bool(key_target[onset]))
        complete_contains_exact_onset += int(bool(complete_target[onset]))
        key_overlaps_reviewed_problem_span += int(
            any(
                bool(key_value) and bool(target_value) and bool(mask_value)
                for key_value, target_value, mask_value in zip(
                    key_target,
                    sparse["token_hallucination_target"],
                    sparse["token_hallucination_mask"],
                )
            )
        )

    report = {
        "schema_version": "clir-dual-prior-evidence-primary-semantic-audit-v1",
        "scope": "diagnostic_only_before_secondary; hallucination onset is not substituted for dual-prior gold",
        "inputs": {
            "annotation_items_sha256": file_sha256(items_path),
            "primary_labels_sha256": file_sha256(labels_path),
            "selection_manifest_sha256": file_sha256(selection_path),
            "hallucination_adjudicated_labels_sha256": file_sha256(adjudicated_path),
            "hallucination_sparse_span_labels_sha256": file_sha256(span_path),
        },
        "rows": len(labels),
        "position_diagnostic": {
            "key_unit_count_rows": {str(key): value for key, value in sorted(key_unit_counts.items())},
            "selected_key_unit_relative_position_mean": sum(relative_positions) / len(relative_positions),
            "selected_key_unit_relative_position_median": statistics.median(relative_positions),
            "rows_with_every_key_unit_in_last_quarter": rows_key_only_last_quarter,
            "rows_with_every_key_unit_in_first_quarter": rows_key_only_first_quarter,
            "interpretation": "A position baseline is mandatory in the training pilot; primary targets alone have a strong late-trajectory skew.",
        },
        "wrong_path_guide_alignment_diagnostic": {
            "previously_adjudicated_hallucinated_rows": hallucinated_rows,
            "key_contains_exact_hallucination_onset": key_contains_exact_onset,
            "complete_contains_exact_hallucination_onset": complete_contains_exact_onset,
            "key_overlaps_any_reviewed_unsupported_or_contradicted_span": key_overlaps_reviewed_problem_span,
            "interpretation": "Primary often selected an answer-producing late calculation instead of the guide's earliest/causally decisive flaw. This does not turn onset into prior gold, but primary labels cannot be adopted without independent secondary annotation and adjudication of the key semantics.",
        },
        "allowed_conclusion": "Primary is structurally valid and nonempty but is not standalone gold; collect the frozen independent secondary labels before any training target is adopted.",
    }
    output = config / "primary_semantic_audit_v1.json"
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
