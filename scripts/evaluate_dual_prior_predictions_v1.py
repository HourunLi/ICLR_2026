#!/usr/bin/env python3
"""Evaluate direct key/complete membership predictions against adjudicated gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/training_protocol_v1.json"
HEAD_FIELDS = {
    "key": ("key_unit_indices", "key_prior_target", "clir_key_prior_membership_probs"),
    "complete": (
        "complete_unit_indices",
        "complete_prior_target",
        "clir_complete_prior_membership_probs",
    ),
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def indexed(rows: Sequence[Mapping[str, Any]], key: str, *, name: str) -> dict[str, dict[str, Any]]:
    output = {str(row[key]): dict(row) for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{name} contains duplicate {key} values")
    return output


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    labels = labels.astype(np.int64, copy=False)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    predicted = np.arange(1, len(sorted_labels) + 1)
    boundaries = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    indices = np.flatnonzero(boundaries)
    precision = true_positives[indices] / predicted[indices]
    recall = true_positives[indices] / int(labels.sum())
    recall_delta = np.diff(np.r_[0.0, recall])
    # Non-interpolated average precision, with tied scores treated as one
    # threshold exactly as the standard binary ranking definition requires.
    return float(np.sum(recall_delta * precision))


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    labels = labels.astype(np.int64, copy=False)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-indexed; tied observations receive their average rank.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive = labels == 1
    positive_count = int(positive.sum())
    negative_count = int(labels.size - positive_count)
    rank_sum = float(ranks[positive].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = labels.astype(bool, copy=False)
    predictions = predictions.astype(bool, copy=False)
    true_positive = int(np.sum(labels & predictions))
    false_positive = int(np.sum(~labels & predictions))
    false_negative = int(np.sum(labels & ~predictions))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if labels.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Cannot calibrate F1 on empty or non-finite predictions")
    candidates = np.unique(scores)
    if candidates.size == 1:
        threshold = float(candidates[0])
        return threshold, binary_f1(labels, scores >= threshold)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    true_positives = np.cumsum(sorted_labels)
    predicted = np.arange(1, len(sorted_labels) + 1)
    total_positive = int(sorted_labels.sum())
    denominator = predicted + total_positive
    f1_values = np.divide(
        2 * true_positives,
        denominator,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=denominator > 0,
    )
    boundary = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    boundary_indices = np.flatnonzero(boundary)
    best_value = float(np.max(f1_values[boundary_indices]))
    tied = boundary_indices[
        np.isclose(f1_values[boundary_indices], best_value, rtol=0.0, atol=1e-15)
    ]
    # Prefer the highest threshold among exact F1 ties, which avoids gratuitous
    # positives and is deterministic for heavily imbalanced key targets.
    best_index = int(tied[0])
    threshold = float(sorted_scores[best_index])
    return threshold, best_value


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    row_labels: Sequence[np.ndarray],
    row_scores: Sequence[np.ndarray],
) -> dict[str, Any]:
    predictions = scores >= threshold
    macro_values = [
        value
        for value in (
            safe_ap(row_label, row_score)
            for row_label, row_score in zip(row_labels, row_scores)
        )
        if value is not None
    ]
    positives = int(labels.sum())
    return {
        "examples": int(labels.size),
        "positives": positives,
        "positive_fraction": positives / int(labels.size),
        "average_precision_micro": safe_ap(labels, scores),
        "average_precision_row_macro": (
            float(sum(macro_values) / len(macro_values)) if macro_values else None
        ),
        "roc_auc_micro": safe_auc(labels, scores),
        "threshold": float(threshold),
        "f1": binary_f1(labels, predictions),
        "predicted_positive_fraction": float(predictions.mean()),
    }


def collect_head_arrays(
    scored_rows: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    head: str,
) -> dict[str, Any]:
    unit_field, target_field, prediction_field = HEAD_FIELDS[head]
    token_labels: list[np.ndarray] = []
    token_scores: list[np.ndarray] = []
    token_positions: list[np.ndarray] = []
    unit_labels: list[np.ndarray] = []
    unit_scores: list[np.ndarray] = []
    unit_positions: list[np.ndarray] = []
    for scored in scored_rows:
        source_id = str(scored["id"])
        if source_id not in gold_by_id:
            raise ValueError(f"Scored row {source_id} has no dual-prior gold")
        gold = gold_by_id[source_id]
        token_ids = [int(value) for value in scored["output_token_ids"]]
        if canonical_sha256(token_ids) != gold["output_token_ids_sha256"]:
            raise ValueError(f"{source_id}: scored/gold token identity drift")
        labels = np.asarray(gold[target_field], dtype=np.int64)
        scores = np.asarray(scored.get(prediction_field), dtype=np.float64)
        if labels.shape != scores.shape or labels.shape != (len(token_ids),):
            raise ValueError(f"{source_id}/{head}: prediction axis differs from frozen tokens")
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(f"{source_id}/{head}: membership probabilities are invalid")
        token_labels.append(labels)
        token_scores.append(scores)
        token_positions.append(
            np.arange(len(labels), dtype=np.float64) / max(1, len(labels) - 1)
        )

        selected_units = set(int(value) for value in gold[unit_field])
        mapped_units = gold["mapped_units"]
        row_unit_labels: list[int] = []
        row_unit_scores: list[float] = []
        row_unit_positions: list[float] = []
        for unit_index, unit in enumerate(mapped_units):
            start = int(unit["token_start"])
            end = int(unit["token_end_exclusive"])
            if not 0 <= start < end <= len(scores):
                raise ValueError(f"{source_id}: unit token span is outside scored tokens")
            row_unit_labels.append(int(unit_index in selected_units))
            row_unit_scores.append(float(scores[start:end].mean()))
            row_unit_positions.append(unit_index / max(1, len(mapped_units) - 1))
        unit_labels.append(np.asarray(row_unit_labels, dtype=np.int64))
        unit_scores.append(np.asarray(row_unit_scores, dtype=np.float64))
        unit_positions.append(np.asarray(row_unit_positions, dtype=np.float64))
    return {
        "token_labels": token_labels,
        "token_scores": token_scores,
        "token_positions": token_positions,
        "unit_labels": unit_labels,
        "unit_scores": unit_scores,
        "unit_positions": unit_positions,
    }


def concatenate(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("Cannot concatenate an empty metric collection")
    return np.concatenate(values)


def evaluate_head(
    train_arrays: Mapping[str, Sequence[np.ndarray]],
    dev_arrays: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for granularity in ("token", "unit"):
        labels_key = f"{granularity}_labels"
        scores_key = f"{granularity}_scores"
        positions_key = f"{granularity}_positions"
        train_labels = concatenate(train_arrays[labels_key])
        train_scores = concatenate(train_arrays[scores_key])
        dev_labels = concatenate(dev_arrays[labels_key])
        dev_scores = concatenate(dev_arrays[scores_key])
        head_threshold, head_train_f1 = best_f1_threshold(train_labels, train_scores)
        train_position = concatenate(train_arrays[positions_key])
        dev_position = concatenate(dev_arrays[positions_key])
        position_threshold, position_train_f1 = best_f1_threshold(
            train_labels, train_position
        )
        neural_train = binary_metrics(
            train_labels,
            train_scores,
            threshold=head_threshold,
            row_labels=train_arrays[labels_key],
            row_scores=train_arrays[scores_key],
        )
        neural_dev = binary_metrics(
            dev_labels,
            dev_scores,
            threshold=head_threshold,
            row_labels=dev_arrays[labels_key],
            row_scores=dev_arrays[scores_key],
        )
        position_train_metrics = binary_metrics(
            train_labels,
            train_position,
            threshold=position_threshold,
            row_labels=train_arrays[labels_key],
            row_scores=train_arrays[positions_key],
        )
        position_dev_metrics = binary_metrics(
            dev_labels,
            dev_position,
            threshold=position_threshold,
            row_labels=dev_arrays[labels_key],
            row_scores=dev_arrays[positions_key],
        )
        output[granularity] = {
            "train_calibration": {
                "neural_threshold": head_threshold,
                "neural_train_f1": head_train_f1,
                "position_threshold": position_threshold,
                "position_train_f1": position_train_f1,
            },
            "neural": {"train": neural_train, "dev": neural_dev},
            "position_only": {
                "train": position_train_metrics,
                "dev": position_dev_metrics,
            },
            "dev_delta_neural_minus_position": {
                "average_precision_micro": (
                    neural_dev["average_precision_micro"]
                    - position_dev_metrics["average_precision_micro"]
                ),
                "average_precision_row_macro": (
                    neural_dev["average_precision_row_macro"]
                    - position_dev_metrics["average_precision_row_macro"]
                ),
                "f1": neural_dev["f1"] - position_dev_metrics["f1"],
            },
        }
    return output


def correctness_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["correctness"]) for row in rows], dtype=np.int64)
    logits = np.asarray([float(row["reward_score"]) for row in rows], dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    return {
        "rows": len(rows),
        "positive_rows": int(labels.sum()),
        "roc_auc": safe_auc(labels, logits),
        "accuracy_at_probability_0_5": float(
            np.mean((probabilities >= 0.5) == labels)
        ),
        "mean_probability": float(probabilities.mean()),
    }


def separation_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_threshold: float,
    complete_threshold: float,
) -> dict[str, Any]:
    key = concatenate(
        [
            np.asarray(row["clir_key_prior_membership_probs"], dtype=np.float64)
            for row in rows
        ]
    )
    complete = concatenate(
        [
            np.asarray(row["clir_complete_prior_membership_probs"], dtype=np.float64)
            for row in rows
        ]
    )
    if key.shape != complete.shape:
        raise ValueError("Key and complete membership prediction axes differ")
    if float(key.std()) == 0.0 or float(complete.std()) == 0.0:
        correlation = None
    else:
        correlation = float(np.corrcoef(key, complete)[0, 1])
    key_positive = key >= key_threshold
    complete_positive = complete >= complete_threshold
    violation = key_positive & ~complete_positive
    return {
        "tokens": int(key.size),
        "pearson_correlation": correlation,
        "mean_absolute_probability_difference": float(np.mean(np.abs(key - complete))),
        "mean_key_probability": float(key.mean()),
        "mean_complete_probability": float(complete.mean()),
        "calibrated_key_positive_tokens": int(key_positive.sum()),
        "calibrated_complete_positive_tokens": int(complete_positive.sum()),
        "calibrated_containment_violation_tokens": int(violation.sum()),
        "calibrated_containment_violation_fraction_of_key_positive": (
            float(violation.sum() / key_positive.sum()) if key_positive.any() else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-scored", type=Path, required=True)
    parser.add_argument("--dev-scored", type=Path, required=True)
    parser.add_argument("--expected-train-scored-sha256", required=True)
    parser.add_argument("--expected-dev-scored-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-dual-prior-standalone-training-protocol-v1":
        raise ValueError("Unexpected dual-prior training protocol schema")
    if args.cell not in protocol["cells"]:
        raise ValueError(f"Unknown D0-D3 cell: {args.cell}")
    if args.seed not in [int(value) for value in protocol["matched_training"]["seeds"]]:
        raise ValueError(f"Seed {args.seed} is not frozen in the protocol")
    train_scored_path = args.train_scored.resolve()
    dev_scored_path = args.dev_scored.resolve()
    if file_sha256(train_scored_path) != args.expected_train_scored_sha256:
        raise ValueError("Train scored predictions hash drifted")
    if file_sha256(dev_scored_path) != args.expected_dev_scored_sha256:
        raise ValueError("Dev scored predictions hash drifted")
    gold_path = resolve(protocol["inputs"]["gold_labels"]["path"])
    if file_sha256(gold_path) != protocol["inputs"]["gold_labels"]["sha256"]:
        raise ValueError("Gold labels hash drifted")

    gold_rows = read_jsonl(gold_path)
    gold_by_id = indexed(gold_rows, "id", name="dual-prior gold")
    train_rows = read_jsonl(train_scored_path)
    dev_rows = read_jsonl(dev_scored_path)
    cell = protocol["cells"][args.cell]
    expected_train_ids = [row["id"] for row in read_jsonl(resolve(cell["train"]["path"]))]
    expected_dev_ids = [row["id"] for row in read_jsonl(resolve(cell["dev"]["path"]))]
    if [row["id"] for row in train_rows] != expected_train_ids:
        raise ValueError("Train scored row identities/order differ from the frozen cell")
    if [row["id"] for row in dev_rows] != expected_dev_ids:
        raise ValueError("Dev scored row identities/order differ from the frozen cell")

    heads: dict[str, Any] = {}
    for head in HEAD_FIELDS:
        train_arrays = collect_head_arrays(train_rows, gold_by_id, head)
        dev_arrays = collect_head_arrays(dev_rows, gold_by_id, head)
        heads[head] = evaluate_head(train_arrays, dev_arrays)
    key_threshold = heads["key"]["token"]["train_calibration"]["neural_threshold"]
    complete_threshold = heads["complete"]["token"]["train_calibration"][
        "neural_threshold"
    ]
    report = {
        "schema_version": "clir-dual-prior-cell-evaluation-v1",
        "evidence_tier": protocol["evidence_tier"],
        "cell": args.cell,
        "seed": args.seed,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "gold_labels_sha256": file_sha256(gold_path),
        "train_scored": str(train_scored_path),
        "train_scored_sha256": file_sha256(train_scored_path),
        "dev_scored": str(dev_scored_path),
        "dev_scored_sha256": file_sha256(dev_scored_path),
        "heads": heads,
        "correctness": {
            "train": correctness_metrics(train_rows),
            "dev": correctness_metrics(dev_rows),
        },
        "head_separation": {
            "train": separation_metrics(
                train_rows,
                key_threshold=key_threshold,
                complete_threshold=complete_threshold,
            ),
            "dev": separation_metrics(
                dev_rows,
                key_threshold=key_threshold,
                complete_threshold=complete_threshold,
            ),
        },
        "position_only_baseline_included": True,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(args.output_json.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
