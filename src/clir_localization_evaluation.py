"""Metrics for held-out hallucination path and onset localization labels."""

from __future__ import annotations

from collections import Counter
import math
from statistics import median
from typing import Any, Mapping, Sequence


def _binary_labels(values: Sequence[int]) -> list[int]:
    labels = [int(value) for value in values]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("Binary metrics require labels in {0, 1}")
    if not labels:
        raise ValueError("Binary metrics require at least one label")
    return labels


def binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute tie-aware ROC AUC from average ascending score ranks."""

    targets = _binary_labels(labels)
    values = [float(score) for score in scores]
    if len(targets) != len(values) or any(not math.isfinite(value) for value in values):
        raise ValueError("ROC AUC requires equal finite label and score sequences")
    positives = sum(targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, target in zip(ranks, targets) if target)
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def binary_average_precision(
    labels: Sequence[int],
    scores: Sequence[float],
) -> float | None:
    """Compute stepwise average precision at each distinct score threshold."""

    targets = _binary_labels(labels)
    values = [float(score) for score in scores]
    if len(targets) != len(values) or any(not math.isfinite(value) for value in values):
        raise ValueError("Average precision requires equal finite label and score sequences")
    positives = sum(targets)
    if positives == 0:
        return None
    ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        for position in range(cursor, end):
            if targets[ordered[position]]:
                true_positives += 1
            else:
                false_positives += 1
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = end
    return average_precision


def binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    targets = _binary_labels(labels)
    values = [float(score) for score in scores]
    if len(targets) != len(values):
        raise ValueError("Binary metric label and score lengths differ")
    predictions = [int(value >= threshold) for value in values]
    confusion = Counter(
        ("positive" if target else "negative", "positive" if prediction else "negative")
        for target, prediction in zip(targets, predictions)
    )
    tp = confusion[("positive", "positive")]
    fn = confusion[("positive", "negative")]
    tn = confusion[("negative", "negative")]
    fp = confusion[("negative", "positive")]
    positive_count = tp + fn
    negative_count = tn + fp
    recall = tp / positive_count if positive_count else None
    specificity = tn / negative_count if negative_count else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    return {
        "rows": len(targets),
        "positive": positive_count,
        "negative": negative_count,
        "threshold": threshold,
        "roc_auc": binary_roc_auc(targets, values),
        "average_precision": binary_average_precision(targets, values),
        "accuracy": (tp + tn) / len(targets),
        "balanced_accuracy": (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None
            else None
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    left_mean = sum(first) / len(first)
    right_mean = sum(second) / len(second)
    numerator = sum(
        (left - left_mean) * (right - right_mean)
        for left, right in zip(first, second)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in first))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in second))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def evaluate_localization_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.5,
    negative_tail_margin: float = 0.5,
) -> dict[str, Any]:
    """Evaluate path, contaminated-tail tokens, onset, and value shaping."""

    if not rows:
        raise ValueError("Localization evaluation requires scored rows")
    path_labels: list[int] = []
    noisy_or_log_scores: list[float] = []
    noisy_or_probability_scores: list[float] = []
    max_token_scores: list[float] = []
    lengths: list[int] = []
    correctness: list[int] = []
    token_labels: list[int] = []
    token_scores: list[float] = []
    onset_errors: list[int] = []
    detected_onset_errors: list[int] = []
    onset_detected = 0
    onset_tolerance = Counter()
    pre_values: list[float] = []
    tail_values: list[float] = []
    clean_values: list[float] = []
    tail_margin_violations = 0

    for row in rows:
        label = row.get("path_hallucinated")
        onset = row.get("hallucination_onset")
        probabilities = row.get("clir_token_hallucination_probs")
        values = row.get("clir_token_values")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("Every localization row requires path_hallucinated in {0, 1}")
        if isinstance(onset, bool) or not isinstance(onset, int):
            raise ValueError("Every localization row requires an integer onset")
        if not isinstance(probabilities, list) or not probabilities:
            raise ValueError("Every scored row requires token hallucination probabilities")
        if not isinstance(values, list) or len(values) != len(probabilities):
            raise ValueError("Token values and hallucination probabilities must align")
        probs = [float(value) for value in probabilities]
        token_values = [float(value) for value in values]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probs):
            raise ValueError("Token hallucination probabilities must be finite in [0, 1]")
        if any(not math.isfinite(value) for value in token_values):
            raise ValueError("Token values must be finite")
        length = len(probs)
        if (onset == -1) != (label == 0) or onset >= length:
            raise ValueError("Path label and onset are inconsistent with scored token length")
        path_labels.append(int(label))
        noisy_or_probability_scores.append(
            float(row["clir_path_hallucination_prob"])
        )
        log_survival = float(row["clir_path_no_hallucination_log_prob"])
        if not math.isfinite(log_survival) or log_survival > 0.0:
            raise ValueError("Path log survival must be finite and non-positive")
        noisy_or_log_scores.append(-log_survival)
        max_token_scores.append(max(probs))
        lengths.append(length)
        correctness.append(int(row["correctness"]))

        target = [int(onset >= 0 and position >= onset) for position in range(length)]
        token_labels.extend(target)
        token_scores.extend(probs)
        if onset >= 0:
            predicted = int(row["clir_pseudo_onset"])
            if predicted >= 0:
                error = abs(predicted - onset)
                onset_detected += 1
                detected_onset_errors.append(error)
            else:
                error = length
            onset_errors.append(error)
            for tolerance in (0, 1, 3, 5):
                onset_tolerance[tolerance] += int(error <= tolerance)
            pre_values.extend(token_values[:onset])
            tail_values.extend(token_values[onset:])
            tail_margin_violations += sum(
                value > -negative_tail_margin for value in token_values[onset:]
            )
        else:
            clean_values.extend(token_values)

    stable_path_threshold = (
        -math.log1p(-threshold) if threshold < 1.0 else float("inf")
    )
    path_overall = binary_metrics(
        path_labels,
        noisy_or_log_scores,
        threshold=stable_path_threshold,
    )
    path_overall["equivalent_probability_threshold"] = threshold
    path_probability_diagnostic = binary_metrics(
        path_labels,
        noisy_or_probability_scores,
        threshold=threshold,
    )
    path_max_token = binary_metrics(path_labels, max_token_scores, threshold=threshold)
    subgroup: dict[str, Any] = {}
    for name, wanted in (("incorrect_only", 0), ("correct_only", 1)):
        indices = [index for index, value in enumerate(correctness) if value == wanted]
        if indices:
            subgroup[name] = binary_metrics(
                [path_labels[index] for index in indices],
                [noisy_or_log_scores[index] for index in indices],
                threshold=stable_path_threshold,
            )

    positive_rows = sum(path_labels)
    onset = {
        "positive_rows": positive_rows,
        "detected_rows": onset_detected,
        "detection_rate": onset_detected / positive_rows if positive_rows else None,
        "mean_absolute_error_with_miss_as_length": (
            sum(onset_errors) / len(onset_errors) if onset_errors else None
        ),
        "median_absolute_error_with_miss_as_length": (
            median(onset_errors) if onset_errors else None
        ),
        "mean_absolute_error_detected_only": (
            sum(detected_onset_errors) / len(detected_onset_errors)
            if detected_onset_errors
            else None
        ),
    }
    for tolerance in (0, 1, 3, 5):
        onset[f"within_{tolerance}"] = {
            "count": onset_tolerance[tolerance],
            "rate": onset_tolerance[tolerance] / positive_rows if positive_rows else None,
        }

    clean_indices = [index for index, value in enumerate(path_labels) if value == 0]
    return {
        "rows": len(rows),
        "threshold": threshold,
        "path_noisy_or_log_space": path_overall,
        "path_noisy_or_probability_diagnostic": {
            **path_probability_diagnostic,
            "exact_one_count": sum(
                value == 1.0 for value in noisy_or_probability_scores
            ),
        },
        "path_max_token_diagnostic": path_max_token,
        "path_subgroups": subgroup,
        "contaminated_tail_tokens": binary_metrics(
            token_labels,
            token_scores,
            threshold=threshold,
        ),
        "onset": onset,
        "token_value_shaping": {
            "negative_tail_margin": negative_tail_margin,
            "pre_onset_tokens": len(pre_values),
            "tail_tokens": len(tail_values),
            "clean_tokens": len(clean_values),
            "mean_pre_onset": sum(pre_values) / len(pre_values) if pre_values else None,
            "mean_tail": sum(tail_values) / len(tail_values) if tail_values else None,
            "mean_clean": sum(clean_values) / len(clean_values) if clean_values else None,
            "tail_margin_violation_rate": (
                tail_margin_violations / len(tail_values) if tail_values else None
            ),
        },
        "length_bias_diagnostic": {
            "negative_log_survival_vs_length_pearson": _pearson(
                [float(value) for value in lengths],
                noisy_or_log_scores,
            ),
            "clean_only_negative_log_survival_vs_length_pearson": _pearson(
                [float(lengths[index]) for index in clean_indices],
                [noisy_or_log_scores[index] for index in clean_indices],
            ),
            "probability_vs_length_pearson_saturation_diagnostic": _pearson(
                [float(value) for value in lengths],
                noisy_or_probability_scores,
            ),
        },
    }
