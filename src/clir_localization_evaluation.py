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


def select_binary_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    objective: str,
) -> dict[str, Any]:
    """Select a conservative threshold using only the supplied calibration rows."""

    targets = _binary_labels(labels)
    values = [float(score) for score in scores]
    if len(targets) != len(values) or any(not math.isfinite(value) for value in values):
        raise ValueError("Threshold calibration requires equal finite labels and scores")
    if objective not in {"balanced_accuracy", "f1"}:
        raise ValueError("Threshold objective must be balanced_accuracy or f1")
    positives = sum(targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Threshold calibration requires both binary classes")
    ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    tp = fp = 0
    fn = positives
    tn = negatives

    def objective_value() -> float:
        if objective == "balanced_accuracy":
            return ((tp / positives) + (tn / negatives)) / 2.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives
        return (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    best_threshold = math.nextafter(max(values), math.inf)
    best_value = objective_value()
    best_confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    candidate_count = 1
    cursor = 0
    while cursor < len(ordered):
        threshold = values[ordered[cursor]]
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == threshold:
            end += 1
        for position in range(cursor, end):
            if targets[ordered[position]]:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
        value = objective_value()
        candidate_count += 1
        # Descending traversal and strict improvement deterministically prefer
        # the higher, more conservative threshold when objectives tie.
        if value > best_value:
            best_threshold = threshold
            best_value = value
            best_confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        cursor = end
    return {
        "objective": objective,
        "threshold": best_threshold,
        "objective_value": best_value,
        "confusion": best_confusion,
        "candidate_count": candidate_count,
    }


def select_onset_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose a token threshold minimizing positive-row onset error on calibration data."""

    positive_rows = [row for row in rows if int(row["path_hallucinated"]) == 1]
    if not positive_rows:
        raise ValueError("Onset threshold calibration requires positive rows")
    events: list[tuple[float, int, int]] = []
    true_onsets: list[int] = []
    lengths: list[int] = []
    for row_index, row in enumerate(positive_rows):
        probabilities = row.get("clir_token_hallucination_probs")
        if not isinstance(probabilities, list) or not probabilities:
            raise ValueError("Onset threshold calibration requires token probabilities")
        onset = int(row["hallucination_onset"])
        if onset < 0 or onset >= len(probabilities):
            raise ValueError("Positive calibration onset is outside the token sequence")
        true_onsets.append(onset)
        lengths.append(len(probabilities))
        for position, probability in enumerate(probabilities):
            value = float(probability)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Onset calibration probabilities must be in [0, 1]")
            events.append((value, row_index, position))
    events.sort(key=lambda event: (-event[0], event[1], event[2]))
    predictions = [-1] * len(positive_rows)

    def state() -> tuple[float, int]:
        errors = [
            abs(predicted - onset) if predicted >= 0 else length
            for predicted, onset, length in zip(predictions, true_onsets, lengths)
        ]
        return sum(errors) / len(errors), sum(predicted >= 0 for predicted in predictions)

    best_error, best_detected = state()
    best_threshold = math.nextafter(events[0][0], math.inf)
    candidate_count = 1
    cursor = 0
    while cursor < len(events):
        threshold = events[cursor][0]
        end = cursor + 1
        while end < len(events) and events[end][0] == threshold:
            end += 1
        for _, row_index, position in events[cursor:end]:
            current = predictions[row_index]
            if current < 0 or position < current:
                predictions[row_index] = position
        error, detected = state()
        candidate_count += 1
        if error < best_error:
            best_threshold = threshold
            best_error = error
            best_detected = detected
        cursor = end
    return {
        "objective": "mean_absolute_error_with_miss_as_length",
        "threshold": best_threshold,
        "objective_value": best_error,
        "detected_rows": best_detected,
        "positive_rows": len(positive_rows),
        "candidate_count": candidate_count,
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


def _ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    targets = _binary_labels(labels)
    return {
        "rows": len(targets),
        "positive": sum(targets),
        "negative": len(targets) - sum(targets),
        "roc_auc": binary_roc_auc(targets, scores),
        "average_precision": binary_average_precision(targets, scores),
    }


def evaluate_localization_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.5,
    negative_tail_margin: float = 0.5,
    path_log_threshold: float | None = None,
    token_threshold: float | None = None,
    onset_threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate path, contaminated-tail tokens, onset, and value shaping."""

    if not rows:
        raise ValueError("Localization evaluation requires scored rows")
    token_threshold = threshold if token_threshold is None else token_threshold
    onset_threshold = threshold if onset_threshold is None else onset_threshold
    if not 0.0 <= token_threshold <= 1.0:
        raise ValueError("Token threshold must be in [0, 1]")
    if not 0.0 <= onset_threshold <= 1.0:
        raise ValueError("Onset threshold must be in [0, 1]")
    path_labels: list[int] = []
    noisy_or_log_scores: list[float] = []
    noisy_or_probability_scores: list[float] = []
    max_token_scores: list[float] = []
    lengths: list[int] = []
    correctness: list[int] = []
    token_labels: list[int] = []
    token_scores: list[float] = []
    token_absolute_positions: list[float] = []
    token_normalized_positions: list[float] = []
    explicit_token_labels: list[int] = []
    explicit_token_scores: list[float] = []
    explicit_token_absolute_positions: list[float] = []
    explicit_token_normalized_positions: list[float] = []
    explicit_token_rows = 0
    claim_labels: list[int] = []
    claim_mean_scores: list[float] = []
    claim_max_scores: list[float] = []
    claim_mean_absolute_positions: list[float] = []
    claim_mean_normalized_positions: list[float] = []
    within_positive_aucs: list[float] = []
    within_positive_average_precisions: list[float] = []
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
        token_absolute_positions.extend(float(position) for position in range(length))
        token_normalized_positions.extend(
            position / max(length - 1, 1) for position in range(length)
        )
        explicit_target = row.get("token_hallucination_target")
        explicit_mask = row.get("token_hallucination_mask")
        if (explicit_target is None) != (explicit_mask is None):
            raise ValueError(
                "Explicit token hallucination target and mask must be provided together"
            )
        if explicit_target is not None:
            if not isinstance(explicit_target, list) or not isinstance(explicit_mask, list):
                raise ValueError("Explicit token hallucination labels must be lists")
            if len(explicit_target) != length or len(explicit_mask) != length:
                raise ValueError("Explicit token hallucination labels must align to scored tokens")
            if any(value not in (0, 1, 0.0, 1.0) for value in explicit_target):
                raise ValueError("Explicit token hallucination targets must be binary")
            if any(value not in (0, 1, 0.0, 1.0) for value in explicit_mask):
                raise ValueError("Explicit token hallucination masks must be binary")
            explicit_token_rows += 1
            for position, (explicit_label, known) in enumerate(
                zip(explicit_target, explicit_mask)
            ):
                if not known:
                    if explicit_label:
                        raise ValueError("Explicit positive token lies outside its label mask")
                    continue
                explicit_token_labels.append(int(explicit_label))
                explicit_token_scores.append(probs[position])
                explicit_token_absolute_positions.append(float(position))
                explicit_token_normalized_positions.append(
                    position / max(length - 1, 1)
                )

            spans = row.get("hallucination_claim_spans")
            if spans is not None:
                if not isinstance(spans, list):
                    raise ValueError("hallucination_claim_spans must be a list")
                for span in spans:
                    if not isinstance(span, Mapping):
                        raise ValueError("Every hallucination claim span must be an object")
                    start = span.get("token_start")
                    end = span.get("token_end_exclusive")
                    claim_label = span.get("target")
                    if (
                        isinstance(start, bool)
                        or not isinstance(start, int)
                        or isinstance(end, bool)
                        or not isinstance(end, int)
                        or not 0 <= start < end <= length
                    ):
                        raise ValueError("Hallucination claim span is outside scored tokens")
                    if claim_label not in (0, 1, 0.0, 1.0):
                        raise ValueError("Hallucination claim span target must be binary")
                    if not all(explicit_mask[position] for position in range(start, end)):
                        raise ValueError("Hallucination claim span contains masked-out tokens")
                    if any(
                        int(explicit_target[position]) != int(claim_label)
                        for position in range(start, end)
                    ):
                        raise ValueError("Hallucination claim span disagrees with token targets")
                    span_scores = probs[start:end]
                    claim_labels.append(int(claim_label))
                    claim_mean_scores.append(sum(span_scores) / len(span_scores))
                    claim_max_scores.append(max(span_scores))
                    mean_position = (start + end - 1) / 2.0
                    claim_mean_absolute_positions.append(mean_position)
                    claim_mean_normalized_positions.append(
                        mean_position / max(length - 1, 1)
                    )
        if onset >= 0:
            if onset > 0:
                row_auc = binary_roc_auc(target, probs)
                row_ap = binary_average_precision(target, probs)
                if row_auc is not None and row_ap is not None:
                    within_positive_aucs.append(row_auc)
                    within_positive_average_precisions.append(row_ap)
            predicted = next(
                (
                    position
                    for position, probability in enumerate(probs)
                    if probability >= onset_threshold
                ),
                -1,
            )
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

    stable_path_threshold = path_log_threshold
    if stable_path_threshold is None:
        stable_path_threshold = (
            -math.log1p(-threshold) if threshold < 1.0 else float("inf")
        )
    path_overall = binary_metrics(
        path_labels,
        noisy_or_log_scores,
        threshold=stable_path_threshold,
    )
    path_overall["equivalent_probability_threshold"] = (
        -math.expm1(-stable_path_threshold)
        if math.isfinite(stable_path_threshold)
        else 1.0
    )
    path_probability_diagnostic = binary_metrics(
        path_labels,
        noisy_or_probability_scores,
        threshold=threshold,
    )
    path_max_token = binary_metrics(
        path_labels,
        max_token_scores,
        threshold=token_threshold,
    )
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
    incorrect_indices = [
        index for index, value in enumerate(correctness) if value == 0
    ]
    position_absolute = _ranking_metrics(token_labels, token_absolute_positions)
    position_normalized = _ranking_metrics(token_labels, token_normalized_positions)
    model_token_ranking = _ranking_metrics(token_labels, token_scores)
    explicit_metrics = None
    explicit_shortcuts = None
    if explicit_token_labels:
        explicit_model_ranking = _ranking_metrics(
            explicit_token_labels,
            explicit_token_scores,
        )
        explicit_absolute = _ranking_metrics(
            explicit_token_labels,
            explicit_token_absolute_positions,
        )
        explicit_normalized = _ranking_metrics(
            explicit_token_labels,
            explicit_token_normalized_positions,
        )
        explicit_metrics = {
            **binary_metrics(
                explicit_token_labels,
                explicit_token_scores,
                threshold=token_threshold,
            ),
            "annotated_rows": explicit_token_rows,
        }
        explicit_shortcuts = {
            "absolute_position_ranking": explicit_absolute,
            "normalized_position_ranking": explicit_normalized,
            "model_minus_absolute_position": {
                "roc_auc": (
                    explicit_model_ranking["roc_auc"] - explicit_absolute["roc_auc"]
                    if explicit_model_ranking["roc_auc"] is not None
                    and explicit_absolute["roc_auc"] is not None
                    else None
                ),
                "average_precision": (
                    explicit_model_ranking["average_precision"]
                    - explicit_absolute["average_precision"]
                    if explicit_model_ranking["average_precision"] is not None
                    and explicit_absolute["average_precision"] is not None
                    else None
                ),
            },
        }
        if claim_labels:
            claim_mean_ranking = _ranking_metrics(claim_labels, claim_mean_scores)
            claim_absolute = _ranking_metrics(
                claim_labels,
                claim_mean_absolute_positions,
            )
            claim_normalized = _ranking_metrics(
                claim_labels,
                claim_mean_normalized_positions,
            )
            explicit_shortcuts["claim_level"] = {
                "claims": len(claim_labels),
                "positive": sum(claim_labels),
                "mean_probability_ranking": claim_mean_ranking,
                "max_probability_ranking": _ranking_metrics(
                    claim_labels,
                    claim_max_scores,
                ),
                "absolute_position_ranking": claim_absolute,
                "normalized_position_ranking": claim_normalized,
                "mean_model_minus_strongest_position_average_precision": (
                    claim_mean_ranking["average_precision"]
                    - max(
                        value
                        for value in (
                            claim_absolute["average_precision"],
                            claim_normalized["average_precision"],
                        )
                        if value is not None
                    )
                    if claim_mean_ranking["average_precision"] is not None
                    and any(
                        value is not None
                        for value in (
                            claim_absolute["average_precision"],
                            claim_normalized["average_precision"],
                        )
                    )
                    else None
                ),
            }
    return {
        "rows": len(rows),
        "fixed_default_threshold": threshold,
        "path_log_threshold": stable_path_threshold,
        "token_threshold": token_threshold,
        "onset_threshold": onset_threshold,
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
            threshold=token_threshold,
        ),
        "explicit_claim_span_tokens": explicit_metrics,
        "explicit_claim_span_shortcuts": explicit_shortcuts,
        "shortcut_baselines": {
            "path_length_ranking": _ranking_metrics(path_labels, lengths),
            "path_incorrectness_ranking": _ranking_metrics(
                path_labels,
                [1 - value for value in correctness],
            ),
            "incorrect_only_path_length_ranking": (
                _ranking_metrics(
                    [path_labels[index] for index in incorrect_indices],
                    [lengths[index] for index in incorrect_indices],
                )
                if incorrect_indices
                else None
            ),
            "token_absolute_position_ranking": position_absolute,
            "token_normalized_position_ranking": position_normalized,
            "model_minus_absolute_position": {
                "roc_auc": (
                    model_token_ranking["roc_auc"] - position_absolute["roc_auc"]
                    if model_token_ranking["roc_auc"] is not None
                    and position_absolute["roc_auc"] is not None
                    else None
                ),
                "average_precision": (
                    model_token_ranking["average_precision"]
                    - position_absolute["average_precision"]
                    if model_token_ranking["average_precision"] is not None
                    and position_absolute["average_precision"] is not None
                    else None
                ),
            },
            "within_positive_row_model_ranking": {
                "eligible_rows": len(within_positive_aucs),
                "macro_roc_auc": (
                    sum(within_positive_aucs) / len(within_positive_aucs)
                    if within_positive_aucs
                    else None
                ),
                "macro_average_precision": (
                    sum(within_positive_average_precisions)
                    / len(within_positive_average_precisions)
                    if within_positive_average_precisions
                    else None
                ),
                "absolute_position_roc_auc_baseline": (
                    1.0 if within_positive_aucs else None
                ),
            },
        },
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
