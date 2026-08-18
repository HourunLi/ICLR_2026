from __future__ import annotations

import pytest

from src.clir_localization_evaluation import (
    binary_average_precision,
    binary_metrics,
    binary_roc_auc,
    evaluate_localization_rows,
    select_binary_threshold,
    select_onset_threshold,
)


def test_binary_rank_metrics_are_tie_aware():
    assert binary_roc_auc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.0
    assert binary_average_precision([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.0
    assert binary_roc_auc([0, 1], [0.5, 0.5]) == 0.5
    metrics = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_localization_metrics_cover_path_tokens_onset_and_tail_values():
    rows = [
        {
            "path_hallucinated": 0,
            "hallucination_onset": -1,
            "correctness": 1,
            "clir_path_hallucination_prob": 0.2,
            "clir_path_no_hallucination_log_prob": -0.2,
            "clir_pseudo_onset": -1,
            "clir_token_hallucination_probs": [0.1, 0.2, 0.1],
            "clir_token_values": [0.2, 0.1, 0.3],
        },
        {
            "path_hallucinated": 1,
            "hallucination_onset": 2,
            "correctness": 0,
            "clir_path_hallucination_prob": 0.9,
            "clir_path_no_hallucination_log_prob": -2.0,
            "clir_pseudo_onset": 2,
            "clir_token_hallucination_probs": [0.1, 0.2, 0.8, 0.9],
            "clir_token_values": [0.2, 0.1, -0.6, -0.4],
        },
    ]
    metrics = evaluate_localization_rows(rows)
    assert metrics["path_noisy_or_log_space"]["roc_auc"] == 1.0
    assert metrics["contaminated_tail_tokens"]["f1"] == 1.0
    assert metrics["onset"]["within_0"]["rate"] == 1.0
    assert metrics["token_value_shaping"]["tail_margin_violation_rate"] == 0.5
    assert metrics["token_value_shaping"]["mean_tail"] == pytest.approx(-0.5)
    assert metrics["shortcut_baselines"]["token_absolute_position_ranking"]["roc_auc"] == 0.95
    assert metrics["shortcut_baselines"]["path_incorrectness_ranking"]["roc_auc"] == 1.0


def test_localization_missed_onset_is_penalized_by_row_length():
    row = {
        "path_hallucinated": 1,
        "hallucination_onset": 1,
        "correctness": 0,
        "clir_path_hallucination_prob": 0.4,
        "clir_path_no_hallucination_log_prob": -0.5,
        "clir_pseudo_onset": -1,
        "clir_token_hallucination_probs": [0.1, 0.2, 0.3],
        "clir_token_values": [0.0, 0.0, 0.0],
    }
    metrics = evaluate_localization_rows([row])
    assert metrics["onset"]["detected_rows"] == 0
    assert metrics["onset"]["mean_absolute_error_with_miss_as_length"] == 3
    assert metrics["path_noisy_or_log_space"]["roc_auc"] is None


def test_train_only_threshold_selection_is_deterministic_and_conservative_on_ties():
    selected = select_binary_threshold(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        objective="balanced_accuracy",
    )
    assert selected["threshold"] == 0.8
    assert selected["objective_value"] == 1.0

    rows = [
        {
            "path_hallucinated": 1,
            "hallucination_onset": 2,
            "clir_token_hallucination_probs": [0.1, 0.2, 0.8, 0.9],
        },
        {
            "path_hallucinated": 1,
            "hallucination_onset": 1,
            "clir_token_hallucination_probs": [0.2, 0.7, 0.8],
        },
    ]
    onset = select_onset_threshold(rows)
    assert onset["threshold"] == 0.7
    assert onset["objective_value"] == 0.0
