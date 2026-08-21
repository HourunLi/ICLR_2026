"""Fixed, parameter-free temporal smoothing for frozen H-probe logits."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


PROTOCOL_SCHEMA = "clir-jp-h-temporal-smoother-protocol-v1"


def validate_smoother_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject drift from the user-approved, single-factor engineering pilot."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("Unknown JP-to-H temporal-smoother protocol schema")
    if protocol.get("evidence_tier") != "small-scale real":
        raise ValueError("Temporal-smoother evidence tier drifted")
    if protocol.get("selection_status") != "post_selection_exploratory":
        raise ValueError("Temporal-smoother selection status drifted")

    smoother = protocol.get("smoother", {})
    expected_smoother = {
        "architecture": "fixed_masked_centered_logit_mean",
        "window_tokens": 3,
        "radius_tokens": 1,
        "centered": True,
        "boundary_rule": "mean_over_available_real_tokens_only",
        "mask_role": "generated_token_validity_only_not_supervision_mask",
        "input": "stable_float64_logit_of_persisted_frozen_linear_head_probability",
        "output": "sigmoid_of_smoothed_logit",
        "compute_dtype": "float64",
        "trainable_parameter_count": 0,
        "head_retraining": False,
        "window_search_during_run": False,
    }
    for name, expected in expected_smoother.items():
        if smoother.get(name) != expected:
            raise ValueError(f"Frozen temporal-smoother field drifted: {name}")

    protected = protocol.get("protected_paths", {})
    for name in (
        "jp_checkpoint_read_only",
        "linear_probe_checkpoints_read_only",
        "base_reward_scores_unchanged",
        "base_token_values_unchanged",
        "smoother_enters_reward_score",
        "tail_supervision_enabled",
        "progress_supervision_enabled",
        "reconstruction_supervision_enabled",
        "consistency_supervision_enabled",
    ):
        expected = (
            False
            if name.endswith("enabled") or name == "smoother_enters_reward_score"
            else True
        )
        if protected.get(name) is not expected:
            raise ValueError(f"Protected path drifted: {name}")

    evaluation = protocol.get("evaluation", {})
    if evaluation.get("folds") != [0, 1, 2, 3]:
        raise ValueError("Temporal-smoother folds drifted")
    if evaluation.get("seeds") != [42, 43, 44]:
        raise ValueError("Temporal-smoother seeds drifted")
    if evaluation.get("primary_confirmatory_folds") != [1, 2, 3]:
        raise ValueError("Temporal-smoother primary folds drifted")
    if evaluation.get("primary_rows") != 48:
        raise ValueError("Temporal-smoother primary row count drifted")
    if evaluation.get("all_oof_rows") != 64:
        raise ValueError("Temporal-smoother all-OOF row count drifted")

    rules = protocol.get("engineering_decision_rules", {})
    if rules.get("minimum_passing_seeds") != 2:
        raise ValueError("Temporal-smoother passing-seed rule drifted")
    for name in (
        "span_ap_strictly_above_same_rows_position",
        "claim_ap_strictly_above_same_rows_position",
        "span_ap_strictly_above_paired_raw_linear",
        "claim_ap_strictly_above_paired_raw_linear",
        "mean_all_four_advantages_strictly_positive",
        "raw_source_predictions_bit_identical",
        "base_reward_fields_bit_identical",
        "automatic_method_adoption",
        "automatic_score_coupling",
    ):
        expected = False if name.startswith("automatic_") else True
        if rules.get(name) is not expected:
            raise ValueError(f"Temporal-smoother decision rule drifted: {name}")


def masked_centered_mean_logits(
    logits: Tensor,
    token_mask: Tensor | None = None,
    *,
    window_tokens: int = 3,
) -> Tensor:
    """Average centered logits over valid tokens, without padded edge values.

    ``token_mask`` describes real generated-token positions, not sparse label
    coverage. Invalid output positions are zeroed and never contribute to a
    neighboring average.
    """

    if logits.ndim not in (1, 2):
        raise ValueError("Temporal smoothing expects [tokens] or [batch, tokens]")
    if window_tokens <= 0 or window_tokens % 2 != 1:
        raise ValueError("Temporal smoothing requires a positive odd window")
    squeeze = logits.ndim == 1
    values = logits.unsqueeze(0) if squeeze else logits
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("Temporal-smoother logits must be finite floating values")
    if token_mask is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    else:
        valid = token_mask.unsqueeze(0) if token_mask.ndim == 1 else token_mask
        if valid.shape != values.shape:
            raise ValueError("Temporal-smoother mask must match logits")
        valid = valid.to(device=values.device, dtype=torch.bool)

    radius = window_tokens // 2
    numerator = torch.zeros_like(values)
    denominator = torch.zeros_like(values)
    for offset in range(-radius, radius + 1):
        if offset < 0:
            source_values = values[:, :offset]
            source_valid = valid[:, :offset]
            target_slice = slice(-offset, None)
        elif offset > 0:
            source_values = values[:, offset:]
            source_valid = valid[:, offset:]
            target_slice = slice(None, -offset)
        else:
            source_values = values
            source_valid = valid
            target_slice = slice(None)
        numerator[:, target_slice] += source_values * source_valid
        denominator[:, target_slice] += source_valid

    if torch.any(valid & (denominator == 0)):
        raise AssertionError("A valid token has no temporal-smoothing support")
    smoothed = torch.where(
        valid,
        numerator / denominator.clamp_min(1),
        torch.zeros_like(numerator),
    )
    return smoothed.squeeze(0) if squeeze else smoothed


def smooth_persisted_probabilities(
    probabilities: Sequence[float],
    *,
    window_tokens: int = 3,
) -> tuple[list[float], list[float]]:
    """Apply the frozen logit-space smoother to one unpadded trajectory."""

    if not probabilities:
        raise ValueError("Temporal smoothing requires a non-empty trajectory")
    probability_tensor = torch.tensor(probabilities, dtype=torch.float64)
    if not torch.isfinite(probability_tensor).all() or not torch.all(
        (probability_tensor > 0.0) & (probability_tensor < 1.0)
    ):
        raise ValueError("Persisted sigmoid probabilities must be finite in (0, 1)")
    logits = torch.log(probability_tensor) - torch.log1p(-probability_tensor)
    smoothed_logits = masked_centered_mean_logits(logits, window_tokens=window_tokens)
    smoothed_probabilities = torch.sigmoid(smoothed_logits)
    return (
        [float(value) for value in smoothed_logits.tolist()],
        [float(value) for value in smoothed_probabilities.tolist()],
    )


def build_smoothed_scored_row(
    raw_scored_row: Mapping[str, Any],
    smoothed_logits: Sequence[float],
    smoothed_probabilities: Sequence[float],
    *,
    fold: int,
    seed: int,
    source_dev_scored_sha256: str,
    source_probe_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Replace only H diagnostics while retaining the frozen JP score path."""

    length = len(raw_scored_row.get("output_token_ids", []))
    raw_probabilities = raw_scored_row.get("clir_token_hallucination_probs")
    if length == 0 or len(smoothed_logits) != length:
        raise ValueError("Smoothed logits must align to exact generated tokens")
    if len(smoothed_probabilities) != length or len(raw_probabilities or []) != length:
        raise ValueError("Smoothed probabilities must align to exact generated tokens")
    if any(not math.isfinite(float(value)) for value in smoothed_logits):
        raise ValueError("Smoothed logits must be finite")
    if any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in smoothed_probabilities
    ):
        raise ValueError("Smoothed probabilities must be finite in [0, 1]")

    logit_tensor = torch.tensor(smoothed_logits, dtype=torch.float64)
    log_survival = float(F.logsigmoid(-logit_tensor).sum())
    path_probability = -math.expm1(log_survival)
    pseudo_onset = next(
        (
            index
            for index, value in enumerate(smoothed_probabilities)
            if float(value) >= 0.5
        ),
        -1,
    )
    scored = dict(raw_scored_row)
    scored.update(
        {
            "clir_token_hallucination_probs": [
                float(value) for value in smoothed_probabilities
            ],
            "clir_path_no_hallucination_log_prob": log_survival,
            "clir_path_hallucination_prob": path_probability,
            "clir_pseudo_onset": pseudo_onset,
            "frozen_h_temporal_smoother_provenance": {
                "schema_version": "clir-jp-h-temporal-smoother-scoring-v1",
                "fold": fold,
                "seed": seed,
                "window_tokens": 3,
                "source_dev_scored_sha256": source_dev_scored_sha256,
                "source_probe_checkpoint_sha256": (source_probe_checkpoint_sha256),
                "raw_source_predictions_reused_read_only": True,
                "trainable_parameter_count": 0,
                "smoother_enters_reward_score": False,
            },
        }
    )
    return scored


def metric_delta(
    candidate: Mapping[str, Any], raw: Mapping[str, Any]
) -> dict[str, float]:
    """Return the four predeclared paired localization advantages."""

    return {
        "span_over_position": float(candidate["span_advantage_over_position"]),
        "claim_over_position": float(candidate["claim_advantage_over_position"]),
        "span_over_raw_linear": float(candidate["span_average_precision"])
        - float(raw["span_average_precision"]),
        "claim_over_raw_linear": float(candidate["claim_mean_average_precision"])
        - float(raw["claim_mean_average_precision"]),
    }


def mean_sample_std(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric sequence")
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


__all__ = [
    "PROTOCOL_SCHEMA",
    "build_smoothed_scored_row",
    "masked_centered_mean_logits",
    "mean_sample_std",
    "metric_delta",
    "smooth_persisted_probabilities",
    "validate_smoother_protocol",
]
