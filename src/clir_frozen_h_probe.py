"""Utilities for a hallucination probe on frozen JP token representations."""

from __future__ import annotations

import math
import random
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


PROTOCOL_SCHEMA = "clir-jp-h-frozen-probe-protocol-v1"


def validate_probe_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject silent changes to the frozen linear-probe question."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("Unknown JP-to-H frozen-probe protocol schema")
    representation = protocol.get("representation", {})
    expected_representation = {
        "source": "jp_epoch5_conditioned_token_features",
        "width": 768,
        "base_model_frozen": True,
        "base_checkpoint_read_only": True,
        "base_parameters_in_optimizer": False,
        "probe_input_detached": True,
        "probe_enters_reward_score": False,
    }
    for name, expected in expected_representation.items():
        if representation.get(name) != expected:
            raise ValueError(f"Frozen representation field drifted: {name}")

    probe = protocol.get("probe", {})
    expected_probe = {
        "architecture": "linear_768_to_1_with_bias",
        "parameter_count": 769,
        "loss": "unweighted_explicit_sparse_token_bce",
        "positive_weight": 1.0,
    }
    for name, expected in expected_probe.items():
        if probe.get(name) != expected:
            raise ValueError(f"Frozen probe field drifted: {name}")

    training = protocol.get("training", {})
    expected_training = {
        "seeds": [42, 43, 44],
        "epochs": 100,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "batching": "one_full_batch_of_all_supervised_tokens_per_fold",
        "checkpoint_selection": "fixed_final_epoch_100",
        "early_stopping": False,
        "automatic_hyperparameter_tuning": False,
    }
    for name, expected in expected_training.items():
        if training.get(name) != expected:
            raise ValueError(f"Frozen probe training field drifted: {name}")

    cross_validation = protocol.get("cross_validation", {})
    if list(cross_validation.get("folds", {})) != ["0", "1", "2", "3"]:
        raise ValueError("Frozen probe requires folds 0-3 in order")
    if cross_validation.get("primary_confirmatory_folds") != [1, 2, 3]:
        raise ValueError("Frozen probe confirmatory folds drifted")
    if cross_validation.get("primary_confirmatory_rows") != 48:
        raise ValueError("Frozen probe confirmatory row count drifted")
    if cross_validation.get("all_out_of_fold_rows") != 64:
        raise ValueError("Frozen probe all-fold row count drifted")

    decision = protocol.get("decision_rules", {})
    if decision.get("minimum_passing_seeds") != 2:
        raise ValueError("Frozen probe passing-seed rule drifted")
    for name in (
        "span_ap_strictly_above_same_rows_strongest_position",
        "claim_ap_strictly_above_same_rows_strongest_position",
        "mean_span_advantage_strictly_positive",
        "mean_claim_advantage_strictly_positive",
        "base_checkpoint_sha_unchanged",
        "base_reward_scores_bit_identical",
        "base_token_values_bit_identical",
    ):
        if decision.get(name) is not True:
            raise ValueError(f"Frozen probe decision rule drifted: {name}")


def set_probe_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_linear_probe(
    features: Tensor,
    targets: Tensor,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[nn.Linear, list[dict[str, float | int]]]:
    """Fit only a linear token classifier on detached, frozen features."""

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("Probe features must have non-empty shape [tokens, width]")
    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError("Probe targets must align one-to-one with token features")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("Probe inputs must be finite")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Probe targets must be binary")
    if epochs <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("Invalid frozen-probe optimization parameters")

    set_probe_seed(seed)
    head = nn.Linear(int(features.shape[1]), 1).to(device=device, dtype=torch.float32)
    if sum(parameter.numel() for parameter in head.parameters()) != features.shape[1] + 1:
        raise AssertionError("Linear probe parameter count drifted")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_features = features.detach().to(device=device, dtype=torch.float32)
    train_targets = targets.detach().to(device=device, dtype=torch.float32)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_features).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Frozen-probe training produced a non-finite loss")
        loss.backward()
        if max_grad_norm > 0.0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_grad_norm
            )
        else:
            gradient_norm = torch.sqrt(
                sum(
                    parameter.grad.detach().float().pow(2).sum()
                    for parameter in head.parameters()
                    if parameter.grad is not None
                )
            )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise RuntimeError("Frozen-probe training produced a non-finite gradient")
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "pre_update_train_bce": float(loss.detach().cpu()),
                "pre_clip_gradient_norm": float(
                    torch.as_tensor(gradient_norm).detach().cpu()
                ),
            }
        )

    head.eval()
    with torch.no_grad():
        final_loss = F.binary_cross_entropy_with_logits(
            head(train_features).squeeze(-1), train_targets
        )
    history[-1]["post_update_train_bce"] = float(final_loss.detach().cpu())
    return head, history


@torch.no_grad()
def score_linear_probe(head: nn.Linear, features: Tensor) -> tuple[list[float], list[float]]:
    if features.ndim != 2 or features.shape[1] != head.in_features:
        raise ValueError("Scoring features do not match the frozen linear probe")
    device = next(head.parameters()).device
    logits = head(features.detach().to(device=device, dtype=torch.float32)).squeeze(-1)
    probabilities = torch.sigmoid(logits)
    return (
        [float(value) for value in logits.cpu().tolist()],
        [float(value) for value in probabilities.cpu().tolist()],
    )


def build_probe_scored_row(
    row: Mapping[str, Any],
    canonical_base_row: Mapping[str, Any],
    logits: Sequence[float],
    probabilities: Sequence[float],
    *,
    fold: int,
    seed: int,
    probe_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Publish probe probabilities while copying the untouched JP score path."""

    length = len(row.get("output_token_ids", []))
    values = canonical_base_row.get("clir_token_values")
    if length == 0 or len(logits) != length or len(probabilities) != length:
        raise ValueError("Probe predictions must align to exact generated-token length")
    if not isinstance(values, list) or len(values) != length:
        raise ValueError("Canonical JP token values do not align to the probe row")
    if str(row.get("id")) != str(canonical_base_row.get("id")):
        raise ValueError("Probe row and canonical JP row identifiers differ")
    if any(not math.isfinite(float(value)) for value in logits):
        raise ValueError("Probe logits must be finite")
    if any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in probabilities
    ):
        raise ValueError("Probe probabilities must be finite in [0, 1]")

    logit_tensor = torch.tensor(logits, dtype=torch.float64)
    log_survival = float(F.logsigmoid(-logit_tensor).sum())
    path_probability = -math.expm1(log_survival)
    pseudo_onset = next(
        (index for index, value in enumerate(probabilities) if float(value) >= 0.5),
        -1,
    )
    scored = dict(row)
    scored.update(
        {
            "reward_model_variant": "clir",
            "reward_score": float(canonical_base_row["reward_score"]),
            "clir_score": float(canonical_base_row["clir_score"]),
            "clir_token_values": [float(value) for value in values],
            "clir_token_hallucination_probs": [
                float(value) for value in probabilities
            ],
            "clir_path_no_hallucination_log_prob": log_survival,
            "clir_path_hallucination_prob": path_probability,
            "clir_pseudo_onset": pseudo_onset,
            "frozen_h_probe_provenance": {
                "schema_version": "clir-jp-h-frozen-probe-scoring-v1",
                "fold": fold,
                "seed": seed,
                "probe_checkpoint_sha256": probe_checkpoint_sha256,
                "base_reward_fields_copied_from_canonical_jp_scoring": True,
                "probe_enters_reward_score": False,
            },
        }
    )
    return scored


def localization_metric_summary(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    explicit = metrics["explicit_claim_span_tokens"]
    shortcuts = metrics["explicit_claim_span_shortcuts"]
    claim = shortcuts["claim_level"]
    span_position = max(
        float(shortcuts["absolute_position_ranking"]["average_precision"]),
        float(shortcuts["normalized_position_ranking"]["average_precision"]),
    )
    claim_position = max(
        float(claim["absolute_position_ranking"]["average_precision"]),
        float(claim["normalized_position_ranking"]["average_precision"]),
    )
    span_ap = float(explicit["average_precision"])
    claim_ap = float(claim["mean_probability_ranking"]["average_precision"])
    return {
        "rows": int(metrics["rows"]),
        "span_supervised_tokens": int(explicit["rows"]),
        "span_positive_tokens": int(explicit["positive"]),
        "span_average_precision": span_ap,
        "span_strongest_position_average_precision": span_position,
        "span_advantage_over_position": span_ap - span_position,
        "claims": int(claim["claims"]),
        "positive_claims": int(claim["positive"]),
        "claim_mean_average_precision": claim_ap,
        "claim_strongest_position_average_precision": claim_position,
        "claim_advantage_over_position": claim_ap - claim_position,
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
    "build_probe_scored_row",
    "fit_linear_probe",
    "localization_metric_summary",
    "mean_sample_std",
    "score_linear_probe",
    "set_probe_seed",
    "validate_probe_protocol",
]
