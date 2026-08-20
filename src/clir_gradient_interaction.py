"""Frozen helpers for the no-update CLIR shared-gradient interaction audit."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence


GRADIENT_INTERACTION_SCHEMA = "clir-joint-gradient-interaction-protocol-v1"

EFFECTIVE_OBJECTIVE_WEIGHTS = {
    "final": 1.0,
    "hallucination": 1.0,
    "consistency": 1.0,
    "prior_key": 1.0,
    "prior_complete": 1.0,
    "prior_distill": 0.25,
    "prior_gate": 10.0,
    "prior_total_outer": 1.0,
}

EXPECTED_STREAM_STRUCTURE = {
    "epochs": 5,
    "batches_per_epoch": 992,
    "mechanism_active_batches": [48, 47, 48, 45, 45],
    "consistency_active_batches": [14, 14, 14, 14, 14],
    "mechanism_consistency_overlap_batches": [0, 0, 0, 0, 0],
}

EXPECTED_DECISION_RULES = {
    "negative_cosine_threshold": -0.05,
    "same_batch_negative_fraction_minimum": 0.7,
    "same_batch_median_must_be_negative": True,
    "stable_requires_both_model_states": True,
    "cross_stream_is_not_simultaneous_conflict": True,
    "automatic_repair_authorized": False,
    "automatic_weight_tuning_authorized": False,
    "automatic_sampler_change_authorized": False,
    "additional_training_authorized": False,
}


def validate_gradient_interaction_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != GRADIENT_INTERACTION_SCHEMA:
        raise ValueError("Unexpected gradient-interaction protocol schema")
    if protocol.get("status") != "frozen_before_audit":
        raise ValueError("Gradient-interaction protocol was not frozen before audit")
    if protocol.get("evidence_tier") != "small-scale real":
        raise ValueError("Unexpected gradient-interaction evidence tier")

    weights = protocol.get("effective_objective_weights")
    if not isinstance(weights, Mapping) or {
        name: float(weights.get(name, -1.0)) for name in EFFECTIVE_OBJECTIVE_WEIGHTS
    } != EFFECTIVE_OBJECTIVE_WEIGHTS:
        raise ValueError("Effective objective weights drifted")

    stream = protocol.get("stream_structure")
    if not isinstance(stream, Mapping):
        raise ValueError("Gradient-interaction stream structure is missing")
    for name, expected in EXPECTED_STREAM_STRUCTURE.items():
        if stream.get(name) != expected:
            raise ValueError(f"Frozen stream-structure field drifted: {name}")

    controlled = protocol.get("controlled_batches", {})
    mechanism = controlled.get("mechanism", {})
    consistency = controlled.get("consistency", {})
    expected_mechanism = {
        "rows": 48,
        "batch_size": 4,
        "batches": 12,
        "row_mean_loss": True,
        "coverage": "all mechanism-train rows in frozen manifest order",
    }
    expected_consistency = {
        "rows": 54,
        "semantic_groups": 27,
        "active_batches": 14,
        "positive_pairs": 27,
        "negative_pairs": 26,
        "coverage": "all epoch-1 consistency-active rows with ordinary rows removed",
    }
    for name, expected in expected_mechanism.items():
        if mechanism.get(name) != expected:
            raise ValueError(f"Frozen mechanism batch field drifted: {name}")
    for name, expected in expected_consistency.items():
        if consistency.get(name) != expected:
            raise ValueError(f"Frozen consistency batch field drifted: {name}")

    states = protocol.get("model_states")
    if not isinstance(states, Sequence) or [state.get("name") for state in states] != [
        "initialization_seed42",
        "jp_epoch5",
    ]:
        raise ValueError("Frozen model states/order drifted")
    if states[0].get("kind") != "deterministic_initialization" or int(
        states[0].get("seed", -1)
    ) != 42:
        raise ValueError("Frozen initialization state drifted")
    if states[1].get("kind") != "checkpoint":
        raise ValueError("Frozen JP checkpoint state drifted")

    groups = protocol.get("shared_parameter_groups")
    required_groups = {
        "shared_all",
        "input_encoder_all",
        "input_projection",
        "transformer_block_0",
        "transformer_block_1",
        "layer_pool",
        "encoder_output",
        "condition_all",
        "condition_attention",
        "condition_fusion",
    }
    if not isinstance(groups, Mapping) or set(groups) != required_groups:
        raise ValueError("Frozen shared-parameter groups drifted")
    for name, prefixes in groups.items():
        if not isinstance(prefixes, Sequence) or isinstance(prefixes, (str, bytes)):
            raise ValueError(f"Parameter group {name} must contain prefixes")
        if not prefixes or any(not isinstance(prefix, str) for prefix in prefixes):
            raise ValueError(f"Parameter group {name} contains invalid prefixes")

    rules = protocol.get("decision_rules")
    if not isinstance(rules, Mapping):
        raise ValueError("Gradient-interaction decision rules are missing")
    for name, expected in EXPECTED_DECISION_RULES.items():
        if rules.get(name) != expected:
            raise ValueError(f"Frozen gradient decision rule drifted: {name}")


def summarize_cosines(values: Sequence[float], *, threshold: float) -> dict[str, float | int]:
    finite = [float(value) for value in values]
    if not finite or any(not math.isfinite(value) for value in finite):
        raise ValueError("Cosine summary requires finite non-empty values")
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "negative_fraction": sum(value < 0.0 for value in finite) / len(finite),
        "at_or_below_threshold_fraction": sum(value <= threshold for value in finite)
        / len(finite),
    }


def classify_same_batch_conflict(
    state_metrics: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float,
    negative_fraction_minimum: float,
) -> str:
    if not state_metrics:
        raise ValueError("Same-batch classification requires model states")
    flags: list[bool] = []
    for metrics in state_metrics.values():
        aggregate = float(metrics["aggregate_shared_cosine"])
        batch = metrics["batch_shared_cosines"]
        flags.append(
            aggregate <= threshold
            and float(batch["negative_fraction"]) >= negative_fraction_minimum
            and float(batch["median"]) < 0.0
        )
    if all(flags):
        return "stable_same_batch_conflict"
    if any(flags):
        return "state_specific_same_batch_conflict"
    return "no_stable_same_batch_conflict"


def classify_cross_stream_pressure(
    state_cosines: Mapping[str, float], *, threshold: float
) -> str:
    if not state_cosines:
        raise ValueError("Cross-stream classification requires model states")
    flags = [float(value) <= threshold for value in state_cosines.values()]
    if all(flags):
        return "stable_cross_stream_opposition"
    if any(flags):
        return "state_specific_cross_stream_opposition"
    return "no_stable_cross_stream_opposition"


__all__ = [
    "EFFECTIVE_OBJECTIVE_WEIGHTS",
    "EXPECTED_DECISION_RULES",
    "EXPECTED_STREAM_STRUCTURE",
    "GRADIENT_INTERACTION_SCHEMA",
    "classify_cross_stream_pressure",
    "classify_same_batch_conflict",
    "summarize_cosines",
    "validate_gradient_interaction_protocol",
]
