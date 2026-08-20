"""Frozen contract helpers for the targeted H-to-condition gradient route."""

from __future__ import annotations

import math
from typing import Any, Mapping


CONDITION_ROUTING_PROTOCOL_SCHEMA = "clir-joint-condition-routing-audit-v1"

BLOCKED_PARAMETER_PREFIXES = (
    "condition_query.",
    "condition_key.",
    "condition_value.",
    "condition_fusion.",
)

HALLUCINATION_REQUIRED_NONZERO_PREFIXES = {
    "trajectory_and_condition_encoder": ("input_encoder.",),
    "shared_feature_norm": ("feature_norm.",),
    "hallucination_head": ("hallucination_head.",),
}

INVARIANT_OBJECTIVE_WEIGHTS = {
    "final": 1.0,
    "prior_key": 1.0,
    "prior_complete": 1.0,
    "prior_distill": 0.25,
    "prior_gate": 10.0,
}

FORWARD_INVARIANT_FIELDS = (
    "scores",
    "hallucination_logits",
    "token_features",
    "key_prior",
    "complete_prior",
    "gates",
)

EXPECTED_DECISION_RULES = {
    "forward_max_abs_tolerance": 0.0,
    "objective_loss_max_abs_tolerance": 0.0,
    "blocked_hallucination_gradient_l2_tolerance": 0.0,
    "baseline_blocked_hallucination_gradient_must_be_positive": True,
    "required_hallucination_routes_must_be_positive": True,
    "gradient_max_abs_difference_tolerance": 1e-5,
    "gradient_relative_l2_difference_tolerance": 1e-5,
    "all_controlled_batches_must_pass": True,
    "no_parameter_update_required": True,
    "optimizer_grad_buffers_must_remain_absent": True,
    "automatic_training_authorized": False,
    "automatic_loss_weight_tuning_authorized": False,
    "automatic_sampler_change_authorized": False,
}


def validate_condition_routing_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != CONDITION_ROUTING_PROTOCOL_SCHEMA:
        raise ValueError("Unexpected condition-routing protocol schema")
    if protocol.get("status") != "frozen_before_audit":
        raise ValueError("Condition-routing protocol was not frozen before audit")
    if protocol.get("evidence_tier") != "small-scale real":
        raise ValueError("Unexpected condition-routing evidence tier")

    route = protocol.get("route_contract")
    if not isinstance(route, Mapping):
        raise ValueError("Condition-routing route contract is missing")
    if route.get("baseline_flag") is not False or route.get("candidate_flag") is not True:
        raise ValueError("Condition-routing flags drifted")
    if tuple(route.get("blocked_parameter_prefixes", ())) != BLOCKED_PARAMETER_PREFIXES:
        raise ValueError("Condition-routing blocked parameter set drifted")
    observed_required = {
        name: tuple(prefixes)
        for name, prefixes in route.get(
            "hallucination_required_nonzero_prefixes", {}
        ).items()
    }
    if observed_required != HALLUCINATION_REQUIRED_NONZERO_PREFIXES:
        raise ValueError("Condition-routing required H routes drifted")
    if route.get("feature_norm_blocked") is not False:
        raise ValueError("feature_norm must remain trainable from hallucination BCE")
    if route.get("condition_forward_value_changed") is not False:
        raise ValueError("Condition-routing forward values must remain unchanged")
    if route.get("dual_prior_architecture_or_loss_changed") is not False:
        raise ValueError("Original dual-prior method may not change in this audit")
    if route.get("gate_fused_prior_target_remains_detached") is not True:
        raise ValueError("Original reward-gate detached target must be preserved")
    if route.get("bidirectional_mutual_updates_both_prior_heads") is not True:
        raise ValueError("Original bidirectional mutual routing must be preserved")

    objectives = protocol.get("invariant_objective_weights")
    if not isinstance(objectives, Mapping) or {
        name: float(objectives.get(name, float("nan")))
        for name in INVARIANT_OBJECTIVE_WEIGHTS
    } != INVARIANT_OBJECTIVE_WEIGHTS:
        raise ValueError("Condition-routing invariant objective weights drifted")
    if set(objectives) != set(INVARIANT_OBJECTIVE_WEIGHTS):
        raise ValueError("Condition-routing invariant objective set drifted")
    if tuple(protocol.get("forward_invariant_fields", ())) != FORWARD_INVARIANT_FIELDS:
        raise ValueError("Condition-routing forward invariant fields drifted")

    controlled = protocol.get("controlled_batches")
    expected_controlled = {
        "rows": 48,
        "batch_size": 4,
        "batches": 12,
        "coverage": "all frozen mechanism-train rows in manifest order",
        "row_mean_objectives": True,
    }
    if not isinstance(controlled, Mapping):
        raise ValueError("Condition-routing controlled batches are missing")
    for name, expected in expected_controlled.items():
        if controlled.get(name) != expected:
            raise ValueError(f"Condition-routing controlled field drifted: {name}")

    states = protocol.get("model_states")
    if not isinstance(states, list) or [state.get("name") for state in states] != [
        "initialization_seed42",
        "jp_epoch5",
    ]:
        raise ValueError("Condition-routing model states drifted")
    if states[0].get("kind") != "deterministic_initialization" or int(
        states[0].get("seed", -1)
    ) != 42:
        raise ValueError("Condition-routing initialization state drifted")
    if states[1].get("kind") != "checkpoint" or states[1].get("input") != "jp_epoch5_checkpoint":
        raise ValueError("Condition-routing checkpoint state drifted")

    rules = protocol.get("decision_rules")
    if not isinstance(rules, Mapping):
        raise ValueError("Condition-routing decision rules are missing")
    for name, expected in EXPECTED_DECISION_RULES.items():
        observed = rules.get(name)
        if isinstance(expected, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(f"Condition-routing decision rule drifted: {name}")
        elif observed != expected:
            raise ValueError(f"Condition-routing decision rule drifted: {name}")


__all__ = [
    "BLOCKED_PARAMETER_PREFIXES",
    "CONDITION_ROUTING_PROTOCOL_SCHEMA",
    "EXPECTED_DECISION_RULES",
    "FORWARD_INVARIANT_FIELDS",
    "HALLUCINATION_REQUIRED_NONZERO_PREFIXES",
    "INVARIANT_OBJECTIVE_WEIGHTS",
    "validate_condition_routing_protocol",
]
