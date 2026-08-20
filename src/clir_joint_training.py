"""Frozen configuration helpers for CLIR joint-training diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

from .consistency_localized_reward import RewardConfig


JOINT_PROTOCOL_SCHEMA = "clir-joint-training-pilot-protocol-v1"
JOINT_DROP_ONE_PROTOCOL_SCHEMA = "clir-joint-training-drop-one-protocol-v1"
JOINT_PACKING_PROTOCOL_SCHEMA = "clir-joint-training-packing-protocol-v1"
JOINT_CONDITION_ROUTING_PROTOCOL_SCHEMA = (
    "clir-joint-training-condition-routing-protocol-v1"
)
LOSS_NAMES = (
    "final",
    "consistency",
    "negative_consistency",
    "score_consistency",
    "hallucination",
    "mil",
    "token_reward",
    "tail",
    "relative_tail",
    "pseudo_tail",
    "progress",
    "prior",
    "key_prior",
    "complete_prior",
    "prior_distill",
    "gate_prior",
    "reconstruction",
)

_FROZEN_MODEL = {
    "variant": "clir",
    "hidden_dim": 101376,
    "encoder_type": "layer_transformer",
    "model_dim": 768,
    "num_feature_layers": 33,
    "per_layer_dim": 3072,
    "layer_encoder_dim": 256,
    "layer_encoder_blocks": 2,
    "layer_encoder_heads": 8,
    "layer_pool_queries": 4,
    "projection_dim": 256,
    "encoder_dropout": 0.0,
}
_FROZEN_TRAINING = {
    "seeds": [42],
    "epochs": 5,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "amp_dtype": "bfloat16",
    "group_by_semantic_id": True,
    "hidden_state_source": "precomputed",
    "validation_every_n_epochs": 5,
    "prior_phase_mode": "joint",
    "train_rows": 3968,
    "batches_per_epoch": 992,
}
_FROZEN_SHARED_WEIGHTS = {
    "final": 1.0,
    "consistency": 0.0,
    "negative_consistency": 1.0,
    "score_consistency": 0.1,
    "hallucination": 0.0,
    "mil": 0.0,
    "token_reward": 0.0,
    "tail": 0.0,
    "relative_tail": 0.0,
    "pseudo_tail": 0.0,
    "progress": 0.0,
    "prior": 0.0,
    "key_prior": 1.0,
    "complete_prior": 1.0,
    "prior_distill": 0.25,
    "gate_prior": 10.0,
    "reconstruction": 0.0,
}
_FROZEN_CELL_OVERRIDES_BY_SCHEMA = {
    JOINT_PROTOCOL_SCHEMA: {
        "j0_correctness": {},
        "jp_original_prior": {"prior": 1.0},
        "jall_full_retained": {
            "consistency": 1.0,
            "hallucination": 1.0,
            "prior": 1.0,
        },
    },
    JOINT_DROP_ONE_PROTOCOL_SCHEMA: {
        "jph_prior_plus_hallucination": {
            "hallucination": 1.0,
            "prior": 1.0,
        },
        "jpc_prior_plus_consistency": {
            "consistency": 1.0,
            "prior": 1.0,
        },
    },
    JOINT_PACKING_PROTOCOL_SCHEMA: {
        "jph_supervision_packed": {
            "hallucination": 1.0,
            "prior": 1.0,
        },
    },
    JOINT_CONDITION_ROUTING_PROTOCOL_SCHEMA: {
        "jph_h_condition_stopgrad": {
            "hallucination": 1.0,
            "prior": 1.0,
        },
    },
}


def validate_joint_protocol(protocol: Mapping[str, Any]) -> None:
    schema = protocol.get("schema_version")
    expected_cells = _FROZEN_CELL_OVERRIDES_BY_SCHEMA.get(schema)
    if expected_cells is None:
        raise ValueError("Unexpected joint-training protocol schema")
    cells = protocol.get("cells")
    if not isinstance(cells, Mapping) or list(cells) != list(expected_cells):
        raise ValueError(
            "Joint-training protocol cells/order differ from the frozen schema"
        )
    for name, expected in _FROZEN_MODEL.items():
        if protocol.get("model", {}).get(name) != expected:
            raise ValueError(f"Frozen joint model field drifted: {name}")
    for name, expected in _FROZEN_TRAINING.items():
        if protocol.get("matched_training", {}).get(name) != expected:
            raise ValueError(f"Frozen joint training field drifted: {name}")
    shared = protocol.get("losses", {}).get("shared")
    if not isinstance(shared, Mapping) or {
        name: float(shared.get(name, -1.0)) for name in LOSS_NAMES
    } != _FROZEN_SHARED_WEIGHTS:
        raise ValueError("Frozen joint shared loss weights drifted")
    for cell_name, expected in expected_cells.items():
        overrides = cells[cell_name].get("loss_overrides")
        if not isinstance(overrides, Mapping) or {
            name: float(value) for name, value in overrides.items()
        } != expected:
            raise ValueError(f"Frozen joint cell overrides drifted: {cell_name}")
    method = protocol.get("method", {})
    expected_method = {
        "consistency_margin": 0.2,
        "hallucination_target_mode": "explicit",
        "hallucination_positive_weight": 1.0,
        "prior_fusion_alpha": 0.5,
        "progress_score_weight": 0.0,
        "negative_tail_margin": 0.5,
        "relative_tail_margin": 0.5,
        "pseudo_onset_threshold": 0.5,
    }
    for name, expected in expected_method.items():
        if method.get(name) != expected:
            raise ValueError(f"Frozen joint method field drifted: {name}")
    if schema == JOINT_DROP_ONE_PROTOCOL_SCHEMA:
        rules = protocol.get("drop_one_decision_rules", {})
        expected_rules = {
            "key_ap_drop_threshold_vs_jp": 0.05,
            "hallucination_span_token_ap_min_exclusive": 0.39328067905143455,
            "hallucination_claim_ap_min_exclusive": 0.42198767865054354,
            "ranking_max_absolute_regression_vs_jp": 0.02,
            "no_seed_expansion_from_this_diagnostic": True,
            "automatic_loss_weight_tuning": False,
            "automatic_stream_switch": False,
        }
        for name, expected in expected_rules.items():
            if rules.get(name) != expected:
                raise ValueError(f"Frozen drop-one decision rule drifted: {name}")
    if schema == JOINT_PACKING_PROTOCOL_SCHEMA:
        packing = protocol.get("batch_packing", {})
        expected_packing = {
            "enabled": True,
            "sidecar_schema": "id_and_packing_pool_id_jsonl",
            "pool_id": "mechanism_explicit_hallucination_and_original_prior",
            "rows": 48,
            "pool_size": 48,
            "exclusive_batch_size": 4,
            "exclusive_batches_per_epoch": 12,
            "reshuffle_membership_each_epoch": True,
            "semantic_metadata_unchanged": True,
        }
        for name, expected in expected_packing.items():
            if packing.get(name) != expected:
                raise ValueError(f"Frozen joint packing field drifted: {name}")
        if protocol.get("method", {}).get("sampler_or_batch_packing_changed") is not True:
            raise ValueError("Packing protocol must declare its sampler change")
        if protocol.get("method", {}).get("dual_prior_architecture_changed") is not False:
            raise ValueError("Packing protocol must preserve the original dual-prior architecture")
        rules = protocol.get("packing_decision_rules", {})
        expected_rules = {
            "key_ap_min_recovery_vs_jph": 0.05,
            "key_ap_max_drop_vs_jp": 0.05,
            "hallucination_span_token_ap_min_exclusive": 0.39328067905143455,
            "hallucination_claim_ap_min_exclusive": 0.42198767865054354,
            "ranking_max_absolute_regression_vs_jp": 0.02,
            "success_requires_budget_matched_followup": True,
            "failure_authorizes_condition_branch_design_discussion_only": True,
            "no_seed_expansion_from_this_diagnostic": True,
            "automatic_loss_weight_tuning": False,
            "automatic_gradient_surgery": False,
            "automatic_stream_switch": False,
        }
        for name, expected in expected_rules.items():
            if rules.get(name) != expected:
                raise ValueError(f"Frozen packing decision rule drifted: {name}")
    if schema == JOINT_CONDITION_ROUTING_PROTOCOL_SCHEMA:
        expected_method_flags = {
            "hallucination_condition_stop_gradient": True,
            "dual_prior_architecture_changed": False,
            "mutual_distillation_changed": False,
            "loss_formula_changed": False,
            "sampler_or_batch_packing_changed": False,
            "loss_weight_scan_in_this_protocol": False,
            "multistream_training_in_this_protocol": False,
        }
        for name, expected in expected_method_flags.items():
            if protocol.get("method", {}).get(name) is not expected:
                raise ValueError(
                    f"Frozen condition-routing method field drifted: {name}"
                )
        if protocol.get("batch_packing", {}).get("enabled", False):
            raise ValueError(
                "Condition-routing protocol must restore ordinary JPH batching"
            )
        expected_rules = {
            "key_ap_min_recovery_vs_jph": 0.05,
            "key_ap_max_drop_vs_jp": 0.05,
            "complete_ap_max_drop_vs_jph": 0.05,
            "hallucination_span_token_ap_min_exclusive": 0.39328067905143455,
            "hallucination_claim_ap_min_exclusive": 0.42198767865054354,
            "ranking_max_absolute_regression_vs_jp": 0.02,
            "no_seed_expansion_from_this_diagnostic": True,
            "automatic_loss_weight_tuning": False,
            "automatic_gradient_surgery": False,
            "automatic_sampler_or_stream_change": False,
        }
        rules = protocol.get("condition_routing_decision_rules", {})
        for name, expected in expected_rules.items():
            if rules.get(name) != expected:
                raise ValueError(
                    f"Frozen condition-routing decision rule drifted: {name}"
                )


def resolve_loss_weights(
    protocol: Mapping[str, Any], cell_name: str
) -> dict[str, float]:
    validate_joint_protocol(protocol)
    if cell_name not in protocol["cells"]:
        raise ValueError(f"Unknown joint-training cell {cell_name!r}")
    shared = protocol["losses"]["shared"]
    cell = protocol["cells"][cell_name]
    weights = {name: float(shared[name]) for name in LOSS_NAMES}
    for name, value in cell["loss_overrides"].items():
        if name not in weights:
            raise ValueError(f"Unknown joint-training loss override {name!r}")
        weights[name] = float(value)
    if set(weights) != set(LOSS_NAMES) or any(value < 0.0 for value in weights.values()):
        raise ValueError("Joint-training loss weights are incomplete or negative")
    return weights


def reward_config_from_protocol(
    protocol: Mapping[str, Any], cell_name: str
) -> RewardConfig:
    weights = resolve_loss_weights(protocol, cell_name)
    model = protocol["model"]
    method = protocol["method"]
    return RewardConfig(
        hidden_dim=int(model["hidden_dim"]),
        model_variant=str(model["variant"]),
        encoder_type=str(model["encoder_type"]),
        model_dim=int(model["model_dim"]),
        num_feature_layers=int(model["num_feature_layers"]),
        per_layer_dim=int(model["per_layer_dim"]),
        layer_encoder_dim=int(model["layer_encoder_dim"]),
        layer_encoder_blocks=int(model["layer_encoder_blocks"]),
        layer_encoder_heads=int(model["layer_encoder_heads"]),
        layer_pool_queries=int(model["layer_pool_queries"]),
        encoder_dropout=float(model["encoder_dropout"]),
        projection_dim=int(model["projection_dim"]),
        consistency_margin=float(method["consistency_margin"]),
        hallucination_target_mode=str(method["hallucination_target_mode"]),
        hallucination_positive_weight=float(method["hallucination_positive_weight"]),
        hallucination_condition_stop_gradient=bool(
            method.get("hallucination_condition_stop_gradient", False)
        ),
        prior_fusion_alpha=float(method["prior_fusion_alpha"]),
        progress_score_weight=float(method["progress_score_weight"]),
        negative_tail_margin=float(method["negative_tail_margin"]),
        relative_tail_margin=float(method["relative_tail_margin"]),
        pseudo_onset_threshold=float(method["pseudo_onset_threshold"]),
        final_weight=weights["final"],
        consistency_weight=weights["consistency"],
        negative_consistency_weight=weights["negative_consistency"],
        score_consistency_weight=weights["score_consistency"],
        hallucination_weight=weights["hallucination"],
        mil_weight=weights["mil"],
        token_reward_weight=weights["token_reward"],
        tail_weight=weights["tail"],
        relative_tail_weight=weights["relative_tail"],
        pseudo_tail_weight=weights["pseudo_tail"],
        progress_weight=weights["progress"],
        prior_weight=weights["prior"],
        key_prior_weight=weights["key_prior"],
        complete_prior_weight=weights["complete_prior"],
        prior_distill_weight=weights["prior_distill"],
        gate_prior_weight=weights["gate_prior"],
        reconstruction_weight=weights["reconstruction"],
    )


__all__ = [
    "JOINT_CONDITION_ROUTING_PROTOCOL_SCHEMA",
    "JOINT_DROP_ONE_PROTOCOL_SCHEMA",
    "JOINT_PACKING_PROTOCOL_SCHEMA",
    "JOINT_PROTOCOL_SCHEMA",
    "LOSS_NAMES",
    "resolve_loss_weights",
    "reward_config_from_protocol",
    "validate_joint_protocol",
]
