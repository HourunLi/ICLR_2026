"""Frozen scoring contract for the mixed-domain JP-to-H blind validation."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from .clir_hallucination_annotation import canonical_sha256
from .clir_real_data import validate_rollout_row


PROTOCOL_SCHEMA = "clir-jp-h-blind-scoring-protocol-v1"
LINEAGE_SCHEMA = "clir-jp-h-blind-private-lineage-v1"
SCORING_INPUT_SCHEMA = "clir-jp-h-blind-scoring-input-v1"
PREDICTION_SCHEMA = "clir-jp-h-blind-prediction-v1"


def validate_blind_scoring_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject changes to the candidate after blind annotations exist."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("Unknown JP-to-H blind-scoring protocol schema")
    if protocol.get("evidence_tier") != "blind_validation":
        raise ValueError("Blind-scoring evidence tier drifted")
    if protocol.get("status") != "frozen_before_label_access":
        raise ValueError("Blind-scoring freeze status drifted")

    readout = protocol.get("readout", {})
    expected_readout = {
        "architecture": "linear_768_to_1_with_bias",
        "feature_width": 768,
        "parameter_count_per_head": 769,
        "seeds": [42, 43, 44],
        "head_scope": "one_all64_head_per_predeclared_seed",
        "training_rows": 64,
        "supervised_tokens": 9132,
        "positive_tokens": 1450,
        "loss": "unweighted_explicit_sparse_token_bce",
        "epochs": 100,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "batching": "one_full_batch_of_all_supervised_tokens",
        "checkpoint_selection": "fixed_final_epoch_100",
        "model_selection_across_seeds": False,
        "new_labels_used_for_training": False,
    }
    for name, expected in expected_readout.items():
        if readout.get(name) != expected:
            raise ValueError(f"Frozen blind readout field drifted: {name}")

    representation = protocol.get("representation", {})
    expected_representation = {
        "source": "jp_epoch5_conditioned_token_features",
        "task_model": "microsoft/Phi-3.5-mini-instruct",
        "task_model_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
        "tokenizer_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
        "layer_count": 33,
        "per_layer_hidden_size": 3072,
        "all_layer_feature_width": 101376,
        "extract_from_exact_saved_token_ids": True,
        "base_jp_frozen": True,
        "base_jp_enters_optimizer": False,
        "new_token_features_persisted": True,
    }
    for name, expected in expected_representation.items():
        if representation.get(name) != expected:
            raise ValueError(f"Frozen blind representation field drifted: {name}")

    smoother = protocol.get("smoother", {})
    expected_smoother = {
        "architecture": "fixed_masked_centered_logit_mean",
        "window_tokens": 3,
        "radius_tokens": 1,
        "centered": True,
        "boundary_rule": "mean_over_available_real_tokens_only",
        "input": "stable_float64_logit_of_persisted_float32_sigmoid_probability",
        "trainable_parameter_count": 0,
        "head_retraining": False,
        "window_search": False,
        "enters_reward_score": False,
    }
    for name, expected in expected_smoother.items():
        if smoother.get(name) != expected:
            raise ValueError(f"Frozen blind smoother field drifted: {name}")

    isolation = protocol.get("blind_isolation", {})
    if isolation.get("new_label_access_before_prediction_freeze_forbidden") is not True:
        raise ValueError("Blind label-access prohibition drifted")
    if isolation.get("lineage_private_fields_removed_before_model_forward") is not True:
        raise ValueError("Private-field sanitization drifted")
    if isolation.get("forbidden_private_fields") != [
        "correctness",
        "domain",
        "parsed_answer",
        "reference_answer",
        "answer_type",
        "problem",
        "prompt",
        "trajectory",
    ]:
        raise ValueError("Blind private-field denylist drifted")

    evaluation = protocol.get("post_unseal_evaluation", {})
    expected_evaluation = {
        "candidate": "three_token_centered_logit_smoother",
        "comparators": [
            "paired_raw_all64_linear_head",
            "same_rows_strongest_absolute_or_normalized_position",
        ],
        "primary_units": [
            "explicit_claim_span_token_average_precision",
            "claim_mean_probability_average_precision",
        ],
        "minimum_passing_seeds": 2,
        "require_candidate_above_both_comparators_on_both_primary_units": True,
        "require_mean_of_all_four_paired_advantages_positive": True,
        "pooled_96_is_primary": True,
        "per_domain_results_are_descriptive": True,
        "tail_targets_used_for_adoption": False,
        "threshold_tuning_on_blind_rows": False,
        "automatic_reward_score_coupling": False,
    }
    for name, expected in expected_evaluation.items():
        if evaluation.get(name) != expected:
            raise ValueError(f"Frozen post-unseal evaluation field drifted: {name}")

    input_paths = {
        str(spec.get("path"))
        for spec in protocol.get("inputs", {}).values()
        if isinstance(spec, Mapping)
    }
    sealed_paths = {
        str(spec.get("path"))
        for spec in protocol.get("sealed_annotations", [])
        if isinstance(spec, Mapping)
    }
    if not sealed_paths or input_paths & sealed_paths:
        raise ValueError("Sealed annotations must be declared and absent from scoring inputs")


def sanitize_lineage_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact-token model input without private selection metadata."""

    if row.get("schema_version") != LINEAGE_SCHEMA:
        raise ValueError("Unknown blind private-lineage schema")
    if row.get("exact_token_alignment") is not True:
        raise ValueError("Blind scoring requires exact token alignment")
    item_id = str(row.get("item_id", ""))
    query_id = str(row.get("query_id", ""))
    if not item_id or not query_id:
        raise ValueError("Blind lineage row is missing item/query identity")
    prompt_ids = list(row.get("prompt_token_ids", []))
    output_ids = list(row.get("output_token_ids", []))
    if canonical_sha256(prompt_ids) != row.get("prompt_token_ids_sha256"):
        raise ValueError(f"Prompt-token hash drifted for {item_id}")
    if canonical_sha256(output_ids) != row.get("output_token_ids_sha256"):
        raise ValueError(f"Output-token hash drifted for {item_id}")
    trajectory = str(row.get("trajectory", ""))
    if hashlib.sha256(trajectory.encode("utf-8")).hexdigest() != row.get(
        "trajectory_sha256"
    ):
        raise ValueError(f"Trajectory text hash drifted for {item_id}")
    provenance = dict(row.get("provenance", {}))
    provenance["protocol_version"] = "clir-jp-h-blind-validation-v1"
    sanitized = {
        "schema_version": SCORING_INPUT_SCHEMA,
        "id": item_id,
        "item_id": item_id,
        "query_id": query_id,
        "selection_order": int(row["selection_order"]),
        "prompt_token_ids": prompt_ids,
        "output_token_ids": output_ids,
        "output_token_ids_sha256": str(row["output_token_ids_sha256"]),
        "trajectory_sha256": str(row["trajectory_sha256"]),
        "response": trajectory,
        "provenance": provenance,
    }
    validate_rollout_row(sanitized)
    forbidden = {
        "correctness",
        "domain",
        "parsed_answer",
        "reference_answer",
        "answer_type",
        "problem",
        "prompt",
        "trajectory",
    }
    if forbidden & set(sanitized):
        raise AssertionError("Private lineage fields leaked into the model input")
    return sanitized


def build_prediction_row(
    row: Mapping[str, Any],
    seed_predictions: Sequence[Mapping[str, Any]],
    *,
    protocol_sha256: str,
    jp_feature_sha256: str,
) -> dict[str, Any]:
    """Construct one label-free, exact-token-aligned frozen prediction row."""

    length = len(row.get("output_token_ids", []))
    if length <= 0:
        raise ValueError("Blind prediction requires at least one output token")
    observed_seeds: list[int] = []
    normalized_predictions: list[dict[str, Any]] = []
    for prediction in seed_predictions:
        seed = int(prediction["seed"])
        observed_seeds.append(seed)
        normalized = {
            "seed": seed,
            "head_sha256": str(prediction["head_sha256"]),
            "raw_logits": [float(value) for value in prediction["raw_logits"]],
            "raw_probabilities": [
                float(value) for value in prediction["raw_probabilities"]
            ],
            "smoothed_logits": [
                float(value) for value in prediction["smoothed_logits"]
            ],
            "smoothed_probabilities": [
                float(value) for value in prediction["smoothed_probabilities"]
            ],
        }
        for field in (
            "raw_logits",
            "raw_probabilities",
            "smoothed_logits",
            "smoothed_probabilities",
        ):
            values = normalized[field]
            if len(values) != length or any(not math.isfinite(value) for value in values):
                raise ValueError(f"Blind prediction field {field} is not token aligned")
        for field in ("raw_probabilities", "smoothed_probabilities"):
            if any(not 0.0 < value < 1.0 for value in normalized[field]):
                raise ValueError(f"Blind prediction field {field} is outside (0, 1)")
        normalized_predictions.append(normalized)
    if observed_seeds != [42, 43, 44]:
        raise ValueError("Blind prediction seed order drifted")

    denominator = max(length - 1, 1)
    prediction_row = {
        "schema_version": PREDICTION_SCHEMA,
        "selection_order": int(row["selection_order"]),
        "item_id": str(row["item_id"]),
        "query_id": str(row["query_id"]),
        "output_token_ids": [int(value) for value in row["output_token_ids"]],
        "output_token_ids_sha256": str(row["output_token_ids_sha256"]),
        "trajectory_sha256": str(row["trajectory_sha256"]),
        "token_count": length,
        "absolute_position_scores": [float(position) for position in range(length)],
        "normalized_position_scores": [
            float(position / denominator) for position in range(length)
        ],
        "seed_predictions": normalized_predictions,
        "jp_feature_sha256": jp_feature_sha256,
        "scoring_protocol_sha256": protocol_sha256,
        "new_hallucination_labels_used": False,
        "smoother_enters_reward_score": False,
    }
    return prediction_row


def validate_prediction_row(
    row: Mapping[str, Any],
    *,
    expected_item: Mapping[str, Any] | None = None,
    protocol_sha256: str | None = None,
) -> None:
    """Validate a durable checkpoint without consulting any annotation file."""

    if row.get("schema_version") != PREDICTION_SCHEMA:
        raise ValueError("Unknown blind-prediction schema")
    if protocol_sha256 is not None and row.get("scoring_protocol_sha256") != protocol_sha256:
        raise ValueError("Blind prediction protocol hash drifted")
    if row.get("new_hallucination_labels_used") is not False:
        raise ValueError("Blind prediction label-isolation marker drifted")
    if row.get("smoother_enters_reward_score") is not False:
        raise ValueError("Blind smoother unexpectedly entered reward scoring")
    if expected_item is not None:
        for field in (
            "selection_order",
            "item_id",
            "query_id",
            "output_token_ids_sha256",
            "trajectory_sha256",
        ):
            if row.get(field) != expected_item.get(field):
                raise ValueError(f"Blind prediction identity drifted: {field}")
        if list(row.get("output_token_ids", [])) != list(
            expected_item.get("output_token_ids", [])
        ):
            raise ValueError("Blind prediction exact output-token IDs drifted")
    rebuilt = build_prediction_row(
        row,
        row.get("seed_predictions", []),
        protocol_sha256=str(row.get("scoring_protocol_sha256", "")),
        jp_feature_sha256=str(row.get("jp_feature_sha256", "")),
    )
    for field in (
        "selection_order",
        "item_id",
        "query_id",
        "output_token_ids",
        "output_token_ids_sha256",
        "trajectory_sha256",
        "token_count",
        "absolute_position_scores",
        "normalized_position_scores",
        "seed_predictions",
        "jp_feature_sha256",
        "scoring_protocol_sha256",
        "new_hallucination_labels_used",
        "smoother_enters_reward_score",
    ):
        if row.get(field) != rebuilt.get(field):
            raise ValueError(f"Blind prediction field drifted: {field}")


__all__ = [
    "LINEAGE_SCHEMA",
    "PREDICTION_SCHEMA",
    "PROTOCOL_SCHEMA",
    "SCORING_INPUT_SCHEMA",
    "build_prediction_row",
    "sanitize_lineage_row",
    "validate_blind_scoring_protocol",
    "validate_prediction_row",
]
