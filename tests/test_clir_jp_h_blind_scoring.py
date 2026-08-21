import json
from pathlib import Path

import pytest

from src.clir_jp_h_blind_scoring import (
    build_prediction_row,
    sanitize_lineage_row,
    validate_blind_scoring_protocol,
    validate_prediction_row,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/jp_h_blind_validation_v1/scoring_protocol_v1.json"
LINEAGE = (
    ROOT
    / "run_artifacts/jp_h_blind_validation_v1/acquisition/private_lineage_v1.jsonl"
)
RUNNER = ROOT / "scripts/run_jp_h_blind_scoring_v1.py"


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_blind_scoring_protocol_is_frozen_and_sealed_labels_are_not_inputs() -> None:
    protocol = load_protocol()
    validate_blind_scoring_protocol(protocol)
    input_paths = {spec["path"] for spec in protocol["inputs"].values()}
    sealed_paths = {spec["path"] for spec in protocol["sealed_annotations"]}
    assert input_paths.isdisjoint(sealed_paths)
    assert protocol["readout"]["seeds"] == [42, 43, 44]
    assert protocol["post_unseal_evaluation"]["tail_targets_used_for_adoption"] is False


def test_lineage_sanitizer_preserves_exact_ids_and_removes_private_fields() -> None:
    raw = json.loads(LINEAGE.read_text(encoding="utf-8").splitlines()[0])
    sanitized = sanitize_lineage_row(raw)
    assert sanitized["id"] == raw["item_id"]
    assert sanitized["prompt_token_ids"] == raw["prompt_token_ids"]
    assert sanitized["output_token_ids"] == raw["output_token_ids"]
    assert sanitized["response"] == raw["trajectory"]
    for field in (
        "correctness",
        "domain",
        "parsed_answer",
        "reference_answer",
        "answer_type",
        "problem",
        "prompt",
        "trajectory",
    ):
        assert field not in sanitized


def test_prediction_builder_is_exact_token_aligned_and_validates() -> None:
    item = {
        "selection_order": 0,
        "item_id": "JPHB-test",
        "query_id": "query-test",
        "output_token_ids": [10, 11, 12],
        "output_token_ids_sha256": "token-hash",
        "trajectory_sha256": "text-hash",
    }
    predictions = []
    for seed in (42, 43, 44):
        predictions.append(
            {
                "seed": seed,
                "head_sha256": f"head-{seed}",
                "raw_logits": [-1.0, 0.0, 1.0],
                "raw_probabilities": [0.25, 0.5, 0.75],
                "smoothed_logits": [-0.5, 0.0, 0.5],
                "smoothed_probabilities": [0.375, 0.5, 0.625],
            }
        )
    row = build_prediction_row(
        item,
        predictions,
        protocol_sha256="protocol-hash",
        jp_feature_sha256="feature-hash",
    )
    validate_prediction_row(
        row, expected_item=item, protocol_sha256="protocol-hash"
    )
    assert row["absolute_position_scores"] == [0.0, 1.0, 2.0]
    assert row["normalized_position_scores"] == [0.0, 0.5, 1.0]
    assert len(row["seed_predictions"]) == 3


def test_prediction_validator_rejects_seed_drift() -> None:
    item = {
        "selection_order": 0,
        "item_id": "JPHB-test",
        "query_id": "query-test",
        "output_token_ids": [10],
        "output_token_ids_sha256": "token-hash",
        "trajectory_sha256": "text-hash",
    }
    with pytest.raises(ValueError, match="seed order"):
        build_prediction_row(
            item,
            [
                {
                    "seed": 42,
                    "head_sha256": "head",
                    "raw_logits": [0.0],
                    "raw_probabilities": [0.5],
                    "smoothed_logits": [0.0],
                    "smoothed_probabilities": [0.5],
                }
            ],
            protocol_sha256="protocol-hash",
            jp_feature_sha256="feature-hash",
        )


def test_runner_has_no_secondary_label_filename_or_label_argument() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "labels_secondary" not in source
    assert "--labels" not in source
    assert "sealed labels are not traversed" in source
