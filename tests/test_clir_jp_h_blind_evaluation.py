import json
from pathlib import Path

import pytest

from src.clir_hallucination_annotation import file_sha256
from src.clir_jp_h_blind_evaluation import (
    derive_sparse_claim_targets,
    evaluate_rows,
    seed_gate,
    strongest_position,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/jp_h_blind_validation_v1/evaluation_protocol_v1.json"
RESULT = ROOT / "configs/jp_h_blind_validation_v1/evaluation_result_v1.json"


def label(*, item_id: str = "x") -> dict:
    return {
        "item_id": item_id,
        "query_id": f"q-{item_id}",
        "output_token_ids_sha256": f"hash-{item_id}",
        "path_status": "hallucinated",
        "hallucination_onset": 2,
        "claim_reviews": [
            {
                "claim_index": 0,
                "status": "supported",
                "token_start": 0,
                "token_end_exclusive": 2,
            },
            {
                "claim_index": 1,
                "status": "unsupported",
                "token_start": 2,
                "token_end_exclusive": 4,
            },
            {
                "claim_index": 2,
                "status": "uncertain",
                "token_start": 4,
                "token_end_exclusive": 5,
            },
        ],
    }


def prediction(*, item_id: str = "x") -> dict:
    raw = [0.4, 0.3, 0.6, 0.7, 0.2]
    smooth = [0.2, 0.3, 0.8, 0.9, 0.1]
    return {
        "item_id": item_id,
        "query_id": f"q-{item_id}",
        "output_token_ids_sha256": f"hash-{item_id}",
        "output_token_ids": [1, 2, 3, 4, 5],
        "token_count": 5,
        "absolute_position_scores": [0.0, 1.0, 2.0, 3.0, 4.0],
        "normalized_position_scores": [0.0, 0.25, 0.5, 0.75, 1.0],
        "seed_predictions": [
            {
                "seed": seed,
                "raw_probabilities": raw,
                "smoothed_probabilities": smooth,
            }
            for seed in (42, 43, 44)
        ],
    }


def test_protocol_binds_inputs_before_predictions_are_opened() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"].endswith("before_prediction_value_access")
    assert protocol["prediction_values_accessed_when_protocol_frozen"] is False
    for spec in protocol["inputs"].values():
        assert file_sha256(ROOT / spec["path"]) == spec["sha256"]
    if RESULT.exists():
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        assert result["evaluation_protocol_sha256"] == file_sha256(PROTOCOL)
        assert result["prediction_values_opened_after_protocol_commit"] is True
        assert result["threshold_tuning_performed"] is False


def test_sparse_mapping_masks_unreviewed_and_uncertain_tokens() -> None:
    mapped = derive_sparse_claim_targets(label(), token_count=6)
    assert mapped["positions"] == [0, 1, 2, 3]
    assert mapped["targets"] == [0, 0, 1, 1]
    assert mapped["positive_claims"] == 1
    assert mapped["negative_claims"] == 1


def test_sparse_mapping_rejects_conflicting_overlap() -> None:
    row = label()
    row["claim_reviews"][1]["token_start"] = 1
    row["hallucination_onset"] = 1
    with pytest.raises(ValueError, match="conflicting"):
        derive_sparse_claim_targets(row, token_count=5)


def test_position_tie_prefers_absolute_deterministically() -> None:
    assert strongest_position(0.4, 0.4) == ("absolute", 0.4)


def test_seed_gate_requires_all_four_strict_improvements() -> None:
    passed = seed_gate(
        raw_token_ap=0.4,
        smoothed_token_ap=0.5,
        position_token_ap=0.3,
        raw_claim_ap=0.45,
        smoothed_claim_ap=0.55,
        position_claim_ap=0.35,
    )
    assert passed["passed"] is True
    tied = seed_gate(
        raw_token_ap=0.5,
        smoothed_token_ap=0.5,
        position_token_ap=0.3,
        raw_claim_ap=0.45,
        smoothed_claim_ap=0.55,
        position_claim_ap=0.35,
    )
    assert tied["passed"] is False


def test_toy_evaluation_has_both_units_and_three_paired_seeds() -> None:
    result = evaluate_rows([label()], [prediction()], seeds=[42, 43, 44])
    assert result["rows"] == 1
    assert result["token_samples"] == 4
    assert result["claim_samples"] == 2
    assert set(result["seeds"]) == {"42", "43", "44"}
    assert all(
        seed["token_average_precision"]["smoothed"] is not None
        for seed in result["seeds"].values()
    )


def test_persisted_blind_result_is_not_overridden() -> None:
    if not RESULT.exists():
        pytest.skip("Blind evaluation has not been executed yet")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "blind_gate_failed_smoother_rejected"
    assert result["overall_adoption_gate_passed"] is False
    assert result["view_passes"] == {
        "resolved_primary": False,
        "resolved_secondary": False,
    }
    assert all(
        view["passing_seeds"] == 0
        for view in result["pooled_primary_results"].values()
    )
    assert result["automatic_score_coupling"] is False
