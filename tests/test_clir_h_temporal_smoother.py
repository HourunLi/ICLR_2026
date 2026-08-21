import copy
import json
from pathlib import Path

import pytest
import torch

from src.clir_h_temporal_smoother import (
    build_smoothed_scored_row,
    masked_centered_mean_logits,
    smooth_persisted_probabilities,
    validate_smoother_protocol,
)
from src.clir_real_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/jp_h_temporal_smoother_v1/protocol_v1.json"
RESULT_PATH = ROOT / "configs/jp_h_temporal_smoother_v1/result_v1.json"


def load_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_temporal_smoother_protocol_freezes_one_zero_parameter_factor():
    protocol = load_protocol()
    validate_smoother_protocol(protocol)
    assert protocol["selection_status"] == "post_selection_exploratory"
    assert (
        protocol["selection_history"]["window_was_chosen_after_inspecting_same_64_rows"]
        is True
    )
    assert protocol["smoother"]["window_tokens"] == 3
    assert protocol["smoother"]["trainable_parameter_count"] == 0
    assert protocol["smoother"]["head_retraining"] is False
    assert protocol["protected_paths"]["smoother_enters_reward_score"] is False
    assert protocol["engineering_decision_rules"]["automatic_method_adoption"] is False
    assert protocol["engineering_decision_rules"]["automatic_score_coupling"] is False


def test_temporal_smoother_protocol_rejects_window_or_coupling_drift():
    protocol = load_protocol()
    changed = copy.deepcopy(protocol)
    changed["smoother"]["window_tokens"] = 5
    with pytest.raises(ValueError, match="window_tokens"):
        validate_smoother_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["protected_paths"]["smoother_enters_reward_score"] = True
    with pytest.raises(ValueError, match="smoother_enters_reward_score"):
        validate_smoother_protocol(changed)


def test_temporal_smoother_inputs_are_exact_and_bind_all_twelve_heads():
    protocol = load_protocol()
    for spec in protocol["inputs"].values():
        path = ROOT / spec["path"]
        assert path.is_file()
        assert file_sha256(path) == spec["sha256"]
    result = json.loads(
        (ROOT / protocol["inputs"]["linear_probe_result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert len(result["cell_reports"]) == 12
    assert (
        len(
            {
                report["probe_checkpoint_sha256"]
                for report in result["cell_reports"].values()
            }
        )
        == 12
    )


def test_masked_centered_mean_uses_available_real_neighbors_only():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    observed = masked_centered_mean_logits(logits, window_tokens=3)
    assert torch.equal(
        observed, torch.tensor([1.5, 2.0, 3.0, 3.5], dtype=torch.float64)
    )
    mask = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
    observed_masked = masked_centered_mean_logits(logits, mask, window_tokens=3)
    assert torch.equal(
        observed_masked,
        torch.tensor([1.5, 1.5, 0.0, 4.0], dtype=torch.float64),
    )


def test_logit_smoothing_is_fixed_and_preserves_constant_predictions():
    logits, probabilities = smooth_persisted_probabilities(
        [0.2, 0.2, 0.2, 0.2], window_tokens=3
    )
    assert len(logits) == len(probabilities) == 4
    assert all(value == pytest.approx(0.2, abs=1e-15) for value in probabilities)
    _, varied = smooth_persisted_probabilities([0.1, 0.9, 0.1], window_tokens=3)
    assert varied[0] == pytest.approx(0.5, abs=1e-15)
    assert varied[1] < 0.5
    assert varied[2] == pytest.approx(0.5, abs=1e-15)


def test_smoothed_row_changes_only_h_diagnostics_and_adds_provenance():
    raw = {
        "id": "row-1",
        "output_token_ids": [10, 11, 12],
        "correctness": 1,
        "reward_score": 1.25,
        "clir_score": 1.25,
        "clir_token_values": [-0.5, 0.25, 0.75],
        "clir_token_hallucination_probs": [0.1, 0.9, 0.1],
        "clir_path_no_hallucination_log_prob": -1.0,
        "clir_path_hallucination_prob": 0.5,
        "clir_pseudo_onset": 1,
        "frozen_h_probe_provenance": {"source": "linear"},
    }
    logits, probabilities = smooth_persisted_probabilities(
        raw["clir_token_hallucination_probs"]
    )
    candidate = build_smoothed_scored_row(
        raw,
        logits,
        probabilities,
        fold=2,
        seed=43,
        source_dev_scored_sha256="a" * 64,
        source_probe_checkpoint_sha256="b" * 64,
    )
    for field in (
        "id",
        "output_token_ids",
        "correctness",
        "reward_score",
        "clir_score",
        "clir_token_values",
        "frozen_h_probe_provenance",
    ):
        assert candidate[field] == raw[field]
    assert (
        candidate["clir_token_hallucination_probs"]
        != raw["clir_token_hallucination_probs"]
    )
    provenance = candidate["frozen_h_temporal_smoother_provenance"]
    assert provenance["trainable_parameter_count"] == 0
    assert provenance["smoother_enters_reward_score"] is False


def test_temporal_smoother_result_passes_engineering_gate_without_adoption():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["status"] == "completed_engineering_signal_supported"
    assert result["protocol"]["sha256"] == file_sha256(PROTOCOL_PATH)
    assert result["execution_gate"]["passed"] is True
    assert result["execution_gate"]["trainable_parameter_count"] == 0
    assert result["execution_gate"]["head_retrained"] is False
    assert result["execution_gate"]["protected_non_h_row_views_bit_identical"] == 192
    assert result["decision"]["engineering_signal_supported"] is True
    assert result["decision"]["passing_seeds"] == [42, 43, 44]
    assert result["decision"]["method_adopted"] is False
    assert result["decision"]["score_coupling_authorized"] is False
    for seed_result in result["primary_48_rows_by_seed"].values():
        assert seed_result["engineering_gate"]["all_four_passed"] is True
    assert result["pilot_test_accessed"] is False
    assert result["final_test_accessed"] is False
