import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_dual_prior_gate_predictions_v1 import gate_integration_metrics
from src.clir_hallucination_annotation import file_sha256
from src.consistency_localized_reward import dual_prior_losses


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/dual_prior_reward_gate_v1/training_protocol_v1.json"
AUDIT = ROOT / "configs/dual_prior_reward_gate_v1/loss_scale_audit_v1.json"
RESULT = ROOT / "configs/dual_prior_reward_gate_v1/training_result_v1.json"


def test_gate_alignment_detaches_fused_prior_but_updates_gate():
    gate_logits = torch.tensor([[0.5, -0.5, 0.0]], requires_grad=True)
    key_logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    complete_logits = torch.tensor([[-1.0, 1.0, 0.0]], requires_grad=True)
    key = torch.softmax(key_logits, dim=1)
    complete = torch.softmax(complete_logits, dim=1)
    outputs = {
        "scores": torch.zeros(1),
        "mask": torch.ones(1, 3),
        "key_prior_logits": key_logits,
        "complete_prior_logits": complete_logits,
        "key_prior": key,
        "complete_prior": complete,
        "gates": torch.sigmoid(gate_logits),
        "fused_prior": 0.5 * key + 0.5 * complete,
        "complete_reconstruction": torch.zeros(1, 3),
    }
    batch = {
        "key_prior_target": torch.tensor([[1.0, 0.0, 0.0]]),
        "complete_prior_target": torch.tensor([[1.0, 1.0, 0.0]]),
    }
    losses = dual_prior_losses(
        outputs,
        batch,
        key_weight=0.0,
        complete_weight=0.0,
        distill_weight=0.0,
        gate_weight=1.0,
        reconstruction_weight=0.0,
        phase="joint",
    )
    losses["total"].backward()
    assert gate_logits.grad is not None and torch.count_nonzero(gate_logits.grad)
    # Zero-weight direct/mutual terms keep these tensors in the graph, so
    # PyTorch materializes exact-zero gradients rather than leaving ``None``.
    assert key_logits.grad is not None and not torch.count_nonzero(key_logits.grad)
    assert complete_logits.grad is not None and not torch.count_nonzero(
        complete_logits.grad
    )


def test_gate_scale_audit_freezes_weight_before_gate_training():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "completed_weight_frozen_before_gate_training"
    assert audit["frozen_gate_prior_weight"] == 10.0
    assert audit["new_gate_enabled_training_or_predictions_accessed"] is False
    weak = audit["candidate_weight_diagnostics"]["0.25"][
        "gate_head_gradient_ratio_vs_final"
    ]
    chosen = audit["candidate_weight_diagnostics"]["10.0"][
        "gate_head_gradient_ratio_vs_final"
    ]
    assert weak["median"] < 0.01
    assert 0.15 <= chosen["median"] <= 0.5
    assert chosen["maximum"] < 1.0


def test_reward_gate_protocol_changes_only_gate_alignment_weight():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_training"
    assert protocol["protected_mutual_method"]["weight"] == 0.25
    assert protocol["protected_mutual_method"]["formula_changed"] is False
    assert protocol["protected_mutual_method"]["weight_changed"] is False
    assert protocol["method"]["weight"] == 10.0
    assert protocol["method"]["weight_frozen_before_gate_training"] is True
    assert protocol["method"]["weight_may_be_changed_after_results"] is False
    assert (
        file_sha256(AUDIT)
        == protocol["inputs"]["gate_loss_scale_audit"]["sha256"]
    )
    shared = protocol["matched_training"]["shared_loss_weights"]
    assert shared["prior_distill"] == 0.25
    assert shared["reconstruction"] == 0.0
    control = protocol["cells"][protocol["comparison"]["control_cell"]]
    gate = protocol["cells"][protocol["comparison"]["gate_cell"]]
    for key in (
        "train",
        "dev",
        "prior_weight",
        "key_prior_weight",
        "complete_prior_weight",
        "prior_distill_weight",
    ):
        assert control[key] == gate[key]
    assert control["gate_prior_weight"] == 0.0
    assert gate["gate_prior_weight"] == 10.0
    assert protocol["evaluation"]["ranking_or_best_of_n_claim_available"] is False


def test_gate_integration_metric_matches_training_objective():
    rows = [
        {
            "id": "aligned",
            "clir_gate_attention": [0.75, 0.25],
            "clir_key_prior": [1.0, 0.0],
            "clir_complete_prior": [0.5, 0.5],
            "clir_prior_gate_alignment": 0.625,
            "clir_mean_gate": 0.5,
            "clir_score": 1.5,
            "clir_token_values": [2.0, 0.0],
        },
        {
            "id": "uniform-gate",
            "clir_gate_attention": [0.5, 0.5],
            "clir_key_prior": [1.0, 0.0],
            "clir_complete_prior": [0.5, 0.5],
            "clir_prior_gate_alignment": 0.5,
            "clir_mean_gate": 0.4,
            "clir_score": 0.5,
            "clir_token_values": [1.0, 0.0],
        },
    ]
    metrics = gate_integration_metrics(rows, fusion_alpha=0.5)
    # The second row differs by [-.25, +.25], so its squared L2 is .125;
    # the first row is exact, and the evaluator averages trajectories.
    assert metrics["training_objective_mse"] == pytest.approx(0.0625)
    assert metrics["mean_overlap_mass"] == pytest.approx(0.875)
    assert metrics["raw_mean_gate"]["mean"] == pytest.approx(0.45)
    assert metrics["token_score"]["mean"] == pytest.approx(1.0)


def test_completed_reward_gate_result_is_retained_as_diagnostic_only():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "completed_reward_gate_integration_diagnostic_only"
    assert result["required_matrix_cells"] == result["completed_matrix_cells"] == 6
    assert result["original_mutual_formula_preserved"] is True
    assert result["mutual_distillation_weight"] == 0.25
    assert result["gate_alignment_weight"] == 10.0
    assert result["gate_fused_prior_detached"] is True
    assert result["selection_passed"] is False
    assert result["passing_seed_counts"]["gate_alignment_improves"] == 3
    assert result["passing_seed_counts"]["key_localization_protected"] == 1
    assert result["passing_seed_counts"]["all"] == 0
    mean = result["mean_metrics_and_deltas"]
    assert mean["dev_gate_objective_mse_relative_reduction"] > 0.7
    assert mean["gate_key_unit_ap_vs_control"] < -0.05
    assert result["ranking_or_best_of_n_evaluated"] is False
    assert result["reconstruction_enabled"] is False
    assert result["containment_replacement_used"] is False
    assert result["pilot_test_accessed"] is False
