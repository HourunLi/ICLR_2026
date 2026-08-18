import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_dual_prior_predictions_v1 import prior_collaboration_metrics
from src.consistency_localized_reward import dual_prior_losses


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs/dual_prior_mutual_distillation_v1/training_protocol_v1.json"
)


def _distill_fixture() -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    key_logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    complete_logits = torch.tensor([[-1.0, 1.0, 0.0]], requires_grad=True)
    outputs = {
        "scores": torch.zeros(1),
        "mask": torch.ones(1, 3),
        "key_prior_logits": key_logits,
        "complete_prior_logits": complete_logits,
        "key_prior": torch.softmax(key_logits, dim=1),
        "complete_prior": torch.softmax(complete_logits, dim=1),
        "gates": torch.ones(1, 3),
        "fused_prior": torch.full((1, 3), 1.0 / 3.0),
        "complete_reconstruction": torch.zeros(1, 3),
    }
    batch = {
        "key_prior_target": torch.tensor([[1.0, 0.0, 0.0]]),
        "complete_prior_target": torch.tensor([[1.0, 1.0, 0.0]]),
    }
    return outputs, batch, key_logits, complete_logits


def test_original_mutual_distillation_formula_and_stop_gradient_are_preserved():
    outputs, batch, key_logits, complete_logits = _distill_fixture()
    one_direction = ((outputs["key_prior"] - outputs["complete_prior"]) ** 2).sum()
    joint = dual_prior_losses(
        outputs,
        batch,
        key_weight=0.0,
        complete_weight=0.0,
        distill_weight=1.0,
        gate_weight=0.0,
        reconstruction_weight=0.0,
        phase="joint",
    )
    assert joint["distill"].item() == pytest.approx(
        float(2.0 * one_direction.detach())
    )
    joint["total"].backward()
    assert key_logits.grad is not None and torch.count_nonzero(key_logits.grad)
    assert complete_logits.grad is not None and torch.count_nonzero(complete_logits.grad)

    outputs, batch, key_logits, complete_logits = _distill_fixture()
    key_phase = dual_prior_losses(
        outputs,
        batch,
        key_weight=0.0,
        complete_weight=0.0,
        distill_weight=1.0,
        gate_weight=0.0,
        reconstruction_weight=0.0,
        phase="key",
    )
    key_phase["total"].backward()
    assert key_logits.grad is not None and torch.count_nonzero(key_logits.grad)
    assert complete_logits.grad is None


def test_mutual_protocol_changes_only_original_distillation_weight():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_training"
    assert protocol["method"]["weight"] == 0.25
    assert protocol["method"]["formula_may_be_changed_after_results"] is False
    assert protocol["method"]["containment_replacement_allowed"] is False
    assert protocol["matched_training"]["prior_phase_mode"] == "joint"
    shared = protocol["matched_training"]["shared_loss_weights"]
    assert shared["gate_prior"] == 0.0
    assert shared["reconstruction"] == 0.0
    control = protocol["cells"][protocol["comparison"]["control_cell"]]
    mutual = protocol["cells"][protocol["comparison"]["mutual_cell"]]
    assert control["train"] == mutual["train"]
    assert control["dev"] == mutual["dev"]
    assert control["prior_weight"] == mutual["prior_weight"] == 1.0
    assert control["key_prior_weight"] == mutual["key_prior_weight"] == 1.0
    assert control["complete_prior_weight"] == mutual["complete_prior_weight"] == 1.0
    assert control["prior_distill_weight"] == 0.0
    assert mutual["prior_distill_weight"] == 0.25


def test_collaboration_metric_matches_two_direction_training_value():
    rows = [
        {
            "clir_key_prior": [0.75, 0.25],
            "clir_complete_prior": [0.25, 0.75],
        },
        {
            "clir_key_prior": [0.5, 0.5],
            "clir_complete_prior": [0.5, 0.5],
        },
    ]
    metrics = prior_collaboration_metrics(rows)
    # First row squared L2 is .5, second is zero; row mean is .25.
    assert metrics["one_direction_attention_mse"] == pytest.approx(0.25)
    assert metrics["symmetric_attention_mse"] == pytest.approx(0.5)
    assert metrics["mean_attention_overlap_mass"] == pytest.approx(0.75)
