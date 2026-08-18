import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.evaluate_dual_prior_predictions_v1 import (
    best_f1_threshold,
    binary_f1,
    safe_ap,
    safe_auc,
)
from src.clir_data import write_jsonl
from src.clir_hallucination_annotation import file_sha256, read_jsonl
from src.consistency_localized_reward import RewardConfig, build_reward_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dual_prior_evidence_v1"
TRAINING_PROTOCOL = CONFIG / "training_protocol_v1.json"


def test_frozen_training_protocol_is_standalone_matched_and_test_blind():
    protocol = json.loads(TRAINING_PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_training"
    assert protocol["matched_training"]["seeds"] == [42, 43, 44]
    assert protocol["matched_training"]["checkpoint_selection"] == (
        "fixed final epoch 5"
    )
    shared = protocol["matched_training"]["shared_loss_weights"]
    assert shared["final"] == 1.0
    assert all(
        shared[name] == 0.0
        for name in (
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
            "prior_distill",
            "gate_prior",
            "reconstruction",
        )
    )
    cells = protocol["cells"]
    assert [
        (
            name,
            cell["prior_weight"],
            cell["key_prior_weight"],
            cell["complete_prior_weight"],
        )
        for name, cell in cells.items()
    ] == [
        ("d0_correctness_only", 0.0, 0.0, 0.0),
        ("d1_correctness_plus_key", 1.0, 1.0, 0.0),
        ("d2_correctness_plus_complete", 1.0, 0.0, 1.0),
        ("d3_correctness_plus_key_and_complete", 1.0, 1.0, 1.0),
    ]
    assert protocol["evaluation"]["position_baseline"].startswith(
        "Use normalized token/unit position"
    )
    assert protocol["evaluation"]["pilot_test_access_allowed"] is False
    assert protocol["evaluation"]["final_test_access_allowed"] is False
    assert protocol["formal_mechanism_claim_allowed"] is False


def test_materialized_D0_D3_rows_are_exactly_matched_except_direct_targets():
    protocol = json.loads(TRAINING_PROTOCOL.read_text(encoding="utf-8"))
    forbidden = {
        "path_hallucinated",
        "hallucination_onset",
        "token_hallucination_target",
        "token_hallucination_mask",
        "token_advantage",
        "progress_targets",
        "semantic_id",
        "style_id",
        "domain_id",
        "complete_reconstruction_target",
    }
    target_fields = {"key_prior_target", "complete_prior_target"}
    for split in ("train", "dev"):
        cell_rows = {
            name: read_jsonl(ROOT / cell[split]["path"])
            for name, cell in protocol["cells"].items()
        }
        for name, cell in protocol["cells"].items():
            path = ROOT / cell[split]["path"]
            assert file_sha256(path) == cell[split]["sha256"]
            assert not any(forbidden.intersection(row) for row in cell_rows[name])
        reference = cell_rows["d0_correctness_only"]
        assert all(
            [row["id"] for row in rows] == [row["id"] for row in reference]
            for rows in cell_rows.values()
        )
        for row_index, base in enumerate(reference):
            assert not target_fields.intersection(base)
            for name, rows in cell_rows.items():
                observed = dict(rows[row_index])
                observed.pop("clir_supervision_provenance", None)
                observed_targets = target_fields.intersection(observed)
                expected_targets = {
                    "d0_correctness_only": set(),
                    "d1_correctness_plus_key": {"key_prior_target"},
                    "d2_correctness_plus_complete": {"complete_prior_target"},
                    "d3_correctness_plus_key_and_complete": target_fields,
                }[name]
                assert observed_targets == expected_targets
                for field in observed_targets:
                    target = observed.pop(field)
                    assert len(target) == len(base["output_token_ids"])
                    assert set(target).issubset({0, 1})
                assert observed == base


def test_numpy_binary_metrics_follow_tie_aware_ranking_definitions():
    labels = np.asarray([1, 0], dtype=np.int64)
    tied = np.asarray([0.5, 0.5], dtype=np.float64)
    assert safe_ap(labels, tied) == pytest.approx(0.5)
    assert safe_auc(labels, tied) == pytest.approx(0.5)
    assert binary_f1(labels, tied >= 0.5) == pytest.approx(2 / 3)

    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.9, 0.8, 0.7], dtype=np.float64)
    threshold, f1 = best_f1_threshold(labels, scores)
    assert threshold == pytest.approx(0.7)
    assert f1 == pytest.approx(0.8)
    assert safe_ap(labels, scores) == pytest.approx(5 / 6)
    assert safe_auc(labels, scores) == pytest.approx(0.75)


def test_score_clir_publishes_binary_prior_membership_separately(tmp_path: Path):
    manifest = tmp_path / "rows.jsonl"
    checkpoint_path = tmp_path / "model.pt"
    output = tmp_path / "scores.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "q0-c0",
                "query_id": "q0",
                "hidden_states": [[0.0, 0.0], [0.0, 0.0]],
                "prompt_token_ids": [1],
                "output_token_ids": [10, 11],
                "response": "toy",
                "correctness": 1,
                "provenance": {
                    "model_id": "toy",
                    "model_revision": "test",
                    "tokenizer_revision": "test",
                    "protocol_version": "test",
                },
            }
        ],
    )
    config = RewardConfig(
        hidden_dim=2,
        model_dim=2,
        projection_dim=2,
        model_variant="clir",
        encoder_type="identity",
    )
    model = build_reward_model(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(
        {
            "schema_version": "clir-full-checkpoint-v2",
            "config": config.__dict__,
            "state_dict": model.state_dict(),
            "experiment_protocol": None,
        },
        checkpoint_path,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "score_clir.py"),
            "--input_jsonl",
            str(manifest),
            "--model",
            str(checkpoint_path),
            "--output_jsonl",
            str(output),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    row = read_jsonl(output)[0]
    assert row["clir_key_prior"] == pytest.approx([0.5, 0.5])
    assert row["clir_complete_prior"] == pytest.approx([0.5, 0.5])
    assert row["clir_key_prior_membership_probs"] == pytest.approx([0.5, 0.5])
    assert row["clir_complete_prior_membership_probs"] == pytest.approx([0.5, 0.5])


def test_frozen_three_seed_direct_target_result_passes_without_test_access():
    result = json.loads(
        (CONFIG / "training_result_v1.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "completed_pass_direct_targets_learnable"
    assert result["training_commit"] == (
        "f485e54db0181fbcce677b8ae3a0fa895e4e8f82"
    )
    assert result["required_matrix_cells"] == 12
    assert result["completed_matrix_cells"] == 12
    assert len(result["cell_result_hashes"]) == 12
    assert result["selection_passed"] is True
    assert result["pilot_test_accessed"] is False
    assert result["formal_mechanism_claim_allowed"] is False
    assert result["position_only_baseline_included"] is True
    assert set(result["passing_seed_counts"].values()) == {3}
    assert all(result["across_seed_checks"].values())

    mean = result["mean_metrics_and_deltas"]
    assert mean["d1_key_unit_ap"] > mean["d0_key_unit_ap"]
    assert mean["d1_key_unit_ap"] > mean["position_key_unit_ap"]
    assert mean["d2_complete_unit_ap"] > mean["d0_complete_unit_ap"]
    assert mean["d2_complete_unit_ap"] > mean["position_complete_unit_ap"]
    assert mean["d3_key_unit_ap"] > mean["d1_key_unit_ap"]
    assert mean["d3_complete_vs_d2"] > -0.05
    assert mean["d3_mean_absolute_key_complete_probability_difference"] > 0.02
    assert mean["d3_key_complete_probability_correlation"] < 0.98
