import json
from pathlib import Path

import torch

from src.clir_data import read_jsonl
from src.clir_frozen_h_probe import (
    build_probe_scored_row,
    fit_linear_probe,
    score_linear_probe,
    validate_probe_protocol,
)
from src.clir_localization_evaluation import binary_average_precision
from src.clir_real_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/jp_h_frozen_probe_v1/training_protocol_v1.json"
RESULT_PATH = ROOT / "configs/jp_h_frozen_probe_v1/training_result_v1.json"


def load_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_frozen_probe_protocol_preserves_jp_and_only_trains_769_parameters():
    protocol = load_protocol()
    validate_probe_protocol(protocol)
    assert protocol["protected_original_method"]["original_method_changed"] is False
    assert protocol["protected_original_method"][
        "bidirectional_stop_gradient_mutual_distillation_weight"
    ] == 0.25
    assert protocol["protected_original_method"][
        "shared_gradient_reward_gate_to_detached_half_key_half_complete_prior_weight"
    ] == 10.0
    assert protocol["representation"]["base_parameters_in_optimizer"] is False
    assert protocol["representation"]["probe_enters_reward_score"] is False
    assert protocol["probe"]["parameter_count"] == 769
    assert protocol["probe"]["full_tail_target_used"] is False


def test_frozen_probe_inputs_and_fold_hashes_are_exact():
    protocol = load_protocol()
    for spec in protocol["inputs"].values():
        path = ROOT / spec["path"]
        assert path.is_file()
        assert file_sha256(path) == spec["sha256"]
    dev_ids = []
    for fold, fold_spec in protocol["cross_validation"]["folds"].items():
        for split in ("train", "dev"):
            spec = fold_spec[split]
            path = ROOT / spec["path"]
            assert file_sha256(path) == spec["sha256"]
            rows = read_jsonl(path)
            assert len(rows) == spec["rows"]
            supervised = sum(
                sum(int(value) for value in row["token_hallucination_mask"])
                for row in rows
            )
            positive = sum(
                sum(
                    int(target)
                    for target, known in zip(
                        row["token_hallucination_target"],
                        row["token_hallucination_mask"],
                    )
                    if known
                )
                for row in rows
            )
            assert supervised == spec["supervised_tokens"]
            assert positive == spec["positive_tokens"]
        dev_ids.extend(
            row["id"] for row in read_jsonl(ROOT / fold_spec["dev"]["path"])
        )
    assert len(dev_ids) == len(set(dev_ids)) == 64


def _position_baseline(rows):
    token_labels = []
    token_absolute_positions = []
    token_normalized_positions = []
    claim_labels = []
    claim_absolute_positions = []
    claim_normalized_positions = []
    for row in rows:
        length = len(row["output_token_ids"])
        for position, (target, known) in enumerate(
            zip(
                row["token_hallucination_target"],
                row["token_hallucination_mask"],
            )
        ):
            if known:
                token_labels.append(int(target))
                token_absolute_positions.append(float(position))
                token_normalized_positions.append(position / max(length - 1, 1))
        for span in row["hallucination_claim_spans"]:
            midpoint = (
                int(span["token_start"])
                + int(span["token_end_exclusive"])
                - 1
            ) / 2.0
            claim_labels.append(int(span["target"]))
            claim_absolute_positions.append(midpoint)
            claim_normalized_positions.append(midpoint / max(length - 1, 1))
    return {
        "rows": len(rows),
        "span_supervised_tokens": len(token_labels),
        "span_positive_tokens": sum(token_labels),
        "span_strongest_position_average_precision": max(
            binary_average_precision(token_labels, token_absolute_positions),
            binary_average_precision(token_labels, token_normalized_positions),
        ),
        "claims": len(claim_labels),
        "positive_claims": sum(claim_labels),
        "claim_strongest_position_average_precision": max(
            binary_average_precision(claim_labels, claim_absolute_positions),
            binary_average_precision(claim_labels, claim_normalized_positions),
        ),
    }


def test_frozen_probe_position_baselines_are_computed_on_the_declared_rows():
    protocol = load_protocol()
    folds = protocol["cross_validation"]["folds"]
    confirmatory_rows = [
        row
        for fold in (1, 2, 3)
        for row in read_jsonl(ROOT / folds[str(fold)]["dev"]["path"])
    ]
    all_rows = [
        row
        for fold in range(4)
        for row in read_jsonl(ROOT / folds[str(fold)]["dev"]["path"])
    ]
    assert _position_baseline(confirmatory_rows) == protocol["position_baselines"][
        "confirmatory_48"
    ]
    assert _position_baseline(all_rows) == protocol["position_baselines"][
        "all_oof_64"
    ]


def test_linear_probe_fits_only_a_detached_linear_head():
    generator = torch.Generator().manual_seed(7)
    negative = torch.randn(64, 4, generator=generator) - 1.0
    positive = torch.randn(64, 4, generator=generator) + 1.0
    features = torch.cat([negative, positive], dim=0)
    targets = torch.cat([torch.zeros(64), torch.ones(64)])
    features_before = features.clone()
    head, history = fit_linear_probe(
        features,
        targets,
        seed=42,
        epochs=80,
        learning_rate=0.01,
        weight_decay=0.0,
        max_grad_norm=1.0,
        device=torch.device("cpu"),
    )
    assert sum(parameter.numel() for parameter in head.parameters()) == 5
    assert torch.equal(features, features_before)
    assert history[-1]["post_update_train_bce"] < history[0][
        "pre_update_train_bce"
    ]
    _, probabilities = score_linear_probe(head, features)
    assert sum(probabilities[64:]) / 64 > sum(probabilities[:64]) / 64


def test_probe_scored_row_preserves_canonical_base_reward_fields():
    row = {
        "id": "row-1",
        "output_token_ids": [10, 11],
        "correctness": 1,
    }
    canonical = {
        "id": "row-1",
        "reward_score": 1.25,
        "clir_score": 1.25,
        "clir_token_values": [-0.5, 0.75],
    }
    scored = build_probe_scored_row(
        row,
        canonical,
        logits=[-1.0, 1.0],
        probabilities=[0.25, 0.75],
        fold=2,
        seed=43,
        probe_checkpoint_sha256="a" * 64,
    )
    assert scored["reward_score"] == canonical["reward_score"]
    assert scored["clir_score"] == canonical["clir_score"]
    assert scored["clir_token_values"] == canonical["clir_token_values"]
    assert scored["clir_token_hallucination_probs"] == [0.25, 0.75]
    assert scored["frozen_h_probe_provenance"]["probe_enters_reward_score"] is False
    assert 0.0 <= scored["clir_path_hallucination_prob"] <= 1.0


def test_frozen_probe_result_keeps_base_exact_and_rejects_plain_linear_head():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["status"] == "completed_frozen_linear_probe_not_supported"
    assert result["protocol"]["sha256"] == file_sha256(PROTOCOL_PATH)
    assert result["execution_gate"]["passed"] is True
    assert result["base_invariance"]["passed"] is True
    assert result["base_invariance"]["reward_score_max_absolute_difference"] == 0.0
    assert result["base_invariance"]["token_value_max_absolute_difference"] == 0.0
    assert result["decision"]["passed"] is False
    assert result["decision"]["passing_seed_count"] == 0
    for metrics in result["confirmatory_48_rows_by_seed"].values():
        assert metrics["gate"] == {
            "span_passed": False,
            "claim_passed": True,
            "both_localization_metrics_passed": False,
        }
    assert result["pilot_test_accessed"] is False
    assert result["final_test_accessed"] is False
