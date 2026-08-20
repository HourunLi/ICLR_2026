from pathlib import Path
import json
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from src.clir_data import (
    CLIRTrajectoryDataset,
    SemanticGroupBatchSampler,
    clir_collate,
    load_batch_packing_pools,
    write_jsonl,
)
from src.consistency_localized_reward import (
    ConsistencyLocalizedReward,
    RewardConfig,
    build_reward_model,
    dual_prior_losses,
    hallucination_localization_losses,
    prism_style_consistency_loss,
    path_hallucination_probability,
    path_level_hallucination_mil,
    path_no_hallucination_log_probability,
    relative_tail_margin_loss,
)
from train_clir import _component_counts, make_config, parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_clir_forward_and_loss():
    config = RewardConfig(hidden_dim=8, projection_dim=4)
    model = ConsistencyLocalizedReward(config)
    batch = {
        "hidden_states": torch.randn(4, 6, 8),
        "condition_states": torch.randn(4, 3, 8),
        "condition_mask": torch.ones(4, 3),
        "mask": torch.ones(4, 6),
        "correctness": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "semantic_ids": torch.tensor([1, 1, 2, 2]),
        "style_ids": torch.tensor([1, 2, 1, 2]),
        "hallucination_onset": torch.tensor([-1, 3, -1, 2]),
        "onset_label_mask": torch.tensor([True, True, True, True]),
        "path_hallucinated": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "path_label_mask": torch.tensor([True, True, True, True]),
        "token_advantage": torch.zeros(4, 6),
        "token_advantage_mask": torch.ones(4, 6, dtype=torch.bool),
        "progress_targets": torch.zeros(4, 6),
        "progress_mask": torch.ones(4, 6, dtype=torch.bool),
        "key_prior_target": torch.zeros(4, 6),
        "complete_prior_target": torch.ones(4, 6),
    }

    outputs, losses = model.training_step(batch)
    assert outputs["scores"].shape == (4,)
    assert outputs["token_rewards"].shape == (4, 6)
    assert outputs["key_prior"].shape == (4, 6)
    assert outputs["condition_relevance"].shape == (4, 6)
    assert losses["total"].ndim == 0
    losses["total"].backward()


def test_negative_consistency_weight_preserves_default_and_can_disable_repulsion():
    representations = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    )
    scores = torch.zeros(4)
    semantic = torch.tensor([1, 1, 2, 2])
    styles = torch.tensor([1, 2, 1, 2])
    default = prism_style_consistency_loss(
        representations, scores, semantic, styles,
        margin=0.2, negative_weight=1.0, score_weight=0.1,
    )
    disabled = prism_style_consistency_loss(
        representations, scores, semantic, styles,
        margin=0.2, negative_weight=0.0, score_weight=0.1,
    )
    assert default["negative"].item() == pytest.approx(0.8)
    assert default["total"].item() == pytest.approx(0.8)
    assert disabled["negative"].item() == pytest.approx(0.8)
    assert disabled["total"].item() == pytest.approx(0.0)


def test_toy_generator_relative_paths_load_without_double_prefix(tmp_path: Path):
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "create_toy_clir_data.py"),
            "--output_jsonl",
            "examples/toy_clir.jsonl",
            "--feature_dir",
            "examples/features",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = tmp_path / "examples" / "toy_clir.jsonl"
    dataset = CLIRTrajectoryDataset(manifest)

    assert len(dataset) == 6
    assert dataset[0]["hidden_states"].shape == (5, 8)


def test_score_health_failure_persists_sidecar_without_scored_manifest(tmp_path: Path):
    manifest = tmp_path / "rows.jsonl"
    checkpoint_path = tmp_path / "model.pt"
    output = tmp_path / "scores.jsonl"
    write_jsonl(manifest, [
        {
            "id": "q0-c0",
            "query_id": "q0",
            "hidden_states": [[0.0, 0.0]],
            "correctness": 0,
        },
        {
            "id": "q0-c1",
            "query_id": "q0",
            "hidden_states": [[0.0, 0.0]],
            "correctness": 1,
        },
    ])
    config = RewardConfig(
        hidden_dim=2,
        model_variant="strict_swift",
        encoder_type="identity",
    )
    model = build_reward_model(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save({
        "schema_version": "clir-full-checkpoint-v2",
        "config": config.__dict__,
        "state_dict": model.state_dict(),
        "experiment_protocol": None,
    }, checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "score_clir.py",
            "--input_jsonl", str(manifest),
            "--model", str(checkpoint_path),
            "--output_jsonl", str(output),
            "--batch_size", "2",
            "--device", "cpu",
            "--min_score_std", "0.1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    health_path = output.with_name(f"{output.name}.health.json")
    health = json.loads(health_path.read_text(encoding="utf-8"))
    assert result.returncode != 0
    assert not output.exists()
    assert health["status"] == "health_gate_failed"
    assert health["health_gate"]["passed"] is False
    assert health["scoring_provenance"]["model_variant"] == "strict_swift"
    assert health["scoring_provenance"]["min_score_std"] == 0.1
    assert health["scoring_provenance"]["score_distribution"] == health[
        "health_gate"
    ]["observed_distribution"]


def test_condition_changes_prior_distribution():
    config = RewardConfig(hidden_dim=8, projection_dim=4)
    model = ConsistencyLocalizedReward(config)
    hidden_states = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5)
    condition_a = torch.randn(2, 3, 8)
    condition_b = condition_a.clone()
    condition_b[:, 0] = condition_b[:, 0] + 5.0

    out_a = model(hidden_states, mask=mask, condition_states=condition_a)
    out_b = model(hidden_states, mask=mask, condition_states=condition_b)

    assert not torch.allclose(out_a["key_prior"], out_b["key_prior"])


def test_prior_loss_is_zero_without_external_targets():
    config = RewardConfig(hidden_dim=8, projection_dim=4)
    model = ConsistencyLocalizedReward(config)
    outputs, losses = model.training_step(
        {
            "hidden_states": torch.randn(2, 4, 8),
            "mask": torch.ones(2, 4),
            "correctness": torch.tensor([1.0, 0.0]),
        }
    )

    assert outputs["key_prior"].shape == (2, 4)
    assert losses["prior_total"].item() == 0.0


def test_dual_prior_partial_mask_preserves_full_attention_mass():
    scores = torch.zeros(1)
    mask = torch.ones(1, 4)
    outputs = {
        "scores": scores,
        "mask": mask,
        "key_prior_logits": torch.zeros(1, 4),
        "complete_prior_logits": torch.zeros(1, 4),
        "key_prior": torch.tensor([[0.7, 0.1, 0.1, 0.1]]),
        "complete_prior": torch.tensor([[0.2, 0.2, 0.3, 0.3]]),
        "gates": torch.tensor([[0.7, 0.1, 0.1, 0.1]]),
        "fused_prior": torch.tensor([[0.2, 0.2, 0.3, 0.3]]),
        "complete_reconstruction": torch.zeros(1, 4),
    }
    batch = {
        "key_prior_target": torch.zeros(1, 4),
        "complete_prior_target": torch.zeros(1, 4),
        "key_prior_mask": torch.tensor([[True, True, False, False]]),
        "complete_prior_mask": torch.tensor([[True, True, False, False]]),
    }

    losses = dual_prior_losses(
        outputs,
        batch,
        key_weight=0.0,
        complete_weight=0.0,
        distill_weight=1.0,
        gate_weight=1.0,
        reconstruction_weight=0.0,
        phase="key",
    )

    expected = torch.tensor((0.7 - 0.2) ** 2 + (0.1 - 0.2) ** 2)
    assert torch.isclose(losses["distill"], expected)
    assert torch.isclose(losses["gate"], expected)


def test_invalid_hallucination_onset_raises():
    config = RewardConfig(hidden_dim=8, projection_dim=4)
    model = ConsistencyLocalizedReward(config)
    batch = {
        "hidden_states": torch.randn(1, 3, 8),
        "mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "hallucination_onset": torch.tensor([2]),
        "onset_label_mask": torch.tensor([True]),
    }

    try:
        model.training_step(batch)
    except ValueError as exc:
        assert "hallucination_onset" in str(exc)
    else:
        raise AssertionError("Expected invalid hallucination onset to raise ValueError")


def test_explicit_sparse_hallucination_target_overrides_onset_tail_and_weights_positives():
    common = {
        "hallucination_logits": torch.zeros(1, 4),
        "token_rewards": torch.zeros(1, 4),
        "mask": torch.ones(1, 4),
        "onset": torch.tensor([2]),
        "onset_label_mask": torch.tensor([True]),
        "token_advantage": None,
        "token_advantage_mask": None,
        "negative_tail_margin": 0.5,
    }
    explicit = hallucination_localization_losses(
        **common,
        token_hallucination_target=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        token_hallucination_mask=torch.tensor([[False, True, False, False]]),
        target_mode="explicit",
        positive_weight=2.0,
    )
    tail = hallucination_localization_losses(
        **common,
        token_hallucination_target=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        token_hallucination_mask=torch.tensor([[False, True, False, False]]),
        target_mode="onset_tail",
        positive_weight=1.0,
    )

    assert explicit["token_bce"].item() == pytest.approx(2.0 * torch.log(torch.tensor(2.0)).item())
    assert tail["token_bce"].item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_relative_tail_margin_is_shift_invariant_and_anchors_every_tail_token():
    values = torch.tensor([[2.0, 1.0, 2.0, 1.0]], requires_grad=True)
    mask = torch.ones_like(values)
    onset = torch.tensor([2])
    loss = relative_tail_margin_loss(values, mask, onset, None, margin=0.5)
    shifted = relative_tail_margin_loss(values - 17.0, mask, onset, None, margin=0.5)

    assert loss.item() == pytest.approx(0.5)
    assert shifted.item() == pytest.approx(loss.item())
    loss.backward()
    assert torch.all(values.grad[0, :2] < 0)
    assert values.grad[0, 2] > 0
    assert values.grad[0, 3] == 0
    assert values.grad.sum().item() == pytest.approx(0.0, abs=1e-7)


def test_relative_tail_margin_excludes_onset_zero_and_averages_rows_equally():
    values = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [9.0, 9.0, 9.0, 9.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    mask = torch.ones_like(values)
    onset = torch.tensor([2, 0, -1])
    loss = relative_tail_margin_loss(values, mask, onset, None, margin=0.5)

    assert loss.item() == pytest.approx(2.25)
    loss.backward()
    assert torch.all(values.grad[1:] == 0)


def test_zero_relative_tail_weight_preserves_old_loss_dictionary_and_total():
    shared = {
        "hidden_dim": 8,
        "projection_dim": 4,
        "tail_weight": 0.0,
        "pseudo_tail_weight": 0.0,
        "mil_weight": 0.0,
        "token_reward_weight": 0.0,
        "progress_weight": 0.0,
        "prior_weight": 0.0,
        "consistency_weight": 0.0,
        "hallucination_target_mode": "explicit",
    }
    control = ConsistencyLocalizedReward(
        RewardConfig(**shared, relative_tail_weight=0.0)
    )
    candidate = ConsistencyLocalizedReward(
        RewardConfig(**shared, relative_tail_weight=0.5)
    )
    candidate.load_state_dict(control.state_dict())
    batch = {
        "hidden_states": torch.randn(2, 4, 8),
        "mask": torch.ones(2, 4),
        "correctness": torch.tensor([1.0, 0.0]),
        "hallucination_onset": torch.tensor([2, -1]),
        "onset_label_mask": torch.tensor([True, True]),
        "token_hallucination_target": torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        "token_hallucination_mask": torch.ones(2, 4, dtype=torch.bool),
    }

    _, control_losses = control.training_step(batch)
    _, candidate_losses = candidate.training_step(batch)

    assert "localization_relative_tail_margin" not in control_losses
    assert "localization_relative_tail_margin" in candidate_losses
    for key, value in control_losses.items():
        if key != "total":
            assert candidate_losses[key].item() == pytest.approx(value.item())
    expected = (
        control_losses["total"]
        + 0.5 * candidate_losses["localization_relative_tail_margin"]
    )
    assert candidate_losses["total"].item() == pytest.approx(expected.item())


def test_localization_applicable_count_follows_the_frozen_target_mode():
    batch = {
        "hidden_states": torch.zeros(1, 4, 2),
        "mask": torch.ones(1, 4),
        "hallucination_onset": torch.tensor([2]),
        "onset_label_mask": torch.tensor([True]),
        "token_hallucination_target": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        "token_hallucination_mask": torch.tensor([[True, True, False, False]]),
    }
    losses = {"localization_token_bce": torch.tensor(0.0)}

    assert _component_counts(
        batch,
        losses,
        hallucination_target_mode="onset_tail",
    )["localization_token_bce"] == 4
    assert _component_counts(
        batch,
        losses,
        hallucination_target_mode="explicit",
    )["localization_token_bce"] == 2

    relative_losses = {
        **losses,
        "localization_relative_tail_margin": torch.tensor(0.0),
    }
    assert _component_counts(
        batch,
        relative_losses,
        hallucination_target_mode="explicit",
    )["localization_relative_tail_margin"] == 2


def test_sparse_hallucination_targets_collate_with_their_explicit_mask(tmp_path: Path):
    hidden = tmp_path / "hidden.pt"
    torch.save(torch.zeros(4, 8), hidden)
    manifest = tmp_path / "span.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "sample",
                "query_id": "query",
                "hidden_states_path": str(hidden),
                "token_hallucination_target": [0, 1, 0, 0],
                "token_hallucination_mask": [1, 1, 0, 0],
            }
        ],
    )
    batch = next(
        iter(
            DataLoader(
                CLIRTrajectoryDataset(manifest),
                batch_size=1,
                collate_fn=clir_collate,
            )
        )
    )

    assert batch["token_hallucination_target"].tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert batch["token_hallucination_mask"].tolist() == [[True, True, False, False]]


def test_jsonl_dataset_collate(tmp_path: Path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    rows = []
    for idx in range(2):
        hidden_path = feature_dir / f"sample{idx}.pt"
        torch.save(torch.randn(3 + idx, 8), hidden_path)
        rows.append(
            {
                "id": f"sample{idx}",
                "query_id": "q0",
                "hidden_states_path": str(hidden_path),
                "correctness": idx == 0,
                "semantic_id": "q0",
                "style_id": f"style{idx}",
                "path_hallucinated": idx == 1,
                "key_prior_target": [1] + [0] * (2 + idx),
            }
        )

    jsonl_path = tmp_path / "data.jsonl"
    write_jsonl(jsonl_path, rows)
    dataset = CLIRTrajectoryDataset(jsonl_path)
    loader = DataLoader(dataset, batch_size=2, collate_fn=clir_collate)
    batch = next(iter(loader))

    assert batch["hidden_states"].shape == (2, 4, 8)
    assert batch["mask"].sum().item() == 7
    assert batch["correctness"].tolist() == [1.0, 0.0]
    assert batch["correctness_mask"].tolist() == [True, True]
    assert batch["consistency_mask"].tolist() == [True, True]


def test_dataset_can_skip_unused_condition_payloads(tmp_path: Path):
    manifest = tmp_path / "data.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "sample",
                "query_id": "query",
                "hidden_states": [[0.0, 1.0]],
                "condition_states_path": "does-not-exist.pt",
                "correctness": 1,
            }
        ],
    )

    item = CLIRTrajectoryDataset(manifest, load_condition=False)[0]

    assert "condition_states" not in item


def test_semantic_group_batch_sampler(tmp_path: Path):
    rows = []
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for semantic_id in ["a", "b"]:
        for style_id in ["direct", "verbose"]:
            path = feature_dir / f"{semantic_id}_{style_id}.pt"
            torch.save(torch.randn(2, 8), path)
            rows.append(
                {
                    "id": f"{semantic_id}-{style_id}",
                    "query_id": semantic_id,
                    "hidden_states_path": str(path),
                    "semantic_id": semantic_id,
                    "style_id": style_id,
                }
            )

    jsonl_path = tmp_path / "grouped.jsonl"
    write_jsonl(jsonl_path, rows)
    dataset = CLIRTrajectoryDataset(jsonl_path)
    sampler = SemanticGroupBatchSampler(dataset, batch_size=2, shuffle=False)
    batches = list(iter(sampler))

    assert batches == [[0, 1], [2, 3]]

    sampler = SemanticGroupBatchSampler(dataset, batch_size=4, shuffle=False)
    batches = list(iter(sampler))

    assert batches == [[0, 1, 2, 3]]


def test_semantic_group_batch_sampler_packs_groups(tmp_path: Path):
    rows = []
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for idx, semantic_id in enumerate(["a", "a", "a", "b", "b"]):
        path = feature_dir / f"sample_{idx}.pt"
        torch.save(torch.randn(2, 8), path)
        rows.append(
            {
                "id": f"sample-{idx}",
                "query_id": semantic_id,
                "hidden_states_path": str(path),
                "semantic_id": semantic_id,
                "style_id": f"style-{idx}",
            }
        )

    jsonl_path = tmp_path / "packed.jsonl"
    write_jsonl(jsonl_path, rows)
    dataset = CLIRTrajectoryDataset(jsonl_path)
    sampler = SemanticGroupBatchSampler(dataset, batch_size=4, shuffle=False)
    batches = list(iter(sampler))

    assert batches[0] == [0, 1, 3, 4]
    assert batches[1] == [2]


def test_semantic_group_batch_sampler_len_matches_iter_for_uneven_groups(tmp_path: Path):
    rows = []
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for group_idx in range(5):
        for sample_idx in range(2):
            row_idx = len(rows)
            path = feature_dir / f"sample_{row_idx}.pt"
            torch.save(torch.randn(2, 8), path)
            rows.append(
                {
                    "id": f"sample-{row_idx}",
                    "query_id": f"q-{group_idx}",
                    "hidden_states_path": str(path),
                    "semantic_id": f"q-{group_idx}",
                    "style_id": f"style-{sample_idx}",
                }
            )

    jsonl_path = tmp_path / "uneven.jsonl"
    write_jsonl(jsonl_path, rows)
    dataset = CLIRTrajectoryDataset(jsonl_path)
    sampler = SemanticGroupBatchSampler(dataset, batch_size=5, shuffle=False)
    batches = list(iter(sampler))

    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert len(sampler) == len(batches)


def test_semantic_group_batch_sampler_uses_independent_exclusive_packing_pool(
    tmp_path: Path,
):
    rows = [
        {
            "id": f"mechanism-{index}",
            "query_id": f"mechanism-query-{index}",
            "hidden_states": [[0.0, 1.0]],
            "correctness": index % 2,
            "token_hallucination_target": [index % 2],
            "token_hallucination_mask": [1],
        }
        for index in range(8)
    ]
    rows.extend(
        {
            "id": f"semantic-{semantic_id}-{style_id}",
            "query_id": f"semantic-query-{semantic_id}-{style_id}",
            "hidden_states": [[0.0, 1.0]],
            "correctness": 1,
            "semantic_id": semantic_id,
            "style_id": style_id,
        }
        for semantic_id in ("a", "b")
        for style_id in ("direct", "verbose")
    )
    manifest = tmp_path / "train.jsonl"
    sidecar = tmp_path / "packing.jsonl"
    write_jsonl(manifest, rows)
    write_jsonl(
        sidecar,
        [
            {
                "id": f"mechanism-{index}",
                "packing_pool_id": "mechanism",
            }
            for index in range(8)
        ],
    )
    dataset = CLIRTrajectoryDataset(manifest)
    pools = load_batch_packing_pools(sidecar, dataset)
    sampler = SemanticGroupBatchSampler(
        dataset,
        batch_size=4,
        shuffle=False,
        packing_pools=pools,
    )

    batches = list(sampler)
    assert batches == [[8, 9, 10, 11], [0, 1, 2, 3], [4, 5, 6, 7]]
    assert sorted(index for batch in batches for index in batch) == list(range(12))
    assert all(
        sum(index < 8 for index in batch) in {0, 4}
        for batch in batches
    )
    assert all("semantic_id" not in dataset.rows[index] for index in range(8))


def test_batch_packing_sidecar_rejects_unknown_or_extra_metadata(tmp_path: Path):
    manifest = tmp_path / "train.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "known",
                "query_id": "q",
                "hidden_states": [[0.0, 1.0]],
                "correctness": 1,
            }
        ],
    )
    dataset = CLIRTrajectoryDataset(manifest)
    unknown = tmp_path / "unknown.jsonl"
    write_jsonl(unknown, [{"id": "missing", "packing_pool_id": "mechanism"}])
    with pytest.raises(ValueError, match="unknown row id"):
        load_batch_packing_pools(unknown, dataset)

    extra = tmp_path / "extra.jsonl"
    write_jsonl(
        extra,
        [{"id": "known", "packing_pool_id": "mechanism", "semantic_id": "fake"}],
    )
    with pytest.raises(ValueError, match="must contain only"):
        load_batch_packing_pools(extra, dataset)


def test_collate_preserves_bfloat16_feature_storage():
    batch = clir_collate(
        [
            {
                "row_index": 0,
                "id": "a",
                "query_id": "q",
                "hidden_states": torch.randn(2, 8, dtype=torch.bfloat16),
                "condition_states": torch.randn(1, 8, dtype=torch.bfloat16),
            }
        ]
    )

    assert batch["hidden_states"].dtype == torch.bfloat16
    assert batch["condition_states"].dtype == torch.bfloat16


def test_partial_correctness_batch_is_masked_instead_of_becoming_negative():
    batch = clir_collate(
        [
            {
                "row_index": 0,
                "id": "labeled",
                "query_id": "q",
                "hidden_states": torch.randn(2, 8),
                "correctness": 1,
            },
            {
                "row_index": 1,
                "id": "unlabeled",
                "query_id": "q",
                "hidden_states": torch.randn(2, 8),
            },
        ]
    )
    assert batch["correctness"].tolist() == [1.0, 0.0]
    assert batch["correctness_mask"].tolist() == [True, False]

    model = ConsistencyLocalizedReward(RewardConfig(hidden_dim=8, projection_dim=4))
    outputs, losses = model.training_step(batch)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["scores"][:1].float(),
        torch.ones(1),
    )
    assert torch.allclose(losses["final"], expected)


def test_log_space_noisy_or_keeps_long_negative_path_gradient():
    logits = torch.zeros(1, 1024, requires_grad=True)
    mask = torch.ones_like(logits)
    loss = path_level_hallucination_mil(logits, mask, torch.tensor([0.0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), rel=1e-6)
    assert logits.grad is not None
    assert logits.grad.norm().item() > 0.0

    probability = path_hallucination_probability(logits.detach(), mask)
    log_survival = path_no_hallucination_log_probability(logits.detach(), mask)
    assert probability.item() == 1.0
    assert torch.isfinite(log_survival).all()
    assert log_survival.item() < -100.0


def test_log_space_noisy_or_exact_zero_survival_has_finite_positive_gradient():
    logits = torch.full((1, 4), -200.0, requires_grad=True)
    mask = torch.ones_like(logits)

    loss = path_level_hallucination_mil(logits, mask, torch.tensor([1.0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_log_space_noisy_or_has_finite_gradient_near_zero_path_probability():
    logits = torch.full((2, 162), -30.0, requires_grad=True)
    mask = torch.ones_like(logits)
    labels = torch.tensor([0.0, 1.0])

    loss = path_level_hallucination_mil(logits, mask, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_train_cli_exposes_new_reward_config_fields():
    argv = [
        "train_clir.py",
        "--train_jsonl",
        "train.jsonl",
        "--output_model",
        "model.pt",
        "--hidden_dim",
        "24",
        "--model_variant",
        "encoded_swift",
        "--encoder_type",
        "layer_transformer",
        "--model_dim",
        "12",
        "--num_feature_layers",
        "3",
        "--per_layer_dim",
        "8",
        "--layer_encoder_dim",
        "8",
        "--layer_encoder_blocks",
        "1",
        "--layer_encoder_heads",
        "2",
        "--layer_pool_queries",
        "2",
        "--condition_attention_temperature",
        "0.7",
        "--hallucination_condition_stop_gradient",
        "--progress_score_weight",
        "0.25",
        "--key_prior_weight",
        "0.8",
        "--complete_prior_weight",
        "0.7",
        "--prior_distill_weight",
        "0.0",
        "--gate_prior_weight",
        "0.0",
        "--reconstruction_weight",
        "0.0",
    ]
    with patch("sys.argv", argv):
        args = parse_args()
    config = make_config(args)

    assert config.condition_attention_temperature == 0.7
    assert config.hallucination_condition_stop_gradient is True
    assert config.progress_score_weight == 0.25
    assert config.key_prior_weight == 0.8
    assert config.complete_prior_weight == 0.7
    assert config.prior_distill_weight == 0.0
    assert config.gate_prior_weight == 0.0
    assert config.reconstruction_weight == 0.0
    assert config.model_variant == "encoded_swift"
    assert config.encoder_type == "layer_transformer"
    assert config.model_dim == 12
    assert config.num_feature_layers == 3
    assert config.per_layer_dim == 8
