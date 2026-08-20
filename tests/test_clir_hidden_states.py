import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

import score_clir
from src.clir_data import CLIRTrajectoryDataset, clir_collate, write_jsonl
from src.clir_hidden_states import (
    OnlineHiddenStateConfig,
    OnlineHiddenStateExtractor,
    validate_online_rows,
)
from src.consistency_localized_reward import RewardConfig, build_reward_model
from train_clir import run_epoch, validate_dataset_feature_contract


class RecordingCausalLM(torch.nn.Module):
    """Deterministic embedding-plus-two-block fixture."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.forwarded_token_ids: list[list[int]] = []

    def forward(
        self,
        input_ids,
        attention_mask,
        output_hidden_states,
        use_cache,
        return_dict,
    ):
        assert output_hidden_states is True
        assert use_cache is False
        assert return_dict is True
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        self.forwarded_token_ids.append(input_ids[0].tolist())
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
        base = torch.stack((input_ids.float(), positions.float()), dim=-1)
        return SimpleNamespace(hidden_states=(base, base + 100.0, base + 200.0))


def _extractor_config() -> OnlineHiddenStateConfig:
    return OnlineHiddenStateConfig(
        model_id="fake/model",
        model_revision="revision",
        tokenizer_revision="revision",
        torch_dtype="float32",
        trust_remote_code=False,
        layer_count=3,
        per_layer_hidden_size=2,
        feature_dim=6,
    )


def _rows():
    provenance = {
        "model_id": "fake/model",
        "model_revision": "revision",
        "tokenizer_revision": "revision",
        "protocol_version": "online-test-v1",
    }
    return [
        {
            "id": "q0-c0",
            "query_id": "q0",
            "semantic_id": "q0-c0",
            "style_id": "original",
            "prompt": "prompt",
            "question": "problem",
            "response": "first",
            "prompt_token_ids": [10, 11],
            "output_token_ids": [20],
            "correctness": 0,
            "provenance": provenance,
        },
        {
            "id": "q0-c1",
            "query_id": "q0",
            "semantic_id": "q0-c1",
            "style_id": "original",
            "prompt": "prompt",
            "question": "problem",
            "response": "second",
            "prompt_token_ids": [10, 11],
            "output_token_ids": [30, 31],
            "correctness": 1,
            "provenance": provenance,
        },
    ]


def test_online_dataset_and_extractor_materialize_exact_aligned_features(
    tmp_path: Path,
):
    manifest = tmp_path / "tokens.jsonl"
    rows = _rows()
    write_jsonl(manifest, rows)
    dataset = CLIRTrajectoryDataset(
        manifest,
        require_correctness=True,
        hidden_state_source="online",
    )
    validate_online_rows(dataset.rows, _extractor_config())
    batch = clir_collate([dataset[0], dataset[1]])
    assert "hidden_states" not in batch
    assert batch["mask"].tolist() == [[1.0, 0.0], [1.0, 1.0]]

    causal_lm = RecordingCausalLM()
    extractor = OnlineHiddenStateExtractor(
        causal_lm,
        _extractor_config(),
        device="cpu",
    )
    materialized = extractor.materialize(batch, include_condition=True)

    assert materialized["hidden_states"].shape == (2, 2, 6)
    assert materialized["condition_states"].shape == (2, 2, 6)
    assert materialized["hidden_states"][0, 0].tolist() == [
        20.0,
        2.0,
        120.0,
        102.0,
        220.0,
        202.0,
    ]
    assert materialized["hidden_states"][1, 1].tolist() == [
        31.0,
        3.0,
        131.0,
        103.0,
        231.0,
        203.0,
    ]
    assert torch.equal(
        materialized["condition_states"][0],
        materialized["condition_states"][1],
    )
    # Two trajectory forwards plus one deduplicated prompt-only condition.
    assert causal_lm.forwarded_token_ids == [
        [10, 11, 20],
        [10, 11, 30, 31],
        [10, 11],
    ]


def test_online_provenance_mismatch_fails_before_model_loading():
    rows = _rows()
    rows[0]["provenance"] = dict(rows[0]["provenance"], model_revision="wrong")
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_online_rows(rows, _extractor_config())


def test_online_schema_does_not_require_precomputed_metadata_on_every_view(
    tmp_path: Path,
):
    rows = _rows()
    rows[0]["feature_metadata"] = {
        "layer_count": 3,
        "per_layer_hidden_size": 2,
        "feature_dim": 6,
    }
    manifest = tmp_path / "mixed-storage-metadata.jsonl"
    write_jsonl(manifest, rows)
    dataset = CLIRTrajectoryDataset(manifest, hidden_state_source="online")
    validate_online_rows(dataset.rows, _extractor_config())

    validate_dataset_feature_contract(
        dataset,
        RewardConfig(
            hidden_dim=6,
            model_variant="strict_swift",
            encoder_type="identity",
        ),
        "train",
    )


def test_training_epoch_accepts_token_only_loader(tmp_path: Path):
    manifest = tmp_path / "tokens.jsonl"
    write_jsonl(manifest, _rows())
    dataset = CLIRTrajectoryDataset(
        manifest,
        require_correctness=True,
        hidden_state_source="online",
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=clir_collate)
    extractor = OnlineHiddenStateExtractor(
        RecordingCausalLM(),
        _extractor_config(),
        device="cpu",
    )
    model = build_reward_model(
        RewardConfig(
            hidden_dim=6,
            model_variant="strict_swift",
            encoder_type="identity",
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer,
        prior_phase="joint",
        hidden_state_extractor=extractor,
    )

    assert metrics["examples"] == 2
    assert metrics["batches"] == 1
    assert metrics["applicable_counts"]["final"] == 2
    assert metrics["active_batches"]["final"] == 1
    assert metrics["active_losses"]["final"] == metrics["losses"]["final"]
    assert metrics["active_loss_reduction"] == (
        "mean_of_per_batch_component_means_over_batches_with_nonzero_"
        "applicable_count"
    )


def test_scoring_accepts_token_only_manifest(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "tokens.jsonl"
    checkpoint_path = tmp_path / "reward.pt"
    output_path = tmp_path / "scores.jsonl"
    write_jsonl(manifest, _rows())
    config = RewardConfig(
        hidden_dim=6,
        model_variant="strict_swift",
        encoder_type="identity",
    )
    reward_model = build_reward_model(config)
    torch.save(
        {
            "schema_version": "clir-full-checkpoint-v2",
            "config": config.__dict__,
            "state_dict": reward_model.state_dict(),
            "experiment_protocol": None,
        },
        checkpoint_path,
    )

    def fake_loader(online_config, **kwargs):
        return OnlineHiddenStateExtractor(
            RecordingCausalLM(),
            online_config,
            device=kwargs["device"],
        )

    monkeypatch.setattr(score_clir, "load_online_hidden_state_extractor", fake_loader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_clir.py",
            "--input_jsonl",
            str(manifest),
            "--model",
            str(checkpoint_path),
            "--output_jsonl",
            str(output_path),
            "--hidden_state_source",
            "online",
            "--extractor_model_id",
            "fake/model",
            "--extractor_model_revision",
            "revision",
            "--extractor_tokenizer_revision",
            "revision",
            "--extractor_torch_dtype",
            "float32",
            "--extractor_layer_count",
            "3",
            "--extractor_per_layer_hidden_size",
            "2",
            "--batch_size",
            "2",
            "--device",
            "cpu",
        ],
    )

    score_clir.main()

    scored = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(scored) == 2
    assert all("reward_score" in row for row in scored)
    assert all(
        row["reward_scoring_provenance"]["hidden_state_source"] == "online"
        for row in scored
    )
