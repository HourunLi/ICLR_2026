from pathlib import Path

import pytest
import torch

from score_clir import load_model
from src.clir_data import clir_collate
from src.consistency_localized_reward import (
    RewardConfig,
    build_reward_model,
    count_trainable_parameters,
)


def small_layer_config(model_variant: str = "clir") -> RewardConfig:
    return RewardConfig(
        hidden_dim=24,
        model_variant=model_variant,
        encoder_type="layer_transformer",
        model_dim=12,
        num_feature_layers=3,
        per_layer_dim=8,
        layer_encoder_dim=8,
        layer_encoder_blocks=1,
        layer_encoder_heads=2,
        layer_pool_queries=2,
        projection_dim=6,
    )


def test_strict_swift_real_width_has_only_linear_d_by_two_head():
    config = RewardConfig(hidden_dim=33 * 3072, model_variant="strict_swift")
    model = build_reward_model(config)

    assert count_trainable_parameters(model) == config.hidden_dim * 2 + 2
    assert tuple(model.token_reward_head.weight.shape) == (2, config.hidden_dim)


def test_config_fails_fast_before_accidental_raw_width_quadratic_clir():
    with pytest.raises(ValueError, match="quadratic modules"):
        RewardConfig(hidden_dim=33 * 3072, model_variant="clir")

    learned = RewardConfig(
        hidden_dim=33 * 3072,
        model_variant="clir",
        encoder_type="layer_transformer",
        num_feature_layers=33,
        per_layer_dim=3072,
    )
    assert learned.model_dim == 768


def test_large_layer_encoder_requires_explicit_layer_layout():
    with pytest.raises(ValueError, match="explicit multi-layer contract"):
        RewardConfig(
            hidden_dim=33 * 3072,
            model_variant="clir",
            encoder_type="layer_transformer",
        )


def test_encoded_swift_cannot_silently_become_identity_swift():
    with pytest.raises(ValueError, match="non-identity encoder"):
        RewardConfig(hidden_dim=8, model_variant="encoded_swift")


def test_real_width_clir_has_no_raw_width_squared_parameter():
    config = RewardConfig(
        hidden_dim=33 * 3072,
        model_variant="clir",
        encoder_type="layer_transformer",
        model_dim=768,
        num_feature_layers=33,
        per_layer_dim=3072,
        layer_encoder_dim=256,
        layer_encoder_blocks=2,
        layer_encoder_heads=8,
        layer_pool_queries=4,
        projection_dim=256,
    )
    model = build_reward_model(config)

    assert count_trainable_parameters(model) < 20_000_000
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            assert not (parameter.shape[-1] == config.hidden_dim and parameter.shape[-2] == config.hidden_dim)


def test_layer_encoder_chunks_normalization_without_changing_outputs():
    torch.manual_seed(7)
    config = small_layer_config("encoded_swift")
    model = build_reward_model(config).eval()
    hidden_states = torch.randn(2, 5, config.hidden_dim)
    mask = torch.ones(2, 5)

    with torch.no_grad():
        reference = model(hidden_states, mask=mask)
        model.input_encoder.max_normalization_elements = (
            config.num_feature_layers * config.per_layer_dim * 2
        )
        chunked = model(hidden_states, mask=mask)

    assert torch.allclose(chunked["scores"], reference["scores"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        chunked["trajectory_layer_attention"],
        reference["trajectory_layer_attention"],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("variant", "encoder_type"),
    [
        ("strict_swift", "identity"),
        ("encoded_swift", "layer_transformer"),
        ("clir", "layer_transformer"),
    ],
)
def test_explicit_variants_forward_backward_and_loss_scope(variant: str, encoder_type: str):
    if encoder_type == "identity":
        config = RewardConfig(hidden_dim=24, model_variant=variant)
    else:
        config = small_layer_config(variant)
    model = build_reward_model(config)
    batch = {
        "hidden_states": torch.randn(2, 4, 24),
        "condition_states": torch.randn(2, 3, 24),
        "condition_mask": torch.ones(2, 3),
        "mask": torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]]),
        "correctness": torch.tensor([1.0, 0.0]),
    }

    outputs, losses = model.training_step(batch)

    assert outputs["scores"].shape == (2,)
    assert outputs["token_rewards"].shape == (2, 4)
    assert torch.isfinite(losses["total"])
    if variant == "strict_swift":
        assert set(losses) == {"final", "total"}
        assert "trajectory_layer_attention" not in outputs
    elif variant == "encoded_swift":
        assert set(losses) == {"final", "total"}
        assert outputs["trajectory_layer_attention"].shape == (2, 4, 2, 3)
    else:
        assert "prior_reconstruction" in losses
        assert outputs["complete_reconstruction"].shape == (2, 12)

    losses["total"].backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_layer_encoder_uses_every_layer_and_normalizes_pool_attention():
    model = build_reward_model(small_layer_config("encoded_swift"))
    hidden_states = torch.randn(1, 2, 24, requires_grad=True)

    outputs = model(hidden_states)
    attention = outputs["trajectory_layer_attention"]
    outputs["scores"].sum().backward()

    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention[..., 0]), atol=1e-6)
    layer_gradient = hidden_states.grad.reshape(1, 2, 3, 8).abs().sum(dim=(0, 1, 3))
    assert torch.all(layer_gradient > 0)


def test_reconstruction_requires_external_model_dim_target():
    model = build_reward_model(small_layer_config("clir"))
    base_batch = {
        "hidden_states": torch.randn(2, 4, 24),
        "mask": torch.ones(2, 4),
        "correctness": torch.tensor([1.0, 0.0]),
    }

    _, no_target_losses = model.training_step(base_batch)
    assert no_target_losses["prior_reconstruction"].item() == 0.0

    bad_batch = dict(base_batch)
    bad_batch["complete_reconstruction_target"] = torch.randn(2, 24)
    with pytest.raises(ValueError, match="externally generated fixed vector"):
        model.training_step(bad_batch)

    good_batch = dict(base_batch)
    good_batch["complete_reconstruction_target"] = torch.randn(2, 12)
    _, good_losses = model.training_step(good_batch)
    assert good_losses["prior_reconstruction"].item() > 0.0


def test_collate_reconstruction_target_width_is_independent_of_raw_input_width():
    batch = clir_collate(
        [
            {
                "row_index": 0,
                "id": "a",
                "query_id": "q",
                "hidden_states": torch.randn(2, 24),
                "complete_reconstruction_target": torch.randn(12),
            },
            {
                "row_index": 1,
                "id": "b",
                "query_id": "q",
                "hidden_states": torch.randn(3, 24),
                "complete_reconstruction_target": torch.randn(12),
            },
        ]
    )

    assert batch["hidden_states"].shape == (2, 3, 24)
    assert batch["complete_reconstruction_target"].shape == (2, 12)


@pytest.mark.parametrize("variant", ["strict_swift", "encoded_swift", "clir"])
def test_variant_checkpoint_round_trip(tmp_path: Path, variant: str):
    config = (
        RewardConfig(hidden_dim=24, model_variant=variant)
        if variant == "strict_swift"
        else small_layer_config(variant)
    )
    model = build_reward_model(config)
    checkpoint = tmp_path / f"{variant}.pt"
    torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, checkpoint)

    restored = load_model(checkpoint, torch.device("cpu"))

    assert restored.config.model_variant == variant
    assert count_trainable_parameters(restored) == count_trainable_parameters(model)
