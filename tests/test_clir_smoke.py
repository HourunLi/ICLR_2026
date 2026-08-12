from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.clir_data import CLIRTrajectoryDataset, clir_collate, write_jsonl
from src.consistency_localized_reward import ConsistencyLocalizedReward, RewardConfig


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
    assert losses["total"].ndim == 0
    losses["total"].backward()


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
                "key_prior_target": [1, 0, 0, 0],
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
    assert batch["consistency_mask"].tolist() == [True, True]
