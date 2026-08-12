"""Train the self-contained CLIR reward model.

This script expects JSONL rows with pre-extracted hidden states. It does not
depend on the SWIFT repository; SWIFT only motivates the token reward/gate
architecture implemented in `src/consistency_localized_reward.py`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.clir_data import CLIRTrajectoryDataset, clir_collate, move_batch_to_device
from src.consistency_localized_reward import ConsistencyLocalizedReward, RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIR on pre-extracted hidden states.")
    parser.add_argument("--train_jsonl", required=True, help="Training JSONL file.")
    parser.add_argument("--feature_root", default=None, help="Base directory for relative feature paths.")
    parser.add_argument("--output_model", required=True, help="Where to save the checkpoint.")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden-state dimension.")
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    parser.add_argument("--final_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--score_consistency_weight", type=float, default=0.1)
    parser.add_argument("--hallucination_weight", type=float, default=1.0)
    parser.add_argument("--mil_weight", type=float, default=0.25)
    parser.add_argument("--token_reward_weight", type=float, default=0.5)
    parser.add_argument("--tail_weight", type=float, default=0.5)
    parser.add_argument("--pseudo_tail_weight", type=float, default=0.1)
    parser.add_argument("--progress_weight", type=float, default=0.25)
    parser.add_argument("--prior_weight", type=float, default=0.25)
    parser.add_argument("--negative_tail_margin", type=float, default=0.5)
    parser.add_argument("--pseudo_onset_threshold", type=float, default=0.5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def make_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        final_weight=args.final_weight,
        consistency_weight=args.consistency_weight,
        score_consistency_weight=args.score_consistency_weight,
        hallucination_weight=args.hallucination_weight,
        mil_weight=args.mil_weight,
        token_reward_weight=args.token_reward_weight,
        tail_weight=args.tail_weight,
        pseudo_tail_weight=args.pseudo_tail_weight,
        progress_weight=args.progress_weight,
        prior_weight=args.prior_weight,
        negative_tail_margin=args.negative_tail_margin,
        pseudo_onset_threshold=args.pseudo_onset_threshold,
    )


def split_dataset(dataset: CLIRTrajectoryDataset, val_fraction: float, seed: int):
    if val_fraction <= 0.0:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("val_fraction leaves no training examples")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(
    model: ConsistencyLocalizedReward,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        _, losses = model.training_step(batch)
        if training:
            losses["total"].backward()
            optimizer.step()

        batch_size = int(batch["hidden_states"].shape[0])
        count += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset = CLIRTrajectoryDataset(args.train_jsonl, feature_root=args.feature_root)
    train_dataset, val_dataset = split_dataset(dataset, args.val_fraction, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=clir_collate)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=clir_collate)
        if val_dataset is not None
        else None
    )

    config = make_config(args)
    model = ConsistencyLocalizedReward(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        message = f"epoch={epoch} train_total={train_metrics.get('total', 0.0):.4f}"
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, device, optimizer=None)
            message += f" val_total={val_metrics.get('total', 0.0):.4f}"
        print(message)

    output = Path(args.output_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.__dict__,
        },
        output,
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
