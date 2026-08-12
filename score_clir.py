"""Score trajectories with a trained CLIR reward model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.clir_data import CLIRTrajectoryDataset, clir_collate, move_batch_to_device, write_jsonl
from src.consistency_localized_reward import (
    ConsistencyLocalizedReward,
    RewardConfig,
    infer_pseudo_onsets,
    path_hallucination_probability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CLIR trajectories.")
    parser.add_argument("--input_jsonl", required=True, help="JSONL file to score.")
    parser.add_argument("--model", required=True, help="CLIR checkpoint from train_clir.py.")
    parser.add_argument("--output_jsonl", required=True, help="Where to write scored rows.")
    parser.add_argument("--feature_root", default=None, help="Base directory for relative feature paths.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--onset_threshold", type=float, default=0.5)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_model(path: str | Path, device: torch.device) -> ConsistencyLocalizedReward:
    checkpoint = torch.load(path, map_location=device)
    config = RewardConfig(**checkpoint["config"])
    model = ConsistencyLocalizedReward(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = CLIRTrajectoryDataset(args.input_jsonl, feature_root=args.feature_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=clir_collate)
    model = load_model(args.model, device)

    rows: List[Dict] = [dict(row) for row in dataset.rows]
    best_by_query: Dict[str, tuple[int, float]] = {}

    for batch in loader:
        row_indices = batch["row_index"].tolist()
        query_ids_raw = list(batch["query_ids_raw"])
        batch = move_batch_to_device(batch, device)
        outputs = model(
            batch["hidden_states"],
            mask=batch["mask"],
            condition_states=batch.get("condition_states"),
            condition_mask=batch.get("condition_mask"),
            condition_embedding=batch.get("condition_embedding"),
        )
        path_probs = path_hallucination_probability(outputs["hallucination_logits"], outputs["mask"])
        pseudo_onsets = infer_pseudo_onsets(
            outputs["hallucination_logits"],
            outputs["mask"],
            threshold=args.onset_threshold,
        )

        for local_idx, row_index in enumerate(row_indices):
            row = rows[row_index]
            row["clir_score"] = float(outputs["scores"][local_idx].detach().cpu())
            row["clir_path_hallucination_prob"] = float(path_probs[local_idx].detach().cpu())
            row["clir_pseudo_onset"] = int(pseudo_onsets[local_idx].detach().cpu())
            row["clir_mean_gate"] = float(outputs["gates"][local_idx].mean().detach().cpu())
            query_id = query_ids_raw[local_idx]
            previous = best_by_query.get(query_id)
            if previous is None or row["clir_score"] > previous[1]:
                best_by_query[query_id] = (row_index, row["clir_score"])

    selected_indices = {row_index for row_index, _ in best_by_query.values()}
    for idx, row in enumerate(rows):
        row["clir_selected_best_of_n"] = idx in selected_indices

    write_jsonl(args.output_jsonl, rows)
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
