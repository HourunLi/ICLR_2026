"""Score trajectories with a trained CLIR reward model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.clir_data import CLIRTrajectoryDataset, clir_collate, move_batch_to_device, write_jsonl
from src.consistency_localized_reward import (
    RewardConfig,
    build_reward_model,
    infer_pseudo_onsets,
    path_hallucination_probability,
    select_best_of_n,
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


def load_model(path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device)
    config = RewardConfig(**checkpoint["config"])
    model = build_reward_model(config).to(device)
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
    model_variant = model.config.model_variant

    rows: List[Dict] = [dict(row) for row in dataset.rows]
    scored_row_indices: List[int] = []
    scored_scores: List[float] = []
    scored_query_ids: List[str] = []

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
            condition_embedding_mask=batch.get("condition_embedding_mask"),
        )
        path_probs = None
        pseudo_onsets = None
        if "hallucination_logits" in outputs:
            path_probs = path_hallucination_probability(outputs["hallucination_logits"], outputs["mask"])
            pseudo_onsets = infer_pseudo_onsets(
                outputs["hallucination_logits"],
                outputs["mask"],
                threshold=args.onset_threshold,
            )

        for local_idx, row_index in enumerate(row_indices):
            row = rows[row_index]
            valid_length = int(batch["mask"][local_idx].sum().detach().cpu())
            row["reward_model_variant"] = model_variant
            row["reward_score"] = float(outputs["scores"][local_idx].detach().cpu())
            row["reward_mean_gate"] = float(
                outputs["gates"][local_idx, :valid_length].mean().detach().cpu()
            )
            if model_variant == "clir":
                row["clir_score"] = row["reward_score"]
                row["clir_mean_gate"] = row["reward_mean_gate"]
            else:
                row["swift_score"] = row["reward_score"]
            gate_attention = outputs["gates"][local_idx] / outputs["gates"][local_idx].sum().clamp_min(1e-8)
            row["reward_gate_attention"] = [
                float(x) for x in gate_attention[:valid_length].detach().cpu().tolist()
            ]
            if path_probs is not None and pseudo_onsets is not None:
                row["clir_path_hallucination_prob"] = float(path_probs[local_idx].detach().cpu())
                row["clir_pseudo_onset"] = int(pseudo_onsets[local_idx].detach().cpu())
                prior_alignment = torch.sum(gate_attention * outputs["fused_prior"][local_idx])
                row["clir_prior_gate_alignment"] = float(prior_alignment.detach().cpu())
                row["clir_condition_relevance"] = [
                    float(x)
                    for x in outputs["condition_relevance"][local_idx, :valid_length].detach().cpu().tolist()
                ]
                row["clir_gate_attention"] = row["reward_gate_attention"]
                row["clir_key_prior"] = [
                    float(x) for x in outputs["key_prior"][local_idx, :valid_length].detach().cpu().tolist()
                ]
                row["clir_complete_prior"] = [
                    float(x) for x in outputs["complete_prior"][local_idx, :valid_length].detach().cpu().tolist()
                ]
            if "trajectory_layer_attention" in outputs:
                layer_attention = outputs["trajectory_layer_attention"][local_idx, :valid_length]
                row["mean_layer_pool_attention"] = layer_attention.mean(dim=0).detach().cpu().tolist()
            scored_row_indices.append(row_index)
            scored_scores.append(row["reward_score"])
            scored_query_ids.append(str(query_ids_raw[local_idx]))

    query_to_int: Dict[str, int] = {}
    encoded_groups = []
    for query_id in scored_query_ids:
        query_to_int.setdefault(query_id, len(query_to_int))
        encoded_groups.append(query_to_int[query_id])

    best_local_indices = select_best_of_n(
        torch.tensor(scored_scores, dtype=torch.float32),
        torch.tensor(encoded_groups, dtype=torch.long),
    )
    selected_indices = {scored_row_indices[local_idx] for local_idx in best_local_indices.values()}
    for idx, row in enumerate(rows):
        row["reward_selected_best_of_n"] = idx in selected_indices
        if model_variant == "clir":
            row["clir_selected_best_of_n"] = idx in selected_indices

    write_jsonl(args.output_jsonl, rows)
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
