"""Score trajectories with a trained CLIR reward model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
from torch.utils.data import DataLoader

from src.clir_data import CLIRTrajectoryDataset, clir_collate, move_batch_to_device
from src.clir_real_data import file_sha256
from src.clir_stage_a import atomic_write_jsonl, git_state
from src.consistency_localized_reward import (
    RewardConfig,
    build_reward_model,
    infer_pseudo_onsets,
    path_hallucination_probability,
    path_no_hallucination_log_probability,
    select_best_of_n,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CLIR trajectories.")
    parser.add_argument("--input_jsonl", required=True, help="JSONL file to score.")
    parser.add_argument("--model", required=True, help="CLIR checkpoint from train_clir.py.")
    parser.add_argument("--output_jsonl", required=True, help="Where to write scored rows.")
    parser.add_argument(
        "--experiment_protocol_config",
        default=None,
        help="Optional frozen experiment protocol; must match the training checkpoint.",
    )
    parser.add_argument("--feature_root", default=None, help="Base directory for relative feature paths.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--onset_threshold", type=float, default=0.5)
    parser.add_argument("--amp_dtype", default="none", choices=["none", "bfloat16"])
    parser.add_argument("--skip_feature_finite_check", action="store_true")
    parser.add_argument("--expected_input_sha256", default=None)
    parser.add_argument("--expected_checkpoint_sha256", default=None)
    parser.add_argument(
        "--min_score_std",
        type=float,
        default=0.0,
        help="Fail before writing scores when population score std is below this health threshold.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_model_with_checkpoint(
    path: str | Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Mapping[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = RewardConfig(**checkpoint["config"])
    model = build_reward_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def load_model(path: str | Path, device: torch.device) -> torch.nn.Module:
    model, _ = load_model_with_checkpoint(path, device)
    return model


def _experiment_protocol_state(path: str | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    protocol_path = Path(path).resolve()
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    schema_version = protocol.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("Experiment protocol requires a non-empty schema_version")
    return {
        "path": str(protocol_path),
        "sha256": file_sha256(protocol_path),
        "schema_version": schema_version,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.persistent_workers and args.num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if args.min_score_std < 0:
        raise ValueError("min_score_std must be non-negative")
    input_path = Path(args.input_jsonl).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output_jsonl).resolve()
    if output_path in {input_path, model_path}:
        raise ValueError("Scoring output must differ from the input manifest and checkpoint")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing scored manifest: {output_path}")
    device = resolve_device(args.device)
    model, checkpoint = load_model_with_checkpoint(model_path, device)
    experiment_protocol = _experiment_protocol_state(args.experiment_protocol_config)
    if checkpoint.get("experiment_protocol") != experiment_protocol:
        raise ValueError(
            "Scoring experiment protocol differs from the training checkpoint"
        )
    model_variant = model.config.model_variant
    dataset = CLIRTrajectoryDataset(
        args.input_jsonl,
        feature_root=args.feature_root,
        check_finite=not args.skip_feature_finite_check,
        load_condition=model_variant == "clir",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    if args.amp_dtype == "bfloat16" and device.type != "cuda":
        raise ValueError("--amp_dtype bfloat16 currently requires CUDA")

    rows: List[Dict] = [dict(row) for row in dataset.rows]
    checkpoint_sha256 = file_sha256(model_path)
    input_sha256 = file_sha256(input_path)
    if args.expected_checkpoint_sha256 and checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError(
            "Checkpoint SHA256 mismatch: "
            f"expected {args.expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )
    if args.expected_input_sha256 and input_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"Input manifest SHA256 mismatch: expected {args.expected_input_sha256}, got {input_sha256}"
        )
    scoring_provenance: Dict[str, Any] = {
        "schema_version": "clir-reward-scoring-v2",
        "model_variant": model_variant,
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_schema_version": checkpoint.get("schema_version"),
        "checkpoint_code": checkpoint.get("code"),
        "input_jsonl": str(input_path),
        "input_sha256": input_sha256,
        "batch_size": args.batch_size,
        "amp_dtype": args.amp_dtype,
        "compute_dtype": "bfloat16" if args.amp_dtype == "bfloat16" else "float32",
        "device": str(device),
        "score_code": git_state(PROJECT_ROOT),
        "experiment_protocol": experiment_protocol,
        "min_score_std": args.min_score_std,
    }
    scored_row_indices: List[int] = []
    scored_scores: List[float] = []
    scored_query_ids: List[str] = []

    for batch in loader:
        row_indices = batch["row_index"].tolist()
        query_ids_raw = list(batch["query_ids_raw"])
        batch = move_batch_to_device(batch, device)
        if args.amp_dtype == "none":
            parameter_dtype = next(model.parameters()).dtype
            for key in ("hidden_states", "condition_states", "condition_embedding"):
                if key in batch:
                    batch[key] = batch[key].to(dtype=parameter_dtype)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if args.amp_dtype == "bfloat16"
            else nullcontext()
        )
        with autocast:
            outputs = model(
                batch["hidden_states"],
                mask=batch["mask"],
                condition_states=batch.get("condition_states"),
                condition_mask=batch.get("condition_mask"),
                condition_embedding=batch.get("condition_embedding"),
                condition_embedding_mask=batch.get("condition_embedding_mask"),
            )
        path_probs = None
        path_log_survival = None
        pseudo_onsets = None
        if "hallucination_logits" in outputs:
            path_probs = path_hallucination_probability(outputs["hallucination_logits"], outputs["mask"])
            path_log_survival = path_no_hallucination_log_probability(
                outputs["hallucination_logits"], outputs["mask"]
            )
            pseudo_onsets = infer_pseudo_onsets(
                outputs["hallucination_logits"],
                outputs["mask"],
                threshold=args.onset_threshold,
            )

        for local_idx, row_index in enumerate(row_indices):
            row = rows[row_index]
            valid_length = int(batch["mask"][local_idx].sum().detach().cpu())
            row["reward_model_variant"] = model_variant
            row["reward_scoring_provenance"] = dict(scoring_provenance)
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
            if path_probs is not None and path_log_survival is not None and pseudo_onsets is not None:
                row["clir_path_hallucination_prob"] = float(path_probs[local_idx].detach().cpu())
                row["clir_path_no_hallucination_log_prob"] = float(
                    path_log_survival[local_idx].detach().cpu()
                )
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

    score_tensor = torch.tensor(scored_scores, dtype=torch.float64)
    score_distribution = {
        "count": len(scored_scores),
        "mean": float(score_tensor.mean()),
        "population_std": float(score_tensor.std(unbiased=False)),
        "min": float(score_tensor.min()),
        "max": float(score_tensor.max()),
    }
    if score_distribution["population_std"] < args.min_score_std:
        raise RuntimeError(
            "Scoring health gate failed: population std "
            f"{score_distribution['population_std']:.8g} is below {args.min_score_std:.8g}"
        )
    scoring_provenance["score_distribution"] = score_distribution
    for row in rows:
        row["reward_scoring_provenance"] = dict(scoring_provenance)

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

    atomic_write_jsonl(output_path, rows)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
