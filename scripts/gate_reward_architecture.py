#!/usr/bin/env python3
"""Run an engineering-only forward/backward gate on real extracted features.

This gate deliberately trains only against final correctness. It verifies that
strict SWIFT, encoded SWIFT, and CLIR can consume the same all-layer manifest
without constructing a raw-width D-by-D parameter. It is not an effectiveness
experiment and must not be reported as one.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clir_data import CLIRTrajectoryDataset, clir_collate, move_batch_to_device
from src.consistency_localized_reward import (
    RewardConfig,
    build_reward_model,
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Extracted CLIR JSONL manifest.")
    parser.add_argument("--output_json", default=None, help="Optional machine-readable gate report.")
    parser.add_argument("--rows", type=int, default=2, help="Number of real rows to collate.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete device.")
    parser.add_argument("--input_dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--model_dim", type=int, default=768)
    parser.add_argument("--num_feature_layers", type=int, default=33)
    parser.add_argument("--per_layer_dim", type=int, default=3072)
    parser.add_argument("--layer_encoder_dim", type=int, default=256)
    parser.add_argument("--layer_encoder_blocks", type=int, default=2)
    parser.add_argument("--layer_encoder_heads", type=int, default=8)
    parser.add_argument("--layer_pool_queries", type=int, default=4)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_config(variant: str, input_dim: int, args: argparse.Namespace) -> RewardConfig:
    if variant == "strict_swift":
        return RewardConfig(
            hidden_dim=input_dim,
            model_variant=variant,
            encoder_type="identity",
            num_feature_layers=args.num_feature_layers,
            per_layer_dim=args.per_layer_dim,
        )
    return RewardConfig(
        hidden_dim=input_dim,
        model_variant=variant,
        encoder_type="layer_transformer",
        model_dim=args.model_dim,
        num_feature_layers=args.num_feature_layers,
        per_layer_dim=args.per_layer_dim,
        layer_encoder_dim=args.layer_encoder_dim,
        layer_encoder_blocks=args.layer_encoder_blocks,
        layer_encoder_heads=args.layer_encoder_heads,
        layer_pool_queries=args.layer_pool_queries,
        encoder_dropout=0.0,
    )


def max_parameter_matrix_shape(model: torch.nn.Module) -> list[int]:
    matrices = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    if not matrices:
        return []
    largest = max(matrices, key=lambda parameter: parameter.numel())
    return list(largest.shape)


def run_variant(
    variant: str,
    batch: Dict[str, Any],
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    config = make_config(variant, input_dim, args)
    model = build_reward_model(config).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        outputs, losses = model.training_step(batch)
    losses["total"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite_gradients = bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients)
    result: Dict[str, Any] = {
        "variant": variant,
        "encoder_type": config.encoder_type,
        "input_dim": input_dim,
        "model_dim": config.model_dim,
        "trainable_parameters": count_trainable_parameters(model),
        "largest_parameter_shape": max_parameter_matrix_shape(model),
        "score_shape": list(outputs["scores"].shape),
        "scores_finite": bool(torch.isfinite(outputs["scores"]).all().item()),
        "loss": float(losses["total"].detach().float().cpu()),
        "loss_finite": bool(torch.isfinite(losses["total"]).item()),
        "parameters_with_gradient": len(gradients),
        "gradients_finite": finite_gradients,
        "elapsed_seconds": elapsed,
    }
    if "trajectory_layer_attention" in outputs:
        attention = outputs["trajectory_layer_attention"].detach().float()
        result["trajectory_layer_attention_shape"] = list(attention.shape)
        result["layer_attention_sum_max_error"] = float(
            (attention.sum(dim=-1) - 1.0).abs().max().cpu()
        )
    if device.type == "cuda":
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)

    if not all(
        [
            result["scores_finite"],
            result["loss_finite"],
            result["gradients_finite"],
        ]
    ):
        raise RuntimeError(f"{variant} produced non-finite gate output: {result}")
    return result


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    device = resolve_device(args.device)
    dataset = CLIRTrajectoryDataset(args.manifest)
    if len(dataset) < args.rows:
        raise ValueError(f"manifest has {len(dataset)} rows, requested {args.rows}")

    batch = clir_collate([dataset[index] for index in range(args.rows)])
    input_dim = int(batch["hidden_states"].shape[-1])
    expected_dim = args.num_feature_layers * args.per_layer_dim
    if input_dim != expected_dim:
        raise ValueError(f"feature width is {input_dim}, but layer contract gives {expected_dim}")
    batch = move_batch_to_device(batch, device)
    feature_dtype = torch.bfloat16 if args.input_dtype == "bfloat16" else torch.float32
    if device.type == "cpu" and feature_dtype == torch.bfloat16:
        feature_dtype = torch.float32
    for key in ("hidden_states", "condition_states", "condition_embedding"):
        if key in batch:
            batch[key] = batch[key].to(dtype=feature_dtype)

    report: Dict[str, Any] = {
        "evidence_tier": "engineering_gate_not_effectiveness_evidence",
        "manifest": str(Path(args.manifest).resolve()),
        "row_ids": batch["ids"],
        "trajectory_lengths": [int(value) for value in batch["mask"].sum(dim=1).cpu().tolist()],
        "condition_lengths": (
            [int(value) for value in batch["condition_mask"].sum(dim=1).cpu().tolist()]
            if "condition_mask" in batch
            else []
        ),
        "input_dtype": str(feature_dtype),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "variants": [],
    }
    for variant in ("strict_swift", "encoded_swift", "clir"):
        result = run_variant(variant, batch, input_dim, device, args)
        report["variants"].append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
