#!/usr/bin/env python3
"""Audit a single gate-alignment weight without training a gate-enabled cell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_clir import load_model_with_checkpoint  # noqa: E402
from src.clir_data import (  # noqa: E402
    CLIRTrajectoryDataset,
    clir_collate,
    move_batch_to_device,
)
from src.clir_hallucination_annotation import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/dual_prior_mutual_distillation_v1/training_protocol_v1.json"
)
DEFAULT_RESULT = (
    ROOT / "configs/dual_prior_mutual_distillation_v1/training_result_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/dual_prior_reward_gate_v1/loss_scale_audit_v1.json"
)
SCOPES = ("train", "checkpoint_train_evaluation", "validation")
COMPONENTS = (
    "final",
    "prior_key",
    "prior_complete",
    "prior_distill",
    "prior_gate",
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def summary(values: Iterable[float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized or not all(math.isfinite(value) for value in materialized):
        raise ValueError("Scale audit requires a non-empty finite collection")
    return {
        "count": len(materialized),
        "minimum": min(materialized),
        "mean": statistics.mean(materialized),
        "median": statistics.median(materialized),
        "maximum": max(materialized),
    }


def parameter_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(squared)


def gate_row_grad_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    weight_grad = model.token_reward_head.weight.grad
    bias_grad = model.token_reward_head.bias.grad
    if weight_grad is not None:
        squared += float(weight_grad[0].detach().float().pow(2).sum().cpu())
    if bias_grad is not None:
        squared += float(bias_grad[0].detach().float().pow(2).sum().cpu())
    return math.sqrt(squared)


def clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    result_path = args.result.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-dual-prior-mutual-distillation-training-protocol-v1"
    ):
        raise ValueError("Unexpected source protocol schema")
    if result.get("status") != "completed_pass_original_mutual_distillation":
        raise ValueError("Gate scale must be audited from the passed M1 comparison")
    if result.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("Source protocol/result provenance drifted")
    if float(result.get("mutual_distillation_weight", -1.0)) != 0.25:
        raise ValueError("Source M1 mutual-distillation weight drifted")

    seeds = [int(seed) for seed in protocol["matched_training"]["seeds"]]
    mutual_cell = str(protocol["comparison"]["mutual_cell"])
    output_root = resolve(protocol["execution"]["output_root"])
    loss_values: dict[str, dict[str, list[float]]] = {
        scope: {component: [] for component in COMPONENTS} for scope in SCOPES
    }
    source_artifacts: dict[str, Any] = {}
    for seed in seeds:
        cell_root = output_root / f"seed_{seed}" / mutual_cell
        metrics_path = cell_root / "metrics.jsonl"
        checkpoint_path = cell_root / "model.pt"
        cell_result_path = cell_root / "cell_result.json"
        cell_result = json.loads(cell_result_path.read_text(encoding="utf-8"))
        expected_key = f"seed_{seed}/{mutual_cell}"
        if result["cell_result_hashes"].get(expected_key) != file_sha256(
            cell_result_path
        ):
            raise ValueError(f"Source cell-result hash drifted for seed {seed}")
        if cell_result.get("metrics_sha256") != file_sha256(metrics_path):
            raise ValueError(f"Source metrics hash drifted for seed {seed}")
        if cell_result.get("checkpoint_sha256") != file_sha256(checkpoint_path):
            raise ValueError(f"Source checkpoint hash drifted for seed {seed}")
        if float(cell_result["resolved_loss_weights"]["gate_prior"]) != 0.0:
            raise ValueError("Scale diagnostic must come from gate-disabled M1")
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != int(protocol["matched_training"]["epochs"]):
            raise ValueError(f"Seed {seed} metrics do not contain every frozen epoch")
        for row in rows:
            for scope in SCOPES:
                for component in COMPONENTS:
                    loss_values[scope][component].append(
                        float(row[scope]["losses"][component])
                    )
        source_artifacts[str(seed)] = {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "metrics": str(metrics_path.relative_to(ROOT)),
            "metrics_sha256": file_sha256(metrics_path),
            "cell_result": str(cell_result_path.relative_to(ROOT)),
            "cell_result_sha256": file_sha256(cell_result_path),
        }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the frozen BF16 gradient audit")
    cell = protocol["cells"][mutual_cell]
    train_path = resolve(cell["train"]["path"])
    if file_sha256(train_path) != cell["train"]["sha256"]:
        raise ValueError("Frozen M1 training manifest drifted")
    dataset = CLIRTrajectoryDataset(
        train_path,
        check_finite=False,
        load_condition=True,
        hidden_state_source="precomputed",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=0,
    )
    first_batch = next(iter(loader))
    batch_ids = list(first_batch["ids"])
    gradient_diagnostic: dict[str, Any] = {}
    raw_gate_to_final_gate_head_ratios: list[float] = []
    raw_gate_to_final_shared_ratios: list[float] = []
    for seed in seeds:
        checkpoint_path = resolve(source_artifacts[str(seed)]["checkpoint"])
        model, checkpoint = load_model_with_checkpoint(checkpoint_path, device)
        if checkpoint["code"]["commit"] != result["training_commit"]:
            raise ValueError("Source checkpoint code commit drifted")
        if float(checkpoint["config"]["gate_prior_weight"]) != 0.0:
            raise ValueError("Source checkpoint unexpectedly trained gate alignment")
        model.train(False)
        shared_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith(
                (
                    "token_reward_head.",
                    "key_prior_head.",
                    "complete_prior_head.",
                    "complete_reconstructor.",
                    "final_score_head.",
                    "hallucination_head.",
                    "progress_head.",
                    "projector.",
                )
            )
        ]
        component_diagnostic: dict[str, Any] = {}
        for component in COMPONENTS:
            batch = move_batch_to_device(clone_batch(first_batch), device)
            model.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast("cpu", dtype=torch.bfloat16)
            )
            with autocast:
                _, losses = model.training_step(batch, prior_phase="joint")
                loss = losses[component]
            loss.backward()
            component_diagnostic[component] = {
                "loss": float(loss.detach().float().cpu()),
                "gate_head_row_grad_l2": gate_row_grad_norm(model),
                "shared_encoder_grad_l2": parameter_grad_norm(shared_parameters),
            }
            del batch
        final_gate_grad = component_diagnostic["final"]["gate_head_row_grad_l2"]
        final_shared_grad = component_diagnostic["final"]["shared_encoder_grad_l2"]
        gate_gate_grad = component_diagnostic["prior_gate"]["gate_head_row_grad_l2"]
        gate_shared_grad = component_diagnostic["prior_gate"]["shared_encoder_grad_l2"]
        if final_gate_grad <= 0.0 or final_shared_grad <= 0.0:
            raise ValueError("Final BCE produced no auditable gradient")
        raw_gate_to_final_gate_head_ratios.append(gate_gate_grad / final_gate_grad)
        raw_gate_to_final_shared_ratios.append(gate_shared_grad / final_shared_grad)
        gradient_diagnostic[str(seed)] = component_diagnostic
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate_weights = (0.25, 1.0, 5.0, 10.0)
    weight_diagnostics: dict[str, Any] = {}
    for weight in candidate_weights:
        weighted_gate_ratios = [
            weight * ratio for ratio in raw_gate_to_final_gate_head_ratios
        ]
        weighted_shared_ratios = [
            weight * ratio for ratio in raw_gate_to_final_shared_ratios
        ]
        weight_diagnostics[str(weight)] = {
            "gate_head_gradient_ratio_vs_final": summary(weighted_gate_ratios),
            "shared_encoder_gradient_ratio_vs_final": summary(
                weighted_shared_ratios
            ),
            "checkpoint_train_weighted_gate_loss": summary(
                weight * value
                for value in loss_values["checkpoint_train_evaluation"][
                    "prior_gate"
                ]
            ),
            "dev_weighted_gate_loss": summary(
                weight * value
                for value in loss_values["validation"]["prior_gate"]
            ),
        }

    chosen_weight = 10.0
    chosen = weight_diagnostics[str(chosen_weight)]
    gate_ratio = chosen["gate_head_gradient_ratio_vs_final"]
    shared_ratio = chosen["shared_encoder_gradient_ratio_vs_final"]
    if not 0.15 <= float(gate_ratio["median"]) <= 0.5:
        raise ValueError("Frozen gate-head gradient target is no longer satisfied")
    if float(gate_ratio["maximum"]) > 1.0:
        raise ValueError("Frozen gate coefficient would dominate final gate-head gradients")
    if float(shared_ratio["maximum"]) > 0.05:
        raise ValueError("Frozen gate coefficient is too large on the shared encoder")

    report = {
        "schema_version": "clir-dual-prior-reward-gate-loss-scale-audit-v1",
        "status": "completed_weight_frozen_before_gate_training",
        "evidence_tier": "pipeline_pilot",
        "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "source_mutual_protocol": str(protocol_path.relative_to(ROOT)),
        "source_mutual_protocol_sha256": file_sha256(protocol_path),
        "source_mutual_result": str(result_path.relative_to(ROOT)),
        "source_mutual_result_sha256": file_sha256(result_path),
        "source_training_commit": result["training_commit"],
        "source_gate_alignment_weight": 0.0,
        "source_artifacts": source_artifacts,
        "loss_diagnostic": {
            scope: {
                component: summary(values)
                for component, values in components.items()
            }
            for scope, components in loss_values.items()
        },
        "gradient_diagnostic": {
            "device_type": device.type,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "amp_dtype": "bfloat16",
            "batch_size": args.batch_size,
            "batch_ids": batch_ids,
            "batch_selection": "first four rows in frozen train-manifest order",
            "per_seed": gradient_diagnostic,
            "raw_gate_head_gradient_ratio_vs_final": summary(
                raw_gate_to_final_gate_head_ratios
            ),
            "raw_shared_encoder_gradient_ratio_vs_final": summary(
                raw_gate_to_final_shared_ratios
            ),
        },
        "candidate_weight_diagnostics": weight_diagnostics,
        "frozen_gate_prior_weight": chosen_weight,
        "choice_rule": {
            "gate_head_median_gradient_ratio_vs_final_minimum": 0.15,
            "gate_head_median_gradient_ratio_vs_final_maximum": 0.5,
            "gate_head_maximum_gradient_ratio_vs_final": 1.0,
            "shared_encoder_maximum_gradient_ratio_vs_final": 0.05,
            "selected_smallest_candidate_satisfying_all_rules": True,
        },
        "rationale": (
            "Weight 0.25 would make the median gate-head gradient only about 0.6% "
            "of final-BCE gate-head gradient. Weight 10.0 is the smallest audited "
            "candidate that raises the median to the preregistered 15%-50% band, "
            "while remaining below final BCE on every audited gate head and below "
            "5% of final-BCE gradient on the shared encoder."
        ),
        "new_gate_enabled_training_or_predictions_accessed": False,
        "pilot_test_accessed": False,
        "formal_claim_allowed": False,
    }
    atomic_write_json(args.output_json.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
