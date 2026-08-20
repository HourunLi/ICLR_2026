#!/usr/bin/env python3
"""Audit no-update shared-gradient interactions for the frozen CLIR joint pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import (  # noqa: E402
    CLIRTrajectoryDataset,
    SemanticGroupBatchSampler,
    clir_collate,
    move_batch_to_device,
)
from src.clir_gradient_interaction import (  # noqa: E402
    classify_cross_stream_pressure,
    classify_same_batch_conflict,
    summarize_cosines,
    validate_gradient_interaction_protocol,
)
from src.clir_joint_training import (  # noqa: E402
    reward_config_from_protocol,
    validate_joint_protocol,
)
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402
from src.consistency_localized_reward import build_reward_model  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/joint_gradient_interaction_v1/audit_protocol_v1.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parameter_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_spec(spec: Mapping[str, Any], *, name: str) -> Path:
    path = resolve(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Frozen {name} is missing: {path}")
    observed = file_sha256(path)
    if observed != spec["sha256"]:
        raise ValueError(
            f"Frozen {name} hash drifted: expected {spec['sha256']}, got {observed}"
        )
    return path


def is_mechanism(row: Mapping[str, Any]) -> bool:
    return (
        row.get("token_hallucination_target") is not None
        and row.get("key_prior_target") is not None
        and row.get("complete_prior_target") is not None
    )


def is_consistency(row: Mapping[str, Any]) -> bool:
    return row.get("semantic_id") is not None and row.get("style_id") is not None


def pair_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    positive = 0
    negative = 0
    for left_position, left in enumerate(rows):
        for right in rows[left_position + 1 :]:
            semantic_equal = left["semantic_id"] == right["semantic_id"]
            style_equal = left["style_id"] == right["style_id"]
            positive += int(semantic_equal and not style_equal)
            negative += int(not semantic_equal and style_equal)
    return positive, negative


def audit_stream_structure(
    dataset: CLIRTrajectoryDataset, protocol: Mapping[str, Any]
) -> tuple[list[list[int]], dict[str, Any]]:
    stream = protocol["stream_structure"]
    sampler = SemanticGroupBatchSampler(
        dataset,
        batch_size=int(stream["batch_size"]),
        shuffle=True,
        drop_last=False,
        seed=int(stream["seed"]),
    )
    epoch_one_consistency: list[list[int]] = []
    epochs: list[dict[str, Any]] = []
    for epoch in range(1, int(stream["epochs"]) + 1):
        batches = list(iter(sampler))
        mechanism_active = 0
        consistency_active = 0
        overlap = 0
        for batch in batches:
            mechanism_indices = [index for index in batch if is_mechanism(dataset.rows[index])]
            consistency_indices = [index for index in batch if is_consistency(dataset.rows[index])]
            mechanism_active += int(bool(mechanism_indices))
            consistency_active += int(bool(consistency_indices))
            overlap += int(bool(mechanism_indices) and bool(consistency_indices))
            if epoch == 1 and consistency_indices:
                epoch_one_consistency.append(consistency_indices)
        epochs.append(
            {
                "epoch": epoch,
                "batches": len(batches),
                "mechanism_active_batches": mechanism_active,
                "consistency_active_batches": consistency_active,
                "mechanism_consistency_overlap_batches": overlap,
            }
        )
    expected_mechanism = stream["mechanism_active_batches"]
    expected_consistency = stream["consistency_active_batches"]
    expected_overlap = stream["mechanism_consistency_overlap_batches"]
    for position, epoch in enumerate(epochs):
        if epoch["batches"] != int(stream["batches_per_epoch"]):
            raise ValueError("Frozen stream batch count drifted")
        if epoch["mechanism_active_batches"] != int(expected_mechanism[position]):
            raise ValueError("Frozen mechanism-active batch count drifted")
        if epoch["consistency_active_batches"] != int(expected_consistency[position]):
            raise ValueError("Frozen consistency-active batch count drifted")
        if epoch["mechanism_consistency_overlap_batches"] != int(
            expected_overlap[position]
        ):
            raise ValueError("Frozen auxiliary overlap count drifted")
    return epoch_one_consistency, {"epochs": epochs, "passed": True}


def controlled_batch_report(
    main_dataset: CLIRTrajectoryDataset,
    mechanism_dataset: CLIRTrajectoryDataset,
    consistency_batches: Sequence[Sequence[int]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    controlled = protocol["controlled_batches"]
    mechanism_batch_size = int(controlled["mechanism"]["batch_size"])
    mechanism_batches = [
        list(range(start, min(start + mechanism_batch_size, len(mechanism_dataset))))
        for start in range(0, len(mechanism_dataset), mechanism_batch_size)
    ]
    consistency_ids: list[str] = []
    positive = 0
    negative = 0
    consistency_rows: list[list[str]] = []
    for indices in consistency_batches:
        rows = [main_dataset.rows[index] for index in indices]
        batch_positive, batch_negative = pair_counts(rows)
        positive += batch_positive
        negative += batch_negative
        ids = [str(row["id"]) for row in rows]
        consistency_ids.extend(ids)
        consistency_rows.append(ids)
    mechanism_ids = [str(row["id"]) for row in mechanism_dataset.rows]
    if len(mechanism_ids) != int(controlled["mechanism"]["rows"]):
        raise ValueError("Controlled mechanism row count drifted")
    if len(set(mechanism_ids)) != len(mechanism_ids):
        raise ValueError("Controlled mechanism rows are not unique")
    if len(mechanism_batches) != int(controlled["mechanism"]["batches"]):
        raise ValueError("Controlled mechanism batch count drifted")
    if len(consistency_ids) != int(controlled["consistency"]["rows"]):
        raise ValueError("Controlled consistency row count drifted")
    if len(set(consistency_ids)) != len(consistency_ids):
        raise ValueError("Controlled consistency rows are not unique")
    if len(consistency_batches) != int(controlled["consistency"]["active_batches"]):
        raise ValueError("Controlled consistency batch count drifted")
    if positive != int(controlled["consistency"]["positive_pairs"]):
        raise ValueError("Controlled positive-pair count drifted")
    if negative != int(controlled["consistency"]["negative_pairs"]):
        raise ValueError("Controlled negative-pair count drifted")
    return {
        "mechanism_batch_indices": mechanism_batches,
        "mechanism_batch_rows": [
            [str(mechanism_dataset.rows[index]["id"]) for index in indices]
            for indices in mechanism_batches
        ],
        "consistency_batch_indices": [list(indices) for indices in consistency_batches],
        "consistency_batch_rows": consistency_rows,
        "mechanism_rows": len(mechanism_ids),
        "consistency_rows": len(consistency_ids),
        "positive_pairs": positive,
        "negative_pairs": negative,
        "passed": True,
    }


def slice_first_dimension(
    values: Mapping[str, Any], *, row: int, batch_size: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in values.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            output[name] = value[row : row + 1]
        else:
            output[name] = value
    return output


def row_mean_mechanism_objectives(
    model: torch.nn.Module, batch: Mapping[str, Any]
) -> tuple[Mapping[str, torch.Tensor], dict[str, torch.Tensor]]:
    outputs = model.forward(
        batch["hidden_states"],
        mask=batch.get("mask"),
        condition_states=batch.get("condition_states"),
        condition_mask=batch.get("condition_mask"),
        condition_embedding=batch.get("condition_embedding"),
        condition_embedding_mask=batch.get("condition_embedding_mask"),
    )
    batch_size = int(batch["hidden_states"].shape[0])
    row_losses: list[Mapping[str, torch.Tensor]] = []
    for row in range(batch_size):
        row_outputs = slice_first_dimension(outputs, row=row, batch_size=batch_size)
        row_batch = slice_first_dimension(batch, row=row, batch_size=batch_size)
        row_losses.append(model.loss(row_outputs, row_batch, prior_phase="joint"))

    def mean(name: str) -> torch.Tensor:
        return torch.stack([losses[name] for losses in row_losses]).mean()

    objectives = {
        "final_mechanism": mean("final"),
        "hallucination": mean("localization_token_bce"),
        "prior_key": mean("prior_key"),
        "prior_complete": mean("prior_complete"),
        "prior_distill": 0.25 * mean("prior_distill"),
        "prior_gate": 10.0 * mean("prior_gate"),
    }
    return outputs, objectives


def shared_parameter_layout(
    model: torch.nn.Module, protocol: Mapping[str, Any]
) -> tuple[list[str], list[torch.nn.Parameter], dict[str, list[tuple[int, int]]], dict[str, Any]]:
    definitions = protocol["shared_parameter_groups"]
    shared_prefixes = tuple(definitions["shared_all"])
    names: list[str] = []
    parameters: list[torch.nn.Parameter] = []
    offsets: list[tuple[int, int]] = []
    position = 0
    for name, parameter in model.named_parameters():
        if not name.startswith(shared_prefixes):
            continue
        names.append(name)
        parameters.append(parameter)
        offsets.append((position, position + parameter.numel()))
        position += parameter.numel()
    groups: dict[str, list[tuple[int, int]]] = {}
    group_report: dict[str, Any] = {}
    for group, prefixes in definitions.items():
        group_ranges = [
            offsets[index]
            for index, name in enumerate(names)
            if name.startswith(tuple(prefixes))
        ]
        if not group_ranges:
            raise ValueError(f"Shared parameter group {group} is empty")
        groups[group] = group_ranges
        group_report[group] = {
            "parameter_tensors": len(group_ranges),
            "parameter_elements": sum(end - start for start, end in group_ranges),
        }
    if group_report["shared_all"]["parameter_elements"] != position:
        raise ValueError("shared_all does not cover every selected shared parameter")
    return names, parameters, groups, group_report


def flat_gradient(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    parts: list[torch.Tensor] = []
    for parameter, gradient in zip(parameters, gradients):
        value = torch.zeros_like(parameter, dtype=torch.float32) if gradient is None else gradient.detach().float()
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite shared gradient encountered")
        parts.append(value.reshape(-1))
    flattened = torch.cat(parts)
    if float(flattened.norm().detach().cpu()) <= 0.0:
        raise RuntimeError("Expected a non-zero shared gradient")
    return flattened


def range_dot(left: torch.Tensor, right: torch.Tensor, ranges: Sequence[tuple[int, int]]) -> torch.Tensor:
    value = left.new_zeros(())
    for start, end in ranges:
        value = value + torch.dot(left[start:end], right[start:end])
    return value


def vector_pair_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    groups: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group, ranges in groups.items():
        dot = range_dot(left, right, ranges)
        left_squared = range_dot(left, left, ranges)
        right_squared = range_dot(right, right, ranges)
        left_norm = torch.sqrt(left_squared.clamp_min(0.0))
        right_norm = torch.sqrt(right_squared.clamp_min(0.0))
        denom = left_norm * right_norm
        cosine = float((dot / denom).detach().cpu()) if float(denom.detach().cpu()) > 0.0 else None
        output[group] = {
            "cosine": cosine,
            "dot": float(dot.detach().cpu()),
            "left_norm": float(left_norm.detach().cpu()),
            "right_norm": float(right_norm.detach().cpu()),
        }
    return output


def vector_norms(
    vector: torch.Tensor,
    groups: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[str, float]:
    return {
        group: float(torch.sqrt(range_dot(vector, vector, ranges).clamp_min(0.0)).detach().cpu())
        for group, ranges in groups.items()
    }


def add_accumulator(
    accumulators: dict[str, torch.Tensor],
    counts: dict[str, int],
    name: str,
    gradient: torch.Tensor,
) -> None:
    if name not in accumulators:
        accumulators[name] = torch.zeros_like(gradient)
        counts[name] = 0
    accumulators[name].add_(gradient)
    counts[name] += 1


def effective_means(
    accumulators: Mapping[str, torch.Tensor], counts: Mapping[str, int]
) -> dict[str, torch.Tensor]:
    return {name: value / int(counts[name]) for name, value in accumulators.items()}


def load_state(
    model: torch.nn.Module,
    state: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if state["kind"] == "deterministic_initialization":
        checksum = parameter_checksum(model)
        if checksum != state["expected_parameter_checksum"]:
            raise ValueError("Deterministic initialization checksum drifted")
        return {"kind": state["kind"], "parameter_checksum": checksum}
    spec = protocol["inputs"][state["input"]]
    checkpoint_path = verify_spec(spec, name="JP epoch-5 checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != spec["schema_version"]:
        raise ValueError("JP checkpoint schema drifted")
    if int(checkpoint.get("completed_epoch", -1)) != int(spec["completed_epoch"]):
        raise ValueError("JP checkpoint epoch drifted")
    if checkpoint.get("code", {}).get("commit") != spec["training_code_commit"] or checkpoint.get("code", {}).get("dirty"):
        raise ValueError("JP checkpoint code provenance drifted")
    if checkpoint.get("experiment_protocol", {}).get("sha256") != spec["experiment_protocol_sha256"]:
        raise ValueError("JP checkpoint experiment protocol drifted")
    if checkpoint.get("data_state", {}).get("train_sha256") != spec["train_manifest_sha256"]:
        raise ValueError("JP checkpoint train manifest drifted")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return {
        "kind": state["kind"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": spec["sha256"],
        "completed_epoch": int(checkpoint["completed_epoch"]),
        "training_code": checkpoint["code"],
        "parameter_checksum": parameter_checksum(model),
    }


def audit_model_state(
    *,
    state: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    main_dataset: CLIRTrajectoryDataset,
    mechanism_dataset: CLIRTrajectoryDataset,
    batches: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed = int(protocol["model_states"][0]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = reward_config_from_protocol(parent_protocol, "jall_full_retained")
    model = build_reward_model(config)
    state_provenance = load_state(model, state, protocol)
    checksum_before = parameter_checksum(model)
    model = model.to(device).train()
    _, shared_parameters, groups, group_report = shared_parameter_layout(model, protocol)
    accumulators: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    batch_cosines: dict[str, list[float]] = {
        "hallucination__prior_total": [],
        "hallucination__prior_key": [],
        "hallucination__prior_complete": [],
        "hallucination__prior_distill": [],
        "hallucination__prior_gate": [],
        "final_mechanism__hallucination": [],
        "final_mechanism__prior_total": [],
        "final_consistency__consistency": [],
    }
    batch_records: dict[str, list[dict[str, Any]]] = {"mechanism": [], "consistency": []}
    component_sum_checks: list[dict[str, float]] = []
    mechanism_pairs = [
        ("hallucination", "prior_total"),
        ("hallucination", "prior_key"),
        ("hallucination", "prior_complete"),
        ("hallucination", "prior_distill"),
        ("hallucination", "prior_gate"),
        ("final_mechanism", "hallucination"),
        ("final_mechanism", "prior_total"),
    ]

    for position, indices in enumerate(batches["mechanism_batch_indices"], start=1):
        cpu_batch = clir_collate([mechanism_dataset[index] for index in indices])
        batch = move_batch_to_device(cpu_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs, objectives = row_mean_mechanism_objectives(model, batch)
        objective_order = [
            "final_mechanism",
            "hallucination",
            "prior_key",
            "prior_complete",
            "prior_distill",
            "prior_gate",
        ]
        gradients: dict[str, torch.Tensor] = {}
        for objective_position, objective in enumerate(objective_order):
            gradients[objective] = flat_gradient(
                objectives[objective],
                shared_parameters,
                retain_graph=objective_position < len(objective_order) - 1,
            )
        gradients["prior_total"] = (
            gradients["prior_key"]
            + gradients["prior_complete"]
            + gradients["prior_distill"]
            + gradients["prior_gate"]
        )
        prior_total_loss = (
            objectives["prior_key"]
            + objectives["prior_complete"]
            + objectives["prior_distill"]
            + objectives["prior_gate"]
        )
        component_sum_checks.append(
            {
                "prior_total_loss": float(prior_total_loss.detach().float().cpu()),
                "component_gradient_norm": float(gradients["prior_total"].norm().detach().cpu()),
            }
        )
        for objective, gradient in gradients.items():
            add_accumulator(accumulators, counts, objective, gradient)
        pair_values: dict[str, float] = {}
        for left, right in mechanism_pairs:
            key = f"{left}__{right}"
            metrics = vector_pair_metrics(gradients[left], gradients[right], {"shared_all": groups["shared_all"]})
            cosine = metrics["shared_all"]["cosine"]
            if cosine is None or not math.isfinite(cosine):
                raise RuntimeError(f"Undefined same-batch cosine for {key}")
            batch_cosines[key].append(float(cosine))
            pair_values[key] = float(cosine)
        batch_records["mechanism"].append(
            {
                "position": position,
                "rows": batches["mechanism_batch_rows"][position - 1],
                "losses": {
                    **{
                        name: float(value.detach().float().cpu())
                        for name, value in objectives.items()
                    },
                    "prior_total": float(prior_total_loss.detach().float().cpu()),
                },
                "shared_cosines": pair_values,
            }
        )
        del gradients, objectives, outputs, batch, cpu_batch
        torch.cuda.empty_cache()
        print(f"[{state['name']}] mechanism {position}/{len(batches['mechanism_batch_indices'])}", flush=True)

    for position, indices in enumerate(batches["consistency_batch_indices"], start=1):
        cpu_batch = clir_collate([main_dataset[index] for index in indices])
        batch = move_batch_to_device(cpu_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, losses = model.training_step(batch, prior_phase="joint")
        final_gradient = flat_gradient(losses["final"], shared_parameters, retain_graph=True)
        consistency_gradient = flat_gradient(
            losses["consistency_total"], shared_parameters, retain_graph=False
        )
        add_accumulator(accumulators, counts, "final_consistency", final_gradient)
        add_accumulator(accumulators, counts, "consistency", consistency_gradient)
        metrics = vector_pair_metrics(
            final_gradient,
            consistency_gradient,
            {"shared_all": groups["shared_all"]},
        )
        cosine = metrics["shared_all"]["cosine"]
        if cosine is None or not math.isfinite(cosine):
            raise RuntimeError("Undefined consistency/final same-batch cosine")
        batch_cosines["final_consistency__consistency"].append(float(cosine))
        batch_records["consistency"].append(
            {
                "position": position,
                "rows": batches["consistency_batch_rows"][position - 1],
                "losses": {
                    "final_consistency": float(losses["final"].detach().float().cpu()),
                    "consistency": float(losses["consistency_total"].detach().float().cpu()),
                },
                "shared_cosines": {
                    "final_consistency__consistency": float(cosine)
                },
            }
        )
        del final_gradient, consistency_gradient, losses, batch, cpu_batch
        torch.cuda.empty_cache()
        print(f"[{state['name']}] consistency {position}/{len(batches['consistency_batch_indices'])}", flush=True)

    means = effective_means(accumulators, counts)
    aggregate_norms = {
        objective: vector_norms(gradient, groups)
        for objective, gradient in means.items()
    }
    aggregate_pairs: dict[str, Any] = {}
    for left, right in protocol["same_batch_pairs"] + protocol["cross_stream_pairs"]:
        key = f"{left}__{right}"
        aggregate_pairs[key] = vector_pair_metrics(means[left], means[right], groups)
    threshold = float(protocol["decision_rules"]["negative_cosine_threshold"])
    batch_summaries = {
        name: summarize_cosines(values, threshold=threshold)
        for name, values in batch_cosines.items()
    }
    no_grad_buffers = all(parameter.grad is None for parameter in model.parameters())
    del means, accumulators
    torch.cuda.empty_cache()
    checksum_after = parameter_checksum(model)
    no_update = checksum_before == checksum_after
    del model
    torch.cuda.empty_cache()
    if not no_update or not no_grad_buffers:
        raise RuntimeError("No-update gradient audit mutated model state or .grad buffers")
    return {
        "state": str(state["name"]),
        "provenance": state_provenance,
        "parameter_checksum_before": checksum_before,
        "parameter_checksum_after": checksum_after,
        "no_parameter_update": no_update,
        "optimizer_grad_buffers_absent": no_grad_buffers,
        "shared_parameter_groups": group_report,
        "objective_batch_counts": counts,
        "aggregate_gradient_norms": aggregate_norms,
        "aggregate_pairs": aggregate_pairs,
        "batch_shared_cosines": batch_summaries,
        "batch_records": batch_records,
        "prior_component_sum_checks": component_sum_checks,
        "passed": True,
    }


def classify_results(
    state_reports: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rules = protocol["decision_rules"]
    threshold = float(rules["negative_cosine_threshold"])
    fraction = float(rules["same_batch_negative_fraction_minimum"])
    same_batch: dict[str, Any] = {}
    for left, right in protocol["same_batch_pairs"]:
        pair = f"{left}__{right}"
        metrics = {
            state: {
                "aggregate_shared_cosine": report["aggregate_pairs"][pair]["shared_all"]["cosine"],
                "batch_shared_cosines": report["batch_shared_cosines"][pair],
            }
            for state, report in state_reports.items()
        }
        same_batch[pair] = {
            "states": metrics,
            "classification": classify_same_batch_conflict(
                metrics,
                threshold=threshold,
                negative_fraction_minimum=fraction,
            ),
        }
    cross_stream: dict[str, Any] = {}
    for left, right in protocol["cross_stream_pairs"]:
        pair = f"{left}__{right}"
        cosines = {
            state: float(report["aggregate_pairs"][pair]["shared_all"]["cosine"])
            for state, report in state_reports.items()
        }
        cross_stream[pair] = {
            "state_shared_cosines": cosines,
            "classification": classify_cross_stream_pressure(
                cosines, threshold=threshold
            ),
            "simultaneous_batch_conflict": False,
        }
    return {
        "same_batch": same_batch,
        "cross_stream": cross_stream,
        "threshold": threshold,
        "same_batch_negative_fraction_minimum": fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_gradient_interaction_protocol(protocol)
    output = resolve(protocol["execution"]["output"])
    code = git_state(ROOT)
    preflight = {
        "schema_version": "clir-joint-gradient-interaction-preflight-v1",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "device": args.device,
        "output": str(output),
        "model_states": [state["name"] for state in protocol["model_states"]],
        "code": code,
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite gradient-interaction audit: {output}")
    if protocol["execution"]["clean_committed_worktree_required"] and code["dirty"]:
        raise RuntimeError("Gradient-interaction audit requires clean committed code")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frozen BF16 gradient-interaction audit requires CUDA")

    for name, spec in protocol["parent_artifacts"].items():
        verify_spec(spec, name=f"parent artifact {name}")
    for name, spec in protocol["inputs"].items():
        verify_spec(spec, name=f"input {name}")
    integrity = json.loads(
        resolve(protocol["inputs"]["feature_integrity_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if int(integrity.get("failure_count", -1)) != int(
        protocol["inputs"]["feature_integrity_report"]["required_failure_count"]
    ):
        raise ValueError("Frozen feature-integrity report contains failures")

    parent_path = resolve(protocol["parent_artifacts"]["joint_training_protocol"]["path"])
    parent_protocol = json.loads(parent_path.read_text(encoding="utf-8"))
    validate_joint_protocol(parent_protocol)
    main_dataset = CLIRTrajectoryDataset(
        resolve(protocol["inputs"]["train_manifest"]["path"]),
        check_finite=False,
        require_correctness=True,
        load_condition=True,
        hidden_state_source="precomputed",
    )
    mechanism_dataset = CLIRTrajectoryDataset(
        resolve(protocol["inputs"]["mechanism_manifest"]["path"]),
        check_finite=False,
        require_correctness=True,
        load_condition=True,
        hidden_state_source="precomputed",
    )
    consistency_batches, stream_report = audit_stream_structure(main_dataset, protocol)
    batches = controlled_batch_report(
        main_dataset,
        mechanism_dataset,
        consistency_batches,
        protocol,
    )

    state_reports: dict[str, Any] = {}
    for state in protocol["model_states"]:
        print(f"Starting model state {state['name']}", flush=True)
        state_reports[str(state["name"])] = audit_model_state(
            state=state,
            protocol=protocol,
            parent_protocol=parent_protocol,
            main_dataset=main_dataset,
            mechanism_dataset=mechanism_dataset,
            batches=batches,
            device=device,
        )
    classifications = classify_results(state_reports, protocol)
    passed = stream_report["passed"] and batches["passed"] and all(
        report["passed"] for report in state_reports.values()
    )
    result = {
        **preflight,
        "schema_version": "clir-joint-gradient-interaction-result-v1",
        "status": "completed_no_update_diagnostic" if passed else "failed",
        "evidence_tier": protocol["evidence_tier"],
        "stream_structure": stream_report,
        "controlled_batches": {
            name: value for name, value in batches.items() if not name.endswith("_indices")
        },
        "model_state_results": state_reports,
        "classifications": classifications,
        "measurement_semantics": protocol["measurement_semantics"],
        "no_parameter_update": all(
            report["no_parameter_update"] for report in state_reports.values()
        ),
        "automatic_repair_performed": False,
        "automatic_weight_tuning_performed": False,
        "automatic_sampler_change_performed": False,
        "additional_training_performed": False,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
        "passed": passed,
    }
    atomic_write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Gradient-interaction audit failed")


if __name__ == "__main__":
    main()
