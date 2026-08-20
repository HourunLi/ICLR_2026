#!/usr/bin/env python3
"""Run a no-update real-feature gradient-routing audit for joint CLIR training."""

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
    clir_collate,
    move_batch_to_device,
)
from src.clir_joint_training import (  # noqa: E402
    reward_config_from_protocol,
    validate_joint_protocol,
)
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402
from src.consistency_localized_reward import build_reward_model  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/joint_training_pilot_v1/training_protocol_v1.json"
PARAMETER_GROUPS = {
    "encoder": ("input_encoder.",),
    "condition_fusion": (
        "condition_query.",
        "condition_key.",
        "condition_value.",
        "condition_fusion.",
        "feature_norm.",
    ),
    "token_reward_head": ("token_reward_head.",),
    "hallucination_head": ("hallucination_head.",),
    "progress_head": ("progress_head.",),
    "final_score_head": ("final_score_head.",),
    "key_prior_head": ("key_prior_head.",),
    "complete_prior_head": ("complete_prior_head.",),
    "projector": ("projector.",),
}
OBJECTIVES = {
    "final": "correctness",
    "consistency_total": "consistency",
    "localization_token_bce": "mechanism",
    "prior_key": "mechanism",
    "prior_complete": "mechanism",
    "prior_distill": "mechanism",
    "prior_gate": "mechanism",
}
REQUIRED_NONZERO_GROUPS = {
    "final": ("encoder", "token_reward_head", "final_score_head"),
    "consistency_total": ("encoder", "projector"),
    "localization_token_bce": ("encoder", "hallucination_head"),
    "prior_key": ("encoder", "key_prior_head"),
    "prior_complete": ("encoder", "complete_prior_head"),
    "prior_distill": ("encoder", "key_prior_head", "complete_prior_head"),
    "prior_gate": ("encoder", "token_reward_head"),
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _parameter_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _select_indices(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    ordinary = [
        index
        for index, row in enumerate(rows)
        if row.get("semantic_id") is None
        and row.get("token_hallucination_target") is None
        and row.get("key_prior_target") is None
    ]
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row.get("semantic_id") is not None and row.get("style_id") is not None:
            groups.setdefault(str(row["semantic_id"]), []).append(index)
    complete_groups = [indices for indices in groups.values() if len(indices) == 2]
    mechanism = [
        index
        for index, row in enumerate(rows)
        if row.get("token_hallucination_target") is not None
        and row.get("key_prior_target") is not None
        and row.get("complete_prior_target") is not None
    ]
    if len(ordinary) < 4 or len(complete_groups) < 2 or len(mechanism) < 4:
        raise ValueError("Joint manifest cannot form all three audit batches")
    return {
        "correctness": ordinary[:4],
        "consistency": complete_groups[0] + complete_groups[1],
        "mechanism": mechanism[:4],
    }


def _group_norms(
    names: Sequence[str], gradients: Sequence[torch.Tensor | None]
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for group, prefixes in PARAMETER_GROUPS.items():
        squared = 0.0
        tensors = 0
        elements = 0
        for name, gradient in zip(names, gradients):
            if gradient is None or not name.startswith(prefixes):
                continue
            value = gradient.detach().float()
            squared += float(value.square().sum().cpu())
            tensors += 1
            elements += int(value.numel())
        output[group] = {
            "l2_norm": math.sqrt(squared),
            "gradient_tensors": tensors,
            "gradient_elements": elements,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell", default="jall_full_retained")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_joint_protocol(protocol)
    config = reward_config_from_protocol(protocol, args.cell)
    output = (
        args.output.resolve()
        if args.output is not None
        else resolve(protocol["execution"]["gradient_audit_report"])
    )
    preflight = {
        "schema_version": "clir-joint-gradient-audit-preflight-v1",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "cell": args.cell,
        "device": args.device,
        "output": str(output),
        "objectives": OBJECTIVES,
        "code": git_state(ROOT),
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite gradient audit: {output}")
    if preflight["code"]["dirty"]:
        raise RuntimeError("The real gradient audit requires a clean committed worktree")
    train_spec = protocol["manifests"]["train"]
    train_path = resolve(train_spec["path"])
    if file_sha256(train_path) != train_spec["sha256"]:
        raise ValueError("Frozen joint train manifest hash drifted")
    integrity_spec = protocol["inputs"]["feature_integrity_report"]
    integrity_path = resolve(integrity_spec["path"])
    if file_sha256(integrity_path) != integrity_spec["sha256"]:
        raise ValueError("Feature-integrity report hash drifted")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if int(integrity.get("failure_count", -1)) != int(
        integrity_spec["required_failure_count"]
    ):
        raise ValueError("Feature-integrity report contains failures")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The frozen BF16 real-feature gradient audit requires CUDA")
    seed = int(protocol["matched_training"]["seeds"][0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = CLIRTrajectoryDataset(
        train_path,
        check_finite=False,
        require_correctness=True,
        load_condition=True,
        hidden_state_source="precomputed",
    )
    selected_indices = _select_indices(dataset.rows)
    model = build_reward_model(config).to(device).train()
    before_checksum = _parameter_checksum(model)
    names = [name for name, _ in model.named_parameters()]
    parameters = [parameter for _, parameter in model.named_parameters()]
    batches: dict[str, dict[str, Any]] = {}
    batch_rows: dict[str, list[str]] = {}
    for batch_name, indices in selected_indices.items():
        batch_rows[batch_name] = [str(dataset.rows[index]["id"]) for index in indices]
        batch = clir_collate([dataset[index] for index in indices])
        # Keep the three audit batches on CPU.  A flattened 33x3072 feature
        # trajectory is large, and retaining all three CUDA batches adds no
        # diagnostic value.
        batches[batch_name] = batch

    objective_reports: dict[str, Any] = {}
    for batch_name in ("correctness", "consistency", "mechanism"):
        batch = move_batch_to_device(batches[batch_name], device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        with autocast:
            _, losses = model.training_step(batch, prior_phase="joint")
        selected_objectives = [
            objective
            for objective, source_batch in OBJECTIVES.items()
            if source_batch == batch_name
        ]
        for objective_position, objective in enumerate(selected_objectives):
            if objective not in losses:
                raise ValueError(f"Audit objective {objective} was not produced")
            loss = losses[objective]
            retain_graph = objective_position < len(selected_objectives) - 1
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=retain_graph,
                allow_unused=True,
            )
            non_finite = [
                name
                for name, gradient in zip(names, gradients)
                if gradient is not None and not torch.isfinite(gradient).all()
            ]
            group_norms = _group_norms(names, gradients)
            required = REQUIRED_NONZERO_GROUPS[objective]
            required_nonzero = all(
                float(group_norms[group]["l2_norm"]) > 0.0 for group in required
            )
            detached_prior_target = True
            if objective == "prior_gate":
                detached_prior_target = (
                    float(group_norms["key_prior_head"]["l2_norm"]) == 0.0
                    and float(group_norms["complete_prior_head"]["l2_norm"]) == 0.0
                )
            value = float(loss.detach().float().cpu())
            objective_reports[objective] = {
                "batch": batch_name,
                "loss": value,
                "finite_loss": math.isfinite(value),
                "non_finite_gradient_parameters": non_finite,
                "gradient_groups": group_norms,
                "required_nonzero_groups": list(required),
                "required_nonzero_groups_passed": required_nonzero,
                "gate_fused_prior_target_detached": detached_prior_target,
                "passed": (
                    math.isfinite(value)
                    and value > 0.0
                    and not non_finite
                    and required_nonzero
                    and detached_prior_target
                ),
            }
        del losses
        del batch
        torch.cuda.empty_cache()

    after_checksum = _parameter_checksum(model)
    no_update = before_checksum == after_checksum
    passed = no_update and all(
        bool(report["passed"]) for report in objective_reports.values()
    )
    report = {
        **preflight,
        "schema_version": "clir-joint-gradient-audit-v1",
        "status": "passed" if passed else "failed",
        "seed": seed,
        "batch_size": 4,
        "batch_rows": batch_rows,
        "parameter_checksum_before": before_checksum,
        "parameter_checksum_after": after_checksum,
        "no_parameter_update": no_update,
        "objectives": objective_reports,
        "diagnostic_only_no_automatic_weight_tuning": True,
        "passed": passed,
    }
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Joint-training no-update gradient audit failed")


if __name__ == "__main__":
    main()
