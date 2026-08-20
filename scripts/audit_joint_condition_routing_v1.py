#!/usr/bin/env python3
"""Audit the targeted hallucination-to-condition stop-gradient without updates."""

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

from src.clir_condition_routing import (  # noqa: E402
    BLOCKED_PARAMETER_PREFIXES,
    FORWARD_INVARIANT_FIELDS,
    HALLUCINATION_REQUIRED_NONZERO_PREFIXES,
    INVARIANT_OBJECTIVE_WEIGHTS,
    validate_condition_routing_protocol,
)
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


DEFAULT_PROTOCOL = ROOT / "configs/joint_condition_routing_v1/audit_protocol_v1.json"
OBJECTIVE_ORDER = (
    "hallucination",
    "final",
    "prior_key",
    "prior_complete",
    "prior_distill",
    "prior_gate",
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def parameter_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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


def row_mean_objectives(
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

    return outputs, {
        "hallucination": mean("localization_token_bce"),
        "final": mean("final"),
        "prior_key": mean("prior_key"),
        "prior_complete": mean("prior_complete"),
        "prior_distill": 0.25 * mean("prior_distill"),
        "prior_gate": 10.0 * mean("prior_gate"),
    }


def collect_gradients(
    objectives: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> dict[str, list[torch.Tensor | None]]:
    collected: dict[str, list[torch.Tensor | None]] = {}
    for position, objective in enumerate(OBJECTIVE_ORDER):
        gradients = torch.autograd.grad(
            objectives[objective],
            parameters,
            retain_graph=position < len(OBJECTIVE_ORDER) - 1,
            allow_unused=True,
        )
        values: list[torch.Tensor | None] = []
        for gradient in gradients:
            if gradient is None:
                values.append(None)
                continue
            value = gradient.detach().float()
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite gradient in objective {objective}")
            values.append(value.cpu())
        collected[objective] = values
    return collected


def gradient_norm(
    names: Sequence[str],
    gradients: Sequence[torch.Tensor | None],
    *,
    include_prefixes: tuple[str, ...] | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> float:
    squared = 0.0
    for name, gradient in zip(names, gradients):
        if gradient is None:
            continue
        if include_prefixes is not None and not name.startswith(include_prefixes):
            continue
        if exclude_prefixes and name.startswith(exclude_prefixes):
            continue
        squared += float(gradient.square().sum())
    return math.sqrt(squared)


def gradient_difference(
    names: Sequence[str],
    reference: Sequence[torch.Tensor | None],
    candidate: Sequence[torch.Tensor | None],
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, float]:
    reference_squared = 0.0
    candidate_squared = 0.0
    difference_squared = 0.0
    max_abs = 0.0
    for name, left, right in zip(names, reference, candidate):
        if exclude_prefixes and name.startswith(exclude_prefixes):
            continue
        if left is None and right is None:
            continue
        if left is None:
            assert right is not None
            left = torch.zeros_like(right)
        if right is None:
            right = torch.zeros_like(left)
        difference = right - left
        reference_squared += float(left.square().sum())
        candidate_squared += float(right.square().sum())
        difference_squared += float(difference.square().sum())
        if difference.numel():
            max_abs = max(max_abs, float(difference.abs().max()))
    reference_norm = math.sqrt(reference_squared)
    candidate_norm = math.sqrt(candidate_squared)
    difference_norm = math.sqrt(difference_squared)
    return {
        "reference_l2_norm": reference_norm,
        "candidate_l2_norm": candidate_norm,
        "difference_l2_norm": difference_norm,
        "relative_l2_difference": difference_norm / max(reference_norm, 1e-12),
        "max_abs_difference": max_abs,
    }


def max_abs_tensor_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(
            f"Forward tensor shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    if not left.numel():
        return 0.0
    return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max())


def load_model_state(
    model: torch.nn.Module,
    state: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if state["kind"] == "deterministic_initialization":
        checksum = parameter_checksum(model)
        if checksum != state["expected_parameter_checksum"]:
            raise ValueError("Deterministic initialization checksum drifted")
        return {"kind": state["kind"], "parameter_checksum": checksum}
    checkpoint_spec = protocol["inputs"][str(state["input"])]
    checkpoint_path = verify_spec(checkpoint_spec, name="JP epoch-5 checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != checkpoint_spec["schema_version"]:
        raise ValueError("JP checkpoint schema drifted")
    if int(checkpoint.get("completed_epoch", -1)) != int(
        checkpoint_spec["completed_epoch"]
    ):
        raise ValueError("JP checkpoint epoch drifted")
    if checkpoint.get("code", {}).get("commit") != checkpoint_spec["training_code_commit"]:
        raise ValueError("JP checkpoint training commit drifted")
    if checkpoint.get("code", {}).get("dirty"):
        raise ValueError("JP checkpoint was produced by dirty code")
    if checkpoint.get("experiment_protocol", {}).get("sha256") != checkpoint_spec[
        "experiment_protocol_sha256"
    ]:
        raise ValueError("JP checkpoint experiment protocol drifted")
    if checkpoint.get("data_state", {}).get("train_sha256") != checkpoint_spec[
        "train_manifest_sha256"
    ]:
        raise ValueError("JP checkpoint train manifest drifted")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return {
        "kind": state["kind"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_spec["sha256"],
        "completed_epoch": int(checkpoint["completed_epoch"]),
        "training_code": checkpoint["code"],
        "parameter_checksum": parameter_checksum(model),
    }


def audit_model_state(
    *,
    state: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    dataset: CLIRTrajectoryDataset,
    device: torch.device,
) -> dict[str, Any]:
    seed = int(protocol["model_states"][0]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = reward_config_from_protocol(parent_protocol, "jph_prior_plus_hallucination")
    if config.hallucination_condition_stop_gradient:
        raise ValueError("Frozen JPH baseline unexpectedly enables condition routing")
    if (
        config.prior_distill_weight != 0.25
        or config.gate_prior_weight != 10.0
        or config.key_prior_weight != 1.0
        or config.complete_prior_weight != 1.0
    ):
        raise ValueError("Original dual-prior weights drifted")
    model = build_reward_model(config)
    provenance = load_model_state(model, state, protocol)
    checksum_before = parameter_checksum(model)
    model = model.to(device).eval()
    names = [name for name, _ in model.named_parameters()]
    parameters = [parameter for _, parameter in model.named_parameters()]
    rules = protocol["decision_rules"]
    batch_size = int(protocol["controlled_batches"]["batch_size"])
    records: list[dict[str, Any]] = []

    for batch_position, start in enumerate(range(0, len(dataset), batch_size), start=1):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        rows = [str(dataset.rows[index]["id"]) for index in indices]
        cpu_batch = clir_collate([dataset[index] for index in indices])
        batch = move_batch_to_device(cpu_batch, device)

        model.config.hallucination_condition_stop_gradient = False
        with torch.autocast("cuda", dtype=torch.bfloat16):
            baseline_outputs, baseline_objectives = row_mean_objectives(model, batch)
        baseline_forward = {
            name: baseline_outputs[name].detach().float().cpu()
            for name in FORWARD_INVARIANT_FIELDS
        }
        baseline_losses = {
            name: float(value.detach().float().cpu())
            for name, value in baseline_objectives.items()
        }
        baseline_gradients = collect_gradients(baseline_objectives, parameters)
        del baseline_outputs, baseline_objectives
        torch.cuda.empty_cache()

        model.config.hallucination_condition_stop_gradient = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            routed_outputs, routed_objectives = row_mean_objectives(model, batch)
        forward_differences = {
            name: max_abs_tensor_difference(baseline_forward[name], routed_outputs[name])
            for name in FORWARD_INVARIANT_FIELDS
        }
        loss_differences = {
            name: abs(
                baseline_losses[name] - float(value.detach().float().cpu())
            )
            for name, value in routed_objectives.items()
        }
        routed_gradients = collect_gradients(routed_objectives, parameters)
        del routed_outputs, routed_objectives
        torch.cuda.empty_cache()

        baseline_blocked_norm = gradient_norm(
            names,
            baseline_gradients["hallucination"],
            include_prefixes=BLOCKED_PARAMETER_PREFIXES,
        )
        routed_blocked_norm = gradient_norm(
            names,
            routed_gradients["hallucination"],
            include_prefixes=BLOCKED_PARAMETER_PREFIXES,
        )
        hallucination_nonblocked_difference = gradient_difference(
            names,
            baseline_gradients["hallucination"],
            routed_gradients["hallucination"],
            exclude_prefixes=BLOCKED_PARAMETER_PREFIXES,
        )
        required_route_norms = {
            name: gradient_norm(
                names,
                routed_gradients["hallucination"],
                include_prefixes=prefixes,
            )
            for name, prefixes in HALLUCINATION_REQUIRED_NONZERO_PREFIXES.items()
        }
        invariant_gradients = {
            objective: gradient_difference(
                names,
                baseline_gradients[objective],
                routed_gradients[objective],
            )
            for objective in INVARIANT_OBJECTIVE_WEIGHTS
        }
        original_prior_routes = {
            "mutual_key_head_l2": gradient_norm(
                names,
                routed_gradients["prior_distill"],
                include_prefixes=("key_prior_head.",),
            ),
            "mutual_complete_head_l2": gradient_norm(
                names,
                routed_gradients["prior_distill"],
                include_prefixes=("complete_prior_head.",),
            ),
            "gate_key_head_l2": gradient_norm(
                names,
                routed_gradients["prior_gate"],
                include_prefixes=("key_prior_head.",),
            ),
            "gate_complete_head_l2": gradient_norm(
                names,
                routed_gradients["prior_gate"],
                include_prefixes=("complete_prior_head.",),
            ),
            "gate_reward_head_l2": gradient_norm(
                names,
                routed_gradients["prior_gate"],
                include_prefixes=("token_reward_head.",),
            ),
        }

        gradient_abs_tolerance = float(
            rules["gradient_max_abs_difference_tolerance"]
        )
        gradient_relative_tolerance = float(
            rules["gradient_relative_l2_difference_tolerance"]
        )
        batch_passed = (
            max(forward_differences.values())
            <= float(rules["forward_max_abs_tolerance"])
            and max(loss_differences.values())
            <= float(rules["objective_loss_max_abs_tolerance"])
            and baseline_blocked_norm > 0.0
            and routed_blocked_norm
            <= float(rules["blocked_hallucination_gradient_l2_tolerance"])
            and all(value > 0.0 for value in required_route_norms.values())
            and hallucination_nonblocked_difference["max_abs_difference"]
            <= gradient_abs_tolerance
            and hallucination_nonblocked_difference["relative_l2_difference"]
            <= gradient_relative_tolerance
            and all(
                metrics["max_abs_difference"] <= gradient_abs_tolerance
                and metrics["relative_l2_difference"]
                <= gradient_relative_tolerance
                for metrics in invariant_gradients.values()
            )
            and original_prior_routes["mutual_key_head_l2"] > 0.0
            and original_prior_routes["mutual_complete_head_l2"] > 0.0
            and original_prior_routes["gate_key_head_l2"] == 0.0
            and original_prior_routes["gate_complete_head_l2"] == 0.0
            and original_prior_routes["gate_reward_head_l2"] > 0.0
        )
        records.append(
            {
                "batch": batch_position,
                "rows": rows,
                "forward_max_abs_differences": forward_differences,
                "objective_loss_abs_differences": loss_differences,
                "hallucination": {
                    "baseline_blocked_condition_l2": baseline_blocked_norm,
                    "routed_blocked_condition_l2": routed_blocked_norm,
                    "nonblocked_gradient_difference": hallucination_nonblocked_difference,
                    "required_routed_gradient_norms": required_route_norms,
                },
                "invariant_objective_gradient_differences": invariant_gradients,
                "original_prior_routes": original_prior_routes,
                "passed": batch_passed,
            }
        )
        del baseline_gradients, routed_gradients, baseline_forward, batch, cpu_batch
        torch.cuda.empty_cache()
        print(
            f"[{state['name']}] routing batch {batch_position}/{int(protocol['controlled_batches']['batches'])}",
            flush=True,
        )

    checksum_after = parameter_checksum(model)
    no_update = checksum_before == checksum_after
    no_grad_buffers = all(parameter.grad is None for parameter in model.parameters())
    expected_batches = int(protocol["controlled_batches"]["batches"])
    expected_rows = int(protocol["controlled_batches"]["rows"])
    passed = (
        len(records) == expected_batches
        and sum(len(record["rows"]) for record in records) == expected_rows
        and all(record["passed"] for record in records)
        and no_update
        and no_grad_buffers
    )
    summary = {
        "maximum_forward_abs_difference": max(
            max(record["forward_max_abs_differences"].values()) for record in records
        ),
        "maximum_objective_loss_abs_difference": max(
            max(record["objective_loss_abs_differences"].values()) for record in records
        ),
        "minimum_baseline_blocked_condition_l2": min(
            record["hallucination"]["baseline_blocked_condition_l2"]
            for record in records
        ),
        "maximum_routed_blocked_condition_l2": max(
            record["hallucination"]["routed_blocked_condition_l2"]
            for record in records
        ),
        "maximum_hallucination_nonblocked_relative_l2_difference": max(
            record["hallucination"]["nonblocked_gradient_difference"][
                "relative_l2_difference"
            ]
            for record in records
        ),
        "maximum_hallucination_nonblocked_max_abs_difference": max(
            record["hallucination"]["nonblocked_gradient_difference"][
                "max_abs_difference"
            ]
            for record in records
        ),
        "minimum_required_hallucination_route_norms": {
            route: min(
                record["hallucination"]["required_routed_gradient_norms"][route]
                for record in records
            )
            for route in HALLUCINATION_REQUIRED_NONZERO_PREFIXES
        },
        "maximum_invariant_objective_relative_l2_differences": {
            objective: max(
                record["invariant_objective_gradient_differences"][objective][
                    "relative_l2_difference"
                ]
                for record in records
            )
            for objective in INVARIANT_OBJECTIVE_WEIGHTS
        },
        "maximum_invariant_objective_max_abs_differences": {
            objective: max(
                record["invariant_objective_gradient_differences"][objective][
                    "max_abs_difference"
                ]
                for record in records
            )
            for objective in INVARIANT_OBJECTIVE_WEIGHTS
        },
        "maximum_gate_gradient_to_key_head_l2": max(
            record["original_prior_routes"]["gate_key_head_l2"]
            for record in records
        ),
        "maximum_gate_gradient_to_complete_head_l2": max(
            record["original_prior_routes"]["gate_complete_head_l2"]
            for record in records
        ),
        "minimum_mutual_gradient_to_key_head_l2": min(
            record["original_prior_routes"]["mutual_key_head_l2"]
            for record in records
        ),
        "minimum_mutual_gradient_to_complete_head_l2": min(
            record["original_prior_routes"]["mutual_complete_head_l2"]
            for record in records
        ),
        "minimum_gate_gradient_to_reward_head_l2": min(
            record["original_prior_routes"]["gate_reward_head_l2"]
            for record in records
        ),
    }
    del model
    torch.cuda.empty_cache()
    return {
        "state": state["name"],
        "provenance": provenance,
        "parameter_checksum_before": checksum_before,
        "parameter_checksum_after": checksum_after,
        "no_parameter_update": no_update,
        "optimizer_grad_buffers_absent": no_grad_buffers,
        "controlled_rows": expected_rows,
        "controlled_batches": expected_batches,
        "summary": summary,
        "batch_records": records,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_condition_routing_protocol(protocol)
    output = resolve(protocol["execution"]["output"])
    code = git_state(ROOT)
    preflight = {
        "schema_version": "clir-joint-condition-routing-preflight-v1",
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
        raise FileExistsError(f"Refusing to overwrite condition-routing audit: {output}")
    if protocol["execution"]["clean_committed_worktree_required"] and code["dirty"]:
        raise RuntimeError("Condition-routing audit requires clean committed code")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frozen BF16 condition-routing audit requires CUDA")

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

    parent_path = resolve(protocol["parent_artifacts"]["jph_protocol"]["path"])
    parent_protocol = json.loads(parent_path.read_text(encoding="utf-8"))
    validate_joint_protocol(parent_protocol)
    dataset = CLIRTrajectoryDataset(
        resolve(protocol["inputs"]["mechanism_manifest"]["path"]),
        check_finite=False,
        require_correctness=True,
        load_condition=True,
        hidden_state_source="precomputed",
    )
    if len(dataset) != int(protocol["controlled_batches"]["rows"]):
        raise ValueError("Frozen condition-routing row count drifted")
    if len({str(row["id"]) for row in dataset.rows}) != len(dataset):
        raise ValueError("Condition-routing rows are not unique")
    for row in dataset.rows:
        if not all(
            field in row
            for field in (
                "token_hallucination_target",
                "key_prior_target",
                "complete_prior_target",
            )
        ):
            raise ValueError("Condition-routing row lacks frozen mechanism supervision")

    state_results: dict[str, Any] = {}
    for state in protocol["model_states"]:
        print(f"Starting model state {state['name']}", flush=True)
        state_results[str(state["name"])] = audit_model_state(
            state=state,
            protocol=protocol,
            parent_protocol=parent_protocol,
            dataset=dataset,
            device=device,
        )
    passed = all(result["passed"] for result in state_results.values())
    result = {
        **preflight,
        "schema_version": "clir-joint-condition-routing-result-v1",
        "status": "completed_no_update_routing_audit" if passed else "failed",
        "evidence_tier": protocol["evidence_tier"],
        "route_contract": protocol["route_contract"],
        "decision_rules": protocol["decision_rules"],
        "controlled_batches": protocol["controlled_batches"],
        "model_state_results": state_results,
        "no_parameter_update": all(
            state["no_parameter_update"] for state in state_results.values()
        ),
        "optimizer_grad_buffers_absent": all(
            state["optimizer_grad_buffers_absent"] for state in state_results.values()
        ),
        "original_dual_prior_preserved": passed,
        "additional_training_performed": False,
        "automatic_loss_weight_tuning_performed": False,
        "automatic_sampler_change_performed": False,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
        "next_training_requires_user_approval": True,
        "passed": passed,
    }
    atomic_write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Condition-routing audit failed")


if __name__ == "__main__":
    main()
