#!/usr/bin/env python3
"""Summarize the frozen three-seed dual-prior reward-gate comparison."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import atomic_write_json, file_sha256  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/dual_prior_reward_gate_v1/training_protocol_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/dual_prior_reward_gate_v1/training_result_v1.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def unit_ap(result: Mapping[str, Any], head: str) -> float:
    value = result["dev_metrics"][head]["unit"]["average_precision_micro"]
    if value is None:
        raise ValueError(f"{result['cell']}/seed {result['seed']}: undefined unit AP")
    return float(value)


def position_unit_ap(result: Mapping[str, Any], head: str) -> float:
    value = result["position_baselines"][head]["unit"]["average_precision_micro"]
    if value is None:
        raise ValueError("Position-only unit AP is undefined")
    return float(value)


def correctness_auc(result: Mapping[str, Any]) -> float:
    value = result["correctness"]["dev"]["roc_auc"]
    if value is None:
        raise ValueError("Correctness AUROC is undefined on the frozen dev split")
    return float(value)


def finite_correlation(result: Mapping[str, Any]) -> float:
    value = result["head_separation"]["dev"]["pearson_correlation"]
    return float(value) if value is not None else float("inf")


def collaboration_mse(result: Mapping[str, Any]) -> float:
    value = float(
        result["prior_collaboration"]["dev"]["symmetric_attention_mse"]
    )
    if value <= 0.0:
        raise ValueError("Symmetric prior-attention MSE must be positive")
    return value


def gate_metric(result: Mapping[str, Any], name: str) -> float:
    value = float(result["gate_integration"]["dev"][name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite gate metric {name}")
    return value


def nested_gate_metric(
    result: Mapping[str, Any], group: str, name: str
) -> float:
    value = float(result["gate_integration"]["dev"][group][name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite gate metric {group}/{name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-dual-prior-reward-gate-integration-training-protocol-v1"
    ):
        raise ValueError("Unexpected reward-gate protocol schema")
    protocol_sha = file_sha256(protocol_path)
    output_root = resolve(protocol["execution"]["output_root"])
    seeds = [int(value) for value in protocol["matched_training"]["seeds"]]
    control_name = str(protocol["comparison"]["control_cell"])
    gate_name = str(protocol["comparison"]["gate_cell"])
    cells = [control_name, gate_name]
    gate_weight = float(protocol["method"]["weight"])
    mutual_weight = float(protocol["protected_mutual_method"]["weight"])

    results: dict[int, dict[str, dict[str, Any]]] = {}
    result_hashes: dict[str, str] = {}
    code_commits: set[str] = set()
    for seed in seeds:
        results[seed] = {}
        for cell in cells:
            path = output_root / f"seed_{seed}" / cell / "cell_result.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen matrix result: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("schema_version") != "clir-dual-prior-cell-result-v1":
                raise ValueError(f"Unexpected cell result schema: {path}")
            if result.get("cell") != cell or int(result.get("seed", -1)) != seed:
                raise ValueError(f"Cell result identity drifted: {path}")
            if result.get("protocol_sha256") != protocol_sha:
                raise ValueError(f"Cell result protocol hash drifted: {path}")
            code = result.get("code")
            if not isinstance(code, Mapping) or bool(code.get("dirty")):
                raise ValueError(f"Cell was not produced from a clean worktree: {path}")
            resolved = result.get("resolved_loss_weights")
            if not isinstance(resolved, Mapping):
                raise ValueError(f"Cell lacks resolved loss weights: {path}")
            expected_gate_weight = 0.0 if cell == control_name else gate_weight
            protected_weights = {
                "prior": 1.0,
                "key_prior": 1.0,
                "complete_prior": 1.0,
                "prior_distill": mutual_weight,
                "gate_prior": expected_gate_weight,
                "reconstruction": 0.0,
            }
            if any(
                float(resolved.get(name, -1.0)) != expected
                for name, expected in protected_weights.items()
            ):
                raise ValueError(f"Protected reward-gate weights drifted: {path}")
            if "gate_integration" not in result:
                raise ValueError(f"Cell lacks gate-integration diagnostics: {path}")
            code_commits.add(str(code.get("commit")))
            results[seed][cell] = result
            result_hashes[f"seed_{seed}/{cell}"] = file_sha256(path)
    if len(code_commits) != 1:
        raise ValueError(f"Matched cells used different commits: {sorted(code_commits)}")

    guards = protocol["evaluation"]["selection_guards"]
    seed_reports: dict[str, Any] = {}
    passing_counter: Counter[str] = Counter()
    aggregate_values: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        control = results[seed][control_name]
        gate = results[seed][gate_name]
        control_gate_mse = gate_metric(control, "training_objective_mse")
        gate_gate_mse = gate_metric(gate, "training_objective_mse")
        control_collaboration_mse = collaboration_mse(control)
        gate_collaboration_mse = collaboration_mse(gate)
        if position_unit_ap(control, "key") != position_unit_ap(gate, "key"):
            raise ValueError("Matched cells produced different key position baselines")
        if position_unit_ap(control, "complete") != position_unit_ap(
            gate, "complete"
        ):
            raise ValueError("Matched cells produced different complete position baselines")
        values = {
            "control_gate_objective_mse": control_gate_mse,
            "gate_gate_objective_mse": gate_gate_mse,
            "control_gate_l1_distance": gate_metric(control, "mean_l1_distance"),
            "gate_gate_l1_distance": gate_metric(gate, "mean_l1_distance"),
            "control_gate_overlap_mass": gate_metric(control, "mean_overlap_mass"),
            "gate_gate_overlap_mass": gate_metric(gate, "mean_overlap_mass"),
            "control_gate_cosine_similarity": gate_metric(
                control, "mean_cosine_similarity"
            ),
            "gate_gate_cosine_similarity": gate_metric(
                gate, "mean_cosine_similarity"
            ),
            "control_gate_normalized_entropy": gate_metric(
                control, "gate_normalized_entropy"
            ),
            "gate_gate_normalized_entropy": gate_metric(
                gate, "gate_normalized_entropy"
            ),
            "control_gate_effective_token_fraction": gate_metric(
                control, "gate_effective_token_fraction"
            ),
            "gate_gate_effective_token_fraction": gate_metric(
                gate, "gate_effective_token_fraction"
            ),
            "gate_fused_prior_effective_token_fraction": gate_metric(
                gate, "fused_prior_effective_token_fraction"
            ),
            "gate_raw_mean": nested_gate_metric(gate, "raw_mean_gate", "mean"),
            "gate_score_population_std": nested_gate_metric(
                gate, "score", "population_std"
            ),
            "gate_token_score_population_std": nested_gate_metric(
                gate, "token_score", "population_std"
            ),
            "gate_score_residual_population_std": nested_gate_metric(
                gate, "score_residual", "population_std"
            ),
            "gate_mean_token_absolute_score_fraction": gate_metric(
                gate, "mean_token_absolute_score_fraction"
            ),
            "control_key_unit_ap": unit_ap(control, "key"),
            "gate_key_unit_ap": unit_ap(gate, "key"),
            "control_complete_unit_ap": unit_ap(control, "complete"),
            "gate_complete_unit_ap": unit_ap(gate, "complete"),
            "control_dev_symmetric_attention_mse": control_collaboration_mse,
            "gate_dev_symmetric_attention_mse": gate_collaboration_mse,
            "gate_mean_absolute_key_complete_probability_difference": float(
                gate["head_separation"]["dev"][
                    "mean_absolute_probability_difference"
                ]
            ),
            "gate_key_complete_probability_correlation": finite_correlation(gate),
            "control_correctness_auroc": correctness_auc(control),
            "gate_correctness_auroc": correctness_auc(gate),
        }
        deltas = {
            "dev_gate_objective_mse_relative_reduction": (
                control_gate_mse - gate_gate_mse
            )
            / control_gate_mse,
            "dev_gate_effective_token_fraction_delta": values[
                "gate_gate_effective_token_fraction"
            ]
            - values["control_gate_effective_token_fraction"],
            "gate_key_unit_ap_vs_control": values["gate_key_unit_ap"]
            - values["control_key_unit_ap"],
            "gate_complete_unit_ap_vs_control": values["gate_complete_unit_ap"]
            - values["control_complete_unit_ap"],
            "gate_dev_symmetric_attention_mse_relative_increase": (
                gate_collaboration_mse - control_collaboration_mse
            )
            / control_collaboration_mse,
            "gate_correctness_auroc_vs_control": values["gate_correctness_auroc"]
            - values["control_correctness_auroc"],
        }
        checks = {
            "gate_alignment_improves": deltas[
                "dev_gate_objective_mse_relative_reduction"
            ]
            >= float(guards["dev_gate_objective_mse_relative_reduction_minimum"]),
            "gate_moves_away_from_uniform": deltas[
                "dev_gate_effective_token_fraction_delta"
            ]
            <= float(guards["dev_gate_effective_token_fraction_delta_maximum"]),
            "key_localization_protected": deltas["gate_key_unit_ap_vs_control"]
            >= float(guards["gate_key_unit_ap_delta_vs_control_minimum"]),
            "complete_localization_protected": deltas[
                "gate_complete_unit_ap_vs_control"
            ]
            >= float(guards["gate_complete_unit_ap_delta_vs_control_minimum"]),
            "mutual_collaboration_protected": deltas[
                "gate_dev_symmetric_attention_mse_relative_increase"
            ]
            <= float(
                guards[
                    "gate_dev_symmetric_attention_mse_relative_increase_maximum"
                ]
            ),
            "prior_maps_remain_separated": values[
                "gate_mean_absolute_key_complete_probability_difference"
            ]
            >= float(
                guards[
                    "gate_mean_absolute_key_complete_probability_difference_minimum"
                ]
            )
            and values["gate_key_complete_probability_correlation"]
            <= float(guards["gate_key_complete_probability_correlation_maximum"]),
            "fused_prior_remains_nonuniform": values[
                "gate_fused_prior_effective_token_fraction"
            ]
            <= float(guards["gate_fused_prior_effective_token_fraction_maximum"]),
            "correctness_protected": deltas["gate_correctness_auroc_vs_control"]
            >= float(guards["gate_correctness_auroc_delta_vs_control_minimum"]),
            "gate_and_score_not_collapsed": float(guards["gate_raw_mean_minimum"])
            <= values["gate_raw_mean"]
            <= float(guards["gate_raw_mean_maximum"])
            and values["gate_score_population_std"]
            >= float(guards["gate_score_population_std_minimum"]),
        }
        checks["all"] = all(checks.values())
        for name, passed in checks.items():
            passing_counter[name] += int(passed)
        for name, value in {**values, **deltas}.items():
            if value != float("inf"):
                aggregate_values[name].append(value)
        seed_reports[str(seed)] = {
            "values": values,
            "deltas": deltas,
            "checks": checks,
        }

    required = int(guards["minimum_passing_seeds"])
    named_checks = (
        "gate_alignment_improves",
        "gate_moves_away_from_uniform",
        "key_localization_protected",
        "complete_localization_protected",
        "mutual_collaboration_protected",
        "prior_maps_remain_separated",
        "fused_prior_remains_nonuniform",
        "correctness_protected",
        "gate_and_score_not_collapsed",
    )
    across_seed_checks = {
        name: passing_counter[name] >= required for name in named_checks
    }
    passed = (
        all(across_seed_checks.values()) and passing_counter["all"] >= required
    )
    report = {
        "schema_version": "clir-dual-prior-reward-gate-result-v1",
        "status": (
            "completed_pass_reward_gate_integration"
            if passed
            else "completed_reward_gate_integration_diagnostic_only"
        ),
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "training_commit": next(iter(code_commits)),
        "original_mutual_formula_preserved": True,
        "mutual_distillation_weight": mutual_weight,
        "gate_alignment_weight": gate_weight,
        "gate_fused_prior_detached": True,
        "required_matrix_cells": len(seeds) * len(cells),
        "completed_matrix_cells": len(result_hashes),
        "cell_result_hashes": dict(sorted(result_hashes.items())),
        "seed_results": seed_reports,
        "passing_seed_counts": {
            name: int(passing_counter[name]) for name in sorted(passing_counter)
        },
        "across_seed_checks": across_seed_checks,
        "selection_passed": passed,
        "mean_metrics_and_deltas": {
            name: statistics.mean(values)
            for name, values in sorted(aggregate_values.items())
        },
        "next_step": (
            protocol["evaluation"]["passing_conclusion"]
            if passed
            else protocol["evaluation"]["failing_conclusion"]
        ),
        "ranking_or_best_of_n_evaluated": False,
        "reconstruction_enabled": False,
        "containment_replacement_used": False,
        "formal_mechanism_claim_allowed": False,
        "pilot_test_accessed": False,
    }
    atomic_write_json(args.output_json.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
