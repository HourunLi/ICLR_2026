#!/usr/bin/env python3
"""Summarize the frozen three-seed original mutual-distillation comparison."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import atomic_write_json, file_sha256  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/dual_prior_mutual_distillation_v1/training_protocol_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/dual_prior_mutual_distillation_v1/training_result_v1.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def unit_ap(result: Mapping[str, Any], head: str) -> float:
    value = result["dev_metrics"][head]["unit"]["average_precision_micro"]
    if value is None:
        raise ValueError(f"{result['cell']}/seed {result['seed']}: unit AP is undefined")
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


def collaboration_mse(result: Mapping[str, Any], split: str) -> float:
    value = result["prior_collaboration"][split]["symmetric_attention_mse"]
    value = float(value)
    if value <= 0.0:
        raise ValueError(
            f"{result['cell']}/seed {result['seed']}: symmetric attention MSE must be positive"
        )
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
        != "clir-dual-prior-mutual-distillation-training-protocol-v1"
    ):
        raise ValueError("Unexpected mutual-distillation protocol schema")
    protocol_sha = file_sha256(protocol_path)
    output_root = resolve(protocol["execution"]["output_root"])
    seeds = [int(value) for value in protocol["matched_training"]["seeds"]]
    control_name = str(protocol["comparison"]["control_cell"])
    mutual_name = str(protocol["comparison"]["mutual_cell"])
    cells = [control_name, mutual_name]
    expected_weight = float(protocol["method"]["weight"])

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
                raise ValueError(f"Cell result was not produced from a clean worktree: {path}")
            resolved = result.get("resolved_loss_weights")
            if not isinstance(resolved, Mapping):
                raise ValueError(f"Cell result lacks resolved loss weights: {path}")
            expected_cell_weight = 0.0 if cell == control_name else expected_weight
            if float(resolved.get("prior_distill", -1.0)) != expected_cell_weight:
                raise ValueError(f"Cell used the wrong distillation weight: {path}")
            if any(
                float(resolved.get(name, -1.0)) != expected
                for name, expected in (
                    ("prior", 1.0),
                    ("key_prior", 1.0),
                    ("complete_prior", 1.0),
                    ("gate_prior", 0.0),
                    ("reconstruction", 0.0),
                )
            ):
                raise ValueError(f"Protected dual-prior weights drifted: {path}")
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
        mutual = results[seed][mutual_name]
        control_dev_mse = collaboration_mse(control, "dev")
        mutual_dev_mse = collaboration_mse(mutual, "dev")
        control_train_mse = collaboration_mse(control, "train")
        mutual_train_mse = collaboration_mse(mutual, "train")
        position_key = position_unit_ap(control, "key")
        position_complete = position_unit_ap(control, "complete")
        if position_key != position_unit_ap(mutual, "key") or position_complete != position_unit_ap(
            mutual, "complete"
        ):
            raise ValueError("Matched cells produced different position-only baselines")
        values = {
            "control_key_unit_ap": unit_ap(control, "key"),
            "mutual_key_unit_ap": unit_ap(mutual, "key"),
            "control_complete_unit_ap": unit_ap(control, "complete"),
            "mutual_complete_unit_ap": unit_ap(mutual, "complete"),
            "position_key_unit_ap": position_key,
            "position_complete_unit_ap": position_complete,
            "control_dev_symmetric_attention_mse": control_dev_mse,
            "mutual_dev_symmetric_attention_mse": mutual_dev_mse,
            "control_train_symmetric_attention_mse": control_train_mse,
            "mutual_train_symmetric_attention_mse": mutual_train_mse,
            "mutual_dev_attention_l1_distance": float(
                mutual["prior_collaboration"]["dev"]["mean_attention_l1_distance"]
            ),
            "mutual_dev_attention_overlap_mass": float(
                mutual["prior_collaboration"]["dev"]["mean_attention_overlap_mass"]
            ),
            "mutual_mean_absolute_key_complete_probability_difference": float(
                mutual["head_separation"]["dev"][
                    "mean_absolute_probability_difference"
                ]
            ),
            "mutual_key_complete_probability_correlation": finite_correlation(mutual),
            "control_correctness_auroc": correctness_auc(control),
            "mutual_correctness_auroc": correctness_auc(mutual),
        }
        deltas = {
            "dev_symmetric_attention_mse_relative_reduction": (
                control_dev_mse - mutual_dev_mse
            )
            / control_dev_mse,
            "train_symmetric_attention_mse_relative_reduction": (
                control_train_mse - mutual_train_mse
            )
            / control_train_mse,
            "mutual_key_unit_ap_vs_control": values["mutual_key_unit_ap"]
            - values["control_key_unit_ap"],
            "mutual_complete_unit_ap_vs_control": values["mutual_complete_unit_ap"]
            - values["control_complete_unit_ap"],
            "mutual_key_unit_ap_vs_position": values["mutual_key_unit_ap"]
            - position_key,
            "mutual_complete_unit_ap_vs_position": values["mutual_complete_unit_ap"]
            - position_complete,
            "mutual_correctness_auroc_vs_control": values["mutual_correctness_auroc"]
            - values["control_correctness_auroc"],
        }
        checks = {
            "mutual_reduces_heldout_attention_discrepancy": deltas[
                "dev_symmetric_attention_mse_relative_reduction"
            ]
            >= float(guards["dev_symmetric_attention_mse_relative_reduction_minimum"]),
            "mutual_key_localization_protected": deltas[
                "mutual_key_unit_ap_vs_control"
            ]
            >= float(guards["mutual_key_unit_ap_delta_vs_control_minimum"]),
            "mutual_complete_localization_protected": deltas[
                "mutual_complete_unit_ap_vs_control"
            ]
            >= float(guards["mutual_complete_unit_ap_delta_vs_control_minimum"]),
            "mutual_localization_above_position": min(
                deltas["mutual_key_unit_ap_vs_position"],
                deltas["mutual_complete_unit_ap_vs_position"],
            )
            >= float(guards["mutual_unit_ap_delta_over_position_minimum"]),
            "mutual_maps_remain_separated": values[
                "mutual_mean_absolute_key_complete_probability_difference"
            ]
            >= float(
                guards[
                    "mutual_mean_absolute_key_complete_probability_difference_minimum"
                ]
            )
            and values["mutual_key_complete_probability_correlation"]
            <= float(guards["mutual_key_complete_probability_correlation_maximum"]),
            "mutual_correctness_protected": deltas[
                "mutual_correctness_auroc_vs_control"
            ]
            >= float(guards["mutual_correctness_auroc_delta_vs_control_minimum"]),
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
        "mutual_reduces_heldout_attention_discrepancy",
        "mutual_key_localization_protected",
        "mutual_complete_localization_protected",
        "mutual_localization_above_position",
        "mutual_maps_remain_separated",
        "mutual_correctness_protected",
    )
    across_seed_checks = {
        name: passing_counter[name] >= required for name in named_checks
    }
    passed = all(across_seed_checks.values())
    report = {
        "schema_version": "clir-dual-prior-mutual-distillation-result-v1",
        "status": (
            "completed_pass_original_mutual_distillation"
            if passed
            else "completed_original_mutual_distillation_diagnostic_only"
        ),
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "training_commit": next(iter(code_commits)),
        "method_formula_preserved": True,
        "mutual_distillation_weight": expected_weight,
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
        "gate_alignment_enabled": False,
        "reconstruction_enabled": False,
        "containment_replacement_used": False,
        "formal_mechanism_claim_allowed": False,
        "pilot_test_accessed": False,
    }
    atomic_write_json(args.output_json.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
