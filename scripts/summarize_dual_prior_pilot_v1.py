#!/usr/bin/env python3
"""Summarize the frozen three-seed D0-D3 dual-prior learnability matrix."""

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


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/training_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "configs/dual_prior_evidence_v1/training_result_v1.json"


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
        raise ValueError("Position baseline unit AP is undefined")
    return float(value)


def correctness_auc(result: Mapping[str, Any]) -> float:
    value = result["correctness"]["dev"]["roc_auc"]
    if value is None:
        raise ValueError("Correctness AUROC is undefined on the frozen dev split")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-dual-prior-standalone-training-protocol-v1":
        raise ValueError("Unexpected dual-prior training protocol schema")
    protocol_sha = file_sha256(protocol_path)
    output_root = resolve(protocol["execution"]["output_root"])
    seeds = [int(value) for value in protocol["matched_training"]["seeds"]]
    cells = list(protocol["cells"])
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
            if result.get("cell") != cell or int(result.get("seed")) != seed:
                raise ValueError(f"Cell result identity drifted: {path}")
            if result.get("protocol_sha256") != protocol_sha:
                raise ValueError(f"Cell result protocol hash drifted: {path}")
            code = result.get("code")
            if not isinstance(code, Mapping) or bool(code.get("dirty")):
                raise ValueError(f"Cell result was not produced from a clean worktree: {path}")
            code_commits.add(str(code.get("commit")))
            results[seed][cell] = result
            result_hashes[f"seed_{seed}/{cell}"] = file_sha256(path)
    if len(code_commits) != 1:
        raise ValueError(f"D0-D3 cells used different commits: {sorted(code_commits)}")

    guard = protocol["evaluation"]["selection_guards"]
    delta_min = float(
        guard["single_head_unit_ap_delta_over_both_D0_and_position_minimum"]
    )
    joint_delta_min = float(
        guard["joint_head_unit_ap_delta_over_both_D0_and_position_minimum"]
    )
    noninferiority = float(
        guard["joint_noninferiority_to_matching_single_head_unit_ap"]
    )
    separation_min = float(
        guard["joint_mean_absolute_key_complete_probability_difference_minimum"]
    )
    correlation_max = float(
        guard["joint_key_complete_probability_correlation_maximum"]
    )
    correctness_min = float(guard["joint_correctness_auroc_delta_vs_D0_minimum"])

    seed_reports: dict[str, Any] = {}
    passing_counter: Counter[str] = Counter()
    aggregate_values: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        d0 = results[seed]["d0_correctness_only"]
        d1 = results[seed]["d1_correctness_plus_key"]
        d2 = results[seed]["d2_correctness_plus_complete"]
        d3 = results[seed]["d3_correctness_plus_key_and_complete"]
        position_key = position_unit_ap(d0, "key")
        position_complete = position_unit_ap(d0, "complete")
        values = {
            "d0_key_unit_ap": unit_ap(d0, "key"),
            "d0_complete_unit_ap": unit_ap(d0, "complete"),
            "d1_key_unit_ap": unit_ap(d1, "key"),
            "d2_complete_unit_ap": unit_ap(d2, "complete"),
            "d3_key_unit_ap": unit_ap(d3, "key"),
            "d3_complete_unit_ap": unit_ap(d3, "complete"),
            "position_key_unit_ap": position_key,
            "position_complete_unit_ap": position_complete,
            "d0_correctness_auroc": correctness_auc(d0),
            "d3_correctness_auroc": correctness_auc(d3),
            "d3_mean_absolute_key_complete_probability_difference": float(
                d3["head_separation"]["dev"][
                    "mean_absolute_probability_difference"
                ]
            ),
        }
        correlation = d3["head_separation"]["dev"]["pearson_correlation"]
        values["d3_key_complete_probability_correlation"] = (
            float(correlation) if correlation is not None else float("inf")
        )
        deltas = {
            "d1_key_vs_d0": values["d1_key_unit_ap"] - values["d0_key_unit_ap"],
            "d1_key_vs_position": values["d1_key_unit_ap"] - position_key,
            "d2_complete_vs_d0": values["d2_complete_unit_ap"]
            - values["d0_complete_unit_ap"],
            "d2_complete_vs_position": values["d2_complete_unit_ap"]
            - position_complete,
            "d3_key_vs_d0": values["d3_key_unit_ap"] - values["d0_key_unit_ap"],
            "d3_key_vs_position": values["d3_key_unit_ap"] - position_key,
            "d3_complete_vs_d0": values["d3_complete_unit_ap"]
            - values["d0_complete_unit_ap"],
            "d3_complete_vs_position": values["d3_complete_unit_ap"]
            - position_complete,
            "d3_key_vs_d1": values["d3_key_unit_ap"] - values["d1_key_unit_ap"],
            "d3_complete_vs_d2": values["d3_complete_unit_ap"]
            - values["d2_complete_unit_ap"],
            "d3_correctness_auroc_vs_d0": values["d3_correctness_auroc"]
            - values["d0_correctness_auroc"],
        }
        checks = {
            "d1_key_direct_learnability": min(
                deltas["d1_key_vs_d0"], deltas["d1_key_vs_position"]
            )
            >= delta_min,
            "d2_complete_direct_learnability": min(
                deltas["d2_complete_vs_d0"], deltas["d2_complete_vs_position"]
            )
            >= delta_min,
            "d3_key_direct_learnability": min(
                deltas["d3_key_vs_d0"], deltas["d3_key_vs_position"]
            )
            >= joint_delta_min,
            "d3_complete_direct_learnability": min(
                deltas["d3_complete_vs_d0"], deltas["d3_complete_vs_position"]
            )
            >= joint_delta_min,
            "d3_key_noninferior_to_d1": deltas["d3_key_vs_d1"] >= noninferiority,
            "d3_complete_noninferior_to_d2": deltas["d3_complete_vs_d2"]
            >= noninferiority,
            "d3_maps_separated": values[
                "d3_mean_absolute_key_complete_probability_difference"
            ]
            >= separation_min
            and values["d3_key_complete_probability_correlation"] <= correlation_max,
            "d3_correctness_guard": deltas["d3_correctness_auroc_vs_d0"]
            >= correctness_min,
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

    required = int(guard["minimum_passing_seeds"])
    across_seed_checks = {
        name: passing_counter[name] >= required
        for name in (
            "d1_key_direct_learnability",
            "d2_complete_direct_learnability",
            "d3_key_direct_learnability",
            "d3_complete_direct_learnability",
            "d3_key_noninferior_to_d1",
            "d3_complete_noninferior_to_d2",
            "d3_maps_separated",
            "d3_correctness_guard",
        )
    }
    passed = all(across_seed_checks.values())
    report = {
        "schema_version": "clir-dual-prior-training-result-v1",
        "status": (
            "completed_pass_direct_targets_learnable"
            if passed
            else "completed_fail_do_not_enable_collaboration"
        ),
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "training_commit": next(iter(code_commits)),
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
        "position_only_baseline_included": True,
        "next_step": (
            protocol["evaluation"]["passing_conclusion"]
            if passed
            else protocol["evaluation"]["failing_conclusion"]
        ),
        "formal_mechanism_claim_allowed": False,
        "pilot_test_accessed": False,
    }
    atomic_write_json(args.output_json.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
