#!/usr/bin/env python3
"""Summarize the frozen seed-42 three-cell CLIR joint-training pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_joint_training import validate_joint_protocol  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/joint_training_pilot_v1/training_protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("Paired ranking arrays have incompatible shapes")
    delta = right_array - left_array
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, delta.size, size=(replicates, delta.size))
    means = delta[samples].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "query_count": int(delta.size),
        "mean_delta": float(delta.mean()),
        "bootstrap_ci": [
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1.0 - alpha)),
        ],
        "replicates": replicates,
        "confidence_level": confidence_level,
        "bootstrap_unit": "query",
        "right_minus_left": True,
    }


def _ranking_values(
    reports: Mapping[str, Mapping[str, Any]], *, k: str
) -> tuple[list[str], dict[str, list[float]]]:
    reference_cell = next(iter(reports))
    reference_rows = reports[reference_cell]["per_query"]
    query_ids = [str(row["query_id"]) for row in reference_rows]
    values: dict[str, list[float]] = {}
    reference_baselines = [
        (
            row["k"][k]["random_expected"],
            row["k"][k]["oracle"],
            row["k"][k]["correct_candidates"],
        )
        for row in reference_rows
    ]
    for cell, report in reports.items():
        rows = report["per_query"]
        if [str(row["query_id"]) for row in rows] != query_ids:
            raise ValueError("Ranking query order differs across joint cells")
        baselines = [
            (
                row["k"][k]["random_expected"],
                row["k"][k]["oracle"],
                row["k"][k]["correct_candidates"],
            )
            for row in rows
        ]
        if baselines != reference_baselines:
            raise ValueError("Ranking candidate outcomes differ across joint cells")
        values[cell] = [float(row["k"][k]["reward"]) for row in rows]
    return query_ids, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_joint_protocol(protocol)
    output = resolve(protocol["execution"]["result"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite joint pilot summary: {output}")
    seed = int(protocol["matched_training"]["seeds"][0])
    output_root = resolve(protocol["execution"]["output_root"])
    results: dict[str, Mapping[str, Any]] = {}
    ranking_reports: dict[str, Mapping[str, Any]] = {}
    result_hashes: dict[str, str] = {}
    for cell in protocol["cells"]:
        cell_root = output_root / f"seed_{seed}" / cell
        result_path = cell_root / "cell_result.json"
        ranking_path = cell_root / "ranking_evaluation.json"
        if not result_path.is_file() or not ranking_path.is_file():
            raise FileNotFoundError(f"Joint cell is incomplete: {cell}/seed {seed}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("schema_version") != "clir-joint-training-cell-result-v1"
            or result.get("cell") != cell
            or int(result.get("seed", -1)) != seed
            or result.get("protocol_sha256") != file_sha256(protocol_path)
        ):
            raise ValueError(f"Joint cell result provenance drifted: {cell}")
        results[cell] = result
        ranking_reports[cell] = json.loads(ranking_path.read_text(encoding="utf-8"))
        result_hashes[cell] = file_sha256(result_path)

    primary_k = str(protocol["evaluation"]["primary_k"])
    query_ids, ranking_values = _ranking_values(ranking_reports, k=primary_k)
    bootstrap = protocol["evaluation"]
    comparisons = {}
    for left, right, name in (
        ("j0_correctness", "jp_original_prior", "jp_minus_j0"),
        ("jp_original_prior", "jall_full_retained", "jall_minus_jp"),
        ("j0_correctness", "jall_full_retained", "jall_minus_j0"),
    ):
        comparisons[name] = _paired_bootstrap(
            ranking_values[left],
            ranking_values[right],
            replicates=int(bootstrap["bootstrap_replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=seed,
        )

    gates = protocol["expansion_gates"]
    full = results["jall_full_retained"]
    prior_control = results["jp_original_prior"]
    hallucination_span = float(
        full["hallucination_dev"]["span_tokens"]["average_precision"]
    )
    hallucination_claim = float(
        full["hallucination_dev"]["claim_mean"]["average_precision"]
    )
    full_key = float(full["prior_dev"]["key_unit"]["average_precision_micro"])
    full_complete = float(
        full["prior_dev"]["complete_unit"]["average_precision_micro"]
    )
    control_key = float(
        prior_control["prior_dev"]["key_unit"]["average_precision_micro"]
    )
    control_complete = float(
        prior_control["prior_dev"]["complete_unit"]["average_precision_micro"]
    )
    consistency_gap = float(
        full["consistency_train_diagnostic"]["mean_cosine_gap"]
    )
    jall_bon = float(
        full["ranking_metrics"][primary_k]["reward_bon_accuracy"]["value"]
    )
    j0_bon = float(
        results["j0_correctness"]["ranking_metrics"][primary_k][
            "reward_bon_accuracy"
        ]["value"]
    )
    jp_bon = float(
        prior_control["ranking_metrics"][primary_k]["reward_bon_accuracy"]["value"]
    )
    checks = {
        "all_three_cells_completed": len(results) == 3,
        "hallucination_span_ap_beats_position": hallucination_span
        > float(gates["hallucination_span_token_ap_min_exclusive"]),
        "hallucination_claim_ap_beats_position": hallucination_claim
        > float(gates["hallucination_claim_ap_min_exclusive"]),
        "key_unit_ap_beats_position": full_key
        > float(gates["key_unit_ap_min_exclusive"]),
        "complete_unit_ap_beats_position": full_complete
        > float(gates["complete_unit_ap_min_exclusive"]),
        "key_unit_ap_drop_vs_jp_within_tolerance": full_key - control_key
        >= -float(gates["prior_ap_max_drop_vs_jp"]),
        "complete_unit_ap_drop_vs_jp_within_tolerance": full_complete
        - control_complete
        >= -float(gates["prior_ap_max_drop_vs_jp"]),
        "same_semantic_cosine_exceeds_same_style_negative": consistency_gap > 0.0,
        "jall_bon_not_more_than_two_points_below_j0": jall_bon - j0_bon
        >= -float(gates["ranking_max_absolute_regression"]),
        "jall_bon_not_more_than_two_points_below_jp": jall_bon - jp_bon
        >= -float(gates["ranking_max_absolute_regression"]),
    }
    expansion_authorized = all(checks.values())
    report = {
        "schema_version": "clir-joint-training-pilot-result-v1",
        "status": (
            "completed_seed42_expansion_gates_passed"
            if expansion_authorized
            else "completed_seed42_expansion_gates_failed"
        ),
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "seed": seed,
        "cells": {
            cell: {
                "result_sha256": result_hashes[cell],
                "bon_at_primary_k": float(
                    results[cell]["ranking_metrics"][primary_k][
                        "reward_bon_accuracy"
                    ]["value"]
                ),
                "pairwise_accuracy": results[cell]["ranking_health"][
                    "pairwise_accuracy"
                ],
                "score_population_std": results[cell][
                    "ranking_score_distribution"
                ]["population_std"],
                "hallucination_span_token_ap": results[cell][
                    "hallucination_dev"
                ]["span_tokens"]["average_precision"],
                "hallucination_claim_mean_ap": results[cell][
                    "hallucination_dev"
                ]["claim_mean"]["average_precision"],
                "key_unit_ap": results[cell]["prior_dev"]["key_unit"][
                    "average_precision_micro"
                ],
                "complete_unit_ap": results[cell]["prior_dev"][
                    "complete_unit"
                ]["average_precision_micro"],
                "consistency_mean_cosine_gap": results[cell][
                    "consistency_train_diagnostic"
                ]["mean_cosine_gap"],
            }
            for cell in results
        },
        "primary_ranking": {
            "k": int(primary_k),
            "query_count": len(query_ids),
            "paired_comparisons": comparisons,
        },
        "expansion_gate_values": {
            "hallucination_span_token_ap": hallucination_span,
            "hallucination_claim_mean_ap": hallucination_claim,
            "jall_key_unit_ap": full_key,
            "jall_complete_unit_ap": full_complete,
            "jp_key_unit_ap": control_key,
            "jp_complete_unit_ap": control_complete,
            "consistency_mean_cosine_gap": consistency_gap,
            "jall_bon": jall_bon,
            "j0_bon": j0_bon,
            "jp_bon": jp_bon,
        },
        "expansion_gate_checks": checks,
        "expand_to_seeds_43_44_authorized": expansion_authorized,
        "automatic_loss_weight_tuning_performed": False,
        "automatic_multistream_switch_performed": False,
        "interpretation": (
            "One-seed small-scale real integration pilot; passing establishes expansion "
            "eligibility, not CLIR efficacy. Consistency geometry is in-training only."
        ),
        "code": git_state(ROOT),
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
