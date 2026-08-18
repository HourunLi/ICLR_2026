#!/usr/bin/env python3
"""Summarize original shared-gradient dual-prior localization and ranking evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/dual_prior_original_scale_v2/training_protocol_v2.json"
)
SCHEMA = "clir-dual-prior-original-scale-training-protocol-v2"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def stable_seed(base: int, label: str) -> int:
    payload = f"{base}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_ci(
    values: Sequence[float], *, replicates: int, confidence: float, seed: int
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Paired bootstrap requires a finite non-empty vector")
    if replicates <= 0:
        value = float(array.mean())
        return [value, value]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    ]


def stats(values: Mapping[int, float]) -> dict[str, Any]:
    ordered = [float(values[seed]) for seed in sorted(values)]
    return {
        "by_seed": {str(seed): float(values[seed]) for seed in sorted(values)},
        "mean": float(statistics.mean(ordered)),
        "sample_std": float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def query_reward(report: Mapping[str, Any], k: int) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in report["per_query"]:
        query_id = str(row["query_id"])
        if query_id in output:
            raise ValueError(f"Duplicate ranking query {query_id}")
        output[query_id] = float(row["k"][str(k)]["reward"])
    return output


def nested(row: Mapping[str, Any], *keys: str) -> float:
    value: Any = row
    for key in keys:
        value = value[key]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Missing/non-finite metric at {'.'.join(keys)}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("Unexpected original dual-prior scale protocol schema")
    protocol_sha = file_sha256(protocol_path)
    cells = list(protocol["cells"])
    seeds = [int(value) for value in protocol["matched_training"]["seeds"]]
    if cells != ["g0_original_mutual_control", "g1_original_shared_gate"]:
        raise ValueError("Original scale protocol requires the frozen G0/G1 order")

    root = resolve(protocol["execution"]["output_root"])
    cell_results: dict[str, dict[int, Mapping[str, Any]]] = {
        cell: {} for cell in cells
    }
    ranking_reports: dict[str, dict[int, Mapping[str, Any]]] = {
        cell: {} for cell in cells
    }
    result_hashes: dict[str, str] = {}
    training_commits: set[str] = set()
    checkpoint_hashes: set[str] = set()
    for seed in seeds:
        reference_baselines: dict[str, Any] | None = None
        for cell in cells:
            cell_root = root / f"seed_{seed}" / cell
            result_path = cell_root / "cell_result.json"
            ranking_path = cell_root / "ranking_evaluation.json"
            if not result_path.is_file() or not ranking_path.is_file():
                raise FileNotFoundError(f"Missing completed output for seed={seed}/{cell}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
            if (
                result.get("schema_version") != "clir-dual-prior-cell-result-v1"
                or result.get("experiment_schema_version") != SCHEMA
                or result.get("cell") != cell
                or int(result.get("seed", -1)) != seed
                or result.get("protocol_sha256") != protocol_sha
            ):
                raise ValueError(f"Cell result identity/protocol drifted: {result_path}")
            if file_sha256(ranking_path) != result["ranking_evaluation_sha256"]:
                raise ValueError(f"Ranking evaluation hash drifted: {ranking_path}")
            code = result.get("code")
            if not isinstance(code, Mapping) or code.get("dirty"):
                raise ValueError(f"Cell was not trained from a clean commit: {result_path}")
            training_commits.add(str(code.get("commit")))
            checkpoint_sha = str(result["checkpoint_sha256"])
            if checkpoint_sha in checkpoint_hashes:
                raise ValueError("One checkpoint was reused by multiple G0/G1 cells")
            checkpoint_hashes.add(checkpoint_sha)

            if (
                ranking.get("reward_model_variant") != "clir"
                or int(ranking.get("query_count", -1))
                != int(protocol["evaluation"]["ranking_query_count"])
                or [int(value) for value in ranking.get("k", [])]
                != [int(value) for value in protocol["evaluation"]["k"]]
                or ranking.get("candidate_subset")
                != protocol["evaluation"]["candidate_subset"]
                or ranking.get("candidate_index_policy")
                != protocol["evaluation"]["candidate_index_policy"]
            ):
                raise ValueError(f"Ranking evaluation contract drifted: {ranking_path}")
            provenance = ranking.get("reward_scoring_provenance")
            experiment = (
                provenance.get("experiment_protocol")
                if isinstance(provenance, Mapping)
                else None
            )
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("checkpoint_sha256") != checkpoint_sha
                or provenance.get("input_sha256")
                != protocol["manifests"]["ranking_validation"]["sha256"]
                or not isinstance(experiment, Mapping)
                or experiment.get("sha256") != protocol_sha
            ):
                raise ValueError(f"Ranking scoring provenance drifted: {ranking_path}")
            baselines = {
                str(k): {
                    name: ranking["metrics"][str(k)][name]["value"]
                    for name in (
                        "random_expected_accuracy",
                        "random_seeded_accuracy",
                        "oracle_accuracy",
                    )
                }
                for k in protocol["evaluation"]["k"]
            }
            if reference_baselines is None:
                reference_baselines = baselines
            elif baselines != reference_baselines:
                raise ValueError(f"Candidate-independent baselines differ at seed {seed}")

            cell_results[cell][seed] = result
            ranking_reports[cell][seed] = ranking
            result_hashes[f"seed_{seed}/{cell}"] = file_sha256(result_path)
    if len(training_commits) != 1:
        raise ValueError(f"Scale cells used multiple code commits: {training_commits}")

    per_cell: dict[str, Any] = {}
    for cell in cells:
        ranking_by_k: dict[str, Any] = {}
        for k in protocol["evaluation"]["k"]:
            ranking_by_k[str(k)] = stats(
                {
                    seed: nested(
                        ranking_reports[cell][seed],
                        "metrics",
                        str(k),
                        "reward_bon_accuracy",
                        "value",
                    )
                    for seed in seeds
                }
            )
        per_cell[cell] = {
            "ranking": ranking_by_k,
            "within_query_pairwise_accuracy": stats(
                {
                    seed: nested(
                        ranking_reports[cell][seed],
                        "ranking_health",
                        "pairwise_accuracy",
                    )
                    for seed in seeds
                }
            ),
            "dev_key_unit_ap": stats(
                {
                    seed: nested(
                        cell_results[cell][seed],
                        "dev_metrics",
                        "key",
                        "unit",
                        "average_precision_micro",
                    )
                    for seed in seeds
                }
            ),
            "dev_complete_unit_ap": stats(
                {
                    seed: nested(
                        cell_results[cell][seed],
                        "dev_metrics",
                        "complete",
                        "unit",
                        "average_precision_micro",
                    )
                    for seed in seeds
                }
            ),
            "dev_correctness_auroc": stats(
                {
                    seed: nested(
                        cell_results[cell][seed], "dev_correctness", "roc_auc"
                    )
                    for seed in seeds
                }
            ),
            "dev_gate_objective_mse": stats(
                {
                    seed: nested(
                        cell_results[cell][seed],
                        "dev_gate_integration",
                        "training_objective_mse",
                    )
                    for seed in seeds
                }
            ),
        }

    control, gate = cells
    primary_k = max(int(value) for value in protocol["evaluation"]["k"])
    replicates = int(protocol["evaluation"]["bootstrap_replicates"])
    confidence = float(protocol["evaluation"]["confidence_level"])
    paired_by_seed: dict[int, dict[str, float]] = {}
    seed_delta: dict[int, float] = {}
    seed_ci: dict[str, list[float]] = {}
    for seed in seeds:
        control_values = query_reward(ranking_reports[control][seed], primary_k)
        gate_values = query_reward(ranking_reports[gate][seed], primary_k)
        if set(control_values) != set(gate_values):
            raise ValueError(f"G0/G1 query sets differ for seed {seed}")
        paired = {
            query_id: gate_values[query_id] - control_values[query_id]
            for query_id in sorted(control_values)
        }
        paired_by_seed[seed] = paired
        seed_delta[seed] = float(statistics.mean(paired.values()))
        seed_ci[str(seed)] = bootstrap_ci(
            list(paired.values()),
            replicates=replicates,
            confidence=confidence,
            seed=stable_seed(42, f"seed-{seed}-g1-minus-g0"),
        )

    query_ids = sorted(next(iter(paired_by_seed.values())))
    if any(sorted(values) != query_ids for values in paired_by_seed.values()):
        raise ValueError("Across-seed query sets differ")
    aggregate_query_delta = [
        float(statistics.mean(paired_by_seed[seed][query_id] for seed in seeds))
        for query_id in query_ids
    ]
    aggregate_mean = float(statistics.mean(aggregate_query_delta))
    aggregate_ci = bootstrap_ci(
        aggregate_query_delta,
        replicates=replicates,
        confidence=confidence,
        seed=stable_seed(42, "aggregate-g1-minus-g0"),
    )
    positive_seed_count = sum(value > 0.0 for value in seed_delta.values())
    stable_positive = (
        positive_seed_count >= 2 and aggregate_mean > 0.0 and aggregate_ci[0] > 0.0
    )

    def metric_delta(*path: str) -> dict[str, Any]:
        values = {
            seed: nested(cell_results[gate][seed], *path)
            - nested(cell_results[control][seed], *path)
            for seed in seeds
        }
        return stats(values)

    result = {
        "schema_version": "clir-dual-prior-original-scale-result-v2",
        "status": "completed_original_shared_gradient_scale_and_ranking",
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "training_commit": next(iter(training_commits)),
        "required_matrix_cells": len(cells) * len(seeds),
        "completed_matrix_cells": len(result_hashes),
        "cell_result_hashes": result_hashes,
        "original_method_preserved": True,
        "method_invariant": protocol["method_invariant"],
        "head_only_or_other_architecture_attempted": False,
        "per_cell": per_cell,
        "primary_ranking_comparison": {
            "metric": f"reward_bon_accuracy@{primary_k}",
            "direction": "G1_original_shared_gate_minus_G0_original_mutual_control",
            **stats(seed_delta),
            "positive_seed_count": positive_seed_count,
            "paired_query_bootstrap_ci_by_seed": seed_ci,
            "aggregate_query_paired": {
                "query_count": len(query_ids),
                "seed_aggregation_within_query": "arithmetic_mean",
                "mean": aggregate_mean,
                "bootstrap_ci": aggregate_ci,
                "replicates": replicates,
                "confidence_level": confidence,
            },
            "stable_positive_definition": protocol["evaluation"][
                "stable_positive_definition"
            ],
            "stable_positive": stable_positive,
        },
        "localization_deltas_g1_minus_g0": {
            "key_unit_ap": metric_delta(
                "dev_metrics", "key", "unit", "average_precision_micro"
            ),
            "complete_unit_ap": metric_delta(
                "dev_metrics", "complete", "unit", "average_precision_micro"
            ),
            "correctness_auroc": metric_delta("dev_correctness", "roc_auc"),
            "gate_objective_mse": metric_delta(
                "dev_gate_integration", "training_objective_mse"
            ),
        },
        "ranking_improvement_established": stable_positive,
        "decision": (
            "The original shared-gradient method remains the project method. The scale "
            "run establishes a stable positive validation ranking signal."
            if stable_positive
            else "The original shared-gradient method remains the implemented project "
            "method, but this run does not establish a stable positive validation "
            "ranking effect. Preserve the result and diagnose supervision scale or "
            "optimization under a new protocol rather than silently replacing the architecture."
        ),
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    output = resolve(protocol["execution"]["result"])
    atomic_write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
