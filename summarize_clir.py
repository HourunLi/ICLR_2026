"""Summarize multi-seed CLIR evaluations and paired primary-metric contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.clir_real_data import file_sha256
from src.clir_stage_a import atomic_write_json


DEFAULT_VARIANTS = ("strict_swift", "encoded_swift", "clir")
EXPECTED_EVALUATION_SCHEMAS = {
    "clir-query-evaluation-v2",
    "clir-query-evaluation-v3",
}
EXPECTED_CANDIDATE_SUBSET = "first_k_by_vllm_completion_output_index"
EXPECTED_CANDIDATE_INDEX_POLICY = "vllm_completion_output_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--primary-k", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42, help="Paired-bootstrap seed.")
    parser.add_argument(
        "--experiment-protocol-config",
        default=None,
        help=(
            "Optional frozen protocol. When supplied, every included report and every "
            "failure record must match its hash and preregistered health thresholds."
        ),
    )
    parser.add_argument(
        "--allow-failed-cells",
        action="store_true",
        help=(
            "Produce an explicitly incomplete diagnostic summary only when every omitted "
            "cell has persisted preregistered health-gate failure evidence."
        ),
    )
    parser.add_argument("--run-dir", default=None, help="Training run-record root for failure evidence.")
    parser.add_argument("--scored-dir", default=None, help="Scoring output root for failure evidence.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _protocol_contract(path: str | Path) -> Dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = _load_json_object(protocol_path)
    health = protocol.get("training", {}).get("health_gates", {})
    if not isinstance(health, Mapping):
        raise ValueError("Experiment protocol lacks training.health_gates")
    prior_improvement = health.get(
        "constant_class_prior_bce_minimum_relative_improvement",
        health.get("constant_class_prior_bce_relative_tolerance"),
    )
    scoring = protocol.get("scoring", {})
    evaluation = protocol.get("evaluation", {})
    manifests = protocol.get("manifests", {})
    train_manifest = manifests.get("train", {})
    validation_manifest = manifests.get("validation", {})
    required_thresholds = {
        "minimum_prior_relative_improvement": prior_improvement,
        "minimum_score_population_std": health.get(
            "minimum_validation_score_population_std"
        ),
        "minimum_within_query_pairwise_accuracy": health.get(
            "minimum_within_query_pairwise_accuracy", 0.0
        ),
    }
    if any(
        not isinstance(value, (int, float))
        for value in required_thresholds.values()
    ):
        raise ValueError("Experiment protocol has incomplete/non-numeric health thresholds")
    k_values = evaluation.get("k")
    if not isinstance(k_values, list) or not k_values:
        raise ValueError("Experiment protocol requires a non-empty evaluation.k")
    if (
        evaluation.get("candidate_subset") != EXPECTED_CANDIDATE_SUBSET
        or evaluation.get("candidate_index_policy") != EXPECTED_CANDIDATE_INDEX_POLICY
    ):
        raise ValueError("Experiment protocol has an unsupported candidate prefix contract")
    return {
        "path": str(protocol_path),
        "sha256": file_sha256(protocol_path),
        "schema_version": protocol.get("schema_version"),
        **{key: float(value) for key, value in required_thresholds.items()},
        "training_seeds": [int(value) for value in protocol.get("training", {}).get("seeds", [])],
        "variants": [str(value) for value in protocol.get("models", {}).get("variants", [])],
        "train_input_sha256": train_manifest.get("sha256"),
        "validation_input_sha256": validation_manifest.get("sha256"),
        "scoring_batch_size": scoring.get("batch_size"),
        "scoring_amp_dtype": scoring.get("amp_dtype"),
        "scoring_compute_dtype": scoring.get("compute_dtype"),
        "evaluation_query_count": evaluation.get("query_count"),
        "evaluation_k": [int(value) for value in k_values],
        "evaluation_candidate_subset": evaluation.get("candidate_subset"),
        "evaluation_candidate_index_policy": evaluation.get("candidate_index_policy"),
        "bootstrap_replicates": evaluation.get("bootstrap_replicates"),
        "confidence_level": evaluation.get("confidence_level"),
        "failed_cell_policy": evaluation.get("failed_cell_policy"),
    }


def _same_threshold(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
    )


def _validate_report_health_contract(
    report: Mapping[str, Any],
    *,
    protocol_contract: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Validate frozen scoring/ranking gates without deciding pass/fail policy."""

    if protocol_contract is None:
        ranking = report.get("ranking_health")
        return ranking if isinstance(ranking, Mapping) else None

    provenance = report.get("reward_scoring_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Evaluation report lacks reward scoring provenance")
    if provenance.get("model_variant") != report.get("reward_model_variant"):
        raise ValueError("Evaluation report/scoring provenance variant mismatch")
    experiment_protocol = provenance.get("experiment_protocol")
    if (
        not isinstance(experiment_protocol, Mapping)
        or experiment_protocol.get("sha256") != protocol_contract["sha256"]
    ):
        raise ValueError("Evaluation report experiment protocol SHA256 mismatch")
    expected_scoring_values = {
        "input_sha256": protocol_contract["validation_input_sha256"],
        "batch_size": protocol_contract["scoring_batch_size"],
        "amp_dtype": protocol_contract["scoring_amp_dtype"],
        "compute_dtype": protocol_contract["scoring_compute_dtype"],
    }
    for key, expected in expected_scoring_values.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Evaluation report scoring {key} differs from frozen protocol")
    expected_score_std = float(protocol_contract["minimum_score_population_std"])
    if not _same_threshold(provenance.get("min_score_std"), expected_score_std):
        raise ValueError("Evaluation report score-health threshold mismatch")
    distribution = provenance.get("score_distribution")
    observed_std = (
        distribution.get("population_std") if isinstance(distribution, Mapping) else None
    )
    if (
        not isinstance(observed_std, (int, float))
        or not math.isfinite(float(observed_std))
        or float(observed_std) < expected_score_std
    ):
        raise ValueError("Evaluation report did not pass the frozen score-health gate")

    expected_pairwise = float(
        protocol_contract["minimum_within_query_pairwise_accuracy"]
    )
    ranking = report.get("ranking_health")
    if expected_pairwise <= 0.0:
        return ranking if isinstance(ranking, Mapping) else None
    if report.get("schema_version") != "clir-query-evaluation-v3":
        raise ValueError("Frozen ranking gate requires evaluation schema v3")
    if not isinstance(ranking, Mapping) or not ranking.get("enabled"):
        raise ValueError("Evaluation report did not enable the frozen ranking gate")
    if not _same_threshold(ranking.get("minimum_pairwise_accuracy"), expected_pairwise):
        raise ValueError("Evaluation report ranking-health threshold mismatch")
    observed_pairwise = ranking.get("pairwise_accuracy")
    pair_count = ranking.get("pair_count")
    expected_pass = (
        isinstance(observed_pairwise, (int, float))
        and math.isfinite(float(observed_pairwise))
        and isinstance(pair_count, int)
        and pair_count > 0
        and float(observed_pairwise) >= expected_pairwise
    )
    if bool(ranking.get("passed")) != expected_pass:
        raise ValueError("Evaluation ranking-health status is inconsistent with its evidence")
    return ranking


def _seed_statistics(values: Mapping[int, float]) -> Dict[str, Any]:
    ordered = [float(values[seed]) for seed in sorted(values)]
    return {
        "by_seed": {str(seed): float(values[seed]) for seed in sorted(values)},
        "mean": float(statistics.mean(ordered)),
        "sample_std": float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def _stable_bootstrap_seed(base_seed: int, comparison: str, training_seed: int) -> int:
    payload = f"{base_seed}:{comparison}:{training_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_aggregate_bootstrap_seed(base_seed: int, comparison: str) -> int:
    payload = (
        f"{base_seed}:{comparison}:mean_across_training_seeds_within_query"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> list[float]:
    array = np.asarray(differences, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty paired query set")
    mean = float(array.mean())
    if replicates <= 0:
        return [mean, mean]
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[samples].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))]


def _query_values(report: Mapping[str, Any], k: int, field: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for row in report["per_query"]:
        query_id = str(row["query_id"])
        if query_id in values:
            raise ValueError(f"Duplicate query_id in evaluation report: {query_id}")
        values[query_id] = float(row["k"][str(k)][field])
    return values


def _validate_reports(
    reports: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    variants: Sequence[str],
    primary_k: int,
    allow_incomplete: bool = False,
    protocol_contract: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int], int, Dict[str, list[int]]]:
    if not reports:
        raise ValueError("No evaluation reports were provided")
    seeds = sorted(int(seed) for seed in reports)
    available_seeds: Dict[str, list[int]] = {variant: [] for variant in variants}
    reference: Mapping[str, Any] | None = None
    reference_query_ids: set[str] | None = None
    reference_scoring_contract: tuple[Any, ...] | None = None
    checkpoint_owners: Dict[str, tuple[int, str]] = {}
    for seed in seeds:
        seed_reports = reports[seed]
        unexpected = sorted(set(seed_reports) - set(variants))
        if unexpected:
            raise ValueError(f"Seed {seed} has unexpected variants: {unexpected}")
        missing = [variant for variant in variants if variant not in seed_reports]
        if missing and not allow_incomplete:
            raise ValueError(f"Seed {seed} is missing variants: {missing}")
        present = [variant for variant in variants if variant in seed_reports]
        if not present:
            continue
        checkpoint_hashes: set[str] = set()
        for variant in present:
            report = seed_reports[variant]
            schema_version = report.get("schema_version")
            if schema_version not in EXPECTED_EVALUATION_SCHEMAS:
                raise ValueError(f"Unsupported evaluation schema for seed {seed}/{variant}")
            if schema_version == "clir-query-evaluation-v3":
                ranking_health = report.get("ranking_health")
                if not isinstance(ranking_health, Mapping):
                    raise ValueError(f"Missing ranking health for seed {seed}/{variant}")
                if not ranking_health.get("passed"):
                    raise ValueError(f"Failed ranking health for seed {seed}/{variant}")
            ranking_health = _validate_report_health_contract(
                report,
                protocol_contract=protocol_contract,
            )
            if isinstance(ranking_health, Mapping) and not ranking_health.get("passed"):
                raise ValueError(f"Failed ranking health for seed {seed}/{variant}")
            if report.get("candidate_subset") != EXPECTED_CANDIDATE_SUBSET:
                raise ValueError(
                    f"Candidate subset mismatch for seed {seed}/{variant}: "
                    f"{report.get('candidate_subset')!r}"
                )
            if report.get("candidate_index_policy") != EXPECTED_CANDIDATE_INDEX_POLICY:
                raise ValueError(
                    f"Candidate index policy mismatch for seed {seed}/{variant}: "
                    f"{report.get('candidate_index_policy')!r}"
                )
            if report.get("reward_model_variant") != variant:
                raise ValueError(
                    f"Reward model variant mismatch for seed {seed}/{variant}"
                )
            provenance = report.get("reward_scoring_provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"Missing scoring provenance for seed {seed}/{variant}")
            checkpoint_sha256 = provenance.get("checkpoint_sha256")
            if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
                raise ValueError(f"Invalid checkpoint SHA256 for seed {seed}/{variant}")
            if checkpoint_sha256 in checkpoint_hashes:
                raise ValueError(
                    f"Seed {seed} reuses one checkpoint across multiple variants"
                )
            checkpoint_hashes.add(checkpoint_sha256)
            previous_owner = checkpoint_owners.get(checkpoint_sha256)
            if previous_owner is not None:
                raise ValueError(
                    "Evaluation matrix reuses one checkpoint across cells: "
                    f"{previous_owner} and {(seed, variant)}"
                )
            checkpoint_owners[checkpoint_sha256] = (seed, variant)
            available_seeds[variant].append(seed)
            scoring_contract = (
                provenance.get("input_sha256"),
                provenance.get("batch_size"),
                provenance.get("amp_dtype"),
                provenance.get("compute_dtype"),
                provenance.get("min_score_std"),
                (
                    provenance.get("experiment_protocol", {}).get("sha256")
                    if isinstance(provenance.get("experiment_protocol"), Mapping)
                    else None
                ),
            )
            if reference_scoring_contract is None:
                reference_scoring_contract = scoring_contract
            elif scoring_contract != reference_scoring_contract:
                raise ValueError(
                    "Evaluation reports do not share one scoring input/batch/dtype contract"
                )
            k_values = [int(k) for k in report["k"]]
            if primary_k not in k_values:
                raise ValueError(f"Primary k={primary_k} is absent for seed {seed}/{variant}")
            query_ids = {str(row["query_id"]) for row in report["per_query"]}
            if len(query_ids) != int(report["query_count"]):
                raise ValueError(f"Query count mismatch for seed {seed}/{variant}")
            if protocol_contract is not None:
                bootstrap = report.get("bootstrap")
                expected_bootstrap = {
                    "replicates": protocol_contract["bootstrap_replicates"],
                    "confidence_level": protocol_contract["confidence_level"],
                    "seed": seed,
                }
                if (
                    k_values != protocol_contract["evaluation_k"]
                    or int(report["query_count"])
                    != protocol_contract["evaluation_query_count"]
                    or not isinstance(bootstrap, Mapping)
                    or any(
                        bootstrap.get(key) != expected
                        for key, expected in expected_bootstrap.items()
                    )
                ):
                    raise ValueError(
                        f"Evaluation report metric/bootstrap contract mismatch for "
                        f"seed {seed}/{variant}"
                    )
            if reference is None:
                reference = report
                reference_query_ids = query_ids
            elif (
                k_values != [int(k) for k in reference["k"]]
                or int(report["query_count"]) != int(reference["query_count"])
                or query_ids != reference_query_ids
            ):
                raise ValueError("Evaluation reports do not share identical k/query sets")

        for k in k_values:
            for metric_name, per_query_field in (
                ("random_expected_accuracy", "random_expected"),
                ("random_seeded_accuracy", "random_seeded"),
                ("oracle_accuracy", "oracle"),
            ):
                values = [
                    float(seed_reports[variant]["metrics"][str(k)][metric_name]["value"])
                    for variant in present
                ]
                if any(not math.isclose(value, values[0], abs_tol=1e-12) for value in values[1:]):
                    raise ValueError(
                        f"Candidate-independent baseline mismatch at seed={seed}, k={k}, "
                        f"metric={metric_name}"
                    )
                reference_values = _query_values(
                    seed_reports[present[0]], k, per_query_field
                )
                for variant in present[1:]:
                    if _query_values(seed_reports[variant], k, per_query_field) != reference_values:
                        raise ValueError(
                            f"Per-query baseline mismatch at seed={seed}, k={k}, "
                            f"metric={metric_name}"
                        )
    if reference is None:
        raise ValueError("No healthy evaluation report is available for aggregation")
    return (
        seeds,
        [int(k) for k in reference["k"]],
        int(reference["query_count"]),
        available_seeds,
    )


def _paired_comparison(
    reports: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
    comparison_name: str,
    from_source: str,
    to_variant: str,
    primary_k: int,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    differences_by_seed: Dict[int, float] = {}
    intervals: Dict[str, list[float]] = {}
    paired_by_seed: Dict[int, Dict[str, float]] = {}
    for training_seed in seeds:
        to_values = _query_values(reports[training_seed][to_variant], primary_k, "reward")
        if from_source == "random_expected":
            from_values = _query_values(
                reports[training_seed][to_variant], primary_k, "random_expected"
            )
        else:
            from_values = _query_values(reports[training_seed][from_source], primary_k, "reward")
        if set(to_values) != set(from_values):
            raise ValueError(f"Unpaired query set for comparison {comparison_name}")
        paired_by_query = {
            query_id: to_values[query_id] - from_values[query_id]
            for query_id in sorted(to_values)
        }
        paired_by_seed[training_seed] = paired_by_query
        paired = list(paired_by_query.values())
        differences_by_seed[training_seed] = float(statistics.mean(paired))
        intervals[str(training_seed)] = _paired_bootstrap_ci(
            paired,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=_stable_bootstrap_seed(bootstrap_seed, comparison_name, training_seed),
        )

    query_ids = sorted(next(iter(paired_by_seed.values())))
    for training_seed, paired_by_query in paired_by_seed.items():
        if sorted(paired_by_query) != query_ids:
            raise ValueError(
                f"Training seed {training_seed} has an unpaired aggregate query set "
                f"for comparison {comparison_name}"
            )
    aggregate_query_differences = [
        float(statistics.mean(paired_by_seed[training_seed][query_id] for training_seed in seeds))
        for query_id in query_ids
    ]
    aggregate_mean = float(statistics.mean(aggregate_query_differences))
    return {
        "from": from_source,
        "to": to_variant,
        "direction": "to_minus_from_accuracy",
        **_seed_statistics(differences_by_seed),
        "paired_query_bootstrap_ci_by_seed": intervals,
        "aggregate_query_paired": {
            "unit": "query",
            "seed_aggregation_within_query": "arithmetic_mean",
            "definition": (
                "For each query, average to-minus-from correctness across training "
                "seeds, then bootstrap those query-level means."
            ),
            "query_count": len(query_ids),
            "training_seed_count": len(seeds),
            "mean": aggregate_mean,
            "bootstrap_ci": _paired_bootstrap_ci(
                aggregate_query_differences,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=_stable_aggregate_bootstrap_seed(bootstrap_seed, comparison_name),
            ),
        },
    }


def summarize_evaluation_reports(
    reports: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    primary_k: int = 16,
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    allow_incomplete: bool = False,
    protocol_contract: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if len(set(variants)) != len(variants) or not variants:
        raise ValueError("variants must be non-empty and unique")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if reports and allow_incomplete and not any(reports.values()):
        seeds = sorted(int(seed) for seed in reports)
        return {
            "schema_version": "clir-multiseed-evaluation-summary-v3",
            "result_status": "all_cells_failed_diagnostic_only",
            "matrix_complete": False,
            "formal_primary_claim_allowed": False,
            "seed_aggregation": "unavailable_no_healthy_cells",
            "seeds": seeds,
            "available_seeds_by_variant": {variant: [] for variant in variants},
            "variants": list(variants),
            "query_count": 0,
            "k": [],
            "primary_k": primary_k,
            "paired_bootstrap": {
                "unit": "query",
                "aggregate_definition": (
                    "mean_across_training_seeds_within_query_then_bootstrap_queries"
                ),
                "replicates": bootstrap_replicates,
                "confidence_level": confidence_level,
                "seed": bootstrap_seed,
                "status": "unavailable_no_healthy_cells",
            },
            "per_variant": {
                variant: {"available_seeds": [], "metrics": {}}
                for variant in variants
            },
            "baselines": {},
            "primary_comparisons": {},
        }
    seeds, k_values, query_count, available_seeds = _validate_reports(
        reports,
        variants=variants,
        primary_k=primary_k,
        allow_incomplete=allow_incomplete,
        protocol_contract=protocol_contract,
    )
    matrix_complete = all(available_seeds[variant] == seeds for variant in variants)

    per_variant: Dict[str, Any] = {}
    for variant in variants:
        variant_seeds = available_seeds[variant]
        metrics: Dict[str, Any] = {}
        if variant_seeds:
            for k in k_values:
                values = {
                    seed: float(
                        reports[seed][variant]["metrics"][str(k)][
                            "reward_bon_accuracy"
                        ]["value"]
                    )
                    for seed in variant_seeds
                }
                metrics[str(k)] = {
                    "reward_bon_accuracy": {
                        **_seed_statistics(values),
                        "query_bootstrap_ci_by_seed": {
                            str(seed): list(
                                reports[seed][variant]["metrics"][str(k)][
                                    "reward_bon_accuracy"
                                ]["query_bootstrap_ci"]
                            )
                            for seed in variant_seeds
                        },
                    }
                }
        per_variant[variant] = {
            "available_seeds": variant_seeds,
            "metrics": metrics,
        }

    baselines: Dict[str, Any] = {}
    baseline_seeds = [
        seed for seed in seeds if any(variant in reports[seed] for variant in variants)
    ]
    for metric_name in (
        "random_expected_accuracy",
        "random_seeded_accuracy",
        "oracle_accuracy",
    ):
        baselines[metric_name] = {}
        for k in k_values:
            values = {
                seed: float(
                    reports[seed][
                        next(variant for variant in variants if variant in reports[seed])
                    ]["metrics"][str(k)][metric_name]["value"]
                )
                for seed in baseline_seeds
            }
            baselines[metric_name][str(k)] = _seed_statistics(values)

    comparisons: Dict[str, Any] = {}
    for variant in variants:
        comparison_seeds = available_seeds[variant]
        if not comparison_seeds:
            continue
        name = f"random_expected_to_{variant}"
        comparisons[name] = _paired_comparison(
            reports,
            seeds=comparison_seeds,
            comparison_name=name,
            from_source="random_expected",
            to_variant=variant,
            primary_k=primary_k,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
    if "strict_swift" in variants and "encoded_swift" in variants:
        comparison_seeds = sorted(
            set(available_seeds["strict_swift"])
            & set(available_seeds["encoded_swift"])
        )
        name = "strict_swift_to_encoded_swift"
        if comparison_seeds:
            comparisons[name] = _paired_comparison(
                reports,
                seeds=comparison_seeds,
                comparison_name=name,
                from_source="strict_swift",
                to_variant="encoded_swift",
                primary_k=primary_k,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                bootstrap_seed=bootstrap_seed,
            )
    if "encoded_swift" in variants and "clir" in variants:
        comparison_seeds = sorted(
            set(available_seeds["encoded_swift"])
            & set(available_seeds["clir"])
        )
        name = "encoded_swift_to_clir"
        if comparison_seeds:
            comparisons[name] = _paired_comparison(
                reports,
                seeds=comparison_seeds,
                comparison_name=name,
                from_source="encoded_swift",
                to_variant="clir",
                primary_k=primary_k,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                bootstrap_seed=bootstrap_seed,
            )

    return {
        "schema_version": "clir-multiseed-evaluation-summary-v3",
        "result_status": "complete" if matrix_complete else "incomplete_diagnostic_only",
        "matrix_complete": matrix_complete,
        "formal_primary_claim_allowed": matrix_complete,
        "seed_aggregation": "arithmetic_mean_and_sample_standard_deviation",
        "seeds": seeds,
        "available_seeds_by_variant": available_seeds,
        "variants": list(variants),
        "query_count": query_count,
        "k": k_values,
        "primary_k": primary_k,
        "paired_bootstrap": {
            "unit": "query",
            "aggregate_definition": (
                "mean_across_training_seeds_within_query_then_bootstrap_queries"
            ),
            "replicates": bootstrap_replicates,
            "confidence_level": confidence_level,
            "seed": bootstrap_seed,
        },
        "per_variant": per_variant,
        "baselines": baselines,
        "primary_comparisons": comparisons,
    }


def _load_json_object(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _health_failure_evidence(
    *,
    seed: int,
    variant: str,
    run_dir: Path,
    scored_dir: Path,
    protocol_contract: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    run_path = run_dir / f"seed_{seed}" / f"{variant}.run.json"
    if run_path.is_file():
        run = _load_json_object(run_path)
        health = run.get("health_gate")
        if (
            run.get("status") == "health_gate_failed"
            and isinstance(health, Mapping)
            and health.get("enabled")
            and not health.get("passed")
        ):
            if protocol_contract is not None:
                experiment_protocol = run.get("experiment_protocol")
                data_state = run.get("data_state")
                prior_bce = health.get("constant_prior_bce")
                observed_bce = health.get("observed_train_correctness_bce")
                observed_improvement = health.get(
                    "relative_improvement_over_prior_bce"
                )
                expected_improvement = (
                    (float(prior_bce) - float(observed_bce)) / float(prior_bce)
                    if isinstance(prior_bce, (int, float))
                    and isinstance(observed_bce, (int, float))
                    and float(prior_bce) > 0.0
                    else None
                )
                threshold = float(
                    protocol_contract["minimum_prior_relative_improvement"]
                )
                if (
                    run.get("model_variant") != variant
                    or not isinstance(experiment_protocol, Mapping)
                    or experiment_protocol.get("sha256") != protocol_contract["sha256"]
                    or not isinstance(data_state, Mapping)
                    or data_state.get("train_sha256")
                    != protocol_contract["train_input_sha256"]
                    or data_state.get("val_sha256")
                    != protocol_contract["validation_input_sha256"]
                    or health.get("schema_version") != "clir-training-health-v2"
                    or health.get("gate") != "constant_class_prior_bce"
                    or not _same_threshold(
                        health.get(
                            "minimum_relative_improvement",
                            health.get("relative_tolerance"),
                        ),
                        threshold,
                    )
                    or expected_improvement is None
                    or not math.isfinite(expected_improvement)
                    or not _same_threshold(observed_improvement, expected_improvement)
                    or expected_improvement >= threshold
                ):
                    raise ValueError(
                        f"Training failure evidence does not match the frozen protocol for "
                        f"seed={seed}, variant={variant}"
                    )
            return {
                "seed": seed,
                "variant": variant,
                "stage": "train",
                "reason": "constant_prior_health_gate_failed",
                "evidence_path": str(run_path),
                "evidence_sha256": file_sha256(run_path),
                "health_gate": dict(health),
            }

    score_health_path = (
        scored_dir / f"seed_{seed}" / f"{variant}.jsonl.health.json"
    )
    if score_health_path.is_file():
        score_health = _load_json_object(score_health_path)
        health = score_health.get("health_gate")
        if (
            score_health.get("schema_version") == "clir-scoring-health-v1"
            and score_health.get("status") == "health_gate_failed"
            and isinstance(health, Mapping)
            and health.get("enabled")
            and not health.get("passed")
        ):
            if protocol_contract is not None:
                experiment_protocol = score_health.get("experiment_protocol")
                provenance = score_health.get("scoring_provenance")
                observed_distribution = health.get("observed_distribution")
                observed_std = (
                    observed_distribution.get("population_std")
                    if isinstance(observed_distribution, Mapping)
                    else None
                )
                threshold = float(protocol_contract["minimum_score_population_std"])
                model_path = run_dir / f"seed_{seed}" / f"{variant}.pt"
                checkpoint_matches = (
                    model_path.is_file()
                    and score_health.get("checkpoint_sha256") == file_sha256(model_path)
                )
                if (
                    score_health.get("model_variant") != variant
                    or not isinstance(experiment_protocol, Mapping)
                    or experiment_protocol.get("sha256") != protocol_contract["sha256"]
                    or not isinstance(provenance, Mapping)
                    or provenance.get("experiment_protocol") != experiment_protocol
                    or provenance.get("model_variant") != variant
                    or provenance.get("checkpoint_sha256")
                    != score_health.get("checkpoint_sha256")
                    or provenance.get("input_sha256")
                    != protocol_contract["validation_input_sha256"]
                    or score_health.get("input_sha256")
                    != protocol_contract["validation_input_sha256"]
                    or provenance.get("batch_size")
                    != protocol_contract["scoring_batch_size"]
                    or provenance.get("amp_dtype")
                    != protocol_contract["scoring_amp_dtype"]
                    or provenance.get("compute_dtype")
                    != protocol_contract["scoring_compute_dtype"]
                    or not _same_threshold(provenance.get("min_score_std"), threshold)
                    or provenance.get("score_distribution") != observed_distribution
                    or health.get("gate")
                    != "minimum_validation_score_population_std"
                    or not _same_threshold(
                        health.get("minimum_population_std"),
                        threshold,
                    )
                    or not isinstance(observed_std, (int, float))
                    or not math.isfinite(float(observed_std))
                    or float(observed_std) >= threshold
                    or not checkpoint_matches
                ):
                    raise ValueError(
                        f"Scoring failure evidence does not match the frozen protocol/cell "
                        f"for seed={seed}, variant={variant}"
                    )
            return {
                "seed": seed,
                "variant": variant,
                "stage": "score",
                "reason": "score_distribution_health_gate_failed",
                "evidence_path": str(score_health_path),
                "evidence_sha256": file_sha256(score_health_path),
                "health_gate": dict(health),
            }

    raise FileNotFoundError(
        "Evaluation is absent without explicit preregistered health-gate failure "
        f"evidence for seed={seed}, variant={variant}; checked {run_path} and "
        f"{score_health_path}"
    )


def _evaluation_failure_evidence(
    report: Mapping[str, Any],
    *,
    path: Path,
    seed: int,
    variant: str,
    protocol_contract: Mapping[str, Any],
) -> Dict[str, Any] | None:
    ranking_health = _validate_report_health_contract(
        report,
        protocol_contract=protocol_contract,
    )
    if not isinstance(ranking_health, Mapping):
        return None
    if ranking_health.get("passed") or not ranking_health.get("enabled"):
        return None
    bootstrap = report.get("bootstrap")
    provenance = report.get("reward_scoring_provenance")
    k_values = [int(value) for value in report.get("k", [])]
    per_query = report.get("per_query")
    query_ids = (
        {str(row.get("query_id")) for row in per_query if isinstance(row, Mapping)}
        if isinstance(per_query, list)
        else set()
    )
    if (
        report.get("reward_model_variant") != variant
        or not isinstance(provenance, Mapping)
        or provenance.get("model_variant") != variant
        or report.get("candidate_subset") != EXPECTED_CANDIDATE_SUBSET
        or report.get("candidate_index_policy") != EXPECTED_CANDIDATE_INDEX_POLICY
        or k_values != protocol_contract["evaluation_k"]
        or report.get("query_count") != protocol_contract["evaluation_query_count"]
        or len(query_ids) != protocol_contract["evaluation_query_count"]
        or ranking_health.get("max_k") != max(protocol_contract["evaluation_k"])
        or not isinstance(bootstrap, Mapping)
        or int(bootstrap.get("seed", -1)) != seed
        or bootstrap.get("replicates") != protocol_contract["bootstrap_replicates"]
        or bootstrap.get("confidence_level") != protocol_contract["confidence_level"]
    ):
        raise ValueError(
            f"Evaluation failure evidence does not identify seed={seed}, variant={variant}"
        )
    return {
        "seed": seed,
        "variant": variant,
        "stage": "evaluate",
        "reason": "within_query_pairwise_ranking_health_gate_failed",
        "evidence_path": str(path),
        "evidence_sha256": file_sha256(path),
        "health_gate": dict(ranking_health),
    }


def main() -> None:
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing summary: {output}")
    if args.allow_failed_cells and (
        args.run_dir is None
        or args.scored_dir is None
        or args.experiment_protocol_config is None
    ):
        raise ValueError(
            "--allow-failed-cells requires --run-dir, --scored-dir, and "
            "--experiment-protocol-config"
        )
    if not args.allow_failed_cells and (args.run_dir is not None or args.scored_dir is not None):
        raise ValueError("--run-dir/--scored-dir require --allow-failed-cells")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    scored_dir = Path(args.scored_dir).resolve() if args.scored_dir else None
    protocol_contract = (
        _protocol_contract(args.experiment_protocol_config)
        if args.experiment_protocol_config
        else None
    )
    if protocol_contract is not None:
        if (
            sorted(set(args.seeds)) != sorted(protocol_contract["training_seeds"])
            or list(args.variants) != protocol_contract["variants"]
            or args.primary_k != max(protocol_contract["evaluation_k"])
            or args.bootstrap_replicates != protocol_contract["bootstrap_replicates"]
            or not _same_threshold(
                args.confidence_level,
                float(protocol_contract["confidence_level"]),
            )
        ):
            raise ValueError("Summary CLI differs from the frozen experiment protocol")
        if args.allow_failed_cells:
            failure_policy = protocol_contract.get("failed_cell_policy")
            if (
                not isinstance(failure_policy, Mapping)
                or failure_policy.get("allow_explicit_health_gate_failures") is not True
                or failure_policy.get("unknown_or_unrun_cells") != "error"
                or failure_policy.get("incomplete_summary")
                != "diagnostic_only_no_formal_primary_claim"
            ):
                raise ValueError(
                    "Experiment protocol does not authorize the frozen failed-cell policy"
                )

    reports: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    inputs: Dict[str, Any] = {}
    failed_cells: list[Dict[str, Any]] = []
    for seed in sorted(set(args.seeds)):
        reports[seed] = {}
        inputs[str(seed)] = {}
        for variant in args.variants:
            path = evaluation_dir / f"seed_{seed}" / f"{variant}.json"
            if path.resolve() == output:
                raise ValueError("Summary output must differ from every evaluation input")
            if not path.is_file():
                if not args.allow_failed_cells:
                    raise FileNotFoundError(f"Evaluation report is absent: {path}")
                assert run_dir is not None and scored_dir is not None
                evidence = _health_failure_evidence(
                    seed=seed,
                    variant=variant,
                    run_dir=run_dir,
                    scored_dir=scored_dir,
                    protocol_contract=protocol_contract,
                )
                failed_cells.append(evidence)
                inputs[str(seed)][variant] = {
                    "status": "excluded_explicit_health_failure",
                    "failure_evidence": evidence,
                }
                continue
            report = _load_json_object(path)
            evaluation_failure = (
                _evaluation_failure_evidence(
                    report,
                    path=path,
                    seed=seed,
                    variant=variant,
                    protocol_contract=protocol_contract,
                )
                if args.allow_failed_cells and protocol_contract is not None
                else None
            )
            if evaluation_failure is not None:
                evidence = evaluation_failure
                failed_cells.append(evidence)
                inputs[str(seed)][variant] = {
                    "status": "excluded_explicit_health_failure",
                    "failure_evidence": evidence,
                }
                continue
            reports[seed][variant] = report
            inputs[str(seed)][variant] = {
                "status": "included",
                "path": str(path),
                "sha256": file_sha256(path),
            }

    summary = summarize_evaluation_reports(
        reports,
        variants=args.variants,
        primary_k=args.primary_k,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.seed,
        allow_incomplete=args.allow_failed_cells,
        protocol_contract=protocol_contract,
    )
    summary["inputs"] = inputs
    summary["matrix_status"] = {
        "expected_cell_count": len(set(args.seeds)) * len(args.variants),
        "included_cell_count": sum(len(seed_reports) for seed_reports in reports.values()),
        "failed_cell_count": len(failed_cells),
        "failed_cells": failed_cells,
        "omission_policy": (
            "only_explicit_preregistered_health_gate_failures"
            if args.allow_failed_cells
            else "none"
        ),
        "unknown_or_unrun_cells_allowed": False,
    }
    if protocol_contract is not None:
        summary["experiment_protocol"] = dict(protocol_contract)
    atomic_write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
