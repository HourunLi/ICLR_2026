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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
) -> tuple[list[int], list[int], int]:
    if not reports:
        raise ValueError("No evaluation reports were provided")
    seeds = sorted(int(seed) for seed in reports)
    reference: Mapping[str, Any] | None = None
    reference_query_ids: set[str] | None = None
    for seed in seeds:
        seed_reports = reports[seed]
        missing = [variant for variant in variants if variant not in seed_reports]
        if missing:
            raise ValueError(f"Seed {seed} is missing variants: {missing}")
        for variant in variants:
            report = seed_reports[variant]
            if report.get("schema_version") != "clir-query-evaluation-v1":
                raise ValueError(f"Unsupported evaluation schema for seed {seed}/{variant}")
            k_values = [int(k) for k in report["k"]]
            if primary_k not in k_values:
                raise ValueError(f"Primary k={primary_k} is absent for seed {seed}/{variant}")
            query_ids = {str(row["query_id"]) for row in report["per_query"]}
            if len(query_ids) != int(report["query_count"]):
                raise ValueError(f"Query count mismatch for seed {seed}/{variant}")
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
                    for variant in variants
                ]
                if any(not math.isclose(value, values[0], abs_tol=1e-12) for value in values[1:]):
                    raise ValueError(
                        f"Candidate-independent baseline mismatch at seed={seed}, k={k}, "
                        f"metric={metric_name}"
                    )
                reference_values = _query_values(
                    seed_reports[variants[0]], k, per_query_field
                )
                for variant in variants[1:]:
                    if _query_values(seed_reports[variant], k, per_query_field) != reference_values:
                        raise ValueError(
                            f"Per-query baseline mismatch at seed={seed}, k={k}, "
                            f"metric={metric_name}"
                        )
    assert reference is not None
    return seeds, [int(k) for k in reference["k"]], int(reference["query_count"])


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
        paired = [to_values[query_id] - from_values[query_id] for query_id in sorted(to_values)]
        differences_by_seed[training_seed] = float(statistics.mean(paired))
        intervals[str(training_seed)] = _paired_bootstrap_ci(
            paired,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=_stable_bootstrap_seed(bootstrap_seed, comparison_name, training_seed),
        )
    return {
        "from": from_source,
        "to": to_variant,
        "direction": "to_minus_from_accuracy",
        **_seed_statistics(differences_by_seed),
        "paired_query_bootstrap_ci_by_seed": intervals,
    }


def summarize_evaluation_reports(
    reports: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    primary_k: int = 16,
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    if len(set(variants)) != len(variants) or not variants:
        raise ValueError("variants must be non-empty and unique")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    seeds, k_values, query_count = _validate_reports(
        reports, variants=variants, primary_k=primary_k
    )

    per_variant: Dict[str, Any] = {}
    for variant in variants:
        metrics: Dict[str, Any] = {}
        for k in k_values:
            values = {
                seed: float(reports[seed][variant]["metrics"][str(k)]["reward_bon_accuracy"]["value"])
                for seed in seeds
            }
            metrics[str(k)] = {
                "reward_bon_accuracy": {
                    **_seed_statistics(values),
                    "query_bootstrap_ci_by_seed": {
                        str(seed): list(
                            reports[seed][variant]["metrics"][str(k)]["reward_bon_accuracy"][
                                "query_bootstrap_ci"
                            ]
                        )
                        for seed in seeds
                    },
                }
            }
        per_variant[variant] = {"metrics": metrics}

    baselines: Dict[str, Any] = {}
    reference_variant = variants[0]
    for metric_name in (
        "random_expected_accuracy",
        "random_seeded_accuracy",
        "oracle_accuracy",
    ):
        baselines[metric_name] = {}
        for k in k_values:
            values = {
                seed: float(
                    reports[seed][reference_variant]["metrics"][str(k)][metric_name]["value"]
                )
                for seed in seeds
            }
            baselines[metric_name][str(k)] = _seed_statistics(values)

    comparisons: Dict[str, Any] = {}
    for variant in variants:
        name = f"random_expected_to_{variant}"
        comparisons[name] = _paired_comparison(
            reports,
            seeds=seeds,
            comparison_name=name,
            from_source="random_expected",
            to_variant=variant,
            primary_k=primary_k,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
    if "strict_swift" in variants and "encoded_swift" in variants:
        name = "strict_swift_to_encoded_swift"
        comparisons[name] = _paired_comparison(
            reports,
            seeds=seeds,
            comparison_name=name,
            from_source="strict_swift",
            to_variant="encoded_swift",
            primary_k=primary_k,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
    if "encoded_swift" in variants and "clir" in variants:
        name = "encoded_swift_to_clir"
        comparisons[name] = _paired_comparison(
            reports,
            seeds=seeds,
            comparison_name=name,
            from_source="encoded_swift",
            to_variant="clir",
            primary_k=primary_k,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )

    return {
        "schema_version": "clir-multiseed-evaluation-summary-v1",
        "seed_aggregation": "arithmetic_mean_and_sample_standard_deviation",
        "seeds": seeds,
        "variants": list(variants),
        "query_count": query_count,
        "k": k_values,
        "primary_k": primary_k,
        "paired_bootstrap": {
            "unit": "query",
            "replicates": bootstrap_replicates,
            "confidence_level": confidence_level,
            "seed": bootstrap_seed,
        },
        "per_variant": per_variant,
        "baselines": baselines,
        "primary_comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing summary: {output}")

    reports: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    inputs: Dict[str, Any] = {}
    for seed in sorted(set(args.seeds)):
        reports[seed] = {}
        inputs[str(seed)] = {}
        for variant in args.variants:
            path = evaluation_dir / f"seed_{seed}" / f"{variant}.json"
            with path.open(encoding="utf-8") as handle:
                reports[seed][variant] = json.load(handle)
            inputs[str(seed)][variant] = {
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
    )
    summary["inputs"] = inputs
    atomic_write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
