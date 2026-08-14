#!/usr/bin/env python
"""Run the preregistered Stage 1B mixed-pool diagnostic without tuning models."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl
from src.clir_real_data import file_sha256
from src.clir_stage_a import atomic_write_json, git_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="METHOD:SEED:JSONL",
        help="Repeat once per scored manifest.",
    )
    parser.add_argument(
        "--label-jsonl",
        default=None,
        help="Optional versioned label manifest overlaid by row id.",
    )
    parser.add_argument("--max-k", type=int, default=16)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_run_spec(spec: str) -> tuple[str, int, Path]:
    parts = spec.split(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError(f"Invalid --run {spec!r}; expected METHOD:SEED:JSONL")
    try:
        seed = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Run seed must be an integer in {spec!r}") from exc
    return parts[0], seed, Path(parts[2]).resolve()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _optional_mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return _mean(finite) if finite else None


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float], *, rank: bool = False) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if rank:
        left_array = _average_ranks(left_array)
        right_array = _average_ranks(right_array)
    if np.ptp(left_array) == 0 or np.ptp(right_array) == 0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _pairwise_ranking_accuracy(correct_scores: Sequence[float], wrong_scores: Sequence[float]) -> float:
    comparisons = [
        1.0 if correct > wrong else 0.5 if correct == wrong else 0.0
        for correct in correct_scores
        for wrong in wrong_scores
    ]
    return _mean(comparisons)


def _normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities.astype(np.float64)
    probabilities = probabilities / probabilities.sum()
    if probabilities.size <= 1:
        return 0.0
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log(nonzero)) / np.log(probabilities.size))


def _label_map(path: Path | None) -> tuple[Dict[str, int], str | None, str | None]:
    if path is None:
        return {}, None, None
    rows = read_jsonl(path)
    labels: Dict[str, int] = {}
    versions = set()
    for row in rows:
        identifier = str(row.get("id", ""))
        label = row.get("correctness")
        if not identifier or isinstance(label, bool) or label not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Invalid label row in {path}: {identifier!r}")
        if identifier in labels:
            raise ValueError(f"Duplicate label id {identifier!r} in {path}")
        labels[identifier] = int(label)
        if row.get("checker_version") is not None:
            versions.add(str(row["checker_version"]))
    if len(versions) > 1:
        raise ValueError(f"Label manifest mixes checker versions: {sorted(versions)}")
    return labels, next(iter(versions), None), file_sha256(path)


def _candidate_order_status(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    policies = {
        str(row.get("generation", {}).get("candidate_index_policy", "missing"))
        for row in rows
    }
    valid = policies == {"vllm_completion_output_index"}
    return {
        "policies": sorted(policies),
        "ordered_prefix_valid": valid,
        "diagnostic_scope": (
            "ordered prefixes and full max-k pool"
            if valid
            else "full max-k pool only; legacy candidate indices are likelihood-ranked"
        ),
    }


def _analyze_run(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    seed: int,
    max_k: int,
    labels_by_id: Mapping[str, int],
) -> tuple[Dict[str, Any], Dict[str, tuple[int, int]]]:
    grouped: Dict[str, list[Mapping[str, Any]]] = {}
    query_order: list[str] = []
    for row in rows:
        identifier = str(row.get("id", ""))
        query_id = str(row.get("query_id", ""))
        if not identifier or not query_id:
            raise ValueError("Every scored row requires id and query_id")
        if query_id not in grouped:
            query_order.append(query_id)
            grouped[query_id] = []
        grouped[query_id].append(row)

    pool_counts = {"mixed": 0, "all_correct": 0, "all_wrong": 0}
    per_query: list[Dict[str, Any]] = []
    selections: Dict[str, tuple[int, int]] = {}
    all_mixed_scores: list[float] = []
    all_mixed_lengths: list[float] = []
    all_pairwise: list[float] = []
    attention_all: list[np.ndarray] = []
    attention_correct: list[np.ndarray] = []
    attention_wrong: list[np.ndarray] = []

    for query_id in query_order:
        pool = sorted(grouped[query_id], key=lambda row: int(row.get("candidate_index", -1)))
        indices = [int(row.get("candidate_index", -1)) for row in pool]
        if len(pool) < max_k or indices[:max_k] != list(range(max_k)):
            raise ValueError(f"Query {query_id} lacks a contiguous full {max_k}-candidate pool")
        pool = pool[:max_k]
        labels = [
            int(labels_by_id.get(str(row["id"]), row.get("correctness", -1)))
            for row in pool
        ]
        scores = [float(row.get("reward_score", math.nan)) for row in pool]
        lengths = [float(len(row.get("output_token_ids", []))) for row in pool]
        if any(label not in (0, 1) for label in labels):
            raise ValueError(f"Query {query_id} has a non-binary label")
        if not all(math.isfinite(score) for score in scores):
            raise ValueError(f"Query {query_id} has a non-finite reward score")
        if any(length <= 0 for length in lengths):
            raise ValueError(f"Query {query_id} has an empty output token sequence")

        correct_count = sum(labels)
        if correct_count == 0:
            pool_type = "all_wrong"
        elif correct_count == max_k:
            pool_type = "all_correct"
        else:
            pool_type = "mixed"
        pool_counts[pool_type] += 1
        if pool_type != "mixed":
            continue

        correct_positions = [position for position, label in enumerate(labels) if label]
        wrong_positions = [position for position, label in enumerate(labels) if not label]
        selected_position = max(range(max_k), key=lambda position: scores[position])
        runner_positions = sorted(range(max_k), key=lambda position: scores[position], reverse=True)
        best_correct = max(correct_positions, key=lambda position: scores[position])
        best_wrong = max(wrong_positions, key=lambda position: scores[position])
        correct_scores = [scores[position] for position in correct_positions]
        wrong_scores = [scores[position] for position in wrong_positions]
        pairwise = _pairwise_ranking_accuracy(correct_scores, wrong_scores)
        all_pairwise.append(pairwise)
        all_mixed_scores.extend(scores)
        all_mixed_lengths.extend(lengths)

        layer_rows: list[np.ndarray] = []
        for position, row in enumerate(pool):
            raw_attention = row.get("mean_layer_pool_attention")
            if raw_attention is None:
                continue
            attention = np.asarray(raw_attention, dtype=np.float64)
            if attention.ndim != 2 or not np.isfinite(attention).all():
                raise ValueError(f"Invalid layer attention in {row['id']}")
            attention = attention.mean(axis=0)
            attention = attention / attention.sum()
            layer_rows.append(attention)
            attention_all.append(attention)
            (attention_correct if labels[position] else attention_wrong).append(attention)

        record = {
            "query_id": query_id,
            "selected_candidate_index": indices[selected_position],
            "selected_correctness": labels[selected_position],
            "selected_length": int(lengths[selected_position]),
            "correct_candidates": correct_count,
            "best_correct_candidate_index": indices[best_correct],
            "best_wrong_candidate_index": indices[best_wrong],
            "correct_vs_wrong_margin": scores[best_correct] - scores[best_wrong],
            "selected_vs_runner_up_margin": scores[runner_positions[0]] - scores[runner_positions[1]],
            "pairwise_correct_over_wrong": pairwise,
            "score_length_pearson": _correlation(scores, lengths),
            "score_length_spearman": _correlation(scores, lengths, rank=True),
            "mean_correct_length": _mean([lengths[position] for position in correct_positions]),
            "mean_wrong_length": _mean([lengths[position] for position in wrong_positions]),
            "mean_correct_score": _mean(correct_scores),
            "mean_wrong_score": _mean(wrong_scores),
        }
        if layer_rows:
            record["mean_layer_attention"] = np.mean(layer_rows, axis=0).tolist()
        per_query.append(record)
        selections[query_id] = (indices[selected_position], labels[selected_position])

    if not per_query:
        raise ValueError("No mixed max-k candidate pools were found")
    margins = [float(row["correct_vs_wrong_margin"]) for row in per_query]
    selected_labels = [int(row["selected_correctness"]) for row in per_query]
    run_report: Dict[str, Any] = {
        "method": method,
        "seed": seed,
        "query_count": len(query_order),
        "max_k": max_k,
        "pool_counts": pool_counts,
        "mixed_pool_metrics": {
            "selection_accuracy": _mean(selected_labels),
            "selection_correct": sum(selected_labels),
            "selection_wrong": len(selected_labels) - sum(selected_labels),
            "correct_vs_wrong_margin_mean": _mean(margins),
            "correct_vs_wrong_margin_median": float(np.median(margins)),
            "correct_vs_wrong_margin_min": min(margins),
            "nonpositive_margin_queries": sum(margin <= 0 for margin in margins),
            "pairwise_correct_over_wrong_mean": _mean(all_pairwise),
            "score_length_pearson_global": _correlation(all_mixed_scores, all_mixed_lengths),
            "score_length_spearman_global": _correlation(
                all_mixed_scores, all_mixed_lengths, rank=True
            ),
            "score_length_pearson_query_mean": _optional_mean(
                [row["score_length_pearson"] for row in per_query]
            ),
            "score_length_spearman_query_mean": _optional_mean(
                [row["score_length_spearman"] for row in per_query]
            ),
            "correct_minus_wrong_length_mean": _mean(
                [row["mean_correct_length"] - row["mean_wrong_length"] for row in per_query]
            ),
        },
        "per_mixed_query": per_query,
    }
    if attention_all:
        mean_attention = np.mean(attention_all, axis=0)
        correct_attention = np.mean(attention_correct, axis=0)
        wrong_attention = np.mean(attention_wrong, axis=0)
        top = np.argsort(mean_attention)[::-1][: min(8, mean_attention.size)]
        run_report["layer_attention"] = {
            "layer_count": int(mean_attention.size),
            "mean": mean_attention.tolist(),
            "correct_minus_wrong": (correct_attention - wrong_attention).tolist(),
            "normalized_entropy": _normalized_entropy(mean_attention),
            "top_layers": [
                {"layer_index": int(index), "weight": float(mean_attention[index])}
                for index in top
            ],
        }
    return run_report, selections


def build_diagnostic_report(
    run_specs: Sequence[tuple[str, int, Path]],
    *,
    label_path: Path | None,
    max_k: int,
) -> Dict[str, Any]:
    if max_k <= 1:
        raise ValueError("max_k must be greater than one")
    if not run_specs:
        raise ValueError("At least one run is required")
    labels_by_id, checker_version, label_sha = _label_map(label_path)

    names = [(method, seed) for method, seed, _ in run_specs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate method/seed run")

    base_signature: Dict[str, tuple[str, int, int]] | None = None
    base_order_status: Dict[str, Any] | None = None
    runs: list[Dict[str, Any]] = []
    selections_by_run: Dict[tuple[str, int], Dict[str, tuple[int, int]]] = {}
    inputs: list[Dict[str, Any]] = []
    for method, seed, path in run_specs:
        rows = read_jsonl(path)
        signature = {
            str(row.get("id")): (
                str(row.get("query_id")),
                int(row.get("candidate_index", -1)),
                len(row.get("output_token_ids", [])),
            )
            for row in rows
        }
        if len(signature) != len(rows):
            raise ValueError(f"Duplicate row id in {path}")
        if base_signature is None:
            base_signature = signature
            base_order_status = _candidate_order_status(rows)
            if labels_by_id and set(labels_by_id) != set(signature):
                missing = sorted(set(signature) - set(labels_by_id))[:5]
                extra = sorted(set(labels_by_id) - set(signature))[:5]
                raise ValueError(f"Label/run id mismatch; missing={missing}, extra={extra}")
        elif signature != base_signature:
            raise ValueError(f"Candidate membership differs in {path}")
        run_report, selections = _analyze_run(
            rows,
            method=method,
            seed=seed,
            max_k=max_k,
            labels_by_id=labels_by_id,
        )
        runs.append(run_report)
        selections_by_run[(method, seed)] = selections
        inputs.append({
            "method": method,
            "seed": seed,
            "jsonl": str(path),
            "sha256": file_sha256(path),
        })

    method_aggregates: Dict[str, Any] = {}
    for method in sorted({method for method, _, _ in run_specs}):
        method_runs = [run for run in runs if run["method"] == method]
        accuracies = [run["mixed_pool_metrics"]["selection_accuracy"] for run in method_runs]
        margins = [run["mixed_pool_metrics"]["correct_vs_wrong_margin_mean"] for run in method_runs]
        pairwise = [run["mixed_pool_metrics"]["pairwise_correct_over_wrong_mean"] for run in method_runs]
        query_ids = sorted(set.intersection(*[
            set(selections_by_run[(method, int(run["seed"]))]) for run in method_runs
        ]))
        distinct_counts = [
            len({selections_by_run[(method, int(run["seed"]))][query_id][0] for run in method_runs})
            for query_id in query_ids
        ]
        method_aggregates[method] = {
            "seeds": [int(run["seed"]) for run in method_runs],
            "mixed_selection_accuracy_mean": _mean(accuracies),
            "mixed_selection_accuracy_sample_std": _sample_std(accuracies),
            "correct_vs_wrong_margin_mean_across_seeds": _mean(margins),
            "pairwise_correct_over_wrong_mean_across_seeds": _mean(pairwise),
            "unanimous_selected_candidate_fraction": _mean(
                [float(count == 1) for count in distinct_counts]
            ),
            "mean_distinct_selected_candidates": _mean(distinct_counts),
        }

    matched_seed_comparisons: list[Dict[str, Any]] = []
    for seed in sorted({seed for _, seed, _ in run_specs}):
        methods = sorted(method for method, run_seed, _ in run_specs if run_seed == seed)
        for left, right in combinations(methods, 2):
            left_selections = selections_by_run[(left, seed)]
            right_selections = selections_by_run[(right, seed)]
            query_ids = sorted(set(left_selections) & set(right_selections))
            outcomes = [
                (left_selections[query_id][1], right_selections[query_id][1])
                for query_id in query_ids
            ]
            matched_seed_comparisons.append({
                "seed": seed,
                "left": left,
                "right": right,
                "mixed_queries": len(query_ids),
                "selection_disagreements": sum(
                    left_selections[query_id][0] != right_selections[query_id][0]
                    for query_id in query_ids
                ),
                "both_correct": sum(pair == (1, 1) for pair in outcomes),
                "left_only_correct": sum(pair == (1, 0) for pair in outcomes),
                "right_only_correct": sum(pair == (0, 1) for pair in outcomes),
                "both_wrong": sum(pair == (0, 0) for pair in outcomes),
            })

    return {
        "schema_version": "clir-stage1b-diagnostic-v1",
        "diagnostic_policy": "fixed historical checkpoints; max-k mixed pools only; no tuning",
        "max_k": max_k,
        "candidate_order": base_order_status,
        "label_overlay": {
            "jsonl": str(label_path) if label_path else None,
            "sha256": label_sha,
            "checker_version": checker_version,
        },
        "inputs": inputs,
        "runs": runs,
        "method_aggregates": method_aggregates,
        "matched_seed_comparisons": matched_seed_comparisons,
        "code": git_state(PROJECT_ROOT),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output}")
    run_specs = [_parse_run_spec(spec) for spec in args.run]
    label_path = Path(args.label_jsonl).resolve() if args.label_jsonl else None
    report = build_diagnostic_report(run_specs, label_path=label_path, max_k=args.max_k)
    atomic_write_json(output, report)
    compact = {
        "output_json": str(output),
        "candidate_order": report["candidate_order"],
        "method_aggregates": report["method_aggregates"],
        "matched_seed_comparisons": report["matched_seed_comparisons"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
