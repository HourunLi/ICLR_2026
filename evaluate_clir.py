"""Evaluate scored candidate pools with query-level random, oracle, and Best-of-N metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.clir_data import read_jsonl
from src.clir_real_data import file_sha256
from src.clir_stage_a import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--score-field", default="reward_score")
    parser.add_argument("--k", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _bootstrap_ci(
    values: Sequence[float],
    *,
    replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty query set")
    if replicates <= 0:
        return [_mean(values), _mean(values)]
    samples = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[samples].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))]


def _seeded_random_index(query_id: str, k: int, seed: int) -> int:
    payload = f"{seed}:{k}:{query_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % k


def evaluate_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_field: str,
    k_values: Sequence[int],
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Scored candidate manifest is empty")
    k_values = sorted(set(int(k) for k in k_values))
    if not k_values or k_values[0] <= 0:
        raise ValueError("All k values must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    grouped: Dict[str, list[Mapping[str, Any]]] = {}
    query_order: list[str] = []
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise ValueError("Every row requires query_id")
        if query_id not in grouped:
            query_order.append(query_id)
            grouped[query_id] = []
        grouped[query_id].append(row)

    max_k = max(k_values)
    normalized: Dict[str, list[Mapping[str, Any]]] = {}
    for query_id in query_order:
        query_rows = sorted(grouped[query_id], key=lambda row: int(row.get("candidate_index", -1)))
        indices = [int(row.get("candidate_index", -1)) for row in query_rows]
        if indices[:max_k] != list(range(max_k)):
            raise ValueError(f"Query {query_id} lacks contiguous first-{max_k} candidates")
        for row in query_rows[:max_k]:
            label = row.get("correctness")
            score = row.get(score_field)
            if label not in (0, 1):
                raise ValueError(f"Query {query_id} has a non-binary correctness label")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"Query {query_id} has an invalid {score_field}")
        normalized[query_id] = query_rows

    rng = np.random.default_rng(seed)
    metrics: Dict[str, Any] = {}
    per_query: list[Dict[str, Any]] = [{"query_id": query_id, "k": {}} for query_id in query_order]
    for k in k_values:
        reward_values: list[float] = []
        random_expected_values: list[float] = []
        random_seeded_values: list[float] = []
        oracle_values: list[float] = []
        for query_position, query_id in enumerate(query_order):
            pool = normalized[query_id][:k]
            labels = [int(row["correctness"]) for row in pool]
            scores = [float(row[score_field]) for row in pool]
            selected = max(range(k), key=lambda index: scores[index])
            random_index = _seeded_random_index(query_id, k, seed)
            values = {
                "reward": float(labels[selected]),
                "random_expected": _mean(labels),
                "random_seeded": float(labels[random_index]),
                "oracle": float(any(labels)),
            }
            reward_values.append(values["reward"])
            random_expected_values.append(values["random_expected"])
            random_seeded_values.append(values["random_seeded"])
            oracle_values.append(values["oracle"])
            per_query[query_position]["k"][str(k)] = {
                **values,
                "reward_selected_candidate_index": int(pool[selected]["candidate_index"]),
                "random_seeded_candidate_index": int(pool[random_index]["candidate_index"]),
                "correct_candidates": sum(labels),
            }

        metric_values = {
            "reward_bon_accuracy": reward_values,
            "random_expected_accuracy": random_expected_values,
            "random_seeded_accuracy": random_seeded_values,
            "oracle_accuracy": oracle_values,
        }
        metrics[str(k)] = {
            name: {
                "value": _mean(values),
                "query_bootstrap_ci": _bootstrap_ci(
                    values,
                    replicates=bootstrap_replicates,
                    confidence_level=confidence_level,
                    rng=rng,
                ),
            }
            for name, values in metric_values.items()
        }

    max_pool_correct = [
        sum(int(row["correctness"]) for row in normalized[query_id][:max_k])
        for query_id in query_order
    ]
    return {
        "schema_version": "clir-query-evaluation-v1",
        "score_field": score_field,
        "candidate_subset": "first_k_in_generation_order",
        "query_count": len(query_order),
        "row_count": len(rows),
        "k": k_values,
        "bootstrap": {
            "unit": "query",
            "replicates": bootstrap_replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "max_k_pool_counts": {
            "mixed": sum(0 < count < max_k for count in max_pool_correct),
            "all_correct": sum(count == max_k for count in max_pool_correct),
            "all_wrong": sum(count == 0 for count in max_pool_correct),
        },
        "metrics": metrics,
        "per_query": per_query,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output}")
    rows = read_jsonl(args.input_jsonl)
    report = evaluate_candidate_rows(
        rows,
        score_field=args.score_field,
        k_values=args.k,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    report["input_jsonl"] = str(Path(args.input_jsonl).resolve())
    report["input_sha256"] = file_sha256(args.input_jsonl)
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
