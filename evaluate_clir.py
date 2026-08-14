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


EXPECTED_CANDIDATE_INDEX_POLICY = "vllm_completion_output_index"
EXPECTED_CANDIDATE_SUBSET = "first_k_by_vllm_completion_output_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--score-field", default="reward_score")
    parser.add_argument("--k", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-input-sha256", default=None)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument(
        "--expected-reward-model-variant",
        default=None,
        choices=["strict_swift", "encoded_swift", "clir"],
    )
    parser.add_argument("--expected-scoring-input-sha256", default=None)
    parser.add_argument("--expected-scoring-batch-size", type=int, default=None)
    parser.add_argument("--expected-experiment-protocol-sha256", default=None)
    parser.add_argument(
        "--expected-scoring-amp-dtype",
        default=None,
        choices=["none", "bfloat16"],
    )
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


def _validated_scoring_provenance(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_field: str,
) -> tuple[str, Dict[str, Any]]:
    """Require one checkpoint/variant/scoring contract for the whole manifest."""

    variants = {str(row.get("reward_model_variant", "missing")) for row in rows}
    if len(variants) != 1 or "missing" in variants:
        raise ValueError(
            "Scored manifest must contain exactly one reward_model_variant; "
            f"got {sorted(variants)}"
        )
    variant = next(iter(variants))
    if variant not in {"strict_swift", "encoded_swift", "clir"}:
        raise ValueError(f"Unsupported reward_model_variant in scored manifest: {variant!r}")

    provenances: list[Dict[str, Any]] = []
    for row in rows:
        value = row.get("reward_scoring_provenance")
        if not isinstance(value, Mapping):
            raise ValueError("Every scored row requires reward_scoring_provenance")
        provenance = dict(value)
        required = {
            "schema_version",
            "model_variant",
            "checkpoint_sha256",
            "input_sha256",
            "batch_size",
            "amp_dtype",
            "compute_dtype",
            "min_score_std",
            "score_distribution",
        }
        missing = sorted(required - set(provenance))
        if missing:
            raise ValueError(f"Scoring provenance is missing required fields: {missing}")
        if provenance["schema_version"] != "clir-reward-scoring-v2":
            raise ValueError(
                "Unsupported scoring provenance schema: "
                f"{provenance['schema_version']!r}"
            )
        if provenance["model_variant"] != variant:
            raise ValueError("reward_model_variant disagrees with scoring provenance")
        for key in ("checkpoint_sha256", "input_sha256"):
            digest = provenance[key]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest.lower())
            ):
                raise ValueError(f"Scoring provenance has invalid {key}")
        if isinstance(provenance["batch_size"], bool) or int(provenance["batch_size"]) <= 0:
            raise ValueError("Scoring provenance has invalid batch_size")
        if provenance["amp_dtype"] not in {"none", "bfloat16"}:
            raise ValueError("Scoring provenance has invalid amp_dtype")
        expected_compute = "bfloat16" if provenance["amp_dtype"] == "bfloat16" else "float32"
        if provenance["compute_dtype"] != expected_compute:
            raise ValueError("Scoring provenance amp_dtype/compute_dtype disagree")
        threshold = provenance["min_score_std"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError("Scoring provenance has invalid min_score_std")
        distribution = provenance["score_distribution"]
        if not isinstance(distribution, Mapping):
            raise ValueError("Scoring provenance score_distribution must be an object")
        provenances.append(provenance)

    reference = provenances[0]
    if any(provenance != reference for provenance in provenances[1:]):
        raise ValueError(
            "Scored manifest mixes rows from different scoring provenance/checkpoints"
        )
    scores = []
    for row in rows:
        score = row.get(score_field)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Scored manifest has an invalid {score_field}")
        scores.append(float(score))
    score_array = np.asarray(scores, dtype=np.float64)
    actual_distribution = {
        "count": len(scores),
        "mean": float(score_array.mean()),
        "population_std": float(score_array.std(ddof=0)),
        "min": float(score_array.min()),
        "max": float(score_array.max()),
    }
    recorded_distribution = reference["score_distribution"]
    if set(recorded_distribution) != set(actual_distribution):
        raise ValueError("Scoring provenance score_distribution fields are invalid")
    for key, actual in actual_distribution.items():
        recorded = recorded_distribution[key]
        if key == "count":
            matches = not isinstance(recorded, bool) and recorded == actual
        else:
            matches = (
                not isinstance(recorded, bool)
                and isinstance(recorded, (int, float))
                and math.isfinite(recorded)
                and math.isclose(float(recorded), actual, rel_tol=1e-12, abs_tol=1e-12)
            )
        if not matches:
            raise ValueError(
                f"Scoring provenance score_distribution {key} does not match scores"
            )
    if actual_distribution["population_std"] < float(reference["min_score_std"]):
        raise ValueError("Scored manifest fails its recorded minimum score-std gate")
    return variant, reference


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

    candidate_index_policies = {
        str(row.get("generation", {}).get("candidate_index_policy", "missing"))
        for row in rows
    }
    if candidate_index_policies != {EXPECTED_CANDIDATE_INDEX_POLICY}:
        raise ValueError(
            "Ordered-prefix evaluation requires candidate_index_policy "
            f"{EXPECTED_CANDIDATE_INDEX_POLICY!r}, got {sorted(candidate_index_policies)}"
        )

    reward_model_variant, scoring_provenance = _validated_scoring_provenance(
        rows,
        score_field=score_field,
    )

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
        tie_sizes: list[int] = []
        for query_position, query_id in enumerate(query_order):
            pool = normalized[query_id][:k]
            labels = [int(row["correctness"]) for row in pool]
            scores = [float(row[score_field]) for row in pool]
            best_score = max(scores)
            tied = [index for index, score in enumerate(scores) if score == best_score]
            selected = tied[0]
            tie_sizes.append(len(tied))
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
                "reward_top_tie_size": len(tied),
                "reward_top_tied_candidate_indices": [
                    int(pool[index]["candidate_index"]) for index in tied
                ],
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
        metrics[str(k)]["reward_selection_ties"] = {
            "query_count": sum(size > 1 for size in tie_sizes),
            "query_fraction": _mean([float(size > 1) for size in tie_sizes]),
            "max_tie_size": max(tie_sizes),
            "tie_break_policy": "lowest_candidate_index",
        }

    max_pool_correct = [
        sum(int(row["correctness"]) for row in normalized[query_id][:max_k])
        for query_id in query_order
    ]
    return {
        "schema_version": "clir-query-evaluation-v2",
        "score_field": score_field,
        "reward_model_variant": reward_model_variant,
        "reward_scoring_provenance": scoring_provenance,
        "candidate_subset": EXPECTED_CANDIDATE_SUBSET,
        "candidate_index_policy": EXPECTED_CANDIDATE_INDEX_POLICY,
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
    if output == Path(args.input_jsonl).resolve():
        raise ValueError("Evaluation output must differ from the scored input JSONL")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output}")
    input_sha256 = file_sha256(args.input_jsonl)
    if args.expected_input_sha256 and input_sha256 != args.expected_input_sha256:
        raise ValueError(
            f"Input manifest SHA256 mismatch: expected {args.expected_input_sha256}, got {input_sha256}"
        )
    rows = read_jsonl(args.input_jsonl)
    report = evaluate_candidate_rows(
        rows,
        score_field=args.score_field,
        k_values=args.k,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    provenance = report["reward_scoring_provenance"]
    expected_values = {
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "input_sha256": args.expected_scoring_input_sha256,
        "batch_size": args.expected_scoring_batch_size,
        "amp_dtype": args.expected_scoring_amp_dtype,
    }
    for key, expected in expected_values.items():
        if expected is not None and provenance[key] != expected:
            raise ValueError(
                f"Scoring provenance {key} mismatch: expected {expected!r}, "
                f"got {provenance[key]!r}"
            )
    if args.expected_experiment_protocol_sha256 is not None:
        experiment_protocol = provenance.get("experiment_protocol")
        actual_protocol_sha256 = (
            experiment_protocol.get("sha256")
            if isinstance(experiment_protocol, Mapping)
            else None
        )
        if actual_protocol_sha256 != args.expected_experiment_protocol_sha256:
            raise ValueError(
                "Scoring experiment protocol SHA256 mismatch: expected "
                f"{args.expected_experiment_protocol_sha256!r}, "
                f"got {actual_protocol_sha256!r}"
            )
    if (
        args.expected_reward_model_variant is not None
        and report["reward_model_variant"] != args.expected_reward_model_variant
    ):
        raise ValueError(
            "Reward model variant mismatch: expected "
            f"{args.expected_reward_model_variant!r}, got {report['reward_model_variant']!r}"
        )
    report["input_jsonl"] = str(Path(args.input_jsonl).resolve())
    report["input_sha256"] = input_sha256
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
