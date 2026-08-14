from copy import deepcopy

import numpy as np
import pytest

from evaluate_clir import evaluate_candidate_rows
from summarize_clir import summarize_evaluation_reports


def _report(scores, variant="clir", checkpoint_character="a"):
    flattened_scores = np.asarray(
        [score for query_scores in scores.values() for score in query_scores],
        dtype=np.float64,
    )
    provenance = {
        "schema_version": "clir-reward-scoring-v2",
        "model_variant": variant,
        "checkpoint_sha256": checkpoint_character * 64,
        "input_sha256": "f" * 64,
        "batch_size": 2,
        "amp_dtype": "none",
        "compute_dtype": "float32",
        "min_score_std": 0.0,
        "score_distribution": {
            "count": int(flattened_scores.size),
            "mean": float(flattened_scores.mean()),
            "population_std": float(flattened_scores.std(ddof=0)),
            "min": float(flattened_scores.min()),
            "max": float(flattened_scores.max()),
        },
    }
    rows = []
    labels = {"q0": [0, 1], "q1": [1, 0]}
    for query_id in ("q0", "q1"):
        for candidate_index in range(2):
            rows.append({
                "query_id": query_id,
                "candidate_index": candidate_index,
                "correctness": labels[query_id][candidate_index],
                "reward_score": scores[query_id][candidate_index],
                "generation": {"candidate_index_policy": "vllm_completion_output_index"},
                "reward_model_variant": variant,
                "reward_scoring_provenance": dict(provenance),
            })
    return evaluate_candidate_rows(
        rows,
        score_field="reward_score",
        k_values=[1, 2],
        bootstrap_replicates=10,
        seed=1,
    )


def test_multiseed_summary_reports_sample_std_and_paired_comparisons():
    strict = _report({"q0": [0.9, 0.1], "q1": [0.9, 0.1]}, "strict_swift", "a")
    encoded_good = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b")
    encoded_bad = _report({"q0": [0.9, 0.1], "q1": [0.1, 0.9]}, "encoded_swift", "c")
    clir = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "d")
    strict_second_seed = _report(
        {"q0": [0.9, 0.1], "q1": [0.9, 0.1]}, "strict_swift", "e"
    )
    clir_second_seed = _report(
        {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "f"
    )
    reports = {
        1: {"strict_swift": strict, "encoded_swift": encoded_good, "clir": clir},
        2: {
            "strict_swift": strict_second_seed,
            "encoded_swift": encoded_bad,
            "clir": clir_second_seed,
        },
    }

    summary = summarize_evaluation_reports(
        reports,
        primary_k=2,
        bootstrap_replicates=20,
        bootstrap_seed=7,
    )

    encoded = summary["per_variant"]["encoded_swift"]["metrics"]["2"][
        "reward_bon_accuracy"
    ]
    assert encoded["by_seed"] == {"1": 1.0, "2": 0.0}
    assert encoded["mean"] == 0.5
    assert encoded["sample_std"] == pytest.approx(2 ** -0.5)
    comparison = summary["primary_comparisons"]["encoded_swift_to_clir"]
    assert comparison["by_seed"] == {"1": 0.0, "2": 1.0}
    assert comparison["mean"] == 0.5
    assert set(comparison["paired_query_bootstrap_ci_by_seed"]) == {"1", "2"}
    aggregate = comparison["aggregate_query_paired"]
    assert aggregate["unit"] == "query"
    assert aggregate["seed_aggregation_within_query"] == "arithmetic_mean"
    assert aggregate["query_count"] == 2
    assert aggregate["training_seed_count"] == 2
    assert aggregate["mean"] == 0.5
    assert aggregate["bootstrap_ci"] == [0.5, 0.5]
    assert summary["paired_bootstrap"]["unit"] == "query"
    assert summary["paired_bootstrap"]["aggregate_definition"] == (
        "mean_across_training_seeds_within_query_then_bootstrap_queries"
    )


def test_multiseed_summary_rejects_candidate_baseline_mismatch():
    strict = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a")
    encoded = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b")
    clir = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "c")
    corrupted = deepcopy(encoded)
    corrupted["metrics"]["2"]["oracle_accuracy"]["value"] = 0.0
    reports = {
        1: {
            "strict_swift": strict,
            "encoded_swift": corrupted,
            "clir": clir,
        }
    }

    with pytest.raises(ValueError, match="baseline mismatch"):
        summarize_evaluation_reports(reports, primary_k=2)


def test_multiseed_summary_rejects_per_query_baseline_mismatch_with_same_mean():
    strict = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a")
    encoded = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b")
    clir = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "c")
    corrupted = deepcopy(encoded)
    corrupted["per_query"][0]["k"]["2"]["random_expected"] = 1.0
    corrupted["per_query"][1]["k"]["2"]["random_expected"] = 0.0
    reports = {
        1: {
            "strict_swift": strict,
            "encoded_swift": corrupted,
            "clir": clir,
        }
    }

    with pytest.raises(ValueError, match="Per-query baseline mismatch"):
        summarize_evaluation_reports(reports, primary_k=2)


def test_multiseed_summary_rejects_historical_candidate_subset():
    strict = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a")
    encoded = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b")
    clir = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "c")
    strict["candidate_subset"] = "first_k_in_generation_order"

    with pytest.raises(ValueError, match="Candidate subset mismatch"):
        summarize_evaluation_reports(
            {1: {"strict_swift": strict, "encoded_swift": encoded, "clir": clir}},
            primary_k=2,
        )


def test_multiseed_summary_rejects_checkpoint_reuse_across_seeds():
    reports = {
        1: {
            "strict_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a"
            ),
            "encoded_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b"
            ),
            "clir": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "c"
            ),
        },
        2: {
            "strict_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a"
            ),
            "encoded_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "d"
            ),
            "clir": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "e"
            ),
        },
    }

    with pytest.raises(ValueError, match="reuses one checkpoint across cells"):
        summarize_evaluation_reports(reports, primary_k=2)
