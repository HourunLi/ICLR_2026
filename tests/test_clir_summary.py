from copy import deepcopy

import pytest

from evaluate_clir import evaluate_candidate_rows
from summarize_clir import summarize_evaluation_reports


def _report(scores):
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
            })
    return evaluate_candidate_rows(
        rows,
        score_field="reward_score",
        k_values=[1, 2],
        bootstrap_replicates=10,
        seed=1,
    )


def test_multiseed_summary_reports_sample_std_and_paired_comparisons():
    strict = _report({"q0": [0.9, 0.1], "q1": [0.9, 0.1]})
    encoded_good = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]})
    encoded_bad = _report({"q0": [0.9, 0.1], "q1": [0.1, 0.9]})
    clir = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]})
    reports = {
        1: {"strict_swift": strict, "encoded_swift": encoded_good, "clir": clir},
        2: {"strict_swift": strict, "encoded_swift": encoded_bad, "clir": clir},
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
    report = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]})
    corrupted = deepcopy(report)
    corrupted["metrics"]["2"]["oracle_accuracy"]["value"] = 0.0
    reports = {
        1: {
            "strict_swift": report,
            "encoded_swift": corrupted,
            "clir": report,
        }
    }

    with pytest.raises(ValueError, match="baseline mismatch"):
        summarize_evaluation_reports(reports, primary_k=2)


def test_multiseed_summary_rejects_per_query_baseline_mismatch_with_same_mean():
    report = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]})
    corrupted = deepcopy(report)
    corrupted["per_query"][0]["k"]["2"]["random_expected"] = 1.0
    corrupted["per_query"][1]["k"]["2"]["random_expected"] = 0.0
    reports = {
        1: {
            "strict_swift": report,
            "encoded_swift": corrupted,
            "clir": report,
        }
    }

    with pytest.raises(ValueError, match="Per-query baseline mismatch"):
        summarize_evaluation_reports(reports, primary_k=2)
