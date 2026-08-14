import pytest
import numpy as np

from evaluate_clir import evaluate_candidate_rows


GENERATION = {"candidate_index_policy": "vllm_completion_output_index"}
PROVENANCE = {
    "schema_version": "clir-reward-scoring-v2",
    "model_variant": "clir",
    "checkpoint_sha256": "a" * 64,
    "input_sha256": "b" * 64,
    "batch_size": 2,
    "amp_dtype": "none",
    "compute_dtype": "float32",
    "min_score_std": 0.0,
    "score_distribution": {},
}


def _scored_row(**values):
    return {
        "generation": GENERATION,
        "reward_model_variant": "clir",
        "reward_scoring_provenance": dict(PROVENANCE),
        **values,
    }


def _with_score_distribution(rows):
    scores = np.asarray([row["reward_score"] for row in rows], dtype=np.float64)
    distribution = {
        "count": len(rows),
        "mean": float(scores.mean()),
        "population_std": float(scores.std(ddof=0)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    for row in rows:
        row["reward_scoring_provenance"]["score_distribution"] = dict(distribution)
    return rows


def test_query_level_evaluator_separates_random_oracle_and_reward_bon():
    rows = []
    labels = {
        "q0": [0, 1, 0, 0],
        "q1": [1, 0, 1, 1],
    }
    scores = {
        "q0": [0.1, 0.9, 0.2, 0.3],
        "q1": [0.9, 0.1, 0.2, 0.3],
    }
    for query_id in ("q0", "q1"):
        for candidate_index in range(4):
            rows.append(_scored_row(**{
                "query_id": query_id,
                "candidate_index": candidate_index,
                "correctness": labels[query_id][candidate_index],
                "reward_score": scores[query_id][candidate_index],
            }))

    report = evaluate_candidate_rows(
        _with_score_distribution(rows),
        score_field="reward_score",
        k_values=[1, 2, 4],
        bootstrap_replicates=100,
        seed=3,
    )

    assert report["bootstrap"]["unit"] == "query"
    assert report["metrics"]["1"]["reward_bon_accuracy"]["value"] == 0.5
    assert report["metrics"]["2"]["reward_bon_accuracy"]["value"] == 1.0
    assert report["metrics"]["4"]["oracle_accuracy"]["value"] == 1.0
    assert report["metrics"]["4"]["random_expected_accuracy"]["value"] == 0.5
    assert report["max_k_pool_counts"] == {"mixed": 2, "all_correct": 0, "all_wrong": 0}
    assert report["reward_scoring_provenance"]["checkpoint_sha256"] == "a" * 64
    assert report["metrics"]["4"]["reward_selection_ties"]["query_count"] == 0


def test_query_level_evaluator_requires_frozen_first_k_candidates():
    rows = [
        _scored_row(query_id="q", candidate_index=0, correctness=0, reward_score=0.1),
        _scored_row(query_id="q", candidate_index=2, correctness=1, reward_score=0.9),
    ]
    with pytest.raises(ValueError, match="contiguous"):
        evaluate_candidate_rows(
            _with_score_distribution(rows), score_field="reward_score", k_values=[2]
        )


def test_query_level_evaluator_rejects_unproven_candidate_order():
    rows = [
        {"query_id": "q", "candidate_index": 0, "correctness": 0, "reward_score": 0.1},
    ]
    with pytest.raises(ValueError, match="candidate_index_policy"):
        evaluate_candidate_rows(rows, score_field="reward_score", k_values=[1])


def test_query_level_evaluator_rejects_mixed_checkpoint_provenance():
    rows = [
        _scored_row(query_id="q", candidate_index=0, correctness=0, reward_score=0.1),
        _scored_row(query_id="q", candidate_index=1, correctness=1, reward_score=0.9),
    ]
    _with_score_distribution(rows)
    rows[1]["reward_scoring_provenance"]["checkpoint_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="mixes rows"):
        evaluate_candidate_rows(rows, score_field="reward_score", k_values=[2])


def test_query_level_evaluator_reports_deterministic_score_ties():
    rows = [
        _scored_row(query_id="q", candidate_index=0, correctness=0, reward_score=0.5),
        _scored_row(query_id="q", candidate_index=1, correctness=1, reward_score=0.5),
    ]

    report = evaluate_candidate_rows(
        _with_score_distribution(rows), score_field="reward_score", k_values=[2]
    )

    assert report["metrics"]["2"]["reward_bon_accuracy"]["value"] == 0.0
    assert report["metrics"]["2"]["reward_selection_ties"] == {
        "query_count": 1,
        "query_fraction": 1.0,
        "max_tie_size": 2,
        "tie_break_policy": "lowest_candidate_index",
    }


def test_query_level_evaluator_recomputes_score_distribution_provenance():
    rows = _with_score_distribution([
        _scored_row(query_id="q", candidate_index=0, correctness=0, reward_score=0.1),
        _scored_row(query_id="q", candidate_index=1, correctness=1, reward_score=0.9),
    ])
    for row in rows:
        row["reward_scoring_provenance"]["score_distribution"]["max"] = 9.0

    with pytest.raises(ValueError, match="does not match scores"):
        evaluate_candidate_rows(rows, score_field="reward_score", k_values=[2])
