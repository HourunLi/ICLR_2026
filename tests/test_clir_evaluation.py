import pytest

from evaluate_clir import evaluate_candidate_rows


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
            rows.append({
                "query_id": query_id,
                "candidate_index": candidate_index,
                "correctness": labels[query_id][candidate_index],
                "reward_score": scores[query_id][candidate_index],
            })

    report = evaluate_candidate_rows(
        rows,
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


def test_query_level_evaluator_requires_frozen_first_k_candidates():
    rows = [
        {"query_id": "q", "candidate_index": 0, "correctness": 0, "reward_score": 0.1},
        {"query_id": "q", "candidate_index": 2, "correctness": 1, "reward_score": 0.9},
    ]
    with pytest.raises(ValueError, match="contiguous"):
        evaluate_candidate_rows(rows, score_field="reward_score", k_values=[2])
