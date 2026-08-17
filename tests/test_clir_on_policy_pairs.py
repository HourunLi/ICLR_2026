import copy

import pytest

from src.clir_on_policy_pairs import audit_candidates, mine_candidate_pairs
from scripts.run_on_policy_verifier_v1 import item_projection, silver_status, validate_annotation


def row(query: str, index: int, *, answer: str = "8", correct: int = 1, tokens: int = 4):
    return {
        "id": f"{query}-c{index}", "query_id": query, "candidate_index": index,
        "question": "What is 2 times 4?", "prompt": "prompt",
        "response": f"trajectory {index}", "prompt_token_ids": [1, 2],
        "output_token_ids": list(range(tokens)), "correctness": correct,
        "normalized_candidate_answer": answer, "source_index": 3, "split": "train",
    }


def test_mining_is_deterministic_capped_and_never_prematurely_labeled():
    rows = [row("q1", 0, tokens=2), row("q1", 1, tokens=8), row("q1", 2, tokens=5)]
    first, report = mine_candidate_pairs(rows, allowed_query_ids=["q1"], max_pairs_per_query=2)
    second, second_report = mine_candidate_pairs(reversed(rows), allowed_query_ids=["q1"], max_pairs_per_query=2)
    assert first == second
    assert report == second_report
    assert len(first) == 2
    assert first[0]["selection_features"]["absolute_output_token_gap"] == 6
    assert all(item["relation_label"] is None for item in first)
    assert report["eligible_pair_count_before_cap"] == 3
    assert report["reasoning_equivalence_claim_allowed"] is False
    audit_candidates(first, report)


def test_mining_filters_correctness_outcome_and_membership():
    rows = [
        row("q1", 0), row("q1", 1), row("q1", 2, answer="9"),
        row("q1", 3, correct=0), row("q2", 0), row("q2", 1),
    ]
    candidates, report = mine_candidate_pairs(rows, allowed_query_ids=["q1"], max_pairs_per_query=4)
    assert len(candidates) == 1
    assert candidates[0]["query_id"] == "q1"
    assert candidates[0]["normalized_outcome"] == "8"
    assert report["source_rows_in_membership"] == 4


def test_audit_rejects_relation_verdict_and_membership_is_complete():
    rows = [row("q1", 0), row("q1", 1)]
    candidates, report = mine_candidate_pairs(rows, allowed_query_ids=["q1"])
    polluted = copy.deepcopy(candidates)
    polluted[0]["relation_label"] = "accept"
    with pytest.raises(ValueError, match="relation verdict"):
        audit_candidates(polluted, report)
    with pytest.raises(ValueError, match="missing from source"):
        mine_candidate_pairs(rows, allowed_query_ids=["q1", "q2"])


def annotation(**changes):
    value = {
        "item_id": "OPA-test", "decision": "accept", "same_task_and_goal": True,
        "same_core_premises": True, "same_reasoning_method": True,
        "same_key_inferences": True, "same_intermediate_conclusions": True,
        "same_final_conclusion": True, "entailed_elaboration_present": False,
        "entailed_elaboration_valid": None, "introduced_new_premise_or_evidence": False,
        "omitted_essential_claim": False, "contradicted_source_claim": False,
        "replaced_with_different_solution": False, "introduced_new_error": False,
        "error_alignment_applicable": False, "same_error_mechanism": None,
        "same_semantic_error_location": None, "same_downstream_effect": None,
        "target_style": "native_surface_difference", "style_satisfied": True,
        "confidence": "high", "reason": "Both retain multiplication and the result while reorganizing the steps.",
    }
    value.update(changes)
    return value


def test_silver_acceptance_is_fail_closed_and_prompt_projection_is_blind():
    value = annotation()
    validate_annotation(value, "OPA-test")
    assert silver_status(value) == "silver_accepted"
    assert silver_status(annotation(style_satisfied=False, decision="reject")) == "rejected"
    assert silver_status(annotation(confidence="medium")) == "review_required"
    assert silver_status(annotation(introduced_new_premise_or_evidence=True)) == "rejected"

    candidates, _ = mine_candidate_pairs([row("q1", 0), row("q1", 1)], allowed_query_ids=["q1"])
    projected = item_projection(candidates[0])
    assert "correctness" not in projected
    assert "normalized_outcome" not in projected
    assert "row_sha256" not in projected
