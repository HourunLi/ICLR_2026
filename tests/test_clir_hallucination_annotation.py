from __future__ import annotations

from collections import Counter

import pytest

from scripts.run_hallucination_primary_v1 import parse_annotation, prompt_for
from src.clir_hallucination_annotation import (
    build_annotation_records,
    canonical_sha256,
    char_span_to_token_span,
    content_token_offsets,
    locate_occurrence,
    map_annotation,
    repair_annotation_contract,
    select_stratified_rows,
    validate_annotation,
    whitespace_equivalent_occurrence,
)


def source_row(
    row_id: str,
    *,
    query_id: str,
    correctness: int,
    length: int,
) -> dict:
    return {
        "id": row_id,
        "query_id": query_id,
        "candidate_index": 0,
        "correctness": correctness,
        "question": f"Problem for {query_id}",
        "response": f"Claim for {row_id}.",
        "prompt_token_ids": [1, 2],
        "output_token_ids": list(range(length)),
    }


def test_stratified_selection_is_deterministic_balanced_and_query_unique():
    rows = []
    for correctness in (0, 1):
        for index in range(12):
            rows.append(
                source_row(
                    f"row-{correctness}-{index}",
                    query_id=f"query-{correctness}-{index}",
                    correctness=correctness,
                    length=index + 2,
                )
            )
    rows.append(
        source_row(
            "excluded-row",
            query_id="excluded-query",
            correctness=0,
            length=50,
        )
    )
    selected = select_stratified_rows(
        rows,
        excluded_query_ids={"excluded-query"},
        seed=1729,
        per_correctness=4,
        length_bins=2,
    )
    repeated = select_stratified_rows(
        list(reversed(rows)),
        excluded_query_ids={"excluded-query"},
        seed=1729,
        per_correctness=4,
        length_bins=2,
    )
    assert [row["id"] for row in selected] == [row["id"] for row in repeated]
    assert len(selected) == len({row["query_id"] for row in selected}) == 8
    assert "excluded-query" not in {row["query_id"] for row in selected}
    counts = Counter(
        (row["selection_correctness"], row["selection_length_bin"])
        for row in selected
    )
    assert counts == Counter({(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2})


def test_annotation_records_keep_correctness_out_of_blind_items():
    rows = [
        {
            **source_row("row-a", query_id="query-a", correctness=0, length=4),
            "selection_correctness": 0,
            "selection_length_bin": 1,
            "selection_priority": "abc",
        }
    ]
    selection, items, lineage = build_annotation_records(
        rows,
        source_manifest="source.jsonl",
        source_manifest_sha256="a" * 64,
        protocol_sha256="b" * 64,
        item_salt="test",
    )
    assert selection[0]["correctness_stratum"] == 0
    assert set(items[0]) == {"schema_version", "item_id", "problem", "trajectory"}
    assert "correctness" not in items[0]
    assert lineage[0]["correctness"] == 0
    assert lineage[0]["output_token_ids_sha256"] == canonical_sha256([0, 1, 2, 3])


def test_locate_occurrence_and_annotation_logic():
    item = {
        "item_id": "HLA-test",
        "trajectory": "A is 2. A is 2. Therefore B is 5.",
    }
    annotation = {
        "item_id": "HLA-test",
        "claim_reviews": [
            {
                "claim_text": "A is 2.",
                "occurrence": 0,
                "status": "supported",
                "reason": "It is supplied by the problem.",
            },
            {
                "claim_text": "A is 2.",
                "occurrence": 1,
                "status": "supported",
                "reason": "It repeats the established premise.",
            },
            {
                "claim_text": "B is 5.",
                "occurrence": 0,
                "status": "unsupported",
                "reason": "No relationship from A to B is provided.",
            },
        ],
        "path_status": "hallucinated",
        "earliest_problem_claim_index": 2,
        "confidence": "high",
        "summary": "The final B claim has no supporting inference.",
    }
    assert locate_occurrence(item["trajectory"], "A is 2.", 1) == (8, 15)
    spans = validate_annotation(annotation, item)
    assert spans[-1] == (26, 33)
    invalid = dict(annotation)
    invalid["earliest_problem_claim_index"] = 1
    with pytest.raises(ValueError, match="earliest problem"):
        validate_annotation(invalid, item)


def test_clean_annotation_rejects_uncertain_claim():
    item = {"item_id": "HLA-test", "trajectory": "Maybe X."}
    annotation = {
        "item_id": "HLA-test",
        "claim_reviews": [
            {
                "claim_text": "Maybe X.",
                "occurrence": 0,
                "status": "uncertain",
                "reason": "The problem supplies no context for X.",
            }
        ],
        "path_status": "clean",
        "earliest_problem_claim_index": None,
        "confidence": "low",
        "summary": "The claim cannot be resolved from this item.",
    }
    with pytest.raises(ValueError, match="clean path"):
        validate_annotation(annotation, item)


def test_token_contract_requires_exact_prefix_and_invisible_terminal():
    offsets = content_token_offsets(
        response="abc def",
        output_token_ids=[10, 11, 99],
        encoded_token_ids=[10, 11],
        offsets=[(0, 3), (3, 7)],
        trailing_token_decodes_to_empty=[True],
    )
    assert offsets == [(0, 3), (3, 7)]
    assert char_span_to_token_span((4, 7), offsets) == (1, 2)
    with pytest.raises(ValueError, match="re-tokenization"):
        content_token_offsets(
            response="abc def",
            output_token_ids=[10, 12, 99],
            encoded_token_ids=[10, 11],
            offsets=[(0, 3), (3, 7)],
            trailing_token_decodes_to_empty=[True],
        )


def test_map_annotation_masks_uncertain_and_maps_definite_onset():
    item = {
        "item_id": "HLA-test",
        "trajectory": "First claim. Bad claim.",
    }
    lineage = {
        "source_id": "row-a",
        "query_id": "query-a",
        "output_token_ids_sha256": "c" * 64,
    }
    annotation = {
        "item_id": "HLA-test",
        "claim_reviews": [
            {
                "claim_text": "First claim.",
                "occurrence": 0,
                "status": "supported",
                "reason": "The premise is explicitly supplied.",
            },
            {
                "claim_text": "Bad claim.",
                "occurrence": 0,
                "status": "contradicted",
                "reason": "It conflicts with the supplied premise.",
            },
        ],
        "path_status": "hallucinated",
        "earliest_problem_claim_index": 1,
        "confidence": "high",
        "summary": "The second claim is the first contradiction.",
    }
    mapped = map_annotation(
        annotation,
        item,
        lineage,
        offsets=[(0, 5), (5, 12), (12, 16), (16, 23)],
        annotator={"role": "primary"},
    )
    assert mapped["path_hallucinated"] == 1
    assert mapped["hallucination_onset"] == 2
    uncertain = dict(annotation)
    uncertain["claim_reviews"] = [
        {
            "claim_text": "First claim.",
            "occurrence": 0,
            "status": "uncertain",
            "reason": "The available problem context is ambiguous.",
        }
    ]
    uncertain["path_status"] = "uncertain"
    uncertain["earliest_problem_claim_index"] = 0
    uncertain["summary"] = "The first claim cannot be judged reliably."
    mapped_uncertain = map_annotation(
        uncertain,
        item,
        lineage,
        offsets=[(0, 5), (5, 12), (12, 16), (16, 23)],
        annotator={"role": "primary"},
    )
    assert "path_hallucinated" not in mapped_uncertain
    assert "hallucination_onset" not in mapped_uncertain


def test_primary_parser_accepts_only_strict_json_or_one_exact_fence():
    value = {
        "item_id": "HLA-test",
        "claim_reviews": [],
        "path_status": "clean",
        "earliest_problem_claim_index": None,
        "confidence": "high",
        "summary": "placeholder",
    }
    import json

    payload = json.dumps(value)
    assert parse_annotation(payload) == (value, "strict_json")
    assert parse_annotation(f"```json\n{payload}\n```") == (
        value,
        "single_json_fence",
    )
    with pytest.raises(ValueError):
        parse_annotation(f"Commentary\n{payload}")


def test_primary_prompt_projects_only_blind_item_fields():
    item = {
        "schema_version": "clir-hallucination-annotation-item-v1",
        "item_id": "HLA-test",
        "problem": "A problem",
        "trajectory": "A trajectory",
    }
    messages = prompt_for(item, "Guide text")
    assert len(messages) == 2
    assert "HLA-test" in messages[1]["content"]
    assert "A trajectory" in messages[1]["content"]
    assert "query_id" not in messages[1]["content"]
    assert "reference_answer" not in messages[1]["content"]


def test_whitespace_equivalent_quote_repair_copies_exact_trajectory_span():
    trajectory = "First equation:\n   x = 2\nThen continue."
    resolved, occurrence, span = whitespace_equivalent_occurrence(
        trajectory,
        "First equation: x = 2",
        0,
    )
    assert resolved == "First equation:\n   x = 2"
    assert occurrence == 0
    assert trajectory[slice(*span)] == resolved
    with pytest.raises(ValueError, match="whitespace-equivalent"):
        whitespace_equivalent_occurrence(trajectory, "First equation: x = 3", 0)


def test_contract_repair_changes_only_quote_location_and_derived_index():
    item = {
        "item_id": "HLA-test",
        "trajectory": "Premise:\n  A = 2\nConclusion B = 5.",
    }
    annotation = {
        "item_id": "HLA-test",
        "claim_reviews": [
            {
                "claim_text": "Premise: A = 2",
                "occurrence": 0,
                "status": "supported",
                "reason": "The trajectory establishes this premise.",
            },
            {
                "claim_text": "Conclusion B = 5.",
                "occurrence": 0,
                "status": "unsupported",
                "reason": "No relationship derives B from the premise.",
            },
        ],
        "path_status": "hallucinated",
        "earliest_problem_claim_index": 2,
        "confidence": "high",
        "summary": "The second claim is not derived from the supplied premise.",
    }
    repaired, operations = repair_annotation_contract(annotation, item)
    assert repaired["claim_reviews"][0]["claim_text"] == "Premise:\n  A = 2"
    assert repaired["earliest_problem_claim_index"] == 1
    assert [operation["operation"] for operation in operations] == [
        "whitespace_equivalent_quote_alignment",
        "derive_first_problem_claim_index",
    ]
    assert repaired["path_status"] == annotation["path_status"]
    assert [claim["status"] for claim in repaired["claim_reviews"]] == [
        claim["status"] for claim in annotation["claim_reviews"]
    ]
