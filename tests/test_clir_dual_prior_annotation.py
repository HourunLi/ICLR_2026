import pytest

from src.clir_dual_prior_annotation import (
    ANNOTATION_SCHEMA,
    ITEM_SCHEMA,
    public_units,
    segment_reasoning_units,
    validate_annotation,
    validate_item,
)


def _item():
    trajectory = (
        "Step 1: Use the supplied value. Then compute x = 2 + 3.\n\n"
        "x = 5\nFinal Answer: The answer is 5."
    )
    units = segment_reasoning_units(trajectory)
    return {
        "schema_version": ITEM_SCHEMA,
        "item_id": "DPA-test",
        "problem": "What is 2 + 3?",
        "trajectory": trajectory,
        "units": public_units(units),
    }, units


def test_unitization_is_exact_complete_and_separates_headers():
    item, units = _item()
    validate_item(item)

    assert [unit["text"] for unit in units] == [
        "Step 1:",
        "Use the supplied value.",
        "Then compute x = 2 + 3.",
        "x = 5",
        "Final Answer:",
        "The answer is 5.",
    ]
    covered = set()
    for unit in units:
        assert item["trajectory"][unit["char_start"] : unit["char_end"]] == unit["text"]
        covered.update(range(unit["char_start"], unit["char_end"]))
    assert all(
        index in covered or character.isspace()
        for index, character in enumerate(item["trajectory"])
    )


def test_usable_annotation_requires_sorted_nonempty_nested_sets():
    item, _ = _item()
    annotation = {
        "schema_version": ANNOTATION_SCHEMA,
        "item_id": item["item_id"],
        "eligibility": "usable",
        "key_unit_indices": [2],
        "complete_unit_indices": [1, 2, 3],
        "confidence": "high",
        "rationale": "The calculation is key; the preceding setup and result complete the chain.",
    }

    assert validate_annotation(annotation, item) == annotation

    non_nested = {**annotation, "complete_unit_indices": [1, 3]}
    with pytest.raises(ValueError, match="subset"):
        validate_annotation(non_nested, item)

    duplicate = {**annotation, "key_unit_indices": [2, 2]}
    with pytest.raises(ValueError, match="sorted"):
        validate_annotation(duplicate, item)


def test_ineligible_annotation_must_not_invent_evidence_targets():
    item, _ = _item()
    annotation = {
        "schema_version": ANNOTATION_SCHEMA,
        "item_id": item["item_id"],
        "eligibility": "insufficient_unitization",
        "key_unit_indices": [],
        "complete_unit_indices": [],
        "confidence": "medium",
        "rationale": "A required inference was fused with unrelated content.",
    }
    validate_annotation(annotation, item)

    with pytest.raises(ValueError, match="ineligible"):
        validate_annotation({**annotation, "complete_unit_indices": [2]}, item)
