import json
from pathlib import Path

from scripts.materialize_jp_h_blind_adjudication_v1 import (
    expand_selection,
    validate_selection,
)
from src.clir_hallucination_annotation import file_sha256, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/jp_h_blind_validation_v1"
PROTOCOL = CONFIG / "adjudication_protocol_v1.json"
RUNNER = ROOT / "scripts/materialize_jp_h_blind_adjudication_v1.py"


def test_protocol_binds_every_input_and_keeps_scores_sealed() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"].endswith("before_prediction_value_access")
    for spec in protocol["inputs"].values():
        assert file_sha256(ROOT / spec["path"]) == spec["sha256"]
    assert protocol["prediction_values_accessed"] is False
    assert protocol["adoption_gate_run"] is False
    assert protocol["adoption_gate_contract"][
        "both_views_must_pass_complete_frozen_gate"
    ] is True


def test_frozen_selections_cover_exact_blocking_package_in_order() -> None:
    package = read_jsonl(CONFIG / "adjudication_items_v1.jsonl")
    selections = read_jsonl(CONFIG / "adjudication_selections_v1.jsonl")
    assert len(package) == len(selections) == 42
    assert [row["item_id"] for row in package] == [
        row["item_id"] for row in selections
    ]
    for selection, item in zip(selections, package):
        validate_selection(selection, item)


def test_expand_selection_copies_complete_valid_view() -> None:
    item = {
        "item_id": "x",
        "problem": "What is 1+1?",
        "trajectory": "It is 2.",
        "annotation_a": {
            "path_status": "clean",
            "earliest_problem_claim_index": None,
            "hallucination_onset": -1,
            "confidence": "high",
            "summary": "Correct arithmetic.",
            "claim_reviews": [
                {
                    "claim_text": "It is 2.",
                    "occurrence": 0,
                    "status": "supported",
                    "reason": "One plus one equals two.",
                    "char_start": 0,
                    "char_end": 8,
                    "token_start": 0,
                    "token_end_exclusive": 3,
                }
            ],
        },
        "annotation_b": {},
    }
    selection = {
        "schema_version": "clir-jp-h-blind-adjudication-selection-v1",
        "item_id": "x",
        "adjudicator": "test-reviewer",
        "relation_to_views": "annotation_a",
        "rationale": "The arithmetic statement follows directly from the supplied problem.",
    }
    raw, resolution = expand_selection(selection, item)
    assert raw["claim_reviews"] == [
        {
            "claim_text": "It is 2.",
            "occurrence": 0,
            "status": "supported",
            "reason": "One plus one equals two.",
        }
    ]
    assert resolution["relation_to_views"] == "annotation_a"
    assert resolution["final_annotation"] == raw


def test_adjudicator_limitations_are_explicit() -> None:
    disclosure = json.loads(PROTOCOL.read_text(encoding="utf-8"))[
        "adjudicator_disclosure"
    ]
    assert disclosure["independent_human_adjudicator"] is False
    assert disclosure["authorship_may_be_inferable_from_annotation_style"] is True
    assert "A/B role lineage" in disclosure["not_visible_during_decisions"]


def test_materializer_does_not_name_or_load_prediction_artifact() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "predictions_v1.jsonl" not in source
    assert "frozen_scoring_v1" not in source
