import json
from pathlib import Path

import pytest

from scripts.checkpoint_jp_h_blind_secondary_v1 import (
    append_one,
    progress_record,
    read_valid_prefix,
)
from scripts.generate_jp_h_blind_rollouts_v1 import (
    parse_governed_label,
    parse_yes_no,
)
from scripts.materialize_jp_h_blind_items_v1 import select_domain
from scripts.prepare_jp_h_blind_sources_v1 import format_options, make_prompt


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/jp_h_blind_validation_v1/protocol_v1.json"


def _item(item_id: str) -> dict:
    return {
        "schema_version": "clir-hallucination-annotation-item-v1",
        "item_id": item_id,
        "problem": "Premise: alpha is true.",
        "trajectory": "Alpha follows from the supplied premise.",
    }


def _annotation(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "claim_reviews": [
            {
                "claim_text": "Alpha follows from the supplied premise.",
                "occurrence": 0,
                "status": "supported",
                "reason": "The claim directly restates the supplied premise.",
            }
        ],
        "path_status": "clean",
        "earliest_problem_claim_index": None,
        "confidence": "high",
        "summary": "The only material claim is directly supported by the premise.",
    }


def test_answer_parsers_require_governed_final_answer() -> None:
    assert parse_governed_label("Reason.\nFinal Answer: **B**", ["A", "B", "C"]) == "B"
    assert parse_governed_label("B is discussed but no conclusion", ["A", "B"]) is None
    assert parse_yes_no("Reason.\nFinal Answer: TRUE") == "yes"
    assert parse_yes_no("The passage says yes, perhaps.") is None


def test_problem_rendering_preserves_option_labels_and_domain_instruction() -> None:
    problem, options = format_options(
        "Which option?",
        {"label": ["1", "2"], "text": ["first", "second"]},
    )
    assert options == [
        {"label": "1", "text": "first"},
        {"label": "2", "text": "second"},
    ]
    assert "1. first" in problem and "2. second" in problem
    assert "Final Answer: <option label>" in make_prompt("arc_challenge", problem)


def test_domain_selection_is_balanced_and_query_unique() -> None:
    rows = []
    for correctness in (0, 1):
        for index in range(32):
            rows.append(
                {
                    "id": f"row-{correctness}-{index:02d}",
                    "query_id": f"query-{index:02d}",
                    "correctness": correctness,
                    "answer_parse_valid": True,
                    "output_token_ids": list(range(32 + index)),
                    "generation": {"finish_reason": "stop"},
                }
            )
    selection = {
        "seed": 51721,
        "length_bins_per_correctness_domain": 4,
        "rows_per_length_bin": 3,
        "minimum_output_tokens": 32,
    }
    chosen = select_domain(rows, domain="synthetic", selection=selection)
    assert len(chosen) == 24
    assert len({row["query_id"] for row in chosen}) == 24
    assert sum(row["selection_correctness"] == 0 for row in chosen) == 12
    assert sum(row["selection_correctness"] == 1 for row in chosen) == 12
    for correctness in (0, 1):
        for length_bin in range(4):
            assert sum(
                row["selection_correctness"] == correctness
                and row["selection_length_bin"] == length_bin
                for row in chosen
            ) == 3


def test_checkpoint_validates_one_row_and_resumes(tmp_path: Path) -> None:
    items = [_item("JPHB-a"), _item("JPHB-b")]
    output = tmp_path / "labels.jsonl"
    scratch = tmp_path / "current.json"
    start = progress_record(read_valid_prefix(output, items), items, output)
    assert start["completed_rows"] == 0 and start["next_item_id"] == "JPHB-a"

    scratch.write_text(json.dumps(_annotation("JPHB-a")), encoding="utf-8")
    first = append_one(annotation_file=scratch, output=output, items=items)
    assert first["completed_rows"] == 1 and first["next_item_id"] == "JPHB-b"
    first_payload = output.read_bytes()

    scratch.write_text(json.dumps(_annotation("JPHB-a")), encoding="utf-8")
    with pytest.raises(ValueError, match="item_id drifted"):
        append_one(annotation_file=scratch, output=output, items=items)
    assert output.read_bytes() == first_payload


def test_truncated_checkpoint_is_rejected(tmp_path: Path) -> None:
    items = [_item("JPHB-a"), _item("JPHB-b")]
    output = tmp_path / "labels.jsonl"
    output.write_text(json.dumps(_annotation("JPHB-a")) + "\n{" , encoding="utf-8")
    with pytest.raises(ValueError, match="truncated or invalid JSON"):
        read_valid_prefix(output, items)


def test_protocol_freezes_non_math_majority_and_single_smoother() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["selection"]["expected_rows"] == 96
    assert protocol["selection"]["expected_non_math_rows"] == 72
    assert protocol["frozen_candidate"]["window"] == 3
    assert protocol["frozen_candidate"]["additional_window_or_architecture_sweep_forbidden"] is True
    assert protocol["prohibitions"]["score_coupling_authorized"] is False
