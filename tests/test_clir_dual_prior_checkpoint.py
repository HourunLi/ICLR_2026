import json
from pathlib import Path

import pytest

from scripts.checkpoint_dual_prior_secondary_v1 import (
    append_one,
    progress_record,
    read_valid_prefix,
)
from src.clir_hallucination_annotation import file_sha256, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "configs/dual_prior_evidence_v1/annotation_items_v1.jsonl"
PROMPT = ROOT / "configs/dual_prior_evidence_v1/secondary_prompt_resumable_v1a.md"
ADDENDUM = ROOT / "configs/dual_prior_evidence_v1/secondary_execution_addendum_v1a.json"


def _annotation(item_id: str) -> dict:
    return {
        "schema_version": "clir-dual-prior-evidence-annotation-v1",
        "item_id": item_id,
        "eligibility": "usable",
        "key_unit_indices": [0],
        "complete_unit_indices": [0],
        "confidence": "medium",
        "rationale": "The selected unit is sufficient for this checkpoint transport test.",
    }


def _write_scratch(path: Path, annotation: dict) -> None:
    path.write_text(json.dumps(annotation), encoding="utf-8")


def test_one_row_checkpoint_is_validated_ordered_and_resumable(tmp_path: Path):
    items = read_jsonl(ITEMS)[:2]
    output = tmp_path / "labels.jsonl"
    scratch = tmp_path / "current.json"

    empty = progress_record(read_valid_prefix(output, items), items, output)
    assert empty["completed_rows"] == 0
    assert empty["next_item_id"] == items[0]["item_id"]

    _write_scratch(scratch, _annotation(items[0]["item_id"]))
    first = append_one(annotation_file=scratch, output=output, items=items)
    assert first["completed_rows"] == 1
    assert first["appended_item_id"] == items[0]["item_id"]
    assert first["next_item_id"] == items[1]["item_id"]
    assert output.read_bytes().endswith(b"\n")
    first_payload = output.read_bytes()

    _write_scratch(scratch, _annotation(items[0]["item_id"]))
    with pytest.raises(ValueError, match="item_id does not match"):
        append_one(annotation_file=scratch, output=output, items=items)
    assert output.read_bytes() == first_payload

    _write_scratch(scratch, _annotation(items[1]["item_id"]))
    second = append_one(annotation_file=scratch, output=output, items=items)
    assert second["completed_rows"] == 2
    assert second["remaining_rows"] == 0
    assert second["next_item_id"] is None
    assert second["complete"] is True
    assert [row["item_id"] for row in read_valid_prefix(output, items)] == [
        item["item_id"] for item in items
    ]


def test_truncated_checkpoint_is_rejected_without_guessing(tmp_path: Path):
    items = read_jsonl(ITEMS)[:2]
    output = tmp_path / "labels.jsonl"
    output.write_text(
        json.dumps(_annotation(items[0]["item_id"])) + "\n" + '{"schema_version":',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not complete valid JSON"):
        read_valid_prefix(output, items)


def test_resumable_prompt_requires_immediate_per_row_durable_progress():
    prompt = PROMPT.read_text(encoding="utf-8")
    assert "judge one row, then save that row immediately" in prompt
    assert "Do **not** keep multiple completed judgments only in memory" in prompt
    assert "checkpoint_dual_prior_secondary_v1.py status" in prompt
    assert "checkpoint_dual_prior_secondary_v1.py append" in prompt
    assert "checkpoint_dual_prior_secondary_v1.py finalize" in prompt
    assert "completed_rows" in prompt and "next_item_id" in prompt
    assert "labels_secondary_raw_v1.jsonl" in prompt
    assert "validate_dual_prior_secondary_v1.py" in prompt and "--input" in prompt


def test_execution_addendum_binds_frozen_semantics_and_active_tools():
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    assert addendum["semantics_changed"] is False
    parent = addendum["frozen_semantic_parent"]
    active = addendum["active_execution"]
    for path_key, hash_key in (
        ("protocol", "protocol_sha256"),
        ("annotation_guide", "annotation_guide_sha256"),
        ("annotation_items", "annotation_items_sha256"),
        ("historical_prompt", "historical_prompt_sha256"),
    ):
        assert file_sha256(ROOT / parent[path_key]) == parent[hash_key]
    assert file_sha256(ROOT / active["prompt"]) == active["prompt_sha256"]
    assert (
        file_sha256(ROOT / active["checkpoint_helper"])
        == active["checkpoint_helper_sha256"]
    )
    assert active["unit_of_durable_progress"] == "one validated annotation row"
    assert active["batch_only_write_forbidden"] is True
