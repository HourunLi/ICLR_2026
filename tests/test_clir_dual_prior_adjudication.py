import json
from pathlib import Path

import pytest

from scripts.checkpoint_dual_prior_adjudication_v1 import (
    append_one,
    load_context,
    progress_record,
    read_valid_prefix,
)
from src.clir_dual_prior_adjudication import validate_adjudication
from src.clir_hallucination_annotation import file_sha256


ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "configs/dual_prior_evidence_v1/adjudication_execution_v1.json"
PROMPT = ROOT / "configs/dual_prior_evidence_v1/adjudication_prompt_resumable_v1.md"
PROTOCOL = (
    ROOT
    / "configs/dual_prior_evidence_v1/agreement_adjudication_protocol_v1.json"
)


def _packet(item_id: str = "DPA-adjudication-test") -> dict:
    return {
        "schema_version": "clir-dual-prior-evidence-adjudication-item-v1",
        "item_id": item_id,
        "problem": "Compute a result.",
        "trajectory": "Use the input.\nCompute one result.\nCompute another result.",
        "units": [
            {"unit_index": 0, "text": "Use the input."},
            {"unit_index": 1, "text": "Compute one result."},
            {"unit_index": 2, "text": "Compute another result."},
        ],
        "annotation_a": {
            "eligibility": "usable",
            "key_unit_indices": [1],
            "complete_unit_indices": [0, 1],
            "confidence": "high",
            "rationale": "Candidate A uses the first computation.",
        },
        "annotation_b": {
            "eligibility": "usable",
            "key_unit_indices": [2],
            "complete_unit_indices": [0, 2],
            "confidence": "high",
            "rationale": "Candidate B uses the second computation.",
        },
    }


def _decision(
    packet: dict,
    *,
    resolution: str,
    key: list[int],
    complete: list[int],
) -> dict:
    return {
        "schema_version": "clir-dual-prior-evidence-adjudication-v1",
        "item_id": packet["item_id"],
        "resolution": resolution,
        "eligibility": "usable",
        "key_unit_indices": key,
        "complete_unit_indices": complete,
        "confidence": "high",
        "rationale": "Independent semantic audit selects this nested evidence target.",
    }


def test_resolution_must_match_adopted_candidate_or_new_synthesis():
    packet = _packet()
    adopt_a = _decision(packet, resolution="adopt_a", key=[1], complete=[0, 1])
    adopt_b = _decision(packet, resolution="adopt_b", key=[2], complete=[0, 2])
    synthesize = _decision(
        packet,
        resolution="synthesize",
        key=[1, 2],
        complete=[0, 1, 2],
    )

    assert validate_adjudication(adopt_a, packet)["resolution"] == "adopt_a"
    assert validate_adjudication(adopt_b, packet)["resolution"] == "adopt_b"
    assert validate_adjudication(synthesize, packet)["resolution"] == "synthesize"
    with pytest.raises(ValueError, match="adopt_a"):
        validate_adjudication(
            {
                **adopt_a,
                "key_unit_indices": [2],
                "complete_unit_indices": [0, 1, 2],
            },
            packet,
        )
    with pytest.raises(ValueError, match="synthesize"):
        validate_adjudication({**adopt_a, "resolution": "synthesize"}, packet)


def test_checkpoint_accepts_only_the_next_valid_disagreement(tmp_path: Path):
    packets = [_packet("DPA-one"), _packet("DPA-two")]
    output = tmp_path / "adjudications.jsonl"
    scratch = tmp_path / "current.json"
    empty = progress_record(read_valid_prefix(output, packets), packets, output)
    assert empty["completed_rows"] == 0
    assert empty["next_item_id"] == "DPA-one"

    first = _decision(packets[0], resolution="adopt_a", key=[1], complete=[0, 1])
    scratch.write_text(json.dumps(first), encoding="utf-8")
    record = append_one(adjudication_file=scratch, output=output, packets=packets)
    assert record["completed_rows"] == 1
    assert record["next_item_id"] == "DPA-two"

    scratch.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(ValueError, match="item_id"):
        append_one(adjudication_file=scratch, output=output, packets=packets)
    assert len(read_valid_prefix(output, packets)) == 1


def test_frozen_adjudication_execution_hashes_tools_and_packet():
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    for group_name in ("semantic_parent", "frozen_packet"):
        group = execution[group_name]
        if "path" in group:
            assert file_sha256(ROOT / group["path"]) == group["sha256"]
    parent = execution["semantic_parent"]
    for stem in ("annotation_guide", "agreement_protocol", "agreement_report"):
        assert file_sha256(ROOT / parent[stem]) == parent[f"{stem}_sha256"]
    active = execution["execution"]
    for stem in ("prompt", "validator_module", "checkpoint_helper", "final_validator"):
        assert file_sha256(ROOT / active[stem]) == active[f"{stem}_sha256"]
    packets, output = load_context(PROTOCOL)
    assert len(packets) == execution["frozen_packet"]["rows"] == 48
    assert output.name == "adjudications_raw_v1.jsonl"


def test_prompt_requires_role_blind_one_row_durable_adjudication():
    prompt = PROMPT.read_text(encoding="utf-8")
    assert "Do not inspect `adjudication_lineage_v1.jsonl`" in prompt
    assert "Neither letter has priority" in prompt
    assert "decide one row, then checkpoint it immediately" in prompt
    assert "checkpoint_dual_prior_adjudication_v1.py status" in prompt
    assert "checkpoint_dual_prior_adjudication_v1.py append" in prompt
    assert "checkpoint_dual_prior_adjudication_v1.py finalize" in prompt
    assert "validate_dual_prior_adjudication_v1.py" in prompt
