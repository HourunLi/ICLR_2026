import json
from pathlib import Path

from scripts.run_hallucination_primary_v1 import load_context
from src.clir_real_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs/jp_h_blind_validation_v1/primary_annotation_protocol_v1.json"
)
MODEL = ROOT.parent / "models/Mistral-Small-24B-Instruct-2501-9527884"


def test_primary_protocol_is_bound_to_blind_items_and_guide_only() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_blind_to_secondary_and_scores"
    semantic_inputs = protocol["blind_isolation"]["semantic_inputs"]
    assert semantic_inputs == [
        "configs/jp_h_blind_validation_v1/annotation_guide_v1.md",
        "configs/jp_h_blind_validation_v1/annotation_items_v1.jsonl",
    ]
    assert protocol["primary_annotator"]["decoding"] == {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 2048,
    }
    items = ROOT / semantic_inputs[1]
    guide = ROOT / semantic_inputs[0]
    assert file_sha256(items) == protocol["execution"]["expected_items_sha256"]
    assert file_sha256(guide) == protocol["execution"]["expected_guide_sha256"]


def test_existing_primary_runner_resolves_exact_96_row_contract() -> None:
    protocol, items, guide, frozen = load_context(
        PROTOCOL,
        model_path=MODEL,
        device="cuda:0",
    )
    assert len(items) == 96
    assert len(guide) == 5755
    assert frozen["items_sha256"] == protocol["execution"]["expected_items_sha256"]
    assert frozen["guide_sha256"] == protocol["execution"]["expected_guide_sha256"]
    assert frozen["decoding"] == protocol["primary_annotator"]["decoding"]
