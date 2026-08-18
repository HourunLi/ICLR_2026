import hashlib
import json
from pathlib import Path

from src.clir_dual_prior_annotation import validate_item
from src.clir_hallucination_annotation import file_sha256, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/protocol_v1.json"


def _resolve(value: str) -> Path:
    return ROOT / value


def test_frozen_dual_prior_package_is_blind_exact_and_reproducible():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    package = json.loads(_resolve(protocol["outputs"]["package_report"]).read_text(encoding="utf-8"))
    items = read_jsonl(_resolve(protocol["outputs"]["annotation_items"]))
    lineage = read_jsonl(_resolve(protocol["outputs"]["annotation_lineage"]))
    selection = read_jsonl(_resolve(protocol["outputs"]["selection_manifest"]))

    assert protocol["status"] == "frozen_before_annotation"
    assert len(items) == len(lineage) == len(selection) == 64
    assert package["rows"] == package["queries"] == 64
    assert package["membership_counts"] == {"dense_train": 48, "localization_dev": 16}
    assert package["forbidden_fields_absent_from_blind_items"] is True
    assert file_sha256(_resolve(protocol["outputs"]["annotation_items"])) == package["annotation_items_sha256"]
    assert file_sha256(_resolve(protocol["outputs"]["annotation_lineage"])) == package["annotation_lineage_sha256"]

    forbidden = {"correctness", "path_status", "membership", "query_id", "source_id"}
    for item, identity, private in zip(items, lineage, selection):
        validate_item(item)
        assert not forbidden.intersection(item)
        assert item["item_id"] == identity["item_id"] == private["item_id"]
        assert hashlib.sha256(item["trajectory"].encode()).hexdigest() == identity["trajectory_sha256"]
        assert len(item["units"]) == len(identity["units"]) == private["unit_count"]
        covered = set()
        for public_unit, private_unit in zip(item["units"], identity["units"]):
            assert public_unit["unit_index"] == private_unit["unit_index"]
            assert public_unit["text"] == private_unit["text"]
            assert item["trajectory"][private_unit["char_start"] : private_unit["char_end"]] == public_unit["text"]
            covered.update(range(private_unit["char_start"], private_unit["char_end"]))
        assert all(
            character.isspace() or index in covered
            for index, character in enumerate(item["trajectory"])
        )


def test_protocol_is_standalone_and_secondary_prompt_names_exact_output():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pilot = protocol["first_training_pilot"]
    assert pilot["standalone_cells"] == [
        "D0_correctness_only",
        "D1_correctness_plus_key",
        "D2_correctness_plus_complete",
        "D3_correctness_plus_key_and_complete",
    ]
    assert {
        "mutual_prior_distillation",
        "gate_prior_alignment",
        "complete_reconstruction",
    }.issubset(pilot["disabled_in_all_cells"])
    prompt = _resolve(protocol["outputs"]["secondary_prompt"]).read_text(encoding="utf-8")
    assert protocol["annotation"]["secondary_output"] in prompt
    assert "exactly 64" in prompt
    assert "key_unit_indices" in prompt and "complete_unit_indices" in prompt
