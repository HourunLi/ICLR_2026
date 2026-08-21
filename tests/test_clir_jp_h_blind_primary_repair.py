import json
from pathlib import Path

from scripts.repair_jp_h_blind_primary_contract_v1 import (
    resolve_invalid_annotation,
    semantic_unique_signature,
)
from scripts.run_hallucination_primary_v1 import parse_annotation
from src.clir_hallucination_annotation import read_jsonl, validate_annotation


ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "configs/jp_h_blind_validation_v1/annotation_items_v1.jsonl"
RAW = (
    ROOT
    / "run_artifacts/jp_h_blind_validation_v1/primary_mistral24_v1/all_results.jsonl"
)
OVERRIDES = (
    ROOT
    / "configs/jp_h_blind_validation_v1/primary_contract_repair_overrides_v1.json"
)


def test_all_raw_invalid_primary_rows_resolve_without_semantic_field_changes() -> None:
    items = read_jsonl(ITEMS)
    rows = read_jsonl(RAW)
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))["overrides"]
    repaired = 0
    operations = []
    for item, row in zip(items, rows):
        if row["parse_valid"]:
            validate_annotation(row["annotation"], item)
            continue
        raw_annotation, _ = parse_annotation(row["completion_text"])
        resolved, row_operations, _ = resolve_invalid_annotation(
            raw_annotation,
            item,
            overrides.get(item["item_id"], []),
        )
        validate_annotation(resolved, item)
        assert semantic_unique_signature(raw_annotation) == semantic_unique_signature(
            resolved
        )
        assert resolved["path_status"] == raw_annotation["path_status"]
        assert resolved["confidence"] == raw_annotation["confidence"]
        assert resolved["summary"] == raw_annotation["summary"]
        repaired += 1
        operations.extend(operation["operation"] for operation in row_operations)
    assert repaired == 19
    assert set(overrides) == {row["item_id"] for row in rows if not row["parse_valid"]} - {
        "JPHB-861843bc81e6ae5f",
        "JPHB-c9062966b732a98f",
        "JPHB-27b1300ce2c5b231",
        "JPHB-d44e4bb4046bc992",
        "JPHB-f1e94b790eb8da27",
        "JPHB-8d42d1148d0dbd27",
        "JPHB-87b3a16a21cd728a",
    }
    assert "whitespace_equivalent_quote_alignment" in operations
    assert "derive_first_problem_claim_index" in operations
    assert "drop_byte_identical_duplicate_claim" in operations
    assert "stable_sort_claims_by_exact_character_start" in operations


def test_repair_policy_explicitly_forbids_secondary_and_prediction_access() -> None:
    config = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    assert config["policy"]["secondary_labels_accessed"] is False
    assert config["policy"]["model_predictions_accessed"] is False
    assert config["policy"]["semantic_fields_mutable"] is False
