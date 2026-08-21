import json
from pathlib import Path

from scripts.finalize_jp_h_blind_primary_v1 import build_primary_labels
from src.clir_hallucination_annotation import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "configs/jp_h_blind_validation_v1/annotation_items_v1.jsonl"
RESOLVED = (
    ROOT
    / "run_artifacts/jp_h_blind_validation_v1/primary_mistral24_v1/all_results_resolved_v1.jsonl"
)
RUNNER = ROOT / "scripts/finalize_jp_h_blind_primary_v1.py"


def test_resolved_primary_materializes_96_strict_raw_labels() -> None:
    labels = build_primary_labels(read_jsonl(ITEMS), read_jsonl(RESOLVED))
    assert len(labels) == 96
    assert len({row["item_id"] for row in labels}) == 96
    assert {row["path_status"] for row in labels} >= {"clean", "hallucinated"}
    assert all(
        set(row)
        == {
            "item_id",
            "claim_reviews",
            "path_status",
            "earliest_problem_claim_index",
            "confidence",
            "summary",
        }
        for row in labels
    )


def test_primary_finalizer_has_no_secondary_or_prediction_input() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "labels_secondary" not in source
    assert "predictions_v1" not in source
    protocol = json.loads(
        (
            ROOT
            / "configs/jp_h_blind_validation_v1/primary_annotation_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["outputs"]["primary_labels_raw"].endswith(
        "labels_primary_raw_v1.jsonl"
    )
