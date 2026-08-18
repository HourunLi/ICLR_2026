import ast
import json
from pathlib import Path

from scripts.summarize_dual_prior_agreement_v1 import (
    assert_inherited_gate,
    primary_is_annotation_a,
    summarize,
    unit_set_f1,
    unit_set_jaccard,
)
from src.clir_hallucination_annotation import canonical_sha256, file_sha256, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "configs/dual_prior_evidence_v1/agreement_adjudication_protocol_v1.json"
)
SUMMARIZER = ROOT / "scripts/summarize_dual_prior_agreement_v1.py"


def _item(item_id: str) -> dict:
    return {
        "schema_version": "clir-dual-prior-evidence-item-v1",
        "item_id": item_id,
        "problem": "Compute a two-step result.",
        "trajectory": "Use the input.\nCompute the result.\nState the result.",
        "units": [
            {"unit_index": 0, "text": "Use the input."},
            {"unit_index": 1, "text": "Compute the result."},
            {"unit_index": 2, "text": "State the result."},
        ],
    }


def _annotation(
    item_id: str,
    *,
    key: list[int],
    complete: list[int],
    rationale: str,
) -> dict:
    return {
        "schema_version": "clir-dual-prior-evidence-annotation-v1",
        "item_id": item_id,
        "eligibility": "usable",
        "key_unit_indices": key,
        "complete_unit_indices": complete,
        "confidence": "high",
        "rationale": rationale,
    }


def test_set_metrics_are_symmetric_and_define_empty_agreement():
    assert unit_set_f1([], []) == 1.0
    assert unit_set_jaccard([], []) == 1.0
    assert unit_set_f1([1], [1, 2]) == 2 / 3
    assert unit_set_f1([1, 2], [1]) == 2 / 3
    assert unit_set_jaccard([1], [1, 2]) == 1 / 2


def test_summarizer_uses_python_literals_in_executable_code():
    tree = ast.parse(SUMMARIZER.read_text(encoding="utf-8"))
    forbidden_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in {"false", "true", "null"}
    }
    assert forbidden_names == set()


def test_summary_applies_macro_gate_and_blinds_every_disagreement():
    items = [_item("DPA-a"), _item("DPA-b")]
    primary = [
        _annotation("DPA-a", key=[1], complete=[0, 1], rationale="Exact row."),
        _annotation("DPA-b", key=[2], complete=[0, 2], rationale="Primary view."),
    ]
    secondary = [
        _annotation("DPA-a", key=[1], complete=[0, 1], rationale="Same target."),
        _annotation(
            "DPA-b",
            key=[1, 2],
            complete=[0, 1, 2],
            rationale="Secondary view.",
        ),
    ]
    thresholds = {
        "minimum_usable_rows_per_annotator": 2,
        "minimum_usable_overlap_rows": 2,
        "minimum_eligibility_agreement": 1.0,
        "minimum_macro_unit_f1_on_usable_overlap": {
            "key": 0.45,
            "complete": 0.6,
        },
        "required_nested_fraction": 1.0,
    }
    report, packet, lineage = summarize(
        items=items,
        primary_rows=primary,
        secondary_rows=secondary,
        thresholds=thresholds,
    )

    assert report["agreement_gate_passed"] is True
    assert report["exact_set_agreement_rows"]["both_targets_and_eligibility"] == 1
    assert report["disagreement_rows"] == 1
    assert report["adjudication_authorized"] is True
    assert report["training_gold_authorized"] is False
    assert len(packet) == len(lineage) == 1
    assert packet[0]["item_id"] == lineage[0]["item_id"] == "DPA-b"
    assert not {
        "correctness",
        "query_id",
        "membership",
        "source_id",
        "path_hallucinated",
    }.intersection(packet[0])
    assert {
        lineage[0]["annotation_a_source"],
        lineage[0]["annotation_b_source"],
    } == {"primary", "secondary"}
    expected_a = "primary" if primary_is_annotation_a("DPA-b") else "secondary"
    assert lineage[0]["annotation_a_source"] == expected_a


def test_frozen_agreement_protocol_inherits_parent_gates_and_hashes_inputs():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent_spec = protocol["inputs"]["parent_protocol"]
    parent_path = ROOT / parent_spec["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))

    assert protocol["status"] == "frozen_before_opening_primary_secondary_agreement_metrics"
    assert_inherited_gate(protocol, parent)
    for spec in protocol["inputs"].values():
        assert file_sha256(ROOT / spec["path"]) == spec["sha256"]
    assert protocol["training_gold_authorized_before_adjudication"] is False
    assert protocol["pilot_or_final_test_access_allowed"] is False


def test_real_agreement_passes_but_every_target_disagreement_is_blinded():
    report_path = ROOT / "configs/dual_prior_evidence_v1/agreement_report_v1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    packet = read_jsonl(ROOT / report["outputs"]["adjudication_items"])
    lineage = read_jsonl(ROOT / report["outputs"]["adjudication_lineage"])

    assert report["status"] == "agreement_gate_passed_adjudication_required"
    assert report["execution"]["dirty"] is False
    assert report["agreement_gate_passed"] is True
    assert report["training_gold_authorized"] is False
    assert report["usable_rows"] == {"primary": 64, "secondary": 64}
    assert report["usable_overlap_rows"] == 64
    assert report["eligibility_agreement"] == 1.0
    assert report["macro_unit_set_f1_on_usable_overlap"]["key"] >= 0.45
    assert report["macro_unit_set_f1_on_usable_overlap"]["complete"] >= 0.6
    assert report["exact_set_agreement_rows"]["both_targets_and_eligibility"] == 16
    assert report["disagreement_rows"] == len(packet) == len(lineage) == 48
    assert file_sha256(PROTOCOL) == report["protocol_sha256"]
    assert file_sha256(ROOT / report["outputs"]["adjudication_items"]) == report[
        "outputs"
    ]["adjudication_items_sha256"]
    assert file_sha256(ROOT / report["outputs"]["adjudication_lineage"]) == report[
        "outputs"
    ]["adjudication_lineage_sha256"]

    forbidden = set(protocol["adjudication_blinding"]["forbidden_fields"])
    for public, private in zip(packet, lineage):
        assert public["item_id"] == private["item_id"]
        assert not forbidden.intersection(public)
        assert {
            private["annotation_a_source"],
            private["annotation_b_source"],
        } == {"primary", "secondary"}
        for candidate_name in ("a", "b"):
            candidate = {
                "schema_version": "clir-dual-prior-evidence-annotation-v1",
                "item_id": public["item_id"],
                **public[f"annotation_{candidate_name}"],
            }
            assert canonical_sha256(candidate) == private[
                f"annotation_{candidate_name}_sha256"
            ]
