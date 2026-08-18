import json
from pathlib import Path

from scripts.materialize_dual_prior_gold_v1 import target_for
from src.clir_hallucination_annotation import file_sha256, read_jsonl
from src.clir_supervision import validate_supervision_annotation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dual_prior_evidence_v1"
PROTOCOL = CONFIG / "gold_materialization_protocol_v1.json"


def _indexed(path: Path, key: str) -> dict[str, dict]:
    rows = read_jsonl(path)
    output = {str(row[key]): row for row in rows}
    assert len(output) == len(rows)
    return output


def test_target_for_maps_whole_fixed_units_and_preserves_trailing_tokens():
    units = [
        {"token_start": 0, "token_end_exclusive": 2},
        {"token_start": 3, "token_end_exclusive": 5},
    ]
    assert target_for([1], units, 7) == [0, 0, 0, 1, 1, 0, 0]
    assert target_for([0, 1], units, 7) == [1, 1, 0, 1, 1, 0, 0]


def test_gold_protocol_freezes_semantic_decision_and_mapping_roles_separately():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == (
        "frozen_after_role_blind_adjudication_before_gold_materialization"
    )
    assert protocol["decision_rule"]["exact_agreement_rows"] == 16
    assert protocol["decision_rule"]["adjudicated_disagreement_rows"] == 48
    assert "alignment carrier only" in protocol["inputs"]["primary_mapped_labels"][
        "allowed_use"
    ]
    for spec in protocol["inputs"].values():
        path = ROOT / spec["path"]
        assert file_sha256(path) == spec["sha256"]


def test_materialized_gold_is_exact_nested_split_safe_and_supervision_valid():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report_path = ROOT / protocol["outputs"]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gold_path = ROOT / protocol["outputs"]["mapped_gold_labels"]
    supervision_path = ROOT / protocol["outputs"]["supervision_all"]
    train_path = ROOT / protocol["outputs"]["supervision_dense_train"]
    dev_path = ROOT / protocol["outputs"]["supervision_localization_dev"]
    gold = read_jsonl(gold_path)
    supervision = read_jsonl(supervision_path)
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)

    assert report["status"] == "ready_for_standalone_D0_D3_protocol"
    assert report["ready_for_standalone_D0_D3_protocol"] is True
    assert report["decision_source_counts"] == {
        "exact_independent_agreement": 16,
        "role_blind_adjudication": 48,
    }
    assert report["membership_counts"] == {
        "dense_train": 48,
        "localization_dev": 16,
    }
    assert len(gold) == len(supervision) == 64
    assert len(train) == 48 and len(dev) == 16
    assert file_sha256(gold_path) == report["outputs"]["mapped_gold_labels_sha256"]
    assert file_sha256(supervision_path) == report["outputs"]["supervision_all_sha256"]
    assert file_sha256(train_path) == report["outputs"]["supervision_dense_train_sha256"]
    assert file_sha256(dev_path) == report["outputs"]["supervision_localization_dev_sha256"]

    sources = _indexed(ROOT / protocol["inputs"]["trajectory_manifest"]["path"], "id")
    supervision_by_id = {row["id"]: row for row in supervision}
    assert set(supervision_by_id) == {row["id"] for row in gold}
    for row in gold:
        assert row["eligibility"] == "usable"
        assert set(row["key_unit_indices"]).issubset(row["complete_unit_indices"])
        assert any(row["key_prior_target"])
        assert any(row["complete_prior_target"])
        assert all(
            key <= complete
            for key, complete in zip(
                row["key_prior_target"], row["complete_prior_target"]
            )
        )
        assert row["key_prior_target"] == target_for(
            row["key_unit_indices"],
            row["mapped_units"],
            len(row["key_prior_target"]),
        )
        assert row["complete_prior_target"] == target_for(
            row["complete_unit_indices"],
            row["mapped_units"],
            len(row["complete_prior_target"]),
        )
        assert row["gold_provenance"]["unilateral_primary_semantic_target_used"] is False
        validate_supervision_annotation(supervision_by_id[row["id"]], sources[row["id"]])

    train_ids = {row["id"] for row in train}
    dev_ids = {row["id"] for row in dev}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids | dev_ids == set(supervision_by_id)


def test_gold_unit_decisions_equal_shared_targets_or_validated_adjudications():
    items = _indexed(CONFIG / "annotation_items_v1.jsonl", "item_id")
    primary = _indexed(CONFIG / "labels_primary_v1.jsonl", "item_id")
    secondary = _indexed(CONFIG / "labels_secondary_v1.jsonl", "item_id")
    adjudications = _indexed(CONFIG / "adjudications_v1.jsonl", "item_id")
    agreement = json.loads((CONFIG / "agreement_report_v1.json").read_text(encoding="utf-8"))
    diagnostic = {row["item_id"]: row for row in agreement["row_diagnostics"]}
    gold = _indexed(CONFIG / "labels_gold_v1.jsonl", "item_id")
    assert set(gold) == set(items)

    for item_id, row in gold.items():
        target = (
            row["eligibility"],
            row["key_unit_indices"],
            row["complete_unit_indices"],
        )
        if diagnostic[item_id]["exact_target_agreement"]:
            primary_target = (
                primary[item_id]["eligibility"],
                primary[item_id]["key_unit_indices"],
                primary[item_id]["complete_unit_indices"],
            )
            secondary_target = (
                secondary[item_id]["eligibility"],
                secondary[item_id]["key_unit_indices"],
                secondary[item_id]["complete_unit_indices"],
            )
            assert target == primary_target == secondary_target
        else:
            adjudicated_target = (
                adjudications[item_id]["eligibility"],
                adjudications[item_id]["key_unit_indices"],
                adjudications[item_id]["complete_unit_indices"],
            )
            assert target == adjudicated_target


def test_gold_semantic_audit_records_position_shortcut_and_wrong_path_improvement():
    audit = json.loads(
        (CONFIG / "gold_semantic_audit_v1.json").read_text(encoding="utf-8")
    )
    assert audit["rows"] == 64
    assert audit["path_strata"] == {
        "previously_adjudicated_clean": 41,
        "previously_adjudicated_hallucinated": 23,
    }
    assert audit["semantic_change_from_unilateral_primary"]["key_target_changed_rows"] > 0
    position = audit["position_diagnostic"]
    assert position["all"]["rows_with_every_key_unit_in_last_quarter"] > 0
    assert "position-only baseline" in position["interpretation"]
    overlap = audit["wrong_path_overlap_diagnostic"]
    assert overlap["adjudicated_gold"]["key_contains_exact_hallucination_onset"] > overlap[
        "primary"
    ]["key_contains_exact_hallucination_onset"]
    assert "diagnostic only" in audit["scope"]
