import json
from pathlib import Path

from scripts.evaluate_dual_prior_gate_predictions_v1 import (
    SUPPORTED_PROTOCOL_SCHEMAS,
)
from scripts.summarize_joint_training_drop_one_v1 import (
    classify_key_attribution,
)
from src.clir_data import read_jsonl
from src.clir_joint_training import (
    reward_config_from_protocol,
    resolve_loss_weights,
    validate_joint_protocol,
)
from src.clir_real_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs/joint_training_pilot_v1/training_protocol_v1.json"
)
DROP_ONE_PROTOCOL_PATH = (
    ROOT / "configs/joint_training_drop_one_v1/training_protocol_v1.json"
)


def load_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_drop_one_protocol():
    return json.loads(DROP_ONE_PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_joint_protocol_freezes_the_three_authorized_cells():
    protocol = load_protocol()
    validate_joint_protocol(protocol)
    assert list(protocol["cells"]) == [
        "j0_correctness",
        "jp_original_prior",
        "jall_full_retained",
    ]
    j0 = resolve_loss_weights(protocol, "j0_correctness")
    jp = resolve_loss_weights(protocol, "jp_original_prior")
    jall = resolve_loss_weights(protocol, "jall_full_retained")
    assert j0["final"] == 1.0
    assert j0["consistency"] == j0["hallucination"] == j0["prior"] == 0.0
    assert jp["prior"] == 1.0
    assert jp["consistency"] == jp["hallucination"] == 0.0
    assert jall["consistency"] == jall["hallucination"] == jall["prior"] == 1.0
    for weights in (j0, jp, jall):
        assert weights["mil"] == 0.0
        assert weights["token_reward"] == 0.0
        assert weights["tail"] == 0.0
        assert weights["relative_tail"] == 0.0
        assert weights["pseudo_tail"] == 0.0
        assert weights["progress"] == 0.0
        assert weights["reconstruction"] == 0.0


def test_joint_jall_preserves_original_dual_prior_and_disables_deferred_heads():
    protocol = load_protocol()
    config = reward_config_from_protocol(protocol, "jall_full_retained")
    assert config.key_prior_weight == 1.0
    assert config.complete_prior_weight == 1.0
    assert config.prior_distill_weight == 0.25
    assert config.gate_prior_weight == 10.0
    assert config.prior_fusion_alpha == 0.5
    assert config.progress_score_weight == 0.0
    assert config.reconstruction_weight == 0.0
    assert protocol["schema_version"] in SUPPORTED_PROTOCOL_SCHEMAS


def test_joint_artifact_hashes_and_data_report_are_exact():
    protocol = load_protocol()
    for section in ("inputs", "manifests"):
        for spec in protocol[section].values():
            path = ROOT / spec["path"]
            assert path.is_file()
            assert file_sha256(path) == spec["sha256"]
    report = json.loads(
        (ROOT / protocol["inputs"]["data_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert report["code"]["dirty"] is False
    assert report["row_classes"] == {
        "correctness_only_rows": 3866,
        "correctness_plus_consistency_rows": 54,
        "correctness_plus_hallucination_and_prior_rows": 48,
    }
    assert report["query_overlap"] == {
        "train_mechanism_dev": 0,
        "train_ranking_validation": 0,
        "mechanism_dev_ranking_validation": 0,
    }
    assert report["sampler_epoch_1"]["batches"] == 992
    assert report["sampler_epoch_1"]["positive_pairs"] == 27
    assert report["sampler_epoch_1"]["negative_pairs"] == 26


def test_joint_manifests_have_no_deferred_targets_or_auxiliary_overlap():
    protocol = load_protocol()
    train = read_jsonl(ROOT / protocol["manifests"]["train"]["path"])
    mechanism = {
        row["id"]
        for row in train
        if "token_hallucination_target" in row or "key_prior_target" in row
    }
    consistency = {
        row["id"]
        for row in train
        if "semantic_id" in row or "style_id" in row
    }
    assert len(train) == 3968
    assert len(mechanism) == 48
    assert len(consistency) == 54
    assert mechanism.isdisjoint(consistency)
    assert all("progress_targets" not in row for row in train)
    assert all("complete_reconstruction_target" not in row for row in train)


def test_drop_one_protocol_freezes_only_jph_and_jpc():
    protocol = load_drop_one_protocol()
    validate_joint_protocol(protocol)
    assert list(protocol["cells"]) == [
        "jph_prior_plus_hallucination",
        "jpc_prior_plus_consistency",
    ]
    jph = resolve_loss_weights(protocol, "jph_prior_plus_hallucination")
    jpc = resolve_loss_weights(protocol, "jpc_prior_plus_consistency")
    assert jph["prior"] == jph["hallucination"] == 1.0
    assert jph["consistency"] == 0.0
    assert jpc["prior"] == jpc["consistency"] == 1.0
    assert jpc["hallucination"] == 0.0
    for weights in (jph, jpc):
        assert weights["final"] == 1.0
        assert weights["prior_distill"] == 0.25
        assert weights["gate_prior"] == 10.0
        assert weights["mil"] == weights["tail"] == weights["progress"] == 0.0
        assert weights["reconstruction"] == 0.0
    assert protocol["schema_version"] in SUPPORTED_PROTOCOL_SCHEMAS


def test_drop_one_protocol_matches_parent_data_model_schedule_and_weights():
    parent = load_protocol()
    protocol = load_drop_one_protocol()
    assert protocol["model"] == parent["model"]
    for name, value in parent["matched_training"].items():
        assert protocol["matched_training"][name] == value
    assert protocol["losses"] == parent["losses"]
    for name, value in parent["method"].items():
        assert protocol["method"][name] == value
    assert protocol["manifests"] == parent["manifests"]
    assert protocol["inputs"] == parent["inputs"]
    assert protocol["method"]["loss_weight_scan_in_this_protocol"] is False
    assert protocol["method"]["multistream_training_in_this_protocol"] is False
    assert protocol["method"]["sampler_or_batch_packing_changed"] is False


def test_drop_one_parent_controls_and_all_referenced_hashes_are_exact():
    protocol = load_drop_one_protocol()
    for spec in protocol["parent_experiment"].values():
        path = ROOT / spec["path"]
        assert path.is_file()
        assert file_sha256(path) == spec["sha256"]
    for control in protocol["frozen_controls"]["cells"].values():
        for spec in control.values():
            path = ROOT / spec["path"]
            assert path.is_file()
            assert file_sha256(path) == spec["sha256"]
    for section in ("inputs", "manifests"):
        for spec in protocol[section].values():
            path = ROOT / spec["path"]
            assert path.is_file()
            assert file_sha256(path) == spec["sha256"]


def test_drop_one_attribution_classifier_is_exhaustive():
    assert (
        classify_key_attribution(
            jph_reproduces_drop=True, jpc_reproduces_drop=True
        )
        == "both_auxiliaries_individually_sufficient_at_seed42"
    )
    assert (
        classify_key_attribution(
            jph_reproduces_drop=True, jpc_reproduces_drop=False
        )
        == "hallucination_individually_sufficient_at_seed42"
    )
    assert (
        classify_key_attribution(
            jph_reproduces_drop=False, jpc_reproduces_drop=True
        )
        == "consistency_individually_sufficient_at_seed42"
    )
    assert (
        classify_key_attribution(
            jph_reproduces_drop=False, jpc_reproduces_drop=False
        )
        == "joint_interaction_only_at_frozen_threshold_seed42"
    )
