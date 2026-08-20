import json
from pathlib import Path

import torch

from scripts.evaluate_dual_prior_gate_predictions_v1 import (
    SUPPORTED_PROTOCOL_SCHEMAS,
)
from scripts.summarize_joint_training_drop_one_v1 import (
    classify_key_attribution,
)
from scripts.summarize_joint_training_packing_v1 import (
    classify_packing_outcome,
)
from scripts.audit_joint_gradient_interactions_v1 import (
    audit_stream_structure,
    controlled_batch_report,
)
from scripts.audit_joint_condition_routing_v1 import gradient_difference
from src.clir_condition_routing import (
    BLOCKED_PARAMETER_PREFIXES,
    INVARIANT_OBJECTIVE_WEIGHTS,
    validate_condition_routing_protocol,
)
from src.clir_data import (
    CLIRTrajectoryDataset,
    SemanticGroupBatchSampler,
    load_batch_packing_pools,
)
from src.clir_gradient_interaction import (
    classify_cross_stream_pressure,
    classify_same_batch_conflict,
    validate_gradient_interaction_protocol,
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
GRADIENT_INTERACTION_PROTOCOL_PATH = (
    ROOT / "configs/joint_gradient_interaction_v1/audit_protocol_v1.json"
)
GRADIENT_INTERACTION_RESULT_PATH = (
    ROOT / "configs/joint_gradient_interaction_v1/audit_result_v1.json"
)
PACKING_PROTOCOL_PATH = (
    ROOT / "configs/joint_training_packing_v1/training_protocol_v1.json"
)
CONDITION_ROUTING_PROTOCOL_PATH = (
    ROOT / "configs/joint_condition_routing_v1/audit_protocol_v1.json"
)
CONDITION_ROUTING_RESULT_PATH = (
    ROOT / "configs/joint_condition_routing_v1/audit_result_v1.json"
)


def load_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_drop_one_protocol():
    return json.loads(DROP_ONE_PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_gradient_interaction_protocol():
    return json.loads(
        GRADIENT_INTERACTION_PROTOCOL_PATH.read_text(encoding="utf-8")
    )


def load_packing_protocol():
    return json.loads(PACKING_PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_condition_routing_protocol():
    return json.loads(
        CONDITION_ROUTING_PROTOCOL_PATH.read_text(encoding="utf-8")
    )


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


def test_gradient_interaction_protocol_and_artifact_hashes_are_frozen():
    protocol = load_gradient_interaction_protocol()
    validate_gradient_interaction_protocol(protocol)
    for section in ("parent_artifacts", "inputs"):
        for spec in protocol[section].values():
            path = ROOT / spec["path"]
            assert path.is_file()
            assert file_sha256(path) == spec["sha256"]
    assert protocol["effective_objective_weights"] == {
        "final": 1.0,
        "hallucination": 1.0,
        "consistency": 1.0,
        "prior_key": 1.0,
        "prior_complete": 1.0,
        "prior_distill": 0.25,
        "prior_gate": 10.0,
        "prior_total_outer": 1.0,
    }
    assert protocol["decision_rules"]["automatic_repair_authorized"] is False
    assert protocol["decision_rules"]["additional_training_authorized"] is False


def test_gradient_interaction_controlled_batches_cover_frozen_stream():
    protocol = load_gradient_interaction_protocol()
    main = CLIRTrajectoryDataset(
        ROOT / protocol["inputs"]["train_manifest"]["path"],
        check_finite=False,
        require_correctness=True,
        load_condition=False,
        hidden_state_source="precomputed",
    )
    mechanism = CLIRTrajectoryDataset(
        ROOT / protocol["inputs"]["mechanism_manifest"]["path"],
        check_finite=False,
        require_correctness=True,
        load_condition=False,
        hidden_state_source="precomputed",
    )
    consistency_batches, stream = audit_stream_structure(main, protocol)
    controlled = controlled_batch_report(
        main, mechanism, consistency_batches, protocol
    )
    assert stream["passed"] is True
    assert [row["mechanism_consistency_overlap_batches"] for row in stream["epochs"]] == [0] * 5
    assert controlled["mechanism_rows"] == 48
    assert controlled["consistency_rows"] == 54
    assert controlled["positive_pairs"] == 27
    assert controlled["negative_pairs"] == 26


def test_gradient_interaction_classifiers_require_stability():
    stable = {
        "init": {
            "aggregate_shared_cosine": -0.2,
            "batch_shared_cosines": {"negative_fraction": 0.8, "median": -0.1},
        },
        "jp": {
            "aggregate_shared_cosine": -0.1,
            "batch_shared_cosines": {"negative_fraction": 0.75, "median": -0.02},
        },
    }
    assert (
        classify_same_batch_conflict(
            stable, threshold=-0.05, negative_fraction_minimum=0.7
        )
        == "stable_same_batch_conflict"
    )
    stable["jp"]["aggregate_shared_cosine"] = 0.1
    assert (
        classify_same_batch_conflict(
            stable, threshold=-0.05, negative_fraction_minimum=0.7
        )
        == "state_specific_same_batch_conflict"
    )
    assert (
        classify_cross_stream_pressure(
            {"init": -0.2, "jp": -0.1}, threshold=-0.05
        )
        == "stable_cross_stream_opposition"
    )
    assert (
        classify_cross_stream_pressure(
            {"init": -0.2, "jp": 0.1}, threshold=-0.05
        )
        == "state_specific_cross_stream_opposition"
    )


def test_gradient_interaction_result_is_no_update_and_matches_frozen_rules():
    protocol = load_gradient_interaction_protocol()
    result = json.loads(GRADIENT_INTERACTION_RESULT_PATH.read_text(encoding="utf-8"))
    assert result["schema_version"] == "clir-joint-gradient-interaction-result-v1"
    assert result["status"] == "completed_no_update_diagnostic"
    assert result["passed"] is True
    assert result["no_parameter_update"] is True
    assert result["protocol_sha256"] == file_sha256(
        GRADIENT_INTERACTION_PROTOCOL_PATH
    )
    assert result["code"]["dirty"] is False
    assert result["pilot_test_accessed"] is False
    assert result["final_test_accessed"] is False
    for state in result["model_state_results"].values():
        assert state["parameter_checksum_before"] == state["parameter_checksum_after"]
        assert state["optimizer_grad_buffers_absent"] is True
        assert state["objective_batch_counts"]["hallucination"] == 12
        assert state["objective_batch_counts"]["consistency"] == 14
    classifications = result["classifications"]
    assert (
        classifications["same_batch"]["hallucination__prior_total"]["classification"]
        == "no_stable_same_batch_conflict"
    )
    assert (
        classifications["cross_stream"]["consistency__prior_total"]["classification"]
        == "no_stable_cross_stream_opposition"
    )
    assert (
        classifications["cross_stream"]["consistency__prior_distill"]["classification"]
        == "stable_cross_stream_opposition"
    )
    assert protocol["decision_rules"]["automatic_repair_authorized"] is False


def test_packing_protocol_changes_only_the_declared_sampler_family():
    parent = load_drop_one_protocol()
    protocol = load_packing_protocol()
    validate_joint_protocol(protocol)
    assert list(protocol["cells"]) == ["jph_supervision_packed"]
    packed = resolve_loss_weights(protocol, "jph_supervision_packed")
    jph = resolve_loss_weights(parent, "jph_prior_plus_hallucination")
    assert packed == jph
    assert protocol["model"] == parent["model"]
    assert protocol["losses"]["shared"] == parent["losses"]["shared"]
    assert protocol["manifests"] == parent["manifests"]
    for name in (
        "seeds",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "amp_dtype",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "group_by_semantic_id",
        "hidden_state_source",
        "validation_every_n_epochs",
        "prior_phase_mode",
        "train_rows",
        "batches_per_epoch",
        "each_row_once_per_epoch",
        "auxiliary_oversampling",
    ):
        assert protocol["matched_training"][name] == parent["matched_training"][name]
    for name in (
        "consistency_margin",
        "hallucination_target_mode",
        "hallucination_positive_weight",
        "prior_fusion_alpha",
        "progress_score_weight",
        "negative_tail_margin",
        "relative_tail_margin",
        "pseudo_onset_threshold",
        "dual_prior_formula",
    ):
        assert protocol["method"][name] == parent["method"][name]
    assert protocol["method"]["mutual_distillation_changed"] is False
    assert protocol["method"]["loss_formula_changed"] is False
    assert protocol["method"]["sampler_or_batch_packing_changed"] is True
    assert protocol["scientific_scope"]["not_a_pure_packing_causal_test"] is True
    assert protocol["schema_version"] in SUPPORTED_PROTOCOL_SCHEMAS


def test_packing_sidecar_and_five_epoch_schedule_are_exact():
    protocol = load_packing_protocol()
    for section in (
        "parent_experiment",
        "preceding_gradient_audit",
        "inputs",
        "manifests",
    ):
        for spec in protocol[section].values():
            path = ROOT / spec["path"]
            assert path.is_file()
            assert file_sha256(path) == spec["sha256"]

    train_path = ROOT / protocol["manifests"]["train"]["path"]
    dataset = CLIRTrajectoryDataset(
        train_path,
        check_finite=False,
        require_correctness=True,
        load_condition=False,
        hidden_state_source="precomputed",
    )
    sidecar_path = ROOT / protocol["inputs"]["batch_packing_sidecar"]["path"]
    pools = load_batch_packing_pools(sidecar_path, dataset)
    assert list(pools) == [protocol["batch_packing"]["pool_id"]]
    mechanism_indices = set(next(iter(pools.values())))
    assert len(mechanism_indices) == 48
    assert all(
        "token_hallucination_target" in dataset.rows[index]
        and "key_prior_target" in dataset.rows[index]
        and "complete_prior_target" in dataset.rows[index]
        and "semantic_id" not in dataset.rows[index]
        and "style_id" not in dataset.rows[index]
        for index in mechanism_indices
    )

    for epoch in range(1, 6):
        sampler = SemanticGroupBatchSampler(
            dataset,
            batch_size=4,
            shuffle=True,
            seed=42,
            packing_pools=pools,
        )
        sampler.epoch = epoch - 1
        batches = list(sampler)
        flattened = [index for batch in batches for index in batch]
        assert len(batches) == 992
        assert sorted(flattened) == list(range(3968))
        mechanism_counts = [
            sum(index in mechanism_indices for index in batch) for batch in batches
        ]
        assert set(mechanism_counts) <= {0, 4}
        assert mechanism_counts.count(4) == 12


def test_packing_outcome_classifier_requires_ranking_and_both_target_families():
    assert (
        classify_packing_outcome(
            key_gates=True,
            hallucination_gates=True,
            ranking_gate=True,
        )
        == "packing_schedule_supported_at_seed42_followup_required"
    )
    assert (
        classify_packing_outcome(
            key_gates=True,
            hallucination_gates=False,
            ranking_gate=True,
        )
        == "packing_schedule_partially_supported_at_seed42"
    )
    assert (
        classify_packing_outcome(
            key_gates=True,
            hallucination_gates=True,
            ranking_gate=False,
        )
        == "packing_schedule_not_supported_at_frozen_gates_seed42"
    )


def test_condition_routing_protocol_freezes_only_the_targeted_h_route():
    protocol = load_condition_routing_protocol()
    validate_condition_routing_protocol(protocol)
    route = protocol["route_contract"]
    assert tuple(route["blocked_parameter_prefixes"]) == BLOCKED_PARAMETER_PREFIXES
    assert route["feature_norm_blocked"] is False
    assert route["condition_forward_value_changed"] is False
    assert route["dual_prior_architecture_or_loss_changed"] is False
    assert route["gate_fused_prior_target_remains_detached"] is True
    assert route["bidirectional_mutual_updates_both_prior_heads"] is True
    assert route["packing_enabled"] is False
    assert protocol["invariant_objective_weights"] == INVARIANT_OBJECTIVE_WEIGHTS
    assert protocol["decision_rules"]["automatic_training_authorized"] is False
    assert protocol["scope"]["does_not_authorize_training"] is True


def test_condition_routing_protocol_references_are_exact():
    protocol = load_condition_routing_protocol()
    for section in ("parent_artifacts", "inputs"):
        for spec in protocol[section].values():
            path = ROOT / spec["path"]
            assert path.is_file()
            assert file_sha256(path) == spec["sha256"]
    parent = load_drop_one_protocol()
    config = reward_config_from_protocol(parent, "jph_prior_plus_hallucination")
    assert config.hallucination_condition_stop_gradient is False
    assert config.prior_distill_weight == 0.25
    assert config.gate_prior_weight == 10.0


def test_condition_routing_gradient_difference_handles_blocked_parameters():
    names = ["condition_query.weight", "input_encoder.weight"]
    reference = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
    ]
    candidate = [
        None,
        torch.tensor([3.0, 4.0]),
    ]
    full = gradient_difference(names, reference, candidate)
    nonblocked = gradient_difference(
        names,
        reference,
        candidate,
        exclude_prefixes=BLOCKED_PARAMETER_PREFIXES,
    )
    assert full["difference_l2_norm"] > 0.0
    assert nonblocked["difference_l2_norm"] == 0.0
    assert nonblocked["relative_l2_difference"] == 0.0


def test_condition_routing_real_result_is_exact_no_update_and_no_training():
    result = json.loads(CONDITION_ROUTING_RESULT_PATH.read_text(encoding="utf-8"))
    assert result["schema_version"] == "clir-joint-condition-routing-result-v1"
    assert result["status"] == "completed_no_update_routing_audit"
    assert result["passed"] is True
    assert result["protocol_sha256"] == file_sha256(
        CONDITION_ROUTING_PROTOCOL_PATH
    )
    assert result["code"]["commit"] == "a5bf692b12590bfc439127dd527dc8c5da5901c2"
    assert result["code"]["dirty"] is False
    assert result["no_parameter_update"] is True
    assert result["optimizer_grad_buffers_absent"] is True
    assert result["original_dual_prior_preserved"] is True
    assert result["additional_training_performed"] is False
    assert result["next_training_requires_user_approval"] is True
    assert result["pilot_test_accessed"] is False
    assert result["final_test_accessed"] is False
    assert set(result["model_state_results"]) == {
        "initialization_seed42",
        "jp_epoch5",
    }
    for state in result["model_state_results"].values():
        assert state["passed"] is True
        assert state["controlled_rows"] == 48
        assert state["controlled_batches"] == 12
        assert state["parameter_checksum_before"] == state["parameter_checksum_after"]
        assert state["no_parameter_update"] is True
        assert state["optimizer_grad_buffers_absent"] is True
        summary = state["summary"]
        assert summary["maximum_forward_abs_difference"] == 0.0
        assert summary["maximum_objective_loss_abs_difference"] == 0.0
        assert summary["minimum_baseline_blocked_condition_l2"] > 0.0
        assert summary["maximum_routed_blocked_condition_l2"] == 0.0
        assert summary["maximum_hallucination_nonblocked_relative_l2_difference"] == 0.0
        assert all(
            value == 0.0
            for value in summary[
                "maximum_invariant_objective_relative_l2_differences"
            ].values()
        )
        assert all(
            value > 0.0
            for value in summary[
                "minimum_required_hallucination_route_norms"
            ].values()
        )
        assert summary["maximum_gate_gradient_to_key_head_l2"] == 0.0
        assert summary["maximum_gate_gradient_to_complete_head_l2"] == 0.0
        assert summary["minimum_mutual_gradient_to_key_head_l2"] > 0.0
        assert summary["minimum_mutual_gradient_to_complete_head_l2"] > 0.0
