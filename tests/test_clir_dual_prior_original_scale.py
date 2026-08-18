import json
from pathlib import Path
import subprocess
import sys

from src.clir_hallucination_annotation import file_sha256, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dual_prior_original_scale_v2"
DATA_PROTOCOL = CONFIG / "data_protocol_v2.json"
DATA_REPORT = CONFIG / "data_report_v2.json"
TRAINING_PROTOCOL = CONFIG / "training_protocol_v2.json"
TRAINING_RESULT = CONFIG / "training_result_v2.json"


def test_scaled_data_is_query_disjoint_and_only_gold_rows_receive_priors():
    protocol = json.loads(DATA_PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(DATA_REPORT.read_text(encoding="utf-8"))
    train_path = ROOT / report["train"]["path"]
    assert file_sha256(DATA_PROTOCOL) == report["protocol_sha256"]
    assert file_sha256(train_path) == report["train"]["sha256"]
    assert report["train"]["rows"] == 3968
    assert report["train"]["queries"] == 496
    assert report["train"]["candidates_per_query"] == 8
    assert report["prior_membership"] == {
        "dev_queries": 16,
        "excluded_dev_query_rows": 128,
        "prior_ranking_validation_overlap": 0,
        "train_dev_overlap": 0,
        "train_queries": 48,
    }

    rows = read_jsonl(train_path)
    annotations = read_jsonl(
        ROOT / protocol["inputs"]["prior_train_annotations"]["path"]
    )
    annotation_ids = {str(row["id"]) for row in annotations}
    supervised = {
        str(row["id"])
        for row in rows
        if "key_prior_target" in row or "complete_prior_target" in row
    }
    assert supervised == annotation_ids
    assert len(supervised) == 48
    for row in rows:
        fields = {
            name
            for name in ("key_prior_target", "complete_prior_target")
            if name in row
        }
        if row["id"] in supervised:
            assert fields == {"key_prior_target", "complete_prior_target"}
            assert len(row["key_prior_target"]) == len(row["output_token_ids"])
            assert len(row["complete_prior_target"]) == len(row["output_token_ids"])
        else:
            assert not fields


def test_scale_protocol_preserves_the_original_shared_gradient_method():
    protocol = json.loads(TRAINING_PROTOCOL.read_text(encoding="utf-8"))
    invariant = protocol["method_invariant"]
    assert protocol["status"] == "frozen_before_training"
    assert invariant["mutual_weight"] == 0.25
    assert "stopgrad(A_complete)" in invariant["key_complete_mutual_objective"]
    assert "stopgrad(A_key)" in invariant["key_complete_mutual_objective"]
    assert invariant["gate_gradient_destination"].startswith(
        "reward gate and shared upstream token representation"
    )
    assert invariant["head_only_or_detached_feature_repair"] is False
    assert invariant["containment_replacement"] is False
    assert invariant["reconstruction"] is False

    control = protocol["cells"]["g0_original_mutual_control"]
    gate = protocol["cells"]["g1_original_shared_gate"]
    for key in (
        "prior_weight",
        "key_prior_weight",
        "complete_prior_weight",
        "prior_distill_weight",
    ):
        assert control[key] == gate[key]
    assert control["prior_distill_weight"] == 0.25
    assert control["gate_prior_weight"] == 0.0
    assert gate["gate_prior_weight"] == 10.0
    assert protocol["method"]["weight_scan_in_this_protocol"] is False
    assert protocol["evaluation"]["pilot_test_access_allowed"] is False
    assert protocol["evaluation"]["final_test_access_allowed"] is False


def test_scale_runner_dry_run_resolves_only_the_frozen_gate_difference():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_dual_prior_original_scale_v2.py"),
            "--protocol",
            str(TRAINING_PROTOCOL),
            "--cell",
            "g1_original_shared_gate",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    preflight = json.loads(completed.stdout)
    weights = preflight["resolved_loss_weights"]
    assert weights["final"] == 1.0
    assert weights["prior"] == 1.0
    assert weights["key_prior"] == 1.0
    assert weights["complete_prior"] == 1.0
    assert weights["prior_distill"] == 0.25
    assert weights["gate_prior"] == 10.0
    assert all(
        weights[name] == 0.0
        for name in (
            "consistency",
            "negative_consistency",
            "score_consistency",
            "hallucination",
            "mil",
            "token_reward",
            "tail",
            "relative_tail",
            "pseudo_tail",
            "progress",
            "reconstruction",
        )
    )


def test_scale_result_preserves_method_identity_and_reports_no_ranking_gain():
    result = json.loads(TRAINING_RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "completed_original_shared_gradient_scale_and_ranking"
    assert result["required_matrix_cells"] == 6
    assert result["completed_matrix_cells"] == 6
    assert result["training_commit"] == "ee549aff034acbe1496b7c6f79ac6a9b76502cae"
    assert result["original_method_preserved"] is True
    assert result["head_only_or_other_architecture_attempted"] is False
    assert result["pilot_test_accessed"] is False
    assert result["final_test_accessed"] is False

    invariant = result["method_invariant"]
    assert invariant["name"] == "original_shared_gradient_dual_prior_reward_gate"
    assert invariant["mutual_weight"] == 0.25
    assert invariant["head_only_or_detached_feature_repair"] is False
    assert invariant["containment_replacement"] is False
    assert invariant["reconstruction"] is False

    primary = result["primary_ranking_comparison"]
    assert primary["metric"] == "reward_bon_accuracy@16"
    assert primary["by_seed"] == {"42": -0.008, "43": -0.01, "44": -0.008}
    assert primary["positive_seed_count"] == 0
    assert primary["stable_positive"] is False
    assert result["ranking_improvement_established"] is False
    assert primary["aggregate_query_paired"]["query_count"] == 500
    assert primary["aggregate_query_paired"]["replicates"] == 10000
    low, high = primary["aggregate_query_paired"]["bootstrap_ci"]
    assert low < 0.0 < high
