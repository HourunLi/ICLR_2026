import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from src.clir_data import CLIRTrajectoryDataset, write_jsonl
from src.clir_real_data import file_sha256
from train_clir import (
    LEGACY_RESUME_DEFAULTS,
    RESUME_PINNED_ARGS,
    _constant_prior_health,
    _reconcile_metrics_history,
    _validate_resume_device,
    _validate_resume_training_args,
    split_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_rows(path: Path, query_prefix: str) -> None:
    rows = []
    for index in range(4):
        rows.append({
            "id": f"{query_prefix}-{index}",
            "query_id": f"{query_prefix}-{index // 2}",
            "hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4).add(index).tolist(),
            "correctness": index % 2,
        })
    write_jsonl(path, rows)


def _command(train: Path, val: Path, output: Path, epochs: int) -> list[str]:
    return [
        sys.executable,
        "train_clir.py",
        "--train_jsonl", str(train),
        "--val_jsonl", str(val),
        "--output_model", str(output),
        "--hidden_dim", "4",
        "--model_variant", "strict_swift",
        "--encoder_type", "identity",
        "--batch_size", "2",
        "--epochs", str(epochs),
        "--lr", "0.001",
        "--seed", "7",
        "--device", "cpu",
        "--no-group_by_semantic_id",
    ]


def test_full_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_rows(train, "train-q")
    _write_rows(val, "val-q")
    resumed = tmp_path / "resumed.pt"
    uninterrupted = tmp_path / "uninterrupted.pt"

    subprocess.run(_command(train, val, resumed, 1), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    resume_source_sha256 = file_sha256(resumed)
    subprocess.run(
        _command(train, val, resumed, 2) + ["--resume_from", str(resumed)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(_command(train, val, uninterrupted, 2), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    resumed_state = torch.load(resumed, map_location="cpu", weights_only=False)
    full_state = torch.load(uninterrupted, map_location="cpu", weights_only=False)
    assert resumed_state["completed_epoch"] == 2
    assert len(resumed_state["metrics"]) == 2
    assert resumed_state["metrics"][1]["train"]["applicable_counts"]["final"] == 4
    for name, value in resumed_state["state_dict"].items():
        assert torch.equal(value, full_state["state_dict"][name])

    run_record = json.loads(Path(f"{resumed}.run.json").read_text(encoding="utf-8"))
    assert run_record["status"] == "completed"
    assert run_record["completed_epoch"] == 2
    assert run_record["start_epoch"] == 1
    assert run_record["resumed_from"] == {
        "path": str(resumed.resolve()),
        "sha256": resume_source_sha256,
        "completed_epoch": 1,
        "device": "cpu",
    }
    assert [segment["device"] for segment in run_record["training_segments"]] == [
        "cpu",
        "cpu",
    ]
    assert resumed_state["execution_device"] == "cpu"
    assert resumed_state["training_segments"][-1]["completed_epoch"] == 2


def test_explicit_validation_rejects_query_leakage(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_rows(train, "shared-q")
    _write_rows(val, "shared-q")
    result = subprocess.run(
        _command(train, val, tmp_path / "model.pt", 1),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "query leakage" in result.stderr


def test_validation_interval_keeps_final_epoch_validation(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_rows(train, "train-q")
    _write_rows(val, "val-q")
    output = tmp_path / "model.pt"

    subprocess.run(
        _command(train, val, output, 3) + ["--val_every_n_epochs", "3"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    assert checkpoint["metrics"][0]["validation"] is None
    assert checkpoint["metrics"][1]["validation"] is None
    assert checkpoint["metrics"][2]["validation"] is not None


def test_epoch_checkpoints_and_experiment_protocol_are_recorded(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_rows(train, "train-q")
    _write_rows(val, "val-q")
    output = tmp_path / "model.pt"
    epoch_dir = tmp_path / "epochs"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"schema_version": "test-experiment-v1"}),
        encoding="utf-8",
    )

    subprocess.run(
        _command(train, val, output, 2)
        + [
            "--epoch_checkpoint_dir", str(epoch_dir),
            "--experiment_protocol_config", str(protocol),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    first = torch.load(epoch_dir / "epoch_001.pt", map_location="cpu", weights_only=False)
    second = torch.load(epoch_dir / "epoch_002.pt", map_location="cpu", weights_only=False)
    final = torch.load(output, map_location="cpu", weights_only=False)
    run = json.loads(Path(f"{output}.run.json").read_text(encoding="utf-8"))
    assert first["completed_epoch"] == 1
    assert second["completed_epoch"] == 2
    assert final["experiment_protocol"]["schema_version"] == "test-experiment-v1"
    assert run["epoch_checkpoints"] == [
        str((epoch_dir / "epoch_001.pt").resolve()),
        str((epoch_dir / "epoch_002.pt").resolve()),
    ]


def test_resume_rejects_changed_optimizer_contract(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_rows(train, "train-q")
    _write_rows(val, "val-q")
    output = tmp_path / "model.pt"
    subprocess.run(
        _command(train, val, output, 1),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        _command(train, val, output, 2)
        + ["--resume_from", str(output), "--lr", "0.002"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Resume training arguments differ" in result.stderr


def test_legacy_val_fraction_splits_whole_query_groups(tmp_path: Path):
    manifest = tmp_path / "rows.jsonl"
    rows = []
    for query_index in range(5):
        for candidate_index in range(query_index + 1):
            rows.append({
                "id": f"q{query_index}-c{candidate_index}",
                "query_id": f"q{query_index}",
                "hidden_states": [[0.0, 1.0]],
                "correctness": candidate_index % 2,
            })
    write_jsonl(manifest, rows)
    dataset = CLIRTrajectoryDataset(manifest, require_correctness=True)

    train_indices, val_indices = split_indices(dataset, val_fraction=0.4, seed=11)
    train_queries = {dataset.rows[index]["query_id"] for index in train_indices}
    val_queries = {dataset.rows[index]["query_id"] for index in val_indices}

    assert len(val_queries) == 2
    assert train_queries.isdisjoint(val_queries)
    assert len(train_indices) + len(val_indices) == len(rows)


def test_force_restarts_only_failed_zero_epoch_record(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_rows(train, "train-q")
    _write_rows(val, "val-q")
    output = tmp_path / "model.pt"
    run_path = Path(f"{output}.run.json")
    previous = {
        "schema_version": "clir-training-run-v1",
        "status": "failed",
        "completed_epoch": 0,
        "error_type": "KeyboardInterrupt",
    }
    run_path.write_text(json.dumps(previous), encoding="utf-8")

    subprocess.run(
        _command(train, val, output, 1) + ["--force"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    current = json.loads(run_path.read_text(encoding="utf-8"))
    assert current["status"] == "completed"
    assert current["restarted_from_failed_zero_epoch"] == previous

    refused = subprocess.run(
        _command(train, val, output, 1) + ["--force"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "never overwrites a checkpoint" in refused.stderr


def test_resume_uses_declared_defaults_for_legacy_missing_arguments():
    current_values = {
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "val_fraction": 0.0,
        "seed": 7,
        "device": "cpu",
        "amp_dtype": "none",
        "skip_feature_finite_check": False,
        "group_by_semantic_id": False,
        "prior_phase_mode": "joint",
        **LEGACY_RESUME_DEFAULTS,
    }
    current = SimpleNamespace(**current_values)
    checkpoint_args = {
        key: value
        for key, value in current_values.items()
        if key not in LEGACY_RESUME_DEFAULTS
    }

    _validate_resume_training_args(checkpoint_args, current)

    changed = SimpleNamespace(**{**current_values, "val_every_n_epochs": 5})
    with pytest.raises(ValueError, match="val_every_n_epochs"):
        _validate_resume_training_args(checkpoint_args, changed)


def test_metrics_history_recovers_only_from_checkpoint_truth(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    checkpoint_rows = [{"epoch": 1}, {"epoch": 2}]
    metrics.write_text(
        "\n".join(json.dumps(row) for row in [*checkpoint_rows, {"epoch": 3}]) + "\n",
        encoding="utf-8",
    )

    recovered, audit = _reconcile_metrics_history(metrics, checkpoint_rows, 2)

    assert recovered == checkpoint_rows
    assert audit == {
        "action": "reconciled_metrics_to_checkpoint",
        "recorded_rows": 3,
        "checkpoint_rows": 2,
    }
    assert [json.loads(line) for line in metrics.read_text().splitlines()] == checkpoint_rows


def test_final_checkpoint_resume_recovers_missing_secondary_artifacts(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_rows(train, "train-q")
    _write_rows(val, "val-q")
    output = tmp_path / "model.pt"
    command = _command(train, val, output, 1)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = Path(f"{output}.metrics.jsonl")
    run_path = Path(f"{output}.run.json")
    metrics.unlink()
    run_path.unlink()

    subprocess.run(
        command + ["--resume_from", str(output)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["completed_epoch"] == 1
    assert run["recovered_completed_checkpoint"] is True
    assert run["metrics_recovery"] == {
        "action": "restored_missing_metrics_from_checkpoint"
    }
    assert len(metrics.read_text(encoding="utf-8").splitlines()) == 1


def test_constant_prior_health_gate_detects_review_collapse_case():
    collapsed = _constant_prior_health(
        positive_count=3666,
        example_count=4096,
        final_correctness_bce=0.3366,
        tolerance=0.02,
    )
    learned = _constant_prior_health(
        positive_count=3666,
        example_count=4096,
        final_correctness_bce=0.2032,
        tolerance=0.02,
    )
    worse_than_prior = _constant_prior_health(
        positive_count=3666,
        example_count=4096,
        final_correctness_bce=0.7,
        tolerance=0.02,
    )

    assert collapsed["constant_prior_bce"] == pytest.approx(0.33589, abs=5e-5)
    assert collapsed["passed"] is False
    assert learned["passed"] is True
    assert worse_than_prior["passed"] is False
    assert worse_than_prior["reason"] == "not_better_than_constant_prior"
    assert worse_than_prior["relative_improvement_over_prior_bce"] < 0.0
    assert set(RESUME_PINNED_ARGS) >= {
        "prior_collapse_tolerance",
        "fail_on_prior_collapse",
        "device",
    }


def test_resume_rejects_cross_device_or_ambiguous_checkpoint():
    with pytest.raises(ValueError, match="Cross-device resume"):
        _validate_resume_device(
            {"execution_device": "cpu"},
            torch.device("cuda"),
        )
    with pytest.raises(ValueError, match="lacks an auditable resolved execution device"):
        _validate_resume_device(
            {"training_args": {"device": "auto"}, "rng_state": {"cuda": None}},
            torch.device("cpu"),
        )
