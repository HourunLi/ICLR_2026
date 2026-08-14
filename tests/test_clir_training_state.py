import json
from pathlib import Path
import subprocess
import sys

import torch

from src.clir_data import CLIRTrajectoryDataset, write_jsonl
from train_clir import split_indices


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
