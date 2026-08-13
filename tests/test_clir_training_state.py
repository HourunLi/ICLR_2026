import json
from pathlib import Path
import subprocess
import sys

import torch

from src.clir_data import write_jsonl


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
