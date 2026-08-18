#!/usr/bin/env python3
"""Run one frozen sparse-span localization cell through train, score, and evaluate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/hallucination_localization_v2/training_protocol_v2.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen input hash drifted for {path}: expected {expected_sha256}, got {observed}"
        )


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def weight_args(protocol: dict[str, Any], cell_name: str) -> list[str]:
    shared = protocol["matched_training"]["shared_loss_weights"]
    cell = protocol["cells"][cell_name]
    values = {
        "final_weight": cell.get("final_weight", shared["final"]),
        "consistency_weight": cell.get(
            "consistency_weight", shared["consistency"]
        ),
        "hallucination_weight": cell.get(
            "hallucination_weight", shared["hallucination"]
        ),
        "mil_weight": cell.get("mil_weight", shared.get("mil", 0.0)),
        "token_reward_weight": cell.get(
            "token_reward_weight", shared["token_reward"]
        ),
        "tail_weight": cell.get("tail_weight", shared["tail"]),
        "pseudo_tail_weight": cell.get(
            "pseudo_tail_weight", shared["pseudo_tail"]
        ),
        "progress_weight": cell.get("progress_weight", shared["progress"]),
        "prior_weight": cell.get("prior_weight", shared["prior"]),
    }
    return [value for key, number in values.items() for value in (f"--{key}", str(number))]


def resolve_training_seed(protocol: dict[str, Any], requested: int | None) -> int:
    training = protocol["matched_training"]
    if "seeds" in training:
        allowed = [int(seed) for seed in training["seeds"]]
        if requested is None:
            if len(allowed) != 1:
                raise ValueError("Multi-seed protocol requires an explicit --seed")
            requested = allowed[0]
        if requested not in allowed:
            raise ValueError(f"Seed {requested} is not frozen in protocol seeds {allowed}")
        return requested
    fixed = int(training["seed"])
    if requested is not None and requested != fixed:
        raise ValueError(f"Legacy protocol freezes seed {fixed}, not {requested}")
    return fixed


def resolve_fold_inputs(
    protocol: dict[str, Any], requested: int | None
) -> tuple[int | None, dict[str, Any], dict[str, Any]]:
    cross_validation = protocol.get("cross_validation")
    if cross_validation is None:
        if requested is not None:
            raise ValueError("Non-CV protocol does not accept --fold")
        return (
            None,
            protocol["inputs"]["dense_train"],
            protocol["inputs"]["localization_dev"],
        )
    if requested is None:
        raise ValueError("Cross-validation protocol requires an explicit --fold")
    folds = cross_validation["folds"]
    if str(requested) not in folds:
        raise ValueError(f"Fold {requested} is not frozen in the protocol")
    fold = folds[str(requested)]
    return requested, fold["train"], fold["dev"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-hallucination-localization-training-protocol-v2"
    ):
        raise ValueError("Unknown hallucination localization training protocol schema")
    if args.cell not in protocol["cells"]:
        raise ValueError(f"Unknown Pilot v2 cell {args.cell!r}")
    training_seed = resolve_training_seed(protocol, args.seed)
    fold, train_input, dev_input = resolve_fold_inputs(protocol, args.fold)
    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Localization training requires a clean committed worktree")
    for value in protocol["inputs"].values():
        verify(resolve(value["path"]), value["sha256"])
    for value in (train_input, dev_input):
        verify(resolve(value["path"]), value["sha256"])
    integrity = protocol["feature_integrity_gate"]
    integrity_path = resolve(integrity["report"])
    verify(integrity_path, integrity["report_sha256"])
    integrity_report = json.loads(integrity_path.read_text(encoding="utf-8"))
    failure_count = integrity_report.get("failure_count")
    if failure_count is None:
        failure_count = integrity_report.get("summary", {}).get("failure_count")
    if int(failure_count) != int(integrity["failure_count"]):
        raise ValueError("Feature integrity report contains failures")

    output_root = resolve(protocol["execution"]["output_root"])
    if fold is not None:
        output_root = output_root / f"fold_{fold}"
    if "seeds" in protocol["matched_training"]:
        output_root = output_root / f"seed_{training_seed}"
    output_root = output_root / args.cell
    paths = {
        "model": output_root / "model.pt",
        "metrics": output_root / "metrics.jsonl",
        "run": output_root / "run.json",
        "train_scored": output_root / "localization_train_scored.jsonl",
        "dev_scored": output_root / "localization_dev_scored.jsonl",
        "evaluation": output_root / "localization_dev_evaluation.json",
        "calibration": output_root / "threshold_calibration_v2.json",
        "result": output_root / "cell_result.json",
    }
    preflight = {
        "schema_version": "clir-hallucination-localization-cell-preflight-v2",
        "cell": args.cell,
        "training_seed": training_seed,
        "cross_validation_fold": fold,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "code": code,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to reuse an existing {args.cell} output directory")
    output_root.mkdir(parents=True, exist_ok=True)

    model = protocol["model"]
    training = protocol["matched_training"]
    cell = protocol["cells"][args.cell]
    train_command = [
        sys.executable,
        str(ROOT / "train_clir.py"),
        "--train_jsonl",
        str(resolve(train_input["path"])),
        "--val_jsonl",
        str(resolve(dev_input["path"])),
        "--output_model",
        str(paths["model"]),
        "--metrics_jsonl",
        str(paths["metrics"]),
        "--run_json",
        str(paths["run"]),
        "--experiment_protocol_config",
        str(protocol_path),
        "--expected_train_sha256",
        train_input["sha256"],
        "--expected_val_sha256",
        dev_input["sha256"],
        "--hidden_state_source",
        training["hidden_state_source"],
        "--hidden_dim",
        str(model["hidden_dim"]),
        "--model_variant",
        model["variant"],
        "--encoder_type",
        model["encoder_type"],
        "--model_dim",
        str(model["model_dim"]),
        "--num_feature_layers",
        str(model["num_feature_layers"]),
        "--per_layer_dim",
        str(model["per_layer_dim"]),
        "--layer_encoder_dim",
        str(model["layer_encoder_dim"]),
        "--layer_encoder_blocks",
        str(model["layer_encoder_blocks"]),
        "--layer_encoder_heads",
        str(model["layer_encoder_heads"]),
        "--layer_pool_queries",
        str(model["layer_pool_queries"]),
        "--projection_dim",
        str(model["projection_dim"]),
        "--encoder_dropout",
        str(model["encoder_dropout"]),
        "--batch_size",
        str(training["batch_size"]),
        "--epochs",
        str(training["epochs"]),
        "--val_every_n_epochs",
        str(training["validation_every_n_epochs"]),
        "--lr",
        str(training["learning_rate"]),
        "--weight_decay",
        str(training["weight_decay"]),
        "--max_grad_norm",
        str(training["max_grad_norm"]),
        "--seed",
        str(training_seed),
        "--amp_dtype",
        training["amp_dtype"],
        "--num_workers",
        str(training["num_workers"]),
        "--pin_memory",
        "--persistent_workers",
        "--no-group_by_semantic_id",
        "--prior_phase_mode",
        "joint",
        "--skip_feature_finite_check",
        "--negative_tail_margin",
        str(training["negative_tail_margin"]),
        "--hallucination_target_mode",
        cell["hallucination_target_mode"],
        "--hallucination_positive_weight",
        str(cell["hallucination_positive_weight"]),
        "--pseudo_onset_threshold",
        str(training["pseudo_onset_threshold"]),
        "--progress_score_weight",
        str(training["progress_score_weight"]),
        *weight_args(protocol, args.cell),
    ]
    run(train_command)
    run_record = json.loads(paths["run"].read_text(encoding="utf-8"))
    if run_record.get("status") != "completed":
        raise RuntimeError(f"Training cell {args.cell} did not complete")

    checkpoint_sha256 = file_sha256(paths["model"])

    def score(input_spec: dict[str, Any], output: Path) -> str:
        run(
            [
                sys.executable,
                str(ROOT / "score_clir.py"),
                "--input_jsonl",
                str(resolve(input_spec["path"])),
                "--model",
                str(paths["model"]),
                "--output_jsonl",
                str(output),
                "--experiment_protocol_config",
                str(protocol_path),
                "--hidden_state_source",
                training["hidden_state_source"],
                "--batch_size",
                "2",
                "--num_workers",
                str(training["num_workers"]),
                "--pin_memory",
                "--persistent_workers",
                "--device",
                "cuda",
                "--amp_dtype",
                "none",
                "--skip_feature_finite_check",
                "--expected_input_sha256",
                input_spec["sha256"],
                "--expected_checkpoint_sha256",
                checkpoint_sha256,
                "--onset_threshold",
                str(protocol["evaluation"]["fixed_threshold"]),
            ]
        )
        return file_sha256(output)

    train_scored_sha256 = score(train_input, paths["train_scored"])
    dev_scored_sha256 = score(dev_input, paths["dev_scored"])
    run(
        [
            sys.executable,
            str(ROOT / "evaluate_hallucination_localization.py"),
            "--input-jsonl",
            str(paths["dev_scored"]),
            "--output-json",
            str(paths["evaluation"]),
            "--expected-input-sha256",
            dev_scored_sha256,
            "--threshold",
            str(protocol["evaluation"]["fixed_threshold"]),
            "--negative-tail-margin",
            str(training["negative_tail_margin"]),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts/calibrate_hallucination_span_thresholds_v2.py"),
            "--cell",
            args.cell,
            "--train-scored",
            str(paths["train_scored"]),
            "--dev-scored",
            str(paths["dev_scored"]),
            "--output-json",
            str(paths["calibration"]),
            "--expected-train-sha256",
            train_scored_sha256,
            "--expected-dev-sha256",
            dev_scored_sha256,
            "--negative-tail-margin",
            str(training["negative_tail_margin"]),
        ]
    )
    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    result = {
        "schema_version": "clir-hallucination-localization-cell-result-v2",
        "evidence_tier": "pipeline_pilot",
        "cell": args.cell,
        "training_seed": training_seed,
        "cross_validation_fold": fold,
        "description": cell["description"],
        "protocol_sha256": file_sha256(protocol_path),
        "train_manifest_sha256": train_input["sha256"],
        "dev_manifest_sha256": dev_input["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "metrics_sha256": file_sha256(paths["metrics"]),
        "run_sha256": file_sha256(paths["run"]),
        "train_scored_sha256": train_scored_sha256,
        "dev_scored_sha256": dev_scored_sha256,
        "train_scoring_health_sha256": file_sha256(
            paths["train_scored"].with_name(f"{paths['train_scored'].name}.health.json")
        ),
        "dev_scoring_health_sha256": file_sha256(
            paths["dev_scored"].with_name(f"{paths['dev_scored'].name}.health.json")
        ),
        "localization_evaluation_sha256": file_sha256(paths["evaluation"]),
        "threshold_calibration_sha256": file_sha256(paths["calibration"]),
        "fixed_dev_metrics": evaluation["metrics"],
        "train_calibrated_dev_metrics": calibration["dev_calibrated_once"],
        "code": code,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(paths["result"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
