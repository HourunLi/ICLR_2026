#!/usr/bin/env python3
"""Run one frozen seed-42 CLIR joint-training cell end to end."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import read_jsonl  # noqa: E402
from src.clir_joint_training import (  # noqa: E402
    resolve_loss_weights,
    reward_config_from_protocol,
    validate_joint_protocol,
)
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/joint_training_pilot_v1/training_protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify(path: Path, expected: str, *, name: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"Frozen {name} hash drifted: expected {expected}, got {observed}"
        )


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def bool_flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def score_command(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    input_spec: Mapping[str, Any],
    checkpoint: Path,
    checkpoint_sha: str,
    output: Path,
) -> list[str]:
    scoring = protocol["scoring"]
    return [
        sys.executable,
        str(ROOT / "score_clir.py"),
        "--input_jsonl",
        str(resolve(input_spec["path"])),
        "--model",
        str(checkpoint),
        "--output_jsonl",
        str(output),
        "--experiment_protocol_config",
        str(protocol_path),
        "--hidden_state_source",
        protocol["matched_training"]["hidden_state_source"],
        "--batch_size",
        str(scoring["batch_size"]),
        "--num_workers",
        str(scoring["num_workers"]),
        bool_flag("pin_memory", bool(scoring["pin_memory"])),
        bool_flag("persistent_workers", bool(scoring["persistent_workers"])),
        "--device",
        "cuda",
        "--amp_dtype",
        scoring["amp_dtype"],
        "--skip_feature_finite_check",
        "--min_score_std",
        str(scoring["minimum_score_population_std"]),
        "--expected_input_sha256",
        str(input_spec["sha256"]),
        "--expected_checkpoint_sha256",
        checkpoint_sha,
    ]


def _validate_epoch_metrics(
    metrics_path: Path, protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = read_jsonl(metrics_path)
    training = protocol["matched_training"]
    if len(rows) != int(training["epochs"]):
        raise ValueError("Training metrics do not contain every frozen epoch")
    expected_counts = protocol["engineering_gates"]["per_epoch_applicable_counts"]
    for expected_epoch, row in enumerate(rows, start=1):
        if int(row.get("epoch", -1)) != expected_epoch:
            raise ValueError("Training metrics epoch order drifted")
        train = row.get("train")
        if not isinstance(train, Mapping):
            raise ValueError("Training metrics lack the train split")
        if int(train.get("examples", -1)) != int(training["train_rows"]):
            raise ValueError("Training epoch example count drifted")
        if int(train.get("batches", -1)) != int(training["batches_per_epoch"]):
            raise ValueError("Training epoch batch count drifted")
        counts = train.get("applicable_counts")
        active_batches = train.get("active_batches")
        active_losses = train.get("active_losses")
        if not isinstance(counts, Mapping):
            raise ValueError("Training metrics lack applicable supervision counts")
        if not isinstance(active_batches, Mapping) or not isinstance(active_losses, Mapping):
            raise ValueError("Training metrics lack active-supervision logging")
        for name, expected in expected_counts.items():
            if int(counts.get(name, -1)) != int(expected):
                raise ValueError(
                    f"Epoch {expected_epoch} applicable count drifted for {name}"
                )
            if int(expected) > 0 and (
                int(active_batches.get(name, 0)) <= 0 or name not in active_losses
            ):
                raise ValueError(f"Epoch {expected_epoch} lacks active loss for {name}")
        for collection_name, collection in (
            ("coverage-weighted", train.get("losses", {})),
            ("active-supervision", active_losses),
        ):
            for value in collection.values():
                if not isinstance(value, (int, float)) or not (
                    float("-inf") < float(value) < float("inf")
                ):
                    raise ValueError(
                        f"Training metrics contain a non-finite {collection_name} loss"
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_joint_protocol(protocol)
    if args.cell not in protocol["cells"]:
        raise ValueError(f"Unknown joint-training cell {args.cell!r}")
    if args.seed not in [int(value) for value in protocol["matched_training"]["seeds"]]:
        raise ValueError("Seed is not frozen in the joint-training protocol")
    code = git_state(ROOT)
    if args.execute and protocol["execution"]["clean_committed_worktree_required"]:
        if code["dirty"]:
            raise RuntimeError("Joint training requires a clean committed worktree")
    for name, spec in protocol["inputs"].items():
        verify(resolve(spec["path"]), str(spec["sha256"]), name=f"input {name}")
    for name, spec in protocol["manifests"].items():
        verify(resolve(spec["path"]), str(spec["sha256"]), name=f"manifest {name}")
    integrity_spec = protocol["inputs"]["feature_integrity_report"]
    integrity = json.loads(resolve(integrity_spec["path"]).read_text(encoding="utf-8"))
    if int(integrity.get("failure_count", -1)) != int(
        integrity_spec["required_failure_count"]
    ):
        raise ValueError("Frozen feature-integrity report contains failures")

    weights = resolve_loss_weights(protocol, args.cell)
    expected_config = reward_config_from_protocol(protocol, args.cell)
    output_root = (
        resolve(protocol["execution"]["output_root"])
        / f"seed_{args.seed}"
        / args.cell
    )
    paths = {
        "model": output_root / "model.pt",
        "metrics": output_root / "metrics.jsonl",
        "run": output_root / "run.json",
        "epoch_checkpoints": output_root / "epoch_checkpoints",
        "mechanism_train_scored": output_root / "mechanism_train_scored.jsonl",
        "mechanism_dev_scored": output_root / "mechanism_dev_scored.jsonl",
        "hallucination_evaluation": output_root / "hallucination_evaluation.json",
        "prior_evaluation": output_root / "prior_evaluation.json",
        "consistency_evaluation": output_root / "consistency_evaluation.json",
        "ranking_scored": output_root / "ranking_validation_scored.jsonl",
        "ranking_evaluation": output_root / "ranking_evaluation.json",
        "result": output_root / "cell_result.json",
    }
    preflight = {
        "schema_version": "clir-joint-training-cell-preflight-v1",
        "cell": args.cell,
        "seed": args.seed,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "code": code,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resolved_loss_weights": weights,
        "reward_config": expected_config.__dict__,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if any(path.exists() for name, path in paths.items() if name != "epoch_checkpoints"):
        raise FileExistsError(
            f"Refusing to reuse an existing joint cell: {args.cell}/seed {args.seed}"
        )
    if paths["epoch_checkpoints"].exists():
        raise FileExistsError("Refusing to reuse an epoch checkpoint directory")
    output_root.mkdir(parents=True, exist_ok=True)

    training = protocol["matched_training"]
    model = protocol["model"]
    train_spec = protocol["manifests"]["train"]
    dev_spec = protocol["manifests"]["mechanism_dev"]
    train_command = [
        sys.executable,
        str(ROOT / "train_clir.py"),
        "--train_jsonl",
        str(resolve(train_spec["path"])),
        "--val_jsonl",
        str(resolve(dev_spec["path"])),
        "--output_model",
        str(paths["model"]),
        "--metrics_jsonl",
        str(paths["metrics"]),
        "--run_json",
        str(paths["run"]),
        "--epoch_checkpoint_dir",
        str(paths["epoch_checkpoints"]),
        "--experiment_protocol_config",
        str(protocol_path),
        "--expected_train_sha256",
        str(train_spec["sha256"]),
        "--expected_val_sha256",
        str(dev_spec["sha256"]),
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
        str(args.seed),
        "--device",
        "cuda",
        "--amp_dtype",
        training["amp_dtype"],
        "--num_workers",
        str(training["num_workers"]),
        bool_flag("pin_memory", bool(training["pin_memory"])),
        bool_flag("persistent_workers", bool(training["persistent_workers"])),
        "--group_by_semantic_id",
        "--prior_phase_mode",
        training["prior_phase_mode"],
        "--skip_feature_finite_check",
        "--no-fail_on_prior_collapse",
        "--hallucination_target_mode",
        protocol["method"]["hallucination_target_mode"],
        "--hallucination_positive_weight",
        str(protocol["method"]["hallucination_positive_weight"]),
        "--negative_tail_margin",
        str(protocol["method"]["negative_tail_margin"]),
        "--relative_tail_margin",
        str(protocol["method"]["relative_tail_margin"]),
        "--pseudo_onset_threshold",
        str(protocol["method"]["pseudo_onset_threshold"]),
        "--progress_score_weight",
        str(protocol["method"]["progress_score_weight"]),
    ]
    for name in weights:
        train_command.extend([f"--{name}_weight", str(weights[name])])
    run(train_command)
    run_record = json.loads(paths["run"].read_text(encoding="utf-8"))
    if run_record.get("status") != "completed" or int(
        run_record.get("completed_epoch", -1)
    ) != int(training["epochs"]):
        raise RuntimeError(f"Training did not complete for {args.cell}/seed {args.seed}")
    metric_rows = _validate_epoch_metrics(paths["metrics"], protocol)
    checkpoint = torch.load(paths["model"], map_location="cpu", weights_only=False)
    if checkpoint.get("config") != expected_config.__dict__:
        raise ValueError("Published checkpoint config differs from the frozen cell")
    checkpoint_sha = file_sha256(paths["model"])

    scored_hashes: dict[str, str] = {}
    for manifest_name, output_name in (
        ("mechanism_train", "mechanism_train_scored"),
        ("mechanism_dev", "mechanism_dev_scored"),
        ("ranking_validation", "ranking_scored"),
    ):
        run(
            score_command(
                protocol_path=protocol_path,
                protocol=protocol,
                input_spec=protocol["manifests"][manifest_name],
                checkpoint=paths["model"],
                checkpoint_sha=checkpoint_sha,
                output=paths[output_name],
            )
        )
        scored_hashes[output_name] = file_sha256(paths[output_name])

    run(
        [
            sys.executable,
            str(ROOT / protocol["execution"]["hallucination_evaluator"]),
            "--input-jsonl",
            str(paths["mechanism_dev_scored"]),
            "--output-json",
            str(paths["hallucination_evaluation"]),
            "--expected-input-sha256",
            scored_hashes["mechanism_dev_scored"],
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / protocol["execution"]["prior_evaluator"]),
            "--protocol",
            str(protocol_path),
            "--cell",
            args.cell,
            "--seed",
            str(args.seed),
            "--train-scored",
            str(paths["mechanism_train_scored"]),
            "--dev-scored",
            str(paths["mechanism_dev_scored"]),
            "--expected-train-scored-sha256",
            scored_hashes["mechanism_train_scored"],
            "--expected-dev-scored-sha256",
            scored_hashes["mechanism_dev_scored"],
            "--output-json",
            str(paths["prior_evaluation"]),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / protocol["execution"]["consistency_evaluator"]),
            "--protocol",
            str(protocol_path),
            "--cell",
            args.cell,
            "--seed",
            str(args.seed),
            "--model",
            str(paths["model"]),
            "--output",
            str(paths["consistency_evaluation"]),
            "--device",
            "cuda",
            "--batch-size",
            str(protocol["scoring"]["batch_size"]),
            "--num-workers",
            str(protocol["scoring"]["num_workers"]),
        ]
    )
    ranking = protocol["evaluation"]
    run(
        [
            sys.executable,
            str(ROOT / protocol["execution"]["ranking_evaluator"]),
            "--input-jsonl",
            str(paths["ranking_scored"]),
            "--output-json",
            str(paths["ranking_evaluation"]),
            "--k",
            *[str(value) for value in ranking["k"]],
            "--bootstrap-replicates",
            str(ranking["bootstrap_replicates"]),
            "--confidence-level",
            str(ranking["confidence_level"]),
            "--seed",
            str(args.seed),
            "--expected-input-sha256",
            scored_hashes["ranking_scored"],
            "--expected-checkpoint-sha256",
            checkpoint_sha,
            "--expected-reward-model-variant",
            model["variant"],
            "--expected-scoring-input-sha256",
            str(protocol["manifests"]["ranking_validation"]["sha256"]),
            "--expected-scoring-batch-size",
            str(protocol["scoring"]["batch_size"]),
            "--expected-scoring-amp-dtype",
            protocol["scoring"]["amp_dtype"],
            "--expected-experiment-protocol-sha256",
            file_sha256(protocol_path),
            "--minimum-within-query-pairwise-accuracy",
            str(ranking["minimum_within_query_pairwise_accuracy"]),
        ]
    )

    hallucination = json.loads(
        paths["hallucination_evaluation"].read_text(encoding="utf-8")
    )
    prior = json.loads(paths["prior_evaluation"].read_text(encoding="utf-8"))
    consistency_report = json.loads(
        paths["consistency_evaluation"].read_text(encoding="utf-8")
    )
    ranking_report = json.loads(
        paths["ranking_evaluation"].read_text(encoding="utf-8")
    )
    result = {
        "schema_version": "clir-joint-training-cell-result-v1",
        "evidence_tier": protocol["evidence_tier"],
        "cell": args.cell,
        "seed": args.seed,
        "description": protocol["cells"][args.cell]["description"],
        "resolved_loss_weights": weights,
        "protocol_sha256": file_sha256(protocol_path),
        "train_manifest_sha256": train_spec["sha256"],
        "mechanism_dev_manifest_sha256": dev_spec["sha256"],
        "ranking_manifest_sha256": protocol["manifests"]["ranking_validation"][
            "sha256"
        ],
        "checkpoint_sha256": checkpoint_sha,
        "metrics_sha256": file_sha256(paths["metrics"]),
        "run_sha256": file_sha256(paths["run"]),
        "scored_sha256": scored_hashes,
        "evaluation_sha256": {
            name: file_sha256(paths[name])
            for name in (
                "hallucination_evaluation",
                "prior_evaluation",
                "consistency_evaluation",
                "ranking_evaluation",
            )
        },
        "training_engineering": {
            "epochs": len(metric_rows),
            "final_epoch_train": metric_rows[-1]["train"],
            "final_epoch_validation": metric_rows[-1]["validation"],
        },
        "hallucination_dev": {
            "span_tokens": hallucination["metrics"]["explicit_claim_span_tokens"],
            "claim_mean": hallucination["metrics"][
                "explicit_claim_span_shortcuts"
            ]["claim_level"]["mean_probability_ranking"],
            "onset": hallucination["metrics"]["onset"],
        },
        "prior_dev": {
            "key_unit": prior["heads"]["key"]["unit"]["neural"]["dev"],
            "complete_unit": prior["heads"]["complete"]["unit"]["neural"]["dev"],
            "collaboration": prior["prior_collaboration"]["dev"],
            "gate_integration": prior["gate_integration"]["dev"],
        },
        "consistency_train_diagnostic": consistency_report[
            "projected_representation"
        ],
        "ranking_metrics": ranking_report["metrics"],
        "ranking_health": ranking_report["ranking_health"],
        "ranking_score_distribution": ranking_report[
            "reward_scoring_provenance"
        ]["score_distribution"],
        "code": code,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(paths["result"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
