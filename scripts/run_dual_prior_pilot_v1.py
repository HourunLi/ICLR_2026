#!/usr/bin/env python3
"""Run one frozen dual-prior cell through train, score, and evaluate."""

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

from src.clir_hallucination_annotation import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/training_protocol_v1.json"
SUPPORTED_PROTOCOL_SCHEMAS = {
    "clir-dual-prior-standalone-training-protocol-v1",
    "clir-dual-prior-mutual-distillation-training-protocol-v1",
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify(path: Path, expected: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"Frozen input hash drifted for {path}: expected {expected}, got {observed}"
        )


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") not in SUPPORTED_PROTOCOL_SCHEMAS:
        raise ValueError("Unexpected dual-prior training protocol schema")
    if args.cell not in protocol["cells"]:
        raise ValueError(f"Unknown dual-prior cell {args.cell!r}")
    seeds = [int(value) for value in protocol["matched_training"]["seeds"]]
    if args.seed not in seeds:
        raise ValueError(f"Seed {args.seed} is not frozen in {seeds}")
    code = git_state(ROOT)
    if protocol["execution"]["clean_committed_worktree_required"] and code["dirty"]:
        raise RuntimeError("Dual-prior training requires a clean committed worktree")
    for spec in protocol["inputs"].values():
        verify(resolve(spec["path"]), spec["sha256"])
    integrity_spec = protocol["inputs"]["feature_integrity_report"]
    integrity = json.loads(resolve(integrity_spec["path"]).read_text(encoding="utf-8"))
    failure_count = integrity.get("failure_count")
    if failure_count is None:
        failure_count = integrity.get("summary", {}).get("failure_count")
    if int(failure_count) != int(integrity_spec["required_failure_count"]):
        raise ValueError("Feature integrity report contains failures")

    cell = protocol["cells"][args.cell]
    for split in ("train", "dev"):
        verify(resolve(cell[split]["path"]), cell[split]["sha256"])
    training = protocol["matched_training"]
    shared = training["shared_loss_weights"]
    resolved_loss_weights = {
        "final": float(shared["final"]),
        "consistency": float(shared["consistency"]),
        "negative_consistency": float(shared["negative_consistency"]),
        "score_consistency": float(shared["score_consistency"]),
        "hallucination": float(shared["hallucination"]),
        "mil": float(shared["mil"]),
        "token_reward": float(shared["token_reward"]),
        "tail": float(shared["tail"]),
        "relative_tail": float(shared["relative_tail"]),
        "pseudo_tail": float(shared["pseudo_tail"]),
        "progress": float(shared["progress"]),
        "prior": float(cell["prior_weight"]),
        "key_prior": float(cell["key_prior_weight"]),
        "complete_prior": float(cell["complete_prior_weight"]),
        "prior_distill": float(
            cell.get("prior_distill_weight", shared["prior_distill"])
        ),
        "gate_prior": float(shared["gate_prior"]),
        "reconstruction": float(shared["reconstruction"]),
    }
    output_root = (
        resolve(protocol["execution"]["output_root"])
        / f"seed_{args.seed}"
        / args.cell
    )
    paths = {
        "model": output_root / "model.pt",
        "metrics": output_root / "metrics.jsonl",
        "run": output_root / "run.json",
        "train_scored": output_root / "train_scored.jsonl",
        "dev_scored": output_root / "dev_scored.jsonl",
        "evaluation": output_root / "evaluation.json",
        "result": output_root / "cell_result.json",
    }
    preflight = {
        "schema_version": "clir-dual-prior-cell-preflight-v1",
        "cell": args.cell,
        "seed": args.seed,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "code": code,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resolved_loss_weights": resolved_loss_weights,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to reuse an existing output for {args.cell}/seed {args.seed}")
    output_root.mkdir(parents=True, exist_ok=True)

    model = protocol["model"]
    train_command = [
        sys.executable,
        str(ROOT / "train_clir.py"),
        "--train_jsonl",
        str(resolve(cell["train"]["path"])),
        "--val_jsonl",
        str(resolve(cell["dev"]["path"])),
        "--output_model",
        str(paths["model"]),
        "--metrics_jsonl",
        str(paths["metrics"]),
        "--run_json",
        str(paths["run"]),
        "--experiment_protocol_config",
        str(protocol_path),
        "--expected_train_sha256",
        cell["train"]["sha256"],
        "--expected_val_sha256",
        cell["dev"]["sha256"],
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
        "--pin_memory",
        "--persistent_workers",
        "--no-group_by_semantic_id",
        "--prior_phase_mode",
        training["prior_phase_mode"],
        "--skip_feature_finite_check",
        "--final_weight",
        str(resolved_loss_weights["final"]),
        "--consistency_weight",
        str(resolved_loss_weights["consistency"]),
        "--negative_consistency_weight",
        str(resolved_loss_weights["negative_consistency"]),
        "--score_consistency_weight",
        str(resolved_loss_weights["score_consistency"]),
        "--hallucination_weight",
        str(resolved_loss_weights["hallucination"]),
        "--mil_weight",
        str(resolved_loss_weights["mil"]),
        "--token_reward_weight",
        str(resolved_loss_weights["token_reward"]),
        "--tail_weight",
        str(resolved_loss_weights["tail"]),
        "--relative_tail_weight",
        str(resolved_loss_weights["relative_tail"]),
        "--pseudo_tail_weight",
        str(resolved_loss_weights["pseudo_tail"]),
        "--progress_weight",
        str(resolved_loss_weights["progress"]),
        "--prior_weight",
        str(resolved_loss_weights["prior"]),
        "--key_prior_weight",
        str(resolved_loss_weights["key_prior"]),
        "--complete_prior_weight",
        str(resolved_loss_weights["complete_prior"]),
        "--prior_distill_weight",
        str(resolved_loss_weights["prior_distill"]),
        "--gate_prior_weight",
        str(resolved_loss_weights["gate_prior"]),
        "--reconstruction_weight",
        str(resolved_loss_weights["reconstruction"]),
    ]
    run(train_command)
    run_record = json.loads(paths["run"].read_text(encoding="utf-8"))
    if run_record.get("status") != "completed":
        raise RuntimeError(f"Training did not complete for {args.cell}/seed {args.seed}")
    checkpoint_sha = file_sha256(paths["model"])

    def score(split: str, output: Path) -> str:
        spec = cell[split]
        run(
            [
                sys.executable,
                str(ROOT / "score_clir.py"),
                "--input_jsonl",
                str(resolve(spec["path"])),
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
                spec["sha256"],
                "--expected_checkpoint_sha256",
                checkpoint_sha,
            ]
        )
        return file_sha256(output)

    train_scored_sha = score("train", paths["train_scored"])
    dev_scored_sha = score("dev", paths["dev_scored"])
    run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_dual_prior_predictions_v1.py"),
            "--protocol",
            str(protocol_path),
            "--cell",
            args.cell,
            "--seed",
            str(args.seed),
            "--train-scored",
            str(paths["train_scored"]),
            "--dev-scored",
            str(paths["dev_scored"]),
            "--expected-train-scored-sha256",
            train_scored_sha,
            "--expected-dev-scored-sha256",
            dev_scored_sha,
            "--output-json",
            str(paths["evaluation"]),
        ]
    )
    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    result = {
        "schema_version": "clir-dual-prior-cell-result-v1",
        "evidence_tier": protocol["evidence_tier"],
        "cell": args.cell,
        "seed": args.seed,
        "description": cell["description"],
        "resolved_loss_weights": resolved_loss_weights,
        "protocol_sha256": file_sha256(protocol_path),
        "train_manifest_sha256": cell["train"]["sha256"],
        "dev_manifest_sha256": cell["dev"]["sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "metrics_sha256": file_sha256(paths["metrics"]),
        "run_sha256": file_sha256(paths["run"]),
        "train_scored_sha256": train_scored_sha,
        "dev_scored_sha256": dev_scored_sha,
        "train_scoring_health_sha256": file_sha256(
            paths["train_scored"].with_name(f"{paths['train_scored'].name}.health.json")
        ),
        "dev_scoring_health_sha256": file_sha256(
            paths["dev_scored"].with_name(f"{paths['dev_scored'].name}.health.json")
        ),
        "evaluation_sha256": file_sha256(paths["evaluation"]),
        "dev_metrics": {
            head: {
                granularity: evaluation["heads"][head][granularity]["neural"]["dev"]
                for granularity in ("token", "unit")
            }
            for head in ("key", "complete")
        },
        "position_baselines": {
            head: {
                granularity: evaluation["heads"][head][granularity]["position_only"]["dev"]
                for granularity in ("token", "unit")
            }
            for head in ("key", "complete")
        },
        "correctness": evaluation["correctness"],
        "head_separation": evaluation["head_separation"],
        "prior_collaboration": evaluation["prior_collaboration"],
        "code": code,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(paths["result"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
