#!/usr/bin/env python3
"""Run one original shared-gradient dual-prior scale cell end to end."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/dual_prior_original_scale_v2/training_protocol_v2.json"
)
SCHEMA = "clir-dual-prior-original-scale-training-protocol-v2"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify(path: Path, expected: str, *, label: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"Frozen {label} hash drifted for {path}: expected {expected}, got {observed}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("Unexpected original dual-prior scale protocol schema")
    if args.cell not in protocol["cells"]:
        raise ValueError(f"Unknown cell {args.cell!r}")
    seeds = [int(value) for value in protocol["matched_training"]["seeds"]]
    if args.seed not in seeds:
        raise ValueError(f"Seed {args.seed} is not frozen in {seeds}")

    code = git_state(ROOT)
    if args.execute and protocol["execution"]["clean_committed_worktree_required"]:
        if code["dirty"]:
            raise RuntimeError("Scale training requires a clean committed worktree")
    for name, spec in protocol["inputs"].items():
        verify(resolve(spec["path"]), str(spec["sha256"]), label=f"input {name}")
    for name, spec in protocol["manifests"].items():
        verify(resolve(spec["path"]), str(spec["sha256"]), label=f"manifest {name}")
    integrity = json.loads(
        resolve(protocol["inputs"]["feature_integrity_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if int(integrity.get("failure_count", -1)) != int(
        protocol["inputs"]["feature_integrity_report"]["required_failure_count"]
    ):
        raise ValueError("Frozen feature-integrity report contains failures")

    training = protocol["matched_training"]
    cell = protocol["cells"][args.cell]
    shared = training["shared_loss_weights"]
    weights = {
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
        "prior_distill": float(cell["prior_distill_weight"]),
        "gate_prior": float(cell["gate_prior_weight"]),
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
        "localization_train_scored": output_root / "localization_train_scored.jsonl",
        "localization_dev_scored": output_root / "localization_dev_scored.jsonl",
        "localization_evaluation": output_root / "localization_evaluation.json",
        "ranking_scored": output_root / "ranking_validation_scored.jsonl",
        "ranking_evaluation": output_root / "ranking_evaluation.json",
        "result": output_root / "cell_result.json",
    }
    preflight = {
        "schema_version": "clir-dual-prior-original-scale-cell-preflight-v2",
        "cell": args.cell,
        "seed": args.seed,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "code": code,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resolved_loss_weights": weights,
        "original_method_invariant": protocol["method_invariant"],
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(
            f"Refusing to reuse an existing output for {args.cell}/seed {args.seed}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    model = protocol["model"]
    train_spec = protocol["manifests"]["train"]
    dev_spec = protocol["manifests"]["localization_dev"]
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
        "--no-group_by_semantic_id",
        "--prior_phase_mode",
        training["prior_phase_mode"],
        "--skip_feature_finite_check",
        "--no-fail_on_prior_collapse",
    ]
    for name, cli_name in (
        ("final", "final"),
        ("consistency", "consistency"),
        ("negative_consistency", "negative_consistency"),
        ("score_consistency", "score_consistency"),
        ("hallucination", "hallucination"),
        ("mil", "mil"),
        ("token_reward", "token_reward"),
        ("tail", "tail"),
        ("relative_tail", "relative_tail"),
        ("pseudo_tail", "pseudo_tail"),
        ("progress", "progress"),
        ("prior", "prior"),
        ("key_prior", "key_prior"),
        ("complete_prior", "complete_prior"),
        ("prior_distill", "prior_distill"),
        ("gate_prior", "gate_prior"),
        ("reconstruction", "reconstruction"),
    ):
        train_command.extend([f"--{cli_name}_weight", str(weights[name])])
    run(train_command)
    run_record = json.loads(paths["run"].read_text(encoding="utf-8"))
    if run_record.get("status") != "completed":
        raise RuntimeError(f"Training did not complete for {args.cell}/seed {args.seed}")
    checkpoint_sha = file_sha256(paths["model"])

    scored_hashes: dict[str, str] = {}
    for manifest_name, output_name in (
        ("localization_train", "localization_train_scored"),
        ("localization_dev", "localization_dev_scored"),
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
            str(ROOT / protocol["execution"]["localization_evaluator"]),
            "--protocol",
            str(protocol_path),
            "--cell",
            args.cell,
            "--seed",
            str(args.seed),
            "--train-scored",
            str(paths["localization_train_scored"]),
            "--dev-scored",
            str(paths["localization_dev_scored"]),
            "--expected-train-scored-sha256",
            scored_hashes["localization_train_scored"],
            "--expected-dev-scored-sha256",
            scored_hashes["localization_dev_scored"],
            "--output-json",
            str(paths["localization_evaluation"]),
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

    localization = json.loads(
        paths["localization_evaluation"].read_text(encoding="utf-8")
    )
    ranking_report = json.loads(
        paths["ranking_evaluation"].read_text(encoding="utf-8")
    )
    result = {
        "schema_version": "clir-dual-prior-cell-result-v1",
        "experiment_schema_version": SCHEMA,
        "evidence_tier": protocol["evidence_tier"],
        "cell": args.cell,
        "seed": args.seed,
        "description": cell["description"],
        "resolved_loss_weights": weights,
        "original_method_invariant": protocol["method_invariant"],
        "protocol_sha256": file_sha256(protocol_path),
        "train_manifest_sha256": train_spec["sha256"],
        "dev_manifest_sha256": dev_spec["sha256"],
        "ranking_manifest_sha256": protocol["manifests"]["ranking_validation"][
            "sha256"
        ],
        "checkpoint_sha256": checkpoint_sha,
        "metrics_sha256": file_sha256(paths["metrics"]),
        "run_sha256": file_sha256(paths["run"]),
        "scored_sha256": scored_hashes,
        "localization_evaluation_sha256": file_sha256(
            paths["localization_evaluation"]
        ),
        "ranking_evaluation_sha256": file_sha256(paths["ranking_evaluation"]),
        "dev_metrics": {
            head: {
                granularity: localization["heads"][head][granularity]["neural"][
                    "dev"
                ]
                for granularity in ("token", "unit")
            }
            for head in ("key", "complete")
        },
        "dev_correctness": localization["correctness"]["dev"],
        "dev_prior_collaboration": localization["prior_collaboration"]["dev"],
        "dev_gate_integration": localization["gate_integration"]["dev"],
        "ranking_metrics": ranking_report["metrics"],
        "ranking_health": ranking_report["ranking_health"],
        "code": code,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(paths["result"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
