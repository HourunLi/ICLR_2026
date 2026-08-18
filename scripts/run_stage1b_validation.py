#!/usr/bin/env python
"""Validate and execute the frozen Stage 1B matrix without hand-written CLI drift."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_real_data import file_sha256, load_protocol, protocol_hashes
from src.clir_stage_a import atomic_write_json, git_state, load_split_manifest
from src.clir_supervision import audit_supervision_coverage


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "stage1b_validation_v4.json"
VARIANTS = ("strict_swift", "encoded_swift", "clir")
FEATURE_REFERENCE_FIELDS = (
    "hidden_states_path",
    "condition_states_path",
    "feature_sha256",
    "condition_sha256",
    "output_token_ids",
    "prompt_token_ids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "train", "score", "evaluate", "summarize", "pipeline"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Use an epoch snapshot instead of the preregistered final checkpoint.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--python", default=sys.executable, help="Interpreter used by executed commands.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the selected stage. Without this flag, print exact commands only.",
    )
    parser.add_argument(
        "--force-zero-epoch",
        action="store_true",
        help="Allow only failed completed_epoch=0 run records during matrix preflight.",
    )
    parser.add_argument(
        "--allow-failed-cells",
        action="store_true",
        help=(
            "For final-checkpoint summary only, exclude cells with explicit preregistered "
            "health-gate failure evidence and mark the result diagnostic/incomplete."
        ),
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_stage1b_protocol(path: str | Path) -> tuple[Path, Dict[str, Any]]:
    protocol_path = _resolve(path)
    protocol = _load_json(protocol_path)
    schema_version = protocol.get("schema_version")
    if schema_version not in {
        "clir-stage1b-validation-v3",
        "clir-stage1b-validation-v4",
    }:
        raise ValueError(
            "This launcher accepts Stage 1B v3/v4 protocols; got "
            f"{schema_version!r}"
        )
    if tuple(protocol.get("models", {}).get("variants", ())) != VARIANTS:
        raise ValueError("Stage 1B outcome-only validation requires the exact three-variant matrix")
    training = protocol.get("training", {})
    if sorted(training.get("seeds", ())) != [42, 43, 44]:
        raise ValueError("Stage 1B outcome-only validation requires seeds 42, 43, and 44")
    if schema_version == "clir-stage1b-validation-v4":
        health = training.get("health_gates", {})
        frozen_thresholds = {
            "constant_class_prior_bce_minimum_relative_improvement": 0.01,
            "minimum_validation_score_population_std": 0.1,
            "minimum_within_query_pairwise_accuracy": 0.6,
        }
        for key, expected in frozen_thresholds.items():
            if health.get(key) != expected:
                raise ValueError(
                    f"Stage 1B v4 freezes {key}={expected!r}; got {health.get(key)!r}"
                )
        if health.get("fail_on_prior_collapse") is not True:
            raise ValueError("Stage 1B v4 requires fail_on_prior_collapse=true")
        failed_policy = protocol.get("evaluation", {}).get("failed_cell_policy", {})
        expected_policy = {
            "allow_explicit_health_gate_failures": True,
            "unknown_or_unrun_cells": "error",
            "incomplete_summary": "diagnostic_only_no_formal_primary_claim",
            "rerun_requires_new_protocol": True,
        }
        if failed_policy != expected_policy:
            raise ValueError("Stage 1B v4 failed-cell policy differs from the frozen contract")
    return protocol_path, protocol


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _manifest_stats(path: Path) -> tuple[Dict[str, Any], Dict[str, Mapping[str, Any]]]:
    query_labels: Dict[str, list[int]] = defaultdict(list)
    rows_by_id: Dict[str, Mapping[str, Any]] = {}
    checker_versions: set[str] = set()
    label_hashes: set[str] = set()
    candidate_counts: Dict[str, set[int]] = defaultdict(set)
    missing_payloads: list[str] = []
    row_count = 0
    correct = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", ""))
            query_id = str(row.get("query_id", ""))
            if not row_id or not query_id or row_id in rows_by_id:
                raise ValueError(f"Invalid/duplicate identity at {path}:{line_number}")
            label = row.get("correctness")
            if label not in (0, 1):
                raise ValueError(f"Non-binary correctness at {path}:{line_number}")
            provenance = row.get("label_provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"Missing label provenance at {path}:{line_number}")
            checker_versions.add(str(provenance.get("checker_version")))
            label_hashes.add(str(provenance.get("label_protocol_sha256")))
            for field in ("hidden_states_path", "condition_states_path"):
                payload = row.get(field)
                if not isinstance(payload, str) or not Path(payload).is_file():
                    if len(missing_payloads) < 5:
                        missing_payloads.append(f"{row_id}:{field}={payload!r}")
            row_count += 1
            correct += int(label)
            query_labels[query_id].append(int(label))
            candidate_counts[query_id].add(int(row.get("candidate_index", -1)))
            rows_by_id[row_id] = row
    if missing_payloads:
        raise FileNotFoundError(f"Missing feature payloads: {missing_payloads}")
    widths = {len(labels) for labels in query_labels.values()}
    if len(widths) != 1:
        raise ValueError(f"Manifest {path} has non-uniform candidate counts: {sorted(widths)}")
    width = next(iter(widths))
    for query_id, indices in candidate_counts.items():
        if indices != set(range(width)):
            raise ValueError(f"Query {query_id} lacks contiguous candidate indices")
    counts = [sum(labels) for labels in query_labels.values()]
    return (
        {
            "sha256": file_sha256(path),
            "rows": row_count,
            "queries": len(query_labels),
            "candidates_per_query": width,
            "correct": correct,
            "incorrect": row_count - correct,
            "mixed_queries": sum(0 < count < width for count in counts),
            "all_correct_queries": sum(count == width for count in counts),
            "all_wrong_queries": sum(count == 0 for count in counts),
            "checker_versions": sorted(checker_versions),
            "label_protocol_sha256": sorted(label_hashes),
        },
        rows_by_id,
    )


def _validate_unchanged_feature_references(
    current_rows: Mapping[str, Mapping[str, Any]],
    relabel_report_path: Path,
) -> None:
    report = _load_json(relabel_report_path)
    source_path = Path(str(report["input_jsonl"]))
    if file_sha256(source_path) != report.get("input_sha256"):
        raise ValueError(f"Relabel source hash changed: {source_path}")
    seen: set[str] = set()
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            row_id = str(source["id"])
            current = current_rows.get(row_id)
            if current is None:
                raise ValueError(f"Relabel output is missing source row {row_id}")
            for field in FEATURE_REFERENCE_FIELDS:
                if current.get(field) != source.get(field):
                    raise ValueError(f"Relabel changed protected field {field} for {row_id}")
            seen.add(row_id)
    if seen != set(current_rows):
        raise ValueError("Relabel output/source row identities differ")


def validate_supervision_contract(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    split_name: str,
) -> Dict[str, Any]:
    if contract.get("schema_version") != "clir-supervision-audit-contract-v1":
        raise ValueError("Stage 1B outcome-only validation requires supervision audit contract v1")
    if split_name not in contract.get("applies_to", ()):
        raise ValueError(f"Supervision audit contract does not cover {split_name}")
    if contract.get("forbid_correctness_derived_auxiliary_targets") is not True:
        raise ValueError("Stage 1B outcome-only validation forbids correctness-derived targets")
    if contract.get("mechanism_claim_allowed") is not False:
        raise ValueError("Stage 1B validation must remain an outcome-only control")
    coverage = audit_supervision_coverage(
        rows,
        expected_reconstruction_dim=int(contract["expected_reconstruction_dim"]),
        require_provenance=True,
    )
    expected_field_rows = contract["expected_field_rows"]
    _assert_equal(
        {
            field: coverage["field_rows"].get(field, 0)
            for field in expected_field_rows
        },
        expected_field_rows,
        f"{split_name} CLIR supervision field coverage",
    )
    unexpected_nonzero_fields = {
        field: count
        for field, count in coverage["field_rows"].items()
        if field not in expected_field_rows and count != 0
    }
    if unexpected_nonzero_fields:
        raise ValueError(
            f"{split_name} contains auxiliary supervision fields absent from the frozen "
            f"Stage 1B contract: {unexpected_nonzero_fields}"
        )
    _assert_equal(
        coverage["eligible_components"],
        contract["expected_eligible_components"],
        f"{split_name} CLIR supervision eligibility",
    )
    _assert_equal(
        coverage["eligible_component_count"],
        contract["expected_eligible_component_count"],
        f"{split_name} CLIR eligible component count",
    )
    return coverage


def validate_protocol_inputs(
    protocol_path: Path,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    components = protocol["protocol_components"]
    acquisition_path = _resolve(components["acquisition"]["config"])
    acquisition_hashes = protocol_hashes(load_protocol(acquisition_path))
    _assert_equal(
        acquisition_hashes["acquisition_protocol_sha256"],
        components["acquisition"]["acquisition_protocol_sha256"],
        "acquisition protocol hash",
    )
    label_path = _resolve(components["labels"]["config"])
    label_hashes = protocol_hashes(load_protocol(label_path))
    _assert_equal(
        label_hashes["label_protocol_sha256"],
        components["labels"]["label_protocol_sha256"],
        "label protocol hash",
    )
    _assert_equal(
        label_hashes["evaluation_protocol_sha256"],
        components["evaluation"]["evaluation_protocol_sha256"],
        "evaluation protocol hash",
    )

    split = load_split_manifest(_resolve(protocol["split_manifest"]))
    _assert_equal(split["manifest_sha256"], protocol["split_manifest_sha256"], "split hash")
    _assert_equal(
        split["membership_sha256"]["validation"],
        protocol["validation_membership_sha256"],
        "validation membership hash",
    )

    feature_gate = protocol["feature_integrity_gate"]
    feature_report_path = _resolve(feature_gate["report"])
    _assert_equal(
        file_sha256(feature_report_path),
        feature_gate["report_sha256"],
        "feature verification report SHA256",
    )
    feature_report = _load_json(feature_report_path)
    for field in ("rows", "unique_payloads", "verified_payloads", "total_bytes", "failure_count"):
        _assert_equal(feature_report.get(field), feature_gate[field], f"feature gate {field}")
    if feature_report.get("failures") != []:
        raise ValueError("Feature integrity report contains failures")

    manifest_results: Dict[str, Any] = {}
    supervision_contract = protocol["supervision_audit"]
    for split_name in ("train", "validation"):
        frozen = protocol["manifests"][split_name]
        manifest_path = _resolve(frozen["path"])
        stats, rows = _manifest_stats(manifest_path)
        for field in (
            "sha256",
            "rows",
            "queries",
            "candidates_per_query",
            "correct",
            "incorrect",
            "mixed_queries",
            "all_correct_queries",
            "all_wrong_queries",
        ):
            _assert_equal(stats[field], frozen[field], f"{split_name} manifest {field}")
        _assert_equal(
            stats["checker_versions"],
            [components["labels"]["checker"]],
            f"{split_name} checker version",
        )
        _assert_equal(
            stats["label_protocol_sha256"],
            [components["labels"]["label_protocol_sha256"]],
            f"{split_name} label protocol hash",
        )
        relabel_report = manifest_path.with_name(f"{split_name}_relabel_v5.json")
        _validate_unchanged_feature_references(rows, relabel_report)
        stats["clir_supervision"] = validate_supervision_contract(
            list(rows.values()),
            supervision_contract,
            split_name=split_name,
        )
        manifest_results[split_name] = stats

    return {
        "protocol_config": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "component_hashes": {
            "acquisition": acquisition_hashes["acquisition_protocol_sha256"],
            "labels": label_hashes["label_protocol_sha256"],
            "evaluation": label_hashes["evaluation_protocol_sha256"],
        },
        "manifests": manifest_results,
        "feature_integrity_report": str(feature_report_path),
    }


def _output_root(protocol: Mapping[str, Any]) -> Path:
    return _resolve(protocol["execution"]["output_root"])


def _cell_paths(
    protocol: Mapping[str, Any],
    seed: int,
    variant: str,
    checkpoint_epoch: int | None = None,
) -> Dict[str, Path]:
    root = _output_root(protocol)
    model_dir = root / "models" / f"seed_{seed}"
    final_model = model_dir / f"{variant}.pt"
    if checkpoint_epoch is None:
        checkpoint = final_model
        scored = root / "scored" / f"seed_{seed}" / f"{variant}.jsonl"
        evaluation = root / "evaluation" / f"seed_{seed}" / f"{variant}.json"
    else:
        checkpoint = model_dir / f"{variant}_epochs" / f"epoch_{checkpoint_epoch:03d}.pt"
        scored = (
            root / "scored_epochs" / f"epoch_{checkpoint_epoch:03d}"
            / f"seed_{seed}" / f"{variant}.jsonl"
        )
        evaluation = (
            root / "evaluation_epochs" / f"epoch_{checkpoint_epoch:03d}"
            / f"seed_{seed}" / f"{variant}.json"
        )
    return {
        "model": checkpoint,
        "final_model": final_model,
        "metrics": model_dir / f"{variant}.metrics.jsonl",
        "run": model_dir / f"{variant}.run.json",
        "epoch_dir": model_dir / f"{variant}_epochs",
        "scored": scored,
        "score_health": scored.with_name(f"{scored.name}.health.json"),
        "evaluation": evaluation,
    }


def _validate_zero_epoch_record(path: Path) -> None:
    record = _load_json(path)
    if record.get("status") != "failed" or int(record.get("completed_epoch", -1)) != 0:
        raise FileExistsError(f"Run record is not a failed zero-epoch artifact: {path}")


def validate_empty_training_matrix(
    protocol: Mapping[str, Any],
    *,
    force_zero_epoch: bool,
) -> None:
    conflicts: list[str] = []
    for seed in protocol["training"]["seeds"]:
        for variant in protocol["models"]["variants"]:
            paths = _cell_paths(protocol, int(seed), str(variant))
            if paths["final_model"].exists() or paths["metrics"].exists():
                conflicts.append(f"{seed}/{variant}: checkpoint or metrics already exists")
                continue
            snapshots = list(paths["epoch_dir"].glob("epoch_*.pt")) if paths["epoch_dir"].exists() else []
            if snapshots:
                conflicts.append(f"{seed}/{variant}: epoch snapshots already exist")
                continue
            if paths["run"].exists():
                if not force_zero_epoch:
                    conflicts.append(f"{seed}/{variant}: run record already exists")
                    continue
                try:
                    _validate_zero_epoch_record(paths["run"])
                except (ValueError, FileExistsError) as exc:
                    conflicts.append(f"{seed}/{variant}: {exc}")
    if conflicts:
        raise FileExistsError(
            "Stage 1B matrix preflight is atomic; no cell may start:\n- "
            + "\n- ".join(conflicts)
        )


def _git_is_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _preflight_path(protocol: Mapping[str, Any]) -> Path:
    return _output_root(protocol) / "formal_preflight.json"


def write_formal_preflight(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    *,
    force_zero_epoch: bool,
) -> Dict[str, Any]:
    validated = validate_protocol_inputs(protocol_path, protocol)
    validate_empty_training_matrix(protocol, force_zero_epoch=force_zero_epoch)
    if protocol["execution"].get("formal_execution_requires_clean_git") and not _git_is_clean():
        raise RuntimeError("Formal Stage 1B execution requires a clean git worktree")
    report = {
        "schema_version": "clir-stage1b-formal-preflight-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **validated,
        "code": git_state(PROJECT_ROOT),
        "force_zero_epoch": force_zero_epoch,
        "matrix": {
            "seeds": list(protocol["training"]["seeds"]),
            "variants": list(protocol["models"]["variants"]),
            "cells": len(protocol["training"]["seeds"]) * len(protocol["models"]["variants"]),
        },
    }
    if report["code"]["dirty"]:
        raise RuntimeError("Formal Stage 1B preflight unexpectedly recorded dirty code")
    atomic_write_json(_preflight_path(protocol), report)
    return report


def require_formal_preflight(
    protocol_path: Path,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    path = _preflight_path(protocol)
    if not path.exists():
        raise FileNotFoundError(
            f"Formal preflight is absent: run --stage preflight --execute first ({path})"
        )
    report = _load_json(path)
    _assert_equal(report.get("protocol_sha256"), file_sha256(protocol_path), "preflight protocol")
    current_code = git_state(PROJECT_ROOT)
    _assert_equal(report.get("code", {}).get("commit"), current_code["commit"], "preflight code commit")
    if current_code["dirty"] or not _git_is_clean():
        raise RuntimeError("Formal Stage 1B execution requires the preflight's clean code state")
    return report


def _selected_cells(
    protocol: Mapping[str, Any],
    seed: int | None,
    variant: str | None,
) -> list[tuple[int, str]]:
    seeds = [int(seed)] if seed is not None else [int(value) for value in protocol["training"]["seeds"]]
    variants = [variant] if variant is not None else [str(value) for value in protocol["models"]["variants"]]
    for value in seeds:
        if value not in protocol["training"]["seeds"]:
            raise ValueError(f"Seed {value} is outside the frozen matrix")
    return [(value, name) for value in seeds for name in variants]


def _bool_flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def training_command(
    python: str,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    seed: int,
    variant: str,
    device: str,
    *,
    force: bool,
) -> list[str]:
    training = protocol["training"]
    health_gates = training["health_gates"]
    prior_improvement = health_gates.get(
        "constant_class_prior_bce_minimum_relative_improvement",
        health_gates.get("constant_class_prior_bce_relative_tolerance"),
    )
    if prior_improvement is None:
        raise ValueError("Protocol is missing the constant-prior BCE health threshold")
    paths = _cell_paths(protocol, seed, variant)
    command = [
        python,
        str(PROJECT_ROOT / "train_clir.py"),
        "--train_jsonl", str(_resolve(protocol["manifests"]["train"]["path"])),
        "--val_jsonl", str(_resolve(protocol["manifests"]["validation"]["path"])),
        "--output_model", str(paths["final_model"]),
        "--metrics_jsonl", str(paths["metrics"]),
        "--run_json", str(paths["run"]),
        "--epoch_checkpoint_dir", str(paths["epoch_dir"]),
        "--experiment_protocol_config", str(protocol_path),
        "--expected_train_sha256", protocol["manifests"]["train"]["sha256"],
        "--expected_val_sha256", protocol["manifests"]["validation"]["sha256"],
        "--hidden_dim", "101376",
        "--model_variant", variant,
        "--batch_size", str(training["batch_size"]),
        "--num_workers", str(training["num_workers"]),
        _bool_flag("pin_memory", bool(training["pin_memory"])),
        _bool_flag("persistent_workers", bool(training["persistent_workers"])),
        "--epochs", str(training["epochs"]),
        "--val_every_n_epochs", str(training["validation_every_n_epochs"]),
        "--lr", str(training["learning_rate"]),
        "--weight_decay", str(training["weight_decay"]),
        "--max_grad_norm", str(training["max_grad_norm"]),
        "--prior_collapse_tolerance", str(prior_improvement),
        _bool_flag("fail_on_prior_collapse", bool(health_gates["fail_on_prior_collapse"])),
        "--seed", str(seed),
        "--device", device,
        "--amp_dtype", training["amp_dtype"],
        "--no-group_by_semantic_id",
        "--prior_phase_mode", training["prior_phase_mode"],
    ]
    if training["skip_feature_finite_check_after_integrity_gate"]:
        command.append("--skip_feature_finite_check")
    if force:
        command.append("--force")
    if variant == "strict_swift":
        command.extend(["--encoder_type", "identity"])
    else:
        encoder = protocol["models"]["shared_encoder_for_encoded_swift_and_clir"]
        command.extend(
            [
                "--encoder_type", encoder["encoder_type"],
                "--model_dim", str(encoder["model_dim"]),
                "--num_feature_layers", str(encoder["num_feature_layers"]),
                "--per_layer_dim", str(encoder["per_layer_dim"]),
                "--layer_encoder_dim", str(encoder["layer_encoder_dim"]),
                "--layer_encoder_blocks", str(encoder["layer_encoder_blocks"]),
                "--layer_encoder_heads", str(encoder["layer_encoder_heads"]),
                "--layer_pool_queries", str(encoder["layer_pool_queries"]),
                "--projection_dim", str(encoder["projection_dim"]),
                "--encoder_dropout", str(encoder["encoder_dropout"]),
            ]
        )
    return command


def _require_completed_model(
    paths: Mapping[str, Path],
    checkpoint_epoch: int | None = None,
    expected_prior_improvement: float | None = None,
) -> Dict[str, Any]:
    def require_frozen_health(health: Any) -> Mapping[str, Any]:
        if not isinstance(health, Mapping) or not health.get("passed"):
            raise RuntimeError("Training health evidence is absent or failed")
        if expected_prior_improvement is not None:
            recorded_threshold = health.get(
                "minimum_relative_improvement",
                health.get("relative_tolerance"),
            )
            if (
                health.get("schema_version") != "clir-training-health-v2"
                or not health.get("enabled")
                or not isinstance(recorded_threshold, (int, float))
                or abs(float(recorded_threshold) - expected_prior_improvement) > 1e-12
            ):
                raise RuntimeError("Training health evidence does not match the frozen threshold")
        return health

    if not paths["model"].is_file():
        raise FileNotFoundError(f"Checkpoint is absent: {paths['model']}")
    if checkpoint_epoch is not None:
        if not paths["metrics"].is_file():
            raise FileNotFoundError(
                f"Epoch health metrics are absent: {paths['metrics']}"
            )
        with paths["metrics"].open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        matches = [row for row in rows if int(row.get("epoch", -1)) == checkpoint_epoch]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one health row for epoch {checkpoint_epoch} in {paths['metrics']}"
            )
        health = matches[0].get("training_health")
        try:
            health = require_frozen_health(health)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Epoch {checkpoint_epoch} training health evidence is absent/failed: "
                f"{paths['metrics']}"
            ) from exc
        return {
            "status": "epoch_snapshot_health_passed",
            "checkpoint_epoch": checkpoint_epoch,
            "training_health": dict(health),
            "metrics_jsonl": str(paths["metrics"]),
        }
    run = _load_json(paths["run"])
    if run.get("status") != "completed":
        raise RuntimeError(f"Training run did not pass health gates: {paths['run']}")
    try:
        require_frozen_health(run.get("health_gate"))
    except RuntimeError as exc:
        raise RuntimeError(f"Training health evidence is absent/failed: {paths['run']}") from exc
    return run


def scoring_command(
    python: str,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    seed: int,
    variant: str,
    device: str,
    checkpoint_epoch: int | None,
    *,
    require_inputs: bool,
) -> list[str]:
    paths = _cell_paths(protocol, seed, variant, checkpoint_epoch)
    if require_inputs:
        health_gates = protocol["training"]["health_gates"]
        expected_prior_improvement = health_gates.get(
            "constant_class_prior_bce_minimum_relative_improvement",
            health_gates.get("constant_class_prior_bce_relative_tolerance"),
        )
        _require_completed_model(
            paths,
            checkpoint_epoch,
            expected_prior_improvement=float(expected_prior_improvement),
        )
        if paths["scored"].exists() or paths["score_health"].exists():
            raise FileExistsError(
                f"Scored output or health evidence already exists: {paths['scored']} / "
                f"{paths['score_health']}"
            )
        checkpoint_sha = file_sha256(paths["model"])
    else:
        checkpoint_sha = "<CHECKPOINT_SHA256>"
    scoring = protocol["scoring"]
    command = [
        python,
        str(PROJECT_ROOT / "score_clir.py"),
        "--input_jsonl", str(_resolve(protocol["manifests"]["validation"]["path"])),
        "--model", str(paths["model"]),
        "--output_jsonl", str(paths["scored"]),
        "--experiment_protocol_config", str(protocol_path),
        "--expected_input_sha256", protocol["manifests"]["validation"]["sha256"],
        "--expected_checkpoint_sha256", checkpoint_sha,
        "--batch_size", str(scoring["batch_size"]),
        "--num_workers", str(scoring["num_workers"]),
        _bool_flag("pin_memory", bool(scoring["pin_memory"])),
        _bool_flag("persistent_workers", bool(scoring["persistent_workers"])),
        "--device", device,
        "--amp_dtype", scoring["amp_dtype"],
        "--min_score_std", str(protocol["training"]["health_gates"]["minimum_validation_score_population_std"]),
    ]
    if scoring["skip_feature_finite_check_after_integrity_gate"]:
        command.append("--skip_feature_finite_check")
    return command


def evaluation_command(
    python: str,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    seed: int,
    variant: str,
    checkpoint_epoch: int | None,
    *,
    require_inputs: bool,
) -> list[str]:
    paths = _cell_paths(protocol, seed, variant, checkpoint_epoch)
    if require_inputs:
        if not paths["scored"].is_file():
            raise FileNotFoundError(f"Scored manifest is absent: {paths['scored']}")
        if paths["evaluation"].exists():
            raise FileExistsError(f"Evaluation already exists: {paths['evaluation']}")
        scored_sha = file_sha256(paths["scored"])
        checkpoint_sha = file_sha256(paths["model"])
    else:
        scored_sha = "<SCORED_MANIFEST_SHA256>"
        checkpoint_sha = "<CHECKPOINT_SHA256>"
    evaluation = protocol["evaluation"]
    return [
        python,
        str(PROJECT_ROOT / "evaluate_clir.py"),
        "--input-jsonl", str(paths["scored"]),
        "--output-json", str(paths["evaluation"]),
        "--k", *[str(value) for value in evaluation["k"]],
        "--bootstrap-replicates", str(evaluation["bootstrap_replicates"]),
        "--confidence-level", str(evaluation["confidence_level"]),
        "--seed", str(seed),
        "--expected-input-sha256", scored_sha,
        "--expected-checkpoint-sha256", checkpoint_sha,
        "--expected-reward-model-variant", variant,
        "--expected-scoring-input-sha256", protocol["manifests"]["validation"]["sha256"],
        "--expected-scoring-batch-size", str(protocol["scoring"]["batch_size"]),
        "--expected-scoring-amp-dtype", protocol["scoring"]["amp_dtype"],
        "--expected-experiment-protocol-sha256", file_sha256(protocol_path),
        "--minimum-within-query-pairwise-accuracy",
        str(
            protocol["training"]["health_gates"].get(
                "minimum_within_query_pairwise_accuracy", 0.0
            )
        ),
    ]


def summary_command(
    python: str,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    checkpoint_epoch: int | None,
    *,
    allow_failed_cells: bool = False,
) -> list[str]:
    root = _output_root(protocol)
    if checkpoint_epoch is None:
        evaluation_dir = root / "evaluation"
        output = root / "summary.json"
    else:
        evaluation_dir = root / "evaluation_epochs" / f"epoch_{checkpoint_epoch:03d}"
        output = root / "summaries" / f"epoch_{checkpoint_epoch:03d}.json"
    command = [
        python,
        str(PROJECT_ROOT / "summarize_clir.py"),
        "--evaluation-dir", str(evaluation_dir),
        "--output-json", str(output),
        "--experiment-protocol-config", str(protocol_path),
        "--seeds", *[str(value) for value in protocol["training"]["seeds"]],
        "--variants", *[str(value) for value in protocol["models"]["variants"]],
        "--primary-k", str(max(protocol["evaluation"]["k"])),
        "--bootstrap-replicates", str(protocol["evaluation"]["bootstrap_replicates"]),
        "--confidence-level", str(protocol["evaluation"]["confidence_level"]),
        "--seed", "42",
    ]
    if allow_failed_cells:
        command.extend([
            "--allow-failed-cells",
            "--run-dir", str(root / "models"),
            "--scored-dir", str(root / "scored"),
        ])
    return command


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def _print_commands(commands: Iterable[Sequence[str]]) -> None:
    for command in commands:
        print(shlex.join(command))


def main() -> None:
    args = parse_args()
    protocol_path, protocol = load_stage1b_protocol(args.protocol_config)
    if args.checkpoint_epoch is not None and not 1 <= args.checkpoint_epoch <= int(protocol["training"]["epochs"]):
        raise ValueError("checkpoint_epoch is outside the frozen training range")
    if args.allow_failed_cells and args.stage != "summarize":
        raise ValueError("--allow-failed-cells is valid only for --stage summarize")
    if args.allow_failed_cells and args.checkpoint_epoch is not None:
        raise ValueError("Failed-cell degradation is only defined for the final checkpoint")
    if args.allow_failed_cells:
        failure_policy = protocol.get("evaluation", {}).get("failed_cell_policy", {})
        if not isinstance(failure_policy, Mapping) or not failure_policy.get(
            "allow_explicit_health_gate_failures"
        ):
            raise ValueError("The frozen protocol does not authorize incomplete summaries")

    if args.stage == "preflight":
        if args.seed is not None or args.variant is not None or args.checkpoint_epoch is not None:
            raise ValueError("Preflight always covers the complete final-checkpoint matrix")
        if args.execute:
            report = write_formal_preflight(
                protocol_path,
                protocol,
                force_zero_epoch=args.force_zero_epoch,
            )
        else:
            report = validate_protocol_inputs(protocol_path, protocol)
            validate_empty_training_matrix(protocol, force_zero_epoch=args.force_zero_epoch)
            report["formal_execution_would_require_clean_git"] = True
            report["current_git_clean"] = _git_is_clean()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    cells = _selected_cells(protocol, args.seed, args.variant)
    if args.execute:
        require_formal_preflight(protocol_path, protocol)
        if args.stage != "summarize" and (args.seed is None or args.variant is None):
            raise ValueError("Executing a cell stage requires both --seed and --variant")

    commands: list[list[str]] = []
    if args.stage in {"train", "pipeline"}:
        for seed, variant in cells:
            paths = _cell_paths(protocol, seed, variant)
            force = args.force_zero_epoch and paths["run"].exists()
            if args.execute and force:
                _validate_zero_epoch_record(paths["run"])
            command = training_command(
                args.python, protocol_path, protocol, seed, variant, args.device, force=force
            )
            commands.append(command)
            if args.execute:
                _run(command)
    if args.stage in {"score", "pipeline"}:
        for seed, variant in cells:
            command = scoring_command(
                args.python,
                protocol_path,
                protocol,
                seed,
                variant,
                args.device,
                args.checkpoint_epoch,
                require_inputs=args.execute,
            )
            commands.append(command)
            if args.execute:
                _run(command)
    if args.stage in {"evaluate", "pipeline"}:
        for seed, variant in cells:
            command = evaluation_command(
                args.python,
                protocol_path,
                protocol,
                seed,
                variant,
                args.checkpoint_epoch,
                require_inputs=args.execute,
            )
            commands.append(command)
            if args.execute:
                _run(command)
    if args.stage == "summarize":
        if args.seed is not None or args.variant is not None:
            raise ValueError("Summary always consumes the complete matrix")
        command = summary_command(
            args.python,
            protocol_path,
            protocol,
            args.checkpoint_epoch,
            allow_failed_cells=args.allow_failed_cells,
        )
        if args.execute:
            output = Path(command[command.index("--output-json") + 1])
            if output.exists():
                raise FileExistsError(f"Summary already exists: {output}")
            _run(command)
        commands.append(command)
    if not args.execute:
        _print_commands(commands)


if __name__ == "__main__":
    main()
