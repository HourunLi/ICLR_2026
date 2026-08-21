#!/usr/bin/env python3
"""Extract frozen JP token features and run the 4-fold x 3-seed H probe."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import (  # noqa: E402
    CLIRTrajectoryDataset,
    clir_collate,
    move_batch_to_device,
    read_jsonl,
)
from src.clir_frozen_h_probe import (  # noqa: E402
    build_probe_scored_row,
    fit_linear_probe,
    localization_metric_summary,
    score_linear_probe,
    validate_probe_protocol,
)
from src.clir_localization_evaluation import evaluate_localization_rows  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    git_state,
)
from src.consistency_localized_reward import (  # noqa: E402
    RewardConfig,
    build_reward_model,
)


DEFAULT_PROTOCOL = ROOT / "configs/jp_h_frozen_probe_v1/training_protocol_v1.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify_artifact(spec: Mapping[str, Any], *, name: str) -> Path:
    path = resolve(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {name}: {path}")
    observed = file_sha256(path)
    if observed != spec["sha256"]:
        raise ValueError(
            f"Frozen {name} hash drifted: expected {spec['sha256']}, got {observed}"
        )
    return path


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def canonical_rows(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("jp_scored_train48", "jp_scored_dev16"):
        path = verify_artifact(protocol["inputs"][name], name=name)
        rows.extend(read_jsonl(path))
    by_id = {str(row["id"]): row for row in rows}
    if len(rows) != 64 or len(by_id) != 64:
        raise ValueError("Canonical JP mechanism scoring must contain 64 unique rows")
    return by_id


def load_frozen_base(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "clir-full-checkpoint-v2":
        raise ValueError("Frozen JP checkpoint schema drifted")
    if int(checkpoint.get("completed_epoch", -1)) != 5:
        raise ValueError("Frozen JP checkpoint is not epoch 5")
    if checkpoint.get("code", {}).get("dirty") is not False:
        raise ValueError("Frozen JP checkpoint was trained from a dirty worktree")
    config = RewardConfig(**checkpoint["config"])
    expected_weights = {
        "final_weight": 1.0,
        "hallucination_weight": 0.0,
        "prior_weight": 1.0,
        "key_prior_weight": 1.0,
        "complete_prior_weight": 1.0,
        "prior_distill_weight": 0.25,
        "gate_prior_weight": 10.0,
        "reconstruction_weight": 0.0,
        "progress_score_weight": 0.0,
    }
    for name, expected in expected_weights.items():
        if getattr(config, name) != expected:
            raise ValueError(f"Frozen JP config drifted: {name}")
    model = build_reward_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.requires_grad_(False)
    model.eval().to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Frozen JP model unexpectedly has trainable parameters")
    return model, checkpoint


@torch.no_grad()
def extract_frozen_features(
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    output_root: Path,
    device: torch.device,
    code: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    checkpoint_spec = protocol["inputs"]["jp_checkpoint"]
    checkpoint_path = verify_artifact(checkpoint_spec, name="JP epoch-5 checkpoint")
    base_sha_before = file_sha256(checkpoint_path)
    canonical = canonical_rows(protocol)
    model, checkpoint = load_frozen_base(checkpoint_path, device)
    extraction_specs = [
        protocol["cross_validation"]["folds"]["0"]["train"],
        protocol["cross_validation"]["folds"]["0"]["dev"],
    ]
    records: dict[str, Tensor] = {}
    score_mismatches: list[str] = []
    token_value_mismatches: list[str] = []
    seen_order: list[str] = []
    max_score_abs_difference = 0.0
    max_token_value_abs_difference = 0.0

    for manifest_index, spec in enumerate(extraction_specs):
        manifest_path = verify_artifact(
            spec, name=f"feature extraction manifest {manifest_index}"
        )
        dataset = CLIRTrajectoryDataset(
            manifest_path,
            check_finite=False,
            load_condition=True,
            hidden_state_source="precomputed",
        )
        loader = DataLoader(
            dataset,
            batch_size=int(protocol["feature_extraction"]["batch_size"]),
            shuffle=False,
            collate_fn=clir_collate,
            num_workers=0,
        )
        for batch in loader:
            ids = [str(identifier) for identifier in batch["ids"]]
            batch = move_batch_to_device(batch, device)
            parameter_dtype = next(model.parameters()).dtype
            for name in ("hidden_states", "condition_states", "condition_embedding"):
                if name in batch:
                    batch[name] = batch[name].to(dtype=parameter_dtype)
            outputs = model(
                batch["hidden_states"],
                mask=batch["mask"],
                condition_states=batch.get("condition_states"),
                condition_mask=batch.get("condition_mask"),
                condition_embedding=batch.get("condition_embedding"),
                condition_embedding_mask=batch.get("condition_embedding_mask"),
            )
            for index, identifier in enumerate(ids):
                if identifier in records:
                    raise ValueError(f"Duplicate frozen feature row: {identifier}")
                canonical_row = canonical.get(identifier)
                if canonical_row is None:
                    raise ValueError(f"Missing canonical JP score for {identifier}")
                length = int(batch["mask"][index].sum().detach().cpu())
                features = outputs["token_features"][index, :length].detach().cpu()
                if features.shape != (length, 768) or features.dtype != torch.float32:
                    raise ValueError("Frozen JP token feature shape/dtype drifted")
                if not torch.isfinite(features).all():
                    raise ValueError("Frozen JP token features contain non-finite values")
                observed_score = float(outputs["scores"][index].detach().cpu())
                expected_score = float(canonical_row["reward_score"])
                score_difference = abs(observed_score - expected_score)
                max_score_abs_difference = max(
                    max_score_abs_difference, score_difference
                )
                if observed_score != expected_score:
                    score_mismatches.append(identifier)
                observed_values = outputs["token_values"][index, :length].detach().cpu()
                expected_values = torch.tensor(
                    canonical_row["clir_token_values"], dtype=observed_values.dtype
                )
                if observed_values.shape != expected_values.shape:
                    raise ValueError("Canonical JP token-value length drifted")
                value_difference = float(
                    (observed_values - expected_values).abs().max().item()
                )
                max_token_value_abs_difference = max(
                    max_token_value_abs_difference, value_difference
                )
                if not torch.equal(observed_values, expected_values):
                    token_value_mismatches.append(identifier)
                records[identifier] = features.contiguous()
                seen_order.append(identifier)
            del outputs, batch

    if len(records) != 64 or set(records) != set(canonical):
        raise ValueError("Frozen feature cache does not cover the canonical 64 rows")
    if score_mismatches or token_value_mismatches:
        raise RuntimeError(
            "Current frozen JP forward is not bit-identical to canonical JP scoring"
        )
    base_sha_after_extraction = file_sha256(checkpoint_path)
    if base_sha_after_extraction != base_sha_before:
        raise RuntimeError("Read-only JP checkpoint changed during feature extraction")

    cache_path = output_root / "cache" / "frozen_jp_token_features.pt"
    cache = {
        "schema_version": "clir-jp-h-frozen-token-feature-cache-v1",
        "protocol_sha256": protocol_sha256,
        "base_checkpoint_sha256": base_sha_before,
        "base_checkpoint_training_commit": checkpoint["code"]["commit"],
        "model_dim": 768,
        "row_order": seen_order,
        "records": records,
    }
    atomic_torch_save(cache, cache_path)
    cache_sha256 = file_sha256(cache_path)
    report_path = output_root / "cache" / "cache_report.json"
    report = {
        "schema_version": "clir-jp-h-frozen-token-feature-cache-report-v1",
        "status": "passed",
        "protocol_sha256": protocol_sha256,
        "base_checkpoint_sha256_before": base_sha_before,
        "base_checkpoint_sha256_after_extraction": base_sha_after_extraction,
        "base_checkpoint_unchanged": base_sha_before == base_sha_after_extraction,
        "base_model_trainable_parameters_during_extraction": 0,
        "probe_parameters_in_base_model_optimizer": 0,
        "rows": len(records),
        "model_dim": 768,
        "reward_score_bit_identical_rows": len(records) - len(score_mismatches),
        "reward_score_mismatch_rows": score_mismatches,
        "reward_score_max_absolute_difference": max_score_abs_difference,
        "token_value_bit_identical_rows": len(records)
        - len(token_value_mismatches),
        "token_value_mismatch_rows": token_value_mismatches,
        "token_value_max_absolute_difference": max_token_value_abs_difference,
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "code": code,
    }
    atomic_write_json(report_path, report)
    del model, checkpoint, records
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache_path, report_path, canonical


def supervised_token_matrix(
    rows: Sequence[Mapping[str, Any]], records: Mapping[str, Tensor]
) -> tuple[Tensor, Tensor]:
    features: list[Tensor] = []
    targets: list[Tensor] = []
    for row in rows:
        identifier = str(row["id"])
        token_features = records.get(identifier)
        if token_features is None:
            raise ValueError(f"Frozen feature cache misses {identifier}")
        target = torch.tensor(row["token_hallucination_target"], dtype=torch.float32)
        mask = torch.tensor(row["token_hallucination_mask"], dtype=torch.bool)
        if token_features.shape[0] != target.numel() or target.shape != mask.shape:
            raise ValueError("Probe token features/targets/mask are not exactly aligned")
        if not mask.any() or torch.any(target.bool() & ~mask):
            raise ValueError("Invalid sparse hallucination supervision mask")
        features.append(token_features[mask])
        targets.append(target[mask])
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def run_cell(
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    fold: int,
    seed: int,
    output_root: Path,
    records: Mapping[str, Tensor],
    cache_sha256: str,
    canonical: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    code: Mapping[str, Any],
) -> dict[str, Any]:
    fold_spec = protocol["cross_validation"]["folds"][str(fold)]
    train_path = verify_artifact(fold_spec["train"], name=f"fold {fold} train")
    dev_path = verify_artifact(fold_spec["dev"], name=f"fold {fold} dev")
    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    train_features, train_targets = supervised_token_matrix(train_rows, records)
    if train_features.shape[0] != int(fold_spec["train"]["supervised_tokens"]):
        raise ValueError("Frozen probe train supervised-token count drifted")
    if int(train_targets.sum()) != int(fold_spec["train"]["positive_tokens"]):
        raise ValueError("Frozen probe train positive-token count drifted")

    training = protocol["training"]
    head, history = fit_linear_probe(
        train_features,
        train_targets,
        seed=seed,
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        device=device,
    )
    cell_root = output_root / f"fold_{fold}" / f"seed_{seed}"
    if cell_root.exists():
        raise FileExistsError(f"Refusing to overwrite frozen-probe cell {cell_root}")
    cell_root.mkdir(parents=True)
    checkpoint_path = cell_root / "probe.pt"
    checkpoint = {
        "schema_version": "clir-jp-h-frozen-linear-probe-checkpoint-v1",
        "protocol_sha256": protocol_sha256,
        "base_checkpoint_sha256": protocol["inputs"]["jp_checkpoint"]["sha256"],
        "frozen_feature_cache_sha256": cache_sha256,
        "fold": fold,
        "seed": seed,
        "model_dim": 768,
        "probe_parameter_count": sum(
            parameter.numel() for parameter in head.parameters()
        ),
        "base_state_dict_included": False,
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in head.state_dict().items()
        },
        "training": dict(training),
        "code": code,
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    atomic_write_jsonl(cell_root / "metrics.jsonl", history)

    scored_rows: list[dict[str, Any]] = []
    for row in dev_rows:
        identifier = str(row["id"])
        logits, probabilities = score_linear_probe(head, records[identifier])
        scored_rows.append(
            build_probe_scored_row(
                row,
                canonical[identifier],
                logits,
                probabilities,
                fold=fold,
                seed=seed,
                probe_checkpoint_sha256=checkpoint_sha256,
            )
        )
    scored_path = cell_root / "dev_scored.jsonl"
    atomic_write_jsonl(scored_path, scored_rows)
    scored_sha256 = file_sha256(scored_path)
    evaluation_metrics = evaluate_localization_rows(
        scored_rows, threshold=0.5, negative_tail_margin=0.5
    )
    evaluation_path = cell_root / "evaluation.json"
    atomic_write_json(
        evaluation_path,
        {
            "schema_version": "clir-jp-h-frozen-probe-evaluation-v1",
            "evidence_tier": "small-scale real",
            "fold": fold,
            "seed": seed,
            "input_sha256": scored_sha256,
            "metrics": evaluation_metrics,
            "summary": localization_metric_summary(evaluation_metrics),
            "pilot_test_accessed": False,
            "final_test_accessed": False,
            "formal_mechanism_claim_allowed": False,
            "code": code,
        },
    )
    result = {
        "schema_version": "clir-jp-h-frozen-probe-cell-result-v1",
        "status": "completed",
        "evidence_tier": "small-scale real",
        "fold": fold,
        "seed": seed,
        "protocol_sha256": protocol_sha256,
        "train_manifest_sha256": fold_spec["train"]["sha256"],
        "dev_manifest_sha256": fold_spec["dev"]["sha256"],
        "base_checkpoint_sha256": protocol["inputs"]["jp_checkpoint"]["sha256"],
        "frozen_feature_cache_sha256": cache_sha256,
        "probe_checkpoint_sha256": checkpoint_sha256,
        "probe_parameter_count": 769,
        "base_parameters_in_optimizer": 0,
        "train_rows": len(train_rows),
        "train_supervised_tokens": int(train_targets.numel()),
        "train_positive_tokens": int(train_targets.sum().item()),
        "dev_rows": len(dev_rows),
        "metrics_rows": len(history),
        "initial_train_bce": history[0]["pre_update_train_bce"],
        "final_train_bce": history[-1]["post_update_train_bce"],
        "dev_scored_sha256": scored_sha256,
        "evaluation_sha256": file_sha256(evaluation_path),
        "dev_metrics": localization_metric_summary(evaluation_metrics),
        "code": code,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(cell_root / "cell_result.json", result)
    del head, train_features, train_targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_probe_protocol(protocol)
    protocol_sha256 = file_sha256(protocol_path)
    output_root = resolve(protocol["execution"]["output_root"])
    preflight = {
        "schema_version": "clir-jp-h-frozen-probe-preflight-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "output_root": str(output_root),
        "device": args.device,
        "folds": list(protocol["cross_validation"]["folds"]),
        "seeds": protocol["training"]["seeds"],
        "cells": len(protocol["cross_validation"]["folds"])
        * len(protocol["training"]["seeds"]),
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Frozen-probe execution requires a clean committed worktree")
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse frozen-probe output root {output_root}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen-probe protocol requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)

    for name, spec in protocol["inputs"].items():
        verify_artifact(spec, name=name)
    jp_cell_result = json.loads(
        resolve(protocol["inputs"]["jp_cell_result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if jp_cell_result.get("cell") != "jp_original_prior":
        raise ValueError("Frozen base result is not the original JP cell")
    if (
        jp_cell_result.get("checkpoint_sha256")
        != protocol["inputs"]["jp_checkpoint"]["sha256"]
    ):
        raise ValueError("Frozen JP cell result and checkpoint hash disagree")
    resolved_weights = jp_cell_result.get("resolved_loss_weights", {})
    expected_original_weights = {
        "final": 1.0,
        "hallucination": 0.0,
        "prior": 1.0,
        "key_prior": 1.0,
        "complete_prior": 1.0,
        "prior_distill": 0.25,
        "gate_prior": 10.0,
        "reconstruction": 0.0,
    }
    for name, expected in expected_original_weights.items():
        if resolved_weights.get(name) != expected:
            raise ValueError(f"Frozen JP cell weight drifted: {name}")
    fold_audit = json.loads(
        resolve(protocol["inputs"]["fold_audit"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if fold_audit.get("status") != "passed":
        raise ValueError("Frozen fold audit did not pass")
    integrity = json.loads(
        resolve(protocol["inputs"]["feature_integrity_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    failure_count = integrity.get(
        "failure_count", integrity.get("summary", {}).get("failure_count")
    )
    if int(failure_count) != 0:
        raise ValueError("Frozen feature-integrity report contains failures")
    for fold, fold_spec in protocol["cross_validation"]["folds"].items():
        for split in ("train", "dev"):
            verify_artifact(fold_spec[split], name=f"fold {fold} {split}")

    cache_path, cache_report_path, canonical = extract_frozen_features(
        protocol,
        protocol_sha256=protocol_sha256,
        output_root=output_root,
        device=device,
        code=code,
    )
    cache_sha256 = file_sha256(cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache["records"]
    cell_results: list[dict[str, Any]] = []
    for fold in map(int, protocol["cross_validation"]["folds"]):
        for seed in map(int, protocol["training"]["seeds"]):
            cell_results.append(
                run_cell(
                    protocol,
                    protocol_sha256=protocol_sha256,
                    fold=fold,
                    seed=seed,
                    output_root=output_root,
                    records=records,
                    cache_sha256=cache_sha256,
                    canonical=canonical,
                    device=device,
                    code=code,
                )
            )

    checkpoint_path = resolve(protocol["inputs"]["jp_checkpoint"]["path"])
    base_sha_after_matrix = file_sha256(checkpoint_path)
    if base_sha_after_matrix != protocol["inputs"]["jp_checkpoint"]["sha256"]:
        raise RuntimeError("Read-only JP checkpoint changed during probe training")
    matrix_report = {
        "schema_version": "clir-jp-h-frozen-probe-matrix-run-v1",
        "status": "completed",
        "protocol_sha256": protocol_sha256,
        "code": code,
        "device": str(device),
        "cache_report_path": str(cache_report_path),
        "cache_report_sha256": file_sha256(cache_report_path),
        "cache_sha256": cache_sha256,
        "cells": len(cell_results),
        "completed_cells": sum(
            result["status"] == "completed" for result in cell_results
        ),
        "base_checkpoint_sha256_after_matrix": base_sha_after_matrix,
        "base_checkpoint_unchanged": True,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
    }
    atomic_write_json(output_root / "matrix_run.json", matrix_report)
    print(json.dumps(matrix_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
