#!/usr/bin/env python3
"""Audit and summarize the frozen JP-to-H linear-probe matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import read_jsonl  # noqa: E402
from src.clir_frozen_h_probe import (  # noqa: E402
    localization_metric_summary,
    mean_sample_std,
    validate_probe_protocol,
)
from src.clir_localization_evaluation import evaluate_localization_rows  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/jp_h_frozen_probe_v1/training_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "configs/jp_h_frozen_probe_v1/training_result_v1.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify(path: Path, expected_sha256: str, *, name: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen-probe {name} hash drifted: expected {expected_sha256}, got {observed}"
        )


def canonical_rows(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("jp_scored_train48", "jp_scored_dev16"):
        spec = protocol["inputs"][name]
        path = resolve(spec["path"])
        verify(path, spec["sha256"], name=name)
        rows.extend(read_jsonl(path))
    by_id = {str(row["id"]): row for row in rows}
    if len(rows) != 64 or len(by_id) != 64:
        raise ValueError("Canonical JP scoring must contain 64 unique rows")
    return by_id


def same_base_score_fields(
    row: Mapping[str, Any], canonical: Mapping[str, Any]
) -> bool:
    return (
        float(row["reward_score"]) == float(canonical["reward_score"])
        and float(row["clir_score"]) == float(canonical["clir_score"])
        and row["clir_token_values"] == canonical["clir_token_values"]
    )


def pool_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = evaluate_localization_rows(
        rows, threshold=0.5, negative_tail_margin=0.5
    )
    return localization_metric_summary(metrics)


def assert_expected_position_baseline(
    summary: Mapping[str, Any], expected: Mapping[str, Any], *, name: str
) -> None:
    observed = {
        "rows": int(summary["rows"]),
        "span_supervised_tokens": int(summary["span_supervised_tokens"]),
        "span_positive_tokens": int(summary["span_positive_tokens"]),
        "span_strongest_position_average_precision": float(
            summary["span_strongest_position_average_precision"]
        ),
        "claims": int(summary["claims"]),
        "positive_claims": int(summary["positive_claims"]),
        "claim_strongest_position_average_precision": float(
            summary["claim_strongest_position_average_precision"]
        ),
    }
    if observed != expected:
        raise ValueError(
            f"{name} same-row position baseline drifted: {observed} != {expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Frozen-probe summary requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen-probe result {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_probe_protocol(protocol)
    protocol_sha256 = file_sha256(protocol_path)
    for name, spec in protocol["inputs"].items():
        path = resolve(spec["path"])
        verify(path, spec["sha256"], name=name)

    output_root = resolve(protocol["execution"]["output_root"])
    matrix_path = output_root / "matrix_run.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("status") != "completed" or int(matrix.get("completed_cells", -1)) != 12:
        raise ValueError("Frozen-probe matrix is incomplete")
    if matrix.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Frozen-probe matrix protocol hash drifted")
    if matrix.get("code", {}).get("dirty") is not False:
        raise ValueError("Frozen-probe matrix used a dirty worktree")
    if matrix.get("base_checkpoint_unchanged") is not True:
        raise ValueError("Frozen-probe matrix changed the base checkpoint")

    cache_report_path = Path(matrix["cache_report_path"])
    verify(
        cache_report_path,
        matrix["cache_report_sha256"],
        name="cache report",
    )
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    base_invariance = {
        "base_checkpoint_sha256_before": cache_report[
            "base_checkpoint_sha256_before"
        ],
        "base_checkpoint_sha256_after_extraction": cache_report[
            "base_checkpoint_sha256_after_extraction"
        ],
        "base_checkpoint_sha256_after_matrix": matrix[
            "base_checkpoint_sha256_after_matrix"
        ],
        "base_checkpoint_unchanged": (
            cache_report["base_checkpoint_unchanged"] is True
            and matrix["base_checkpoint_unchanged"] is True
            and cache_report["base_checkpoint_sha256_before"]
            == matrix["base_checkpoint_sha256_after_matrix"]
            == protocol["inputs"]["jp_checkpoint"]["sha256"]
        ),
        "reward_score_bit_identical_rows": int(
            cache_report["reward_score_bit_identical_rows"]
        ),
        "reward_score_max_absolute_difference": float(
            cache_report["reward_score_max_absolute_difference"]
        ),
        "token_value_bit_identical_rows": int(
            cache_report["token_value_bit_identical_rows"]
        ),
        "token_value_max_absolute_difference": float(
            cache_report["token_value_max_absolute_difference"]
        ),
        "base_parameters_in_optimizer": 0,
        "probe_checkpoint_contains_base_state_dict": False,
    }
    base_invariance["passed"] = (
        base_invariance["base_checkpoint_unchanged"]
        and base_invariance["reward_score_bit_identical_rows"] == 64
        and base_invariance["reward_score_max_absolute_difference"] == 0.0
        and base_invariance["token_value_bit_identical_rows"] == 64
        and base_invariance["token_value_max_absolute_difference"] == 0.0
    )
    if not base_invariance["passed"]:
        raise ValueError("Frozen JP base invariance gate failed")

    canonical = canonical_rows(protocol)
    folds = [int(value) for value in protocol["cross_validation"]["folds"]]
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    confirmatory_folds = [
        int(value)
        for value in protocol["cross_validation"]["primary_confirmatory_folds"]
    ]
    rows_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    cell_reports: dict[str, Any] = {}
    probe_hashes: set[str] = set()
    execution_commits: set[str] = set()
    base_score_field_mismatches: list[str] = []
    all_cells_converged = True

    for fold in folds:
        fold_spec = protocol["cross_validation"]["folds"][str(fold)]
        expected_ids = [
            str(row["id"]) for row in read_jsonl(resolve(fold_spec["dev"]["path"]))
        ]
        for seed in seeds:
            cell_root = output_root / f"fold_{fold}" / f"seed_{seed}"
            result_path = cell_root / "cell_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                raise ValueError(f"Frozen-probe fold {fold} seed {seed} is incomplete")
            if result.get("protocol_sha256") != protocol_sha256:
                raise ValueError("Frozen-probe cell protocol hash drifted")
            if int(result.get("probe_parameter_count", -1)) != 769:
                raise ValueError("Frozen-probe parameter count drifted")
            if int(result.get("base_parameters_in_optimizer", -1)) != 0:
                raise ValueError("Frozen-probe cell optimized base parameters")
            if result["train_manifest_sha256"] != fold_spec["train"]["sha256"]:
                raise ValueError("Frozen-probe train manifest hash drifted")
            if result["dev_manifest_sha256"] != fold_spec["dev"]["sha256"]:
                raise ValueError("Frozen-probe dev manifest hash drifted")
            if result["base_checkpoint_sha256"] != protocol["inputs"]["jp_checkpoint"]["sha256"]:
                raise ValueError("Frozen-probe base checkpoint hash drifted")
            if result.get("code", {}).get("dirty") is not False:
                raise ValueError("Frozen-probe cell used a dirty worktree")
            execution_commits.add(str(result["code"]["commit"]))
            checkpoint_path = cell_root / "probe.pt"
            verify(
                checkpoint_path,
                result["probe_checkpoint_sha256"],
                name=f"fold {fold} seed {seed} probe checkpoint",
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("base_state_dict_included") is not False:
                raise ValueError("Probe checkpoint unexpectedly embeds the JP base")
            if set(checkpoint["state_dict"]) != {"weight", "bias"}:
                raise ValueError("Probe checkpoint contains more than the linear head")
            if tuple(checkpoint["state_dict"]["weight"].shape) != (1, 768):
                raise ValueError("Probe checkpoint weight shape drifted")
            probe_hash = result["probe_checkpoint_sha256"]
            if probe_hash in probe_hashes:
                raise ValueError("Frozen-probe matrix unexpectedly reused a head checkpoint")
            probe_hashes.add(probe_hash)

            scored_path = cell_root / "dev_scored.jsonl"
            verify(scored_path, result["dev_scored_sha256"], name="dev scoring")
            rows = read_jsonl(scored_path)
            if [str(row["id"]) for row in rows] != expected_ids:
                raise ValueError("Frozen-probe scored-dev row order drifted")
            for row in rows:
                identifier = str(row["id"])
                if not same_base_score_fields(row, canonical[identifier]):
                    base_score_field_mismatches.append(
                        f"fold_{fold}/seed_{seed}/{identifier}"
                    )
            rows_by_cell[(fold, seed)] = rows
            converged = float(result["final_train_bce"]) < float(
                result["initial_train_bce"]
            )
            all_cells_converged = all_cells_converged and converged
            cell_reports[f"fold_{fold}/seed_{seed}"] = {
                "fold": fold,
                "seed": seed,
                "selection_exposed": bool(fold_spec["selection_exposed"]),
                "train_rows": int(result["train_rows"]),
                "train_supervised_tokens": int(result["train_supervised_tokens"]),
                "train_positive_tokens": int(result["train_positive_tokens"]),
                "initial_train_bce": float(result["initial_train_bce"]),
                "final_train_bce": float(result["final_train_bce"]),
                "train_bce_decreased": converged,
                "probe_checkpoint_sha256": probe_hash,
                "dev_scored_sha256": result["dev_scored_sha256"],
                "cell_result_sha256": file_sha256(result_path),
                "dev_metrics": result["dev_metrics"],
            }

    if len(execution_commits) != 1:
        raise ValueError("Frozen-probe cells did not share one clean code commit")
    if base_score_field_mismatches:
        raise ValueError("Scored probe rows changed canonical JP reward fields")

    confirmatory: dict[str, Any] = {}
    all_oof: dict[str, Any] = {}
    passing_seeds: list[int] = []
    span_advantages: list[float] = []
    claim_advantages: list[float] = []
    span_aps: list[float] = []
    claim_aps: list[float] = []
    for seed in seeds:
        confirmatory_rows = [
            row
            for fold in confirmatory_folds
            for row in rows_by_cell[(fold, seed)]
        ]
        all_rows = [row for fold in folds for row in rows_by_cell[(fold, seed)]]
        if len(confirmatory_rows) != 48 or len({row["id"] for row in confirmatory_rows}) != 48:
            raise ValueError("Frozen-probe confirmatory pool is not 48 unique rows")
        if len(all_rows) != 64 or len({row["id"] for row in all_rows}) != 64:
            raise ValueError("Frozen-probe all-fold pool is not 64 unique rows")
        confirmatory_summary = pool_summary(confirmatory_rows)
        all_summary = pool_summary(all_rows)
        assert_expected_position_baseline(
            confirmatory_summary,
            protocol["position_baselines"]["confirmatory_48"],
            name="confirmatory-48",
        )
        assert_expected_position_baseline(
            all_summary,
            protocol["position_baselines"]["all_oof_64"],
            name="all-OOF-64",
        )
        seed_gate = {
            "span_passed": float(
                confirmatory_summary["span_advantage_over_position"]
            )
            > 0.0,
            "claim_passed": float(
                confirmatory_summary["claim_advantage_over_position"]
            )
            > 0.0,
        }
        seed_gate["both_localization_metrics_passed"] = all(seed_gate.values())
        if seed_gate["both_localization_metrics_passed"]:
            passing_seeds.append(seed)
        confirmatory[str(seed)] = {
            **confirmatory_summary,
            "gate": seed_gate,
        }
        all_oof[str(seed)] = all_summary
        span_advantages.append(
            float(confirmatory_summary["span_advantage_over_position"])
        )
        claim_advantages.append(
            float(confirmatory_summary["claim_advantage_over_position"])
        )
        span_aps.append(float(confirmatory_summary["span_average_precision"]))
        claim_aps.append(float(confirmatory_summary["claim_mean_average_precision"]))

    mean_span_advantage = sum(span_advantages) / len(span_advantages)
    mean_claim_advantage = sum(claim_advantages) / len(claim_advantages)
    decision_rules = protocol["decision_rules"]
    decision_passed = (
        len(passing_seeds) >= int(decision_rules["minimum_passing_seeds"])
        and mean_span_advantage > 0.0
        and mean_claim_advantage > 0.0
        and base_invariance["passed"]
        and all_cells_converged
    )
    decision = {
        "passed": decision_passed,
        "passing_seeds": passing_seeds,
        "passing_seed_count": len(passing_seeds),
        "required_passing_seed_count": int(
            decision_rules["minimum_passing_seeds"]
        ),
        "mean_span_advantage_over_same_rows_position": mean_span_advantage,
        "mean_claim_advantage_over_same_rows_position": mean_claim_advantage,
        "mean_span_advantage_passed": mean_span_advantage > 0.0,
        "mean_claim_advantage_passed": mean_claim_advantage > 0.0,
        "base_invariance_passed": base_invariance["passed"],
        "all_cells_train_bce_decreased": all_cells_converged,
        "next_authorized_step": (
            "discuss_explicit_frozen_h_probability_score_coupling"
            if decision_passed
            else "discuss_detached_h_specific_adapter_or_more_labels"
        ),
        "automatic_next_experiment_authorized": False,
    }

    report = {
        "schema_version": "clir-jp-h-frozen-probe-result-v1",
        "status": (
            "completed_frozen_linear_probe_supported"
            if decision_passed
            else "completed_frozen_linear_probe_not_supported"
        ),
        "evidence_tier": "small-scale real",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": protocol_sha256,
        },
        "execution_gate": {
            "matrix_completed_cells": 12,
            "all_cells_share_clean_commit": True,
            "execution_commit": next(iter(execution_commits)),
            "unique_probe_checkpoints": len(probe_hashes),
            "all_cells_train_bce_decreased": all_cells_converged,
            "all_dev_ids_and_hashes_verified": True,
            "base_reward_fields_bit_identical_in_all_192_scored_row_views": True,
            "passed": all_cells_converged,
        },
        "base_invariance": base_invariance,
        "data": {
            "folds": 4,
            "seeds": seeds,
            "cells": 12,
            "primary_confirmatory_folds": confirmatory_folds,
            "primary_confirmatory_rows": 48,
            "all_out_of_fold_rows": 64,
            "fold_0_selection_exposed": True,
            "pilot_test_accessed": False,
            "final_test_accessed": False,
        },
        "confirmatory_48_rows_by_seed": confirmatory,
        "all_64_out_of_fold_diagnostic_by_seed": all_oof,
        "multi_seed_summary": {
            "span_average_precision": mean_sample_std(span_aps),
            "claim_mean_average_precision": mean_sample_std(claim_aps),
            "span_advantage_over_same_rows_position": mean_sample_std(
                span_advantages
            ),
            "claim_advantage_over_same_rows_position": mean_sample_std(
                claim_advantages
            ),
        },
        "historical_fold0_thresholds_report_only": protocol[
            "position_baselines"
        ]["historical_fold0_report_only"],
        "cell_reports": cell_reports,
        "decision": decision,
        "limitations": [
            "The 64 trajectories retain the existing Silver pipeline-pilot hallucination labels.",
            "Folds 1-3 are out-of-fold views of the former 48-row training pool, not a newly annotated external validation set.",
            "Fold 0 was exposed to earlier localization model selection and is excluded from the primary decision.",
            "This tests linear decodability of frozen JP features only; the probe does not enter the reward score and establishes no Best-of-N gain.",
            "No pilot-test or final-test data was accessed.",
        ],
        "code": {
            "execution_commit": next(iter(execution_commits)),
            "summary_commit": state,
        },
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
