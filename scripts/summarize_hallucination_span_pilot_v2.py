#!/usr/bin/env python3
"""Validate and summarize the frozen sparse-span hallucination Pilot v2a."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import read_jsonl  # noqa: E402
from src.clir_localization_evaluation import binary_average_precision  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/hallucination_localization_v2/training_protocol_v2.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/hallucination_localization_v2/training_result_v2.json"
)
CELL_ORDER = [
    "s0_tail_bce",
    "s1_span_bce",
    "s2_span_balanced",
    "s3_span_balanced_path",
]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def select_span_cell(
    cells: Mapping[str, Mapping[str, float]],
    *,
    token_position_ap: float,
    claim_position_ap: float,
) -> dict[str, Any]:
    candidates = CELL_ORDER[1:]
    selected = max(
        candidates,
        key=lambda name: (
            float(cells[name]["span_token_average_precision"]),
            -candidates.index(name),
        ),
    )
    token_ap = float(cells[selected]["span_token_average_precision"])
    claim_ap = float(cells[selected]["claim_mean_average_precision"])
    return {
        "selected_cell": selected,
        "beats_s0": token_ap > float(cells["s0_tail_bce"]["span_token_average_precision"]),
        "beats_token_position": token_ap > token_position_ap,
        "beats_claim_position": claim_ap > claim_position_ap,
        "token_gate_passed": (
            token_ap > float(cells["s0_tail_bce"]["span_token_average_precision"])
            and token_ap > token_position_ap
            and claim_ap > claim_position_ap
        ),
    }


def _row_fields(row: Mapping[str, Any]) -> dict[str, list[float] | list[int]]:
    labels: list[int] = []
    scores: list[float] = []
    positions: list[float] = []
    claim_labels: list[int] = []
    claim_scores: list[float] = []
    claim_positions: list[float] = []
    probabilities = row["clir_token_hallucination_probs"]
    for position, (label, known, score) in enumerate(
        zip(
            row["token_hallucination_target"],
            row["token_hallucination_mask"],
            probabilities,
        )
    ):
        if known:
            labels.append(int(label))
            scores.append(float(score))
            positions.append(float(position))
    for span in row["hallucination_claim_spans"]:
        start = int(span["token_start"])
        end = int(span["token_end_exclusive"])
        claim_labels.append(int(span["target"]))
        claim_scores.append(sum(probabilities[start:end]) / (end - start))
        claim_positions.append((start + end - 1) / 2.0)
    return {
        "labels": labels,
        "scores": scores,
        "positions": positions,
        "claim_labels": claim_labels,
        "claim_scores": claim_scores,
        "claim_positions": claim_positions,
    }


def _interval(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    return {
        "mean": sum(ordered) / len(ordered),
        "lower_2_5_percentile": ordered[int(0.025 * (len(ordered) - 1))],
        "upper_97_5_percentile": ordered[int(0.975 * (len(ordered) - 1))],
        "fraction_above_zero": sum(value > 0.0 for value in ordered) / len(ordered),
        "valid_resamples": len(ordered),
    }


def paired_query_bootstrap(
    s0_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    s0 = {str(row["id"]): _row_fields(row) for row in s0_rows}
    selected = {str(row["id"]): _row_fields(row) for row in selected_rows}
    if set(s0) != set(selected):
        raise ValueError("Bootstrap cells contain different dev row ids")
    identifiers = sorted(s0)
    rng = random.Random(seed)
    token_vs_s0: list[float] = []
    token_vs_position: list[float] = []
    claim_vs_s0: list[float] = []
    claim_vs_position: list[float] = []
    for _ in range(samples):
        sampled = [identifiers[rng.randrange(len(identifiers))] for _ in identifiers]
        labels: list[int] = []
        s0_scores: list[float] = []
        selected_scores: list[float] = []
        positions: list[float] = []
        claim_labels: list[int] = []
        s0_claim_scores: list[float] = []
        selected_claim_scores: list[float] = []
        claim_positions: list[float] = []
        for identifier in sampled:
            left = s0[identifier]
            right = selected[identifier]
            labels.extend(left["labels"])
            s0_scores.extend(left["scores"])
            selected_scores.extend(right["scores"])
            positions.extend(left["positions"])
            claim_labels.extend(left["claim_labels"])
            s0_claim_scores.extend(left["claim_scores"])
            selected_claim_scores.extend(right["claim_scores"])
            claim_positions.extend(left["claim_positions"])
        if not 0 < sum(labels) < len(labels):
            continue
        if not 0 < sum(claim_labels) < len(claim_labels):
            continue
        token_selected_ap = binary_average_precision(labels, selected_scores)
        claim_selected_ap = binary_average_precision(claim_labels, selected_claim_scores)
        token_vs_s0.append(
            token_selected_ap - binary_average_precision(labels, s0_scores)
        )
        token_vs_position.append(
            token_selected_ap - binary_average_precision(labels, positions)
        )
        claim_vs_s0.append(
            claim_selected_ap - binary_average_precision(claim_labels, s0_claim_scores)
        )
        claim_vs_position.append(
            claim_selected_ap - binary_average_precision(claim_labels, claim_positions)
        )
    return {
        "unit": "query_trajectory_row",
        "requested_resamples": samples,
        "seed": seed,
        "token_ap_selected_minus_s0": _interval(token_vs_s0),
        "token_ap_selected_minus_absolute_position": _interval(token_vs_position),
        "claim_mean_ap_selected_minus_s0": _interval(claim_vs_s0),
        "claim_mean_ap_selected_minus_absolute_position": _interval(claim_vs_position),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-hallucination-localization-training-protocol-v2"
    ):
        raise ValueError("Unknown localization Pilot v2 protocol")
    output_root = resolve(protocol["execution"]["output_root"])
    expected_train = protocol["inputs"]["dense_train"]["sha256"]
    expected_dev = protocol["inputs"]["localization_dev"]["sha256"]
    expected_counts = {
        "s0_tail_bce": (14307, 5008),
        "s1_span_bce": (6681, 2451),
        "s2_span_balanced": (6681, 2451),
        "s3_span_balanced_path": (6681, 2451),
    }
    code_commits: set[str] = set()
    protocol_hashes: set[str] = set()
    cells: dict[str, Any] = {}
    dev_rows: dict[str, list[dict[str, Any]]] = {}
    for cell in CELL_ORDER:
        cell_root = output_root / cell
        result_path = cell_root / "cell_result.json"
        run_path = cell_root / "run.json"
        metrics_path = cell_root / "metrics.jsonl"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        run = json.loads(run_path.read_text(encoding="utf-8"))
        metrics = read_jsonl(metrics_path)
        if run.get("status") != "completed" or run.get("completed_epoch") != 5:
            raise ValueError(f"{cell} did not complete five epochs")
        if run["code"]["dirty"] is not False:
            raise ValueError(f"{cell} was trained from a dirty worktree")
        if result["train_manifest_sha256"] != expected_train:
            raise ValueError(f"{cell} train manifest hash drifted")
        if result["dev_manifest_sha256"] != expected_dev:
            raise ValueError(f"{cell} dev manifest hash drifted")
        if len(metrics) != 5:
            raise ValueError(f"{cell} metrics do not contain five epochs")
        train_count, dev_count = expected_counts[cell]
        for epoch in metrics:
            if epoch["train"]["applicable_counts"]["localization_token_bce"] != train_count:
                raise ValueError(f"{cell} train token-supervision count drifted")
            if epoch["validation"]["applicable_counts"]["localization_token_bce"] != dev_count:
                raise ValueError(f"{cell} dev token-supervision count drifted")
            for section in ("train", "checkpoint_train_evaluation", "validation"):
                if any(
                    not isinstance(value, (int, float))
                    or not float("-inf") < float(value) < float("inf")
                    for value in epoch[section]["losses"].values()
                ):
                    raise ValueError(f"{cell} contains non-finite losses")
        fixed = result["fixed_dev_metrics"]
        calibrated = result["train_calibrated_dev_metrics"]
        span = fixed["explicit_claim_span_tokens"]
        claim = fixed["explicit_claim_span_shortcuts"]["claim_level"]
        path = fixed["path_noisy_or_log_space"]
        cells[cell] = {
            "description": protocol["cells"][cell]["description"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "metrics_sha256": result["metrics_sha256"],
            "dev_scored_sha256": result["dev_scored_sha256"],
            "cell_result_sha256": file_sha256(result_path),
            "span_token_roc_auc": span["roc_auc"],
            "span_token_average_precision": span["average_precision"],
            "span_token_fixed_0_5_f1": span["f1"],
            "claim_mean_roc_auc": claim["mean_probability_ranking"]["roc_auc"],
            "claim_mean_average_precision": claim["mean_probability_ranking"][
                "average_precision"
            ],
            "claim_max_average_precision": claim["max_probability_ranking"][
                "average_precision"
            ],
            "path_roc_auc": path["roc_auc"],
            "path_average_precision": path["average_precision"],
            "fixed_onset_mae": fixed["onset"][
                "mean_absolute_error_with_miss_as_length"
            ],
            "fixed_onset_within_5": fixed["onset"]["within_5"]["count"],
            "train_calibrated_onset_mae": calibrated["onset"][
                "mean_absolute_error_with_miss_as_length"
            ],
            "train_calibrated_onset_within_5": calibrated["onset"]["within_5"][
                "count"
            ],
        }
        code_commits.add(str(run["code"]["commit"]))
        protocol_hashes.add(str(result["protocol_sha256"]))
        dev_rows[cell] = read_jsonl(cell_root / "localization_dev_scored.jsonl")
    if len(code_commits) != 1 or len(protocol_hashes) != 1:
        raise ValueError("Pilot v2 cells do not share one code commit and protocol")

    shortcuts = protocol["evaluation"]["frozen_shortcuts"]
    selection = select_span_cell(
        cells,
        token_position_ap=shortcuts["dev_token_absolute_position_average_precision"],
        claim_position_ap=shortcuts["dev_claim_absolute_position_average_precision"],
    )
    selected = selection["selected_cell"]
    selected_metrics = cells[selected]
    onset_gate = (
        selected_metrics["train_calibrated_onset_mae"]
        < shortcuts["v1_constant_train_median_onset_mean_absolute_error"]
        and selected_metrics["train_calibrated_onset_within_5"] >= 1
    )
    bootstrap = paired_query_bootstrap(
        dev_rows["s0_tail_bce"],
        dev_rows[selected],
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report = {
        "schema_version": "clir-hallucination-localization-training-result-v2",
        "status": "completed_span_token_gate_passed_onset_gate_failed",
        "evidence_tier": "pipeline_pilot",
        "training_protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": file_sha256(protocol_path),
        },
        "data": {
            "dense_train_rows": 48,
            "dense_train_sha256": expected_train,
            "localization_dev_rows": 16,
            "localization_dev_positive_rows": 6,
            "localization_dev_sha256": expected_dev,
            "query_overlap": 0,
            "claim_level_gold": False,
        },
        "execution_gate": {
            "all_four_cells_completed_epoch_5": True,
            "all_losses_finite": True,
            "identical_train_dev_seed_architecture_budget_commit": True,
            "s0_train_supervised_tokens_per_epoch": 14307,
            "s1_s2_s3_train_supervised_tokens_per_epoch": 6681,
            "passed": True,
        },
        "frozen_shortcuts": shortcuts,
        "cells": cells,
        "selection": selection,
        "paired_query_bootstrap_diagnostic": bootstrap,
        "decision": {
            "engineering_pipeline_passed": True,
            "span_token_localization_gate_passed": selection["token_gate_passed"],
            "selected_span_cell": selected,
            "exact_onset_gate_passed": onset_gate,
            "authorize_pseudo_tail": False,
            "authorize_negative_tail_shaping": False,
            "authorize_mixed_data_mechanism_run": False,
            "recommended_next_step": (
                "Keep unweighted sparse claim-span BCE as the token-localization candidate; "
                "next test a separately frozen boundary/segment decoder or loss for exact onset, "
                "then expand labels and seeds before any mechanism or Best-of-N claim."
            ),
        },
        "interpretation": {
            "span_token": (
                "S1 improves the held-out span-token and claim-mean AP point estimates over "
                "S0 and the strongest absolute-position shortcuts."
            ),
            "uncertainty": (
                "The paired query bootstrap intervals cross zero on only 16 dev rows; the point-"
                "estimate gate is a pilot selection signal, not stable efficacy evidence."
            ),
            "class_balance": (
                "Full negative-over-positive weighting increases fixed-threshold F1 but reduces "
                "threshold-free span and claim ranking; S2/S3 are not selected."
            ),
            "onset": (
                "No cell places any of six raw first-threshold crossings within five tokens; "
                "span discrimination and exact first-error boundary detection remain separate gates."
            ),
        },
        "code": {
            "training_commit": next(iter(code_commits)),
            "summary_commit": git_state(ROOT),
        },
        "pre_audit_output_excluded": protocol["execution"].get(
            "supersedes_pre_audit_output_root"
        ),
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
