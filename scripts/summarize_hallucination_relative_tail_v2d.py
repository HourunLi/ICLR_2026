#!/usr/bin/env python3
"""Audit the one-cell pre-onset-anchored relative full-tail pilot v2d."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_hallucination_tail_comparison_v2b import (  # noqa: E402
    span_probability_ap,
    tail_gate,
    value_diagnostics,
)
from src.clir_data import read_jsonl  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/hallucination_localization_v2/relative_tail_protocol_v2d.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/hallucination_localization_v2/relative_tail_result_v2d.json"
)
CONTROL_CELL = "t0_span_only"
CANDIDATE_CELL = "r1_span_relative_full_tail"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def relative_tail_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible_rows = 0
    eligible_tail_tokens = 0
    onset_zero_ids: list[str] = []
    hallucinated_rows = 0
    for row in rows:
        values = row.get("output_token_ids", row.get("clir_token_values"))
        if not isinstance(values, list) or not values:
            raise ValueError("Relative-tail coverage requires a non-empty token-axis list")
        onset = int(row["hallucination_onset"])
        if not -1 <= onset < len(values):
            raise ValueError("Relative-tail coverage found an out-of-range onset")
        if onset < 0:
            continue
        hallucinated_rows += 1
        if onset == 0:
            onset_zero_ids.append(str(row["id"]))
            continue
        eligible_rows += 1
        eligible_tail_tokens += len(values) - onset
    return {
        "hallucinated_rows": hallucinated_rows,
        "eligible_rows_with_onset_greater_than_zero": eligible_rows,
        "eligible_tail_tokens": eligible_tail_tokens,
        "excluded_onset_zero_rows": len(onset_zero_ids),
        "excluded_row_ids": onset_zero_ids,
    }


def relative_tail_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    margin: float,
) -> dict[str, Any]:
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("Relative-tail diagnostic margin must be positive")
    coverage = relative_tail_coverage(rows)
    row_gaps: list[float] = []
    row_hinges: list[float] = []
    violations = 0
    tail_tokens = 0
    for row in rows:
        onset = int(row["hallucination_onset"])
        if onset <= 0:
            continue
        values = [float(value) for value in row["clir_token_values"]]
        pre_mean = sum(values[:onset]) / onset
        tail = values[onset:]
        row_gaps.append(sum(tail) / len(tail) - pre_mean)
        squared_hinges = [max(0.0, margin + value - pre_mean) ** 2 for value in tail]
        row_hinges.append(sum(squared_hinges) / len(squared_hinges))
        violations += sum((margin + value - pre_mean) > 0.0 for value in tail)
        tail_tokens += len(tail)
    if tail_tokens != coverage["eligible_tail_tokens"]:
        raise ValueError("Relative-tail diagnostic coverage drifted")
    return {
        **coverage,
        "margin": margin,
        "eligible_tail_token_relative_margin_violation_rate": (
            violations / tail_tokens if tail_tokens else None
        ),
        "row_mean_tail_minus_pre_onset_mean": _mean(row_gaps),
        "row_mean_relative_tail_hinge_squared": _mean(row_hinges),
    }


def _interval(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty bootstrap distribution")
    return {
        "mean": _mean(ordered),
        "lower_2_5_percentile": ordered[int(0.025 * (len(ordered) - 1))],
        "upper_97_5_percentile": ordered[int(0.975 * (len(ordered) - 1))],
        "fraction_above_zero": sum(value > 0.0 for value in ordered) / len(ordered),
        "fraction_below_zero": sum(value < 0.0 for value in ordered) / len(ordered),
        "valid_resamples": len(ordered),
    }


def paired_bootstrap(
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    margin: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    control = {str(row["id"]): row for row in control_rows}
    candidate = {str(row["id"]): row for row in candidate_rows}
    if set(control) != set(candidate):
        raise ValueError("Relative-tail bootstrap cells contain different row ids")
    identifiers = sorted(control)
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {
        "relative_margin_violation_rate": [],
        "row_mean_tail_minus_pre_gap": [],
        "pooled_tail_minus_pre_gap": [],
        "pooled_tail_minus_clean_gap": [],
        "explicit_token_value_risk_average_precision": [],
        "span_hallucination_probability_average_precision": [],
        "reward_score_correctness_roc_auc": [],
    }
    for _ in range(samples):
        selected = [identifiers[rng.randrange(len(identifiers))] for _ in identifiers]
        left_rows = [control[identifier] for identifier in selected]
        right_rows = [candidate[identifier] for identifier in selected]
        left_relative = relative_tail_diagnostics(left_rows, margin=margin)
        right_relative = relative_tail_diagnostics(right_rows, margin=margin)
        left_values = value_diagnostics(left_rows)
        right_values = value_diagnostics(right_rows)
        pairs = {
            "relative_margin_violation_rate": (
                left_relative["eligible_tail_token_relative_margin_violation_rate"],
                right_relative["eligible_tail_token_relative_margin_violation_rate"],
            ),
            "row_mean_tail_minus_pre_gap": (
                left_relative["row_mean_tail_minus_pre_onset_mean"],
                right_relative["row_mean_tail_minus_pre_onset_mean"],
            ),
            "pooled_tail_minus_pre_gap": (
                left_values["token_value_populations"]["tail_mean_minus_pre_onset_mean"],
                right_values["token_value_populations"]["tail_mean_minus_pre_onset_mean"],
            ),
            "pooled_tail_minus_clean_gap": (
                left_values["token_value_populations"]["tail_mean_minus_clean_mean"],
                right_values["token_value_populations"]["tail_mean_minus_clean_mean"],
            ),
            "explicit_token_value_risk_average_precision": (
                left_values["explicit_token_value_localization"]["average_precision"],
                right_values["explicit_token_value_localization"]["average_precision"],
            ),
            "span_hallucination_probability_average_precision": (
                span_probability_ap(left_rows),
                span_probability_ap(right_rows),
            ),
            "reward_score_correctness_roc_auc": (
                left_values["reward_score_correctness"]["roc_auc"],
                right_values["reward_score_correctness"]["roc_auc"],
            ),
        }
        for key, (left, right) in pairs.items():
            delta = _difference(right, left)
            if delta is not None:
                distributions[key].append(delta)
    return {
        "unit": "query_trajectory_row",
        "requested_resamples": samples,
        "seed": seed,
        "candidate_minus_control": {
            key: _interval(values) for key, values in distributions.items()
        },
    }


def _load_cell(root: Path, *, margin: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_path = root / "cell_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_path = root / "run.json"
    metrics_path = root / "metrics.jsonl"
    scored_path = root / "localization_dev_scored.jsonl"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metrics = read_jsonl(metrics_path)
    rows = read_jsonl(scored_path)
    if file_sha256(run_path) != result["run_sha256"]:
        raise ValueError(f"Cell run hash drifted at {root}")
    if file_sha256(metrics_path) != result["metrics_sha256"]:
        raise ValueError(f"Cell metrics hash drifted at {root}")
    if file_sha256(scored_path) != result["dev_scored_sha256"]:
        raise ValueError(f"Cell scored-dev hash drifted at {root}")
    if file_sha256(root / "model.pt") != result["checkpoint_sha256"]:
        raise ValueError(f"Cell checkpoint hash drifted at {root}")
    fixed = result["fixed_dev_metrics"]
    values = value_diagnostics(rows)
    span = fixed["explicit_claim_span_tokens"]
    return (
        {
            "result_path": str(result_path.relative_to(ROOT)),
            "cell_result_sha256": file_sha256(result_path),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "metrics_sha256": result["metrics_sha256"],
            "dev_scored_sha256": result["dev_scored_sha256"],
            "run": run,
            "metrics": metrics,
            "span_token_roc_auc": span["roc_auc"],
            "span_token_average_precision": span["average_precision"],
            "path_roc_auc": fixed["path_noisy_or_log_space"]["roc_auc"],
            "path_average_precision": fixed["path_noisy_or_log_space"]["average_precision"],
            "fixed_onset_mae": fixed["onset"]["mean_absolute_error_with_miss_as_length"],
            "fixed_onset_within_5": fixed["onset"]["within_5"]["count"],
            "value_diagnostics": values,
            "relative_tail_diagnostics": relative_tail_diagnostics(rows, margin=margin),
            "raw_result": result,
        },
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Relative-tail summary requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite relative-tail result: {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != "clir-hallucination-relative-tail-pilot-v2d":
        raise ValueError("Unknown relative-tail protocol")
    for spec in protocol["inputs"].values():
        if file_sha256(resolve(spec["path"])) != spec["sha256"]:
            raise ValueError(f"Frozen relative-tail input drifted: {spec['path']}")

    train_rows = read_jsonl(resolve(protocol["inputs"]["dense_train"]["path"]))
    dev_rows = read_jsonl(resolve(protocol["inputs"]["localization_dev"]["path"]))
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        observed = relative_tail_coverage(rows)
        if observed != protocol["relative_anchor_audit"][split]:
            raise ValueError(f"Relative-tail {split} anchor audit drifted")

    margin = float(protocol["matched_training"]["relative_tail_margin"])
    control_root = resolve(protocol["reuse"]["output_root"])
    candidate_root = resolve(protocol["execution"]["output_root"]) / CANDIDATE_CELL
    control, control_rows = _load_cell(control_root, margin=margin)
    candidate, candidate_rows = _load_cell(candidate_root, margin=margin)
    expected_protocol = file_sha256(protocol_path)
    expected_train = protocol["inputs"]["dense_train"]["sha256"]
    expected_dev = protocol["inputs"]["localization_dev"]["sha256"]

    if control["run"].get("status") != "completed" or control["run"].get("completed_epoch") != 5:
        raise ValueError("Frozen T0 control is incomplete")
    if candidate["run"].get("status") != "completed" or candidate["run"].get("completed_epoch") != 5:
        raise ValueError("R1 relative-tail cell is incomplete")
    result = candidate["raw_result"]
    if result["protocol_sha256"] != expected_protocol:
        raise ValueError("R1 protocol hash drifted")
    if result["train_manifest_sha256"] != expected_train or result["dev_manifest_sha256"] != expected_dev:
        raise ValueError("R1 train/dev hash drifted")
    if set(str(row["id"]) for row in control_rows) != set(str(row["id"]) for row in candidate_rows):
        raise ValueError("T0 and R1 scored different development rows")
    if result.get("cell") != CANDIDATE_CELL or result.get("training_seed") != 42:
        raise ValueError("R1 cell identity or training seed drifted")
    if result.get("cross_validation_fold") is not None:
        raise ValueError("R1 single-cell pilot unexpectedly used a CV fold")
    if candidate["run"].get("code", {}).get("dirty") is not False:
        raise ValueError("R1 training used a dirty worktree")
    if len(candidate["metrics"]) != 5:
        raise ValueError("R1 must contain five epoch records")
    expected_counts = {"train": (6681, 3110), "validation": (2451, 1460)}
    for epoch in candidate["metrics"]:
        for section, (sparse_count, relative_count) in expected_counts.items():
            counts = epoch[section]["applicable_counts"]
            if counts["localization_token_bce"] != sparse_count:
                raise ValueError(f"R1 {section} sparse-token count drifted")
            if counts["localization_relative_tail_margin"] != relative_count:
                raise ValueError(f"R1 {section} relative-tail count drifted")
        for section in ("train", "checkpoint_train_evaluation", "validation"):
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in epoch[section]["losses"].values()
            ):
                raise ValueError("R1 contains a non-finite loss")

    checkpoint = torch.load(candidate_root / "model.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    frozen_cell = protocol["cells"][CANDIDATE_CELL]
    if float(config.get("tail_weight", -1.0)) != 0.0:
        raise ValueError("R1 accidentally enabled the historical absolute tail")
    if float(config.get("pseudo_tail_weight", -1.0)) != 0.0:
        raise ValueError("R1 accidentally enabled pseudo tail")
    if float(config.get("relative_tail_weight", -1.0)) != float(frozen_cell["relative_tail_weight"]):
        raise ValueError("R1 relative-tail weight drifted")
    if float(config.get("relative_tail_margin", -1.0)) != margin:
        raise ValueError("R1 relative-tail margin drifted")

    control_for_gate = {
        "span_token_average_precision": control["span_token_average_precision"],
        "value_diagnostics": control["value_diagnostics"],
    }
    candidate_for_gate = {
        "span_token_average_precision": candidate["span_token_average_precision"],
        "value_diagnostics": candidate["value_diagnostics"],
    }
    standard_gate = tail_gate(control_for_gate, candidate_for_gate)
    control_relative = control["relative_tail_diagnostics"]
    candidate_relative = candidate["relative_tail_diagnostics"]
    violation_passed = (
        candidate_relative["eligible_tail_token_relative_margin_violation_rate"]
        < control_relative["eligible_tail_token_relative_margin_violation_rate"]
    )
    guards = {
        "relative_margin_violation_improved": violation_passed,
        **standard_gate,
        "all_pilot_guards_passed": violation_passed and standard_gate["all_pilot_guards_passed"],
    }

    control_values = control["value_diagnostics"]["token_value_populations"]
    candidate_values = candidate["value_diagnostics"]["token_value_populations"]
    deltas = {
        "relative_margin_violation_rate": _difference(
            candidate_relative["eligible_tail_token_relative_margin_violation_rate"],
            control_relative["eligible_tail_token_relative_margin_violation_rate"],
        ),
        "row_mean_tail_minus_pre_gap": _difference(
            candidate_relative["row_mean_tail_minus_pre_onset_mean"],
            control_relative["row_mean_tail_minus_pre_onset_mean"],
        ),
        "pooled_tail_minus_pre_gap": _difference(
            candidate_values["tail_mean_minus_pre_onset_mean"],
            control_values["tail_mean_minus_pre_onset_mean"],
        ),
        "pooled_tail_minus_clean_gap": _difference(
            candidate_values["tail_mean_minus_clean_mean"],
            control_values["tail_mean_minus_clean_mean"],
        ),
        "mean_pre_onset": _difference(candidate_values["mean_pre_onset"], control_values["mean_pre_onset"]),
        "mean_tail": _difference(candidate_values["mean_tail"], control_values["mean_tail"]),
        "mean_clean": _difference(candidate_values["mean_clean"], control_values["mean_clean"]),
        "explicit_token_value_risk_average_precision": _difference(
            candidate["value_diagnostics"]["explicit_token_value_localization"]["average_precision"],
            control["value_diagnostics"]["explicit_token_value_localization"]["average_precision"],
        ),
        "span_hallucination_probability_average_precision": (
            candidate["span_token_average_precision"] - control["span_token_average_precision"]
        ),
        "reward_score_correctness_roc_auc": _difference(
            candidate["value_diagnostics"]["reward_score_correctness"]["roc_auc"],
            control["value_diagnostics"]["reward_score_correctness"]["roc_auc"],
        ),
    }
    bootstrap = paired_bootstrap(
        control_rows,
        candidate_rows,
        margin=margin,
        samples=int(protocol["evaluation"]["paired_query_bootstrap_samples"]),
        seed=int(protocol["evaluation"]["paired_query_bootstrap_seed"]),
    )
    passed = bool(guards["all_pilot_guards_passed"])

    def publishable(cell: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in cell.items() if key not in {"run", "metrics", "raw_result"}}

    report = {
        "schema_version": "clir-hallucination-relative-tail-result-v2d",
        "status": (
            "completed_pass_single_cell_authorize_confirmatory_expansion"
            if passed
            else "completed_fail_keep_t0"
        ),
        "evidence_tier": "pipeline_pilot",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": expected_protocol,
        },
        "data": {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_sha256": expected_train,
            "dev_sha256": expected_dev,
            "relative_anchor_audit": protocol["relative_anchor_audit"],
        },
        "execution_gate": {
            "r1_completed_epoch_5": True,
            "all_losses_finite": True,
            "sparse_train_tokens_per_epoch": 6681,
            "relative_tail_train_tokens_per_epoch": 3110,
            "sparse_dev_tokens_per_epoch": 2451,
            "relative_tail_dev_tokens_per_epoch": 1460,
            "absolute_and_pseudo_tail_disabled": True,
            "passed": True,
        },
        "cells": {
            CONTROL_CELL: publishable(control),
            CANDIDATE_CELL: publishable(candidate),
        },
        "candidate_minus_control": deltas,
        "predeclared_guards": guards,
        "paired_query_bootstrap_diagnostic": bootstrap,
        "decision": {
            "authorize_confirmatory_expansion": passed,
            "select_r1_as_module_default": False,
            "keep_t0_as_current_default": True,
            "permanently_reject_relative_tail": False,
            "allowed_conclusion": (
                "R1 passed the selection-exposed one-seed point-estimate guards and may enter only the frozen multi-fold, multi-seed confirmatory expansion."
                if passed
                else "R1 failed at least one selection-exposed one-seed point-estimate guard; keep T0 and do not expand this objective unchanged."
            ),
        },
        "limitations": [
            "One training seed and sixteen selection-exposed development trajectories.",
            "The labels remain Silver pipeline-pilot labels rather than independent human Gold.",
            "Full tail intentionally lowers supported and unreviewed post-onset continuation values.",
            "No pilot/final test, mixed training, or Best-of-N result was used."
        ],
        "code": {
            "control_training_commit": control["run"]["code"]["commit"],
            "candidate_training_commit": candidate["run"]["code"]["commit"],
            "summary_commit": state,
        },
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
