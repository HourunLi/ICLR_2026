#!/usr/bin/env python3
"""Audit and summarize the frozen sparse-span versus full-tail Pilot v2b."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import read_jsonl  # noqa: E402
from src.clir_localization_evaluation import (  # noqa: E402
    binary_average_precision,
    binary_roc_auc,
)
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/hallucination_localization_v2/tail_comparison_protocol_v2b.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/hallucination_localization_v2/tail_comparison_result_v2b.json"
)
CELL_ORDER = [
    "t0_span_only",
    "t1_span_tail_light",
    "t2_span_tail_historical",
]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _ranking(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float | None]:
    return {
        "roc_auc": binary_roc_auc(labels, scores),
        "average_precision": binary_average_precision(labels, scores),
    }


def value_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure whether low token values localize reviewed hallucinated spans."""

    pre: list[float] = []
    tail: list[float] = []
    clean: list[float] = []
    explicit_labels: list[int] = []
    explicit_risks: list[float] = []
    explicit_positive: list[float] = []
    explicit_negative: list[float] = []
    post_positive: list[float] = []
    post_supported: list[float] = []
    post_unreviewed: list[float] = []
    correctness: list[int] = []
    reward_scores: list[float] = []

    for row in rows:
        values = row.get("clir_token_values")
        target = row.get("token_hallucination_target")
        mask = row.get("token_hallucination_mask")
        onset = row.get("hallucination_onset")
        if not isinstance(values, list) or not values:
            raise ValueError("Tail diagnostics require non-empty clir_token_values")
        if not isinstance(target, list) or not isinstance(mask, list):
            raise ValueError("Tail diagnostics require sparse target and mask")
        if len(values) != len(target) or len(target) != len(mask):
            raise ValueError("Tail diagnostic token fields are misaligned")
        if isinstance(onset, bool) or not isinstance(onset, int) or onset >= len(values):
            raise ValueError("Tail diagnostic onset is invalid")
        token_values = [float(value) for value in values]
        if any(not math.isfinite(value) for value in token_values):
            raise ValueError("Tail diagnostic token values must be finite")
        correctness_value = row.get("correctness")
        reward_score = float(row.get("reward_score"))
        if correctness_value not in (0, 1) or not math.isfinite(reward_score):
            raise ValueError("Tail diagnostics require correctness and finite reward_score")
        correctness.append(int(correctness_value))
        reward_scores.append(reward_score)

        if onset >= 0:
            pre.extend(token_values[:onset])
            tail.extend(token_values[onset:])
        else:
            clean.extend(token_values)

        for position, (label, known, value) in enumerate(
            zip(target, mask, token_values)
        ):
            if label not in (0, 1) or known not in (0, 1):
                raise ValueError("Sparse token targets and masks must be binary")
            if not known:
                if label:
                    raise ValueError("Sparse positive token lies outside its mask")
                if onset >= 0 and position >= onset:
                    post_unreviewed.append(value)
                continue
            explicit_labels.append(int(label))
            explicit_risks.append(-value)
            if label:
                explicit_positive.append(value)
                if onset >= 0 and position >= onset:
                    post_positive.append(value)
            else:
                explicit_negative.append(value)
                if onset >= 0 and position >= onset:
                    post_supported.append(value)

    mean_pre = _mean(pre)
    mean_tail = _mean(tail)
    mean_clean = _mean(clean)
    mean_explicit_positive = _mean(explicit_positive)
    mean_explicit_negative = _mean(explicit_negative)
    mean_post_positive = _mean(post_positive)
    mean_post_supported = _mean(post_supported)
    return {
        "rows": len(rows),
        "token_value_populations": {
            "pre_onset_tokens": len(pre),
            "tail_tokens": len(tail),
            "clean_tokens": len(clean),
            "mean_pre_onset": mean_pre,
            "mean_tail": mean_tail,
            "mean_clean": mean_clean,
            "tail_mean_minus_pre_onset_mean": _difference(mean_tail, mean_pre),
            "tail_mean_minus_clean_mean": _difference(mean_tail, mean_clean),
        },
        "explicit_token_value_localization": {
            "supervised_tokens": len(explicit_labels),
            "positive_tokens": len(explicit_positive),
            "negative_tokens": len(explicit_negative),
            "risk_definition": "negative_clir_token_value",
            **_ranking(explicit_labels, explicit_risks),
            "mean_hallucinated_token_value": mean_explicit_positive,
            "mean_supported_token_value": mean_explicit_negative,
            "hallucinated_mean_minus_supported_mean": _difference(
                mean_explicit_positive, mean_explicit_negative
            ),
        },
        "post_onset_semantic_audit": {
            "explicit_hallucinated_tokens": len(post_positive),
            "explicit_supported_tokens": len(post_supported),
            "unreviewed_tokens": len(post_unreviewed),
            "mean_explicit_hallucinated_value": mean_post_positive,
            "mean_explicit_supported_value": mean_post_supported,
            "mean_unreviewed_value": _mean(post_unreviewed),
            "hallucinated_mean_minus_supported_mean": _difference(
                mean_post_positive, mean_post_supported
            ),
        },
        "reward_score_correctness": _ranking(correctness, reward_scores),
    }


def tail_label_composition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count exactly what a full post-onset tail would supervise or overwrite."""

    result = {
        "hallucinated_rows": 0,
        "tail_tokens": 0,
        "supervised_tokens": 0,
        "explicit_hallucinated_tokens": 0,
        "explicit_supported_tokens": 0,
        "unreviewed_tokens": 0,
    }
    for row in rows:
        onset = row["hallucination_onset"]
        target = row["token_hallucination_target"]
        mask = row["token_hallucination_mask"]
        if len(target) != len(mask) or not -1 <= onset < len(target):
            raise ValueError("Tail-composition labels are misaligned")
        if onset < 0:
            continue
        result["hallucinated_rows"] += 1
        for label, known in zip(target[onset:], mask[onset:]):
            result["tail_tokens"] += 1
            if known:
                result["supervised_tokens"] += 1
                if label:
                    result["explicit_hallucinated_tokens"] += 1
                else:
                    result["explicit_supported_tokens"] += 1
            else:
                result["unreviewed_tokens"] += 1
    result["explicit_supported_fraction_of_supervised_tail"] = (
        result["explicit_supported_tokens"] / result["supervised_tokens"]
    )
    result["unreviewed_fraction_of_full_tail"] = (
        result["unreviewed_tokens"] / result["tail_tokens"]
    )
    return result


def span_probability_ap(rows: Sequence[Mapping[str, Any]]) -> float | None:
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        probabilities = row["clir_token_hallucination_probs"]
        for label, known, score in zip(
            row["token_hallucination_target"],
            row["token_hallucination_mask"],
            probabilities,
        ):
            if known:
                labels.append(int(label))
                scores.append(float(score))
    return binary_average_precision(labels, scores)


def tail_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    span_tolerance: float = 0.02,
    correctness_tolerance: float = 0.05,
) -> dict[str, Any]:
    control_values = control["value_diagnostics"]["token_value_populations"]
    candidate_values = candidate["value_diagnostics"]["token_value_populations"]
    control_semantic = control["value_diagnostics"][
        "explicit_token_value_localization"
    ]
    candidate_semantic = candidate["value_diagnostics"][
        "explicit_token_value_localization"
    ]
    control_correctness = control["value_diagnostics"]["reward_score_correctness"]
    candidate_correctness = candidate["value_diagnostics"]["reward_score_correctness"]
    relative_pre = (
        candidate_values["tail_mean_minus_pre_onset_mean"]
        < control_values["tail_mean_minus_pre_onset_mean"]
    )
    relative_clean = (
        candidate_values["tail_mean_minus_clean_mean"]
        < control_values["tail_mean_minus_clean_mean"]
    )
    semantic_value = (
        candidate_semantic["average_precision"]
        >= control_semantic["average_precision"]
    )
    span = (
        candidate["span_token_average_precision"]
        >= control["span_token_average_precision"] - span_tolerance
    )
    correctness = (
        candidate_correctness["roc_auc"]
        >= control_correctness["roc_auc"] - correctness_tolerance
    )
    return {
        "relative_tail_vs_pre_improved": relative_pre,
        "relative_tail_vs_clean_improved": relative_clean,
        "relative_locality_passed": relative_pre and relative_clean,
        "semantic_value_guard_passed": semantic_value,
        "span_guard_tolerance": span_tolerance,
        "span_guard_passed": span,
        "correctness_roc_auc_guard_tolerance": correctness_tolerance,
        "correctness_guard_passed": correctness,
        "all_pilot_guards_passed": (
            relative_pre and relative_clean and semantic_value and span and correctness
        ),
    }


def _interval(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
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
    samples: int,
    seed: int,
) -> dict[str, Any]:
    control = {str(row["id"]): row for row in control_rows}
    candidate = {str(row["id"]): row for row in candidate_rows}
    if set(control) != set(candidate):
        raise ValueError("Tail bootstrap cells contain different row ids")
    identifiers = sorted(control)
    rng = random.Random(seed)
    tail_pre: list[float] = []
    value_ap: list[float] = []
    span_ap: list[float] = []
    correctness_auc: list[float] = []
    for _ in range(samples):
        selected_ids = [identifiers[rng.randrange(len(identifiers))] for _ in identifiers]
        left_rows = [control[identifier] for identifier in selected_ids]
        right_rows = [candidate[identifier] for identifier in selected_ids]
        left = value_diagnostics(left_rows)
        right = value_diagnostics(right_rows)
        left_tail_pre = left["token_value_populations"][
            "tail_mean_minus_pre_onset_mean"
        ]
        right_tail_pre = right["token_value_populations"][
            "tail_mean_minus_pre_onset_mean"
        ]
        left_value_ap = left["explicit_token_value_localization"]["average_precision"]
        right_value_ap = right["explicit_token_value_localization"]["average_precision"]
        left_auc = left["reward_score_correctness"]["roc_auc"]
        right_auc = right["reward_score_correctness"]["roc_auc"]
        if left_tail_pre is not None and right_tail_pre is not None:
            tail_pre.append(right_tail_pre - left_tail_pre)
        if left_value_ap is not None and right_value_ap is not None:
            value_ap.append(right_value_ap - left_value_ap)
        if left_auc is not None and right_auc is not None:
            correctness_auc.append(right_auc - left_auc)
        left_span_ap = span_probability_ap(left_rows)
        right_span_ap = span_probability_ap(right_rows)
        if left_span_ap is not None and right_span_ap is not None:
            span_ap.append(right_span_ap - left_span_ap)
    if not all((tail_pre, value_ap, span_ap, correctness_auc)):
        raise ValueError("Tail bootstrap produced an empty metric distribution")
    return {
        "unit": "query_trajectory_row",
        "requested_resamples": samples,
        "seed": seed,
        "candidate_minus_control": {
            "tail_minus_pre_gap": _interval(tail_pre),
            "explicit_token_value_risk_average_precision": _interval(value_ap),
            "span_hallucination_probability_average_precision": _interval(span_ap),
            "reward_score_correctness_roc_auc": _interval(correctness_auc),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Tail comparison summary requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite tail comparison summary: {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != "clir-hallucination-negative-tail-comparison-v2b":
        raise ValueError("Unknown hallucination tail comparison protocol")
    for input_spec in protocol["inputs"].values():
        if file_sha256(resolve(input_spec["path"])) != input_spec["sha256"]:
            raise ValueError("Tail comparison input hash drifted")
    base_cell = {
        key: value
        for key, value in protocol["cells"][CELL_ORDER[0]].items()
        if key not in {"description", "tail_weight"}
    }
    for cell in CELL_ORDER[1:]:
        comparable = {
            key: value
            for key, value in protocol["cells"][cell].items()
            if key not in {"description", "tail_weight"}
        }
        if comparable != base_cell:
            raise ValueError("Tail comparison cells differ beyond tail_weight")
    if [protocol["cells"][cell]["tail_weight"] for cell in CELL_ORDER] != [0.0, 0.1, 0.5]:
        raise ValueError("Tail comparison weights drifted")

    audit_path = resolve(protocol["inputs"]["tail_hypothesis_audit"]["path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for split_name, input_name in (("train", "dense_train"), ("dev", "localization_dev")):
        observed = tail_label_composition(
            read_jsonl(resolve(protocol["inputs"][input_name]["path"]))
        )
        expected = audit["current_full_tail_target_composition"][split_name]
        for key, value in observed.items():
            expected_value = expected[key]
            if isinstance(value, float):
                if not math.isclose(value, float(expected_value), rel_tol=0.0, abs_tol=1e-15):
                    raise ValueError(f"{split_name} tail-composition audit drifted at {key}")
            elif value != expected_value:
                raise ValueError(f"{split_name} tail-composition audit drifted at {key}")

    output_root = resolve(protocol["execution"]["output_root"])
    expected_train = protocol["inputs"]["dense_train"]["sha256"]
    expected_dev = protocol["inputs"]["localization_dev"]["sha256"]
    expected_protocol = file_sha256(protocol_path)
    code_commits: set[str] = set()
    cells: dict[str, Any] = {}
    dev_rows: dict[str, list[dict[str, Any]]] = {}
    for cell in CELL_ORDER:
        root = output_root / cell
        result_path = root / "cell_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        metrics = read_jsonl(root / "metrics.jsonl")
        if run.get("status") != "completed" or run.get("completed_epoch") != 5:
            raise ValueError(f"{cell} did not complete five epochs")
        if run["code"]["dirty"] is not False:
            raise ValueError(f"{cell} used a dirty worktree")
        if result["protocol_sha256"] != expected_protocol:
            raise ValueError(f"{cell} protocol hash drifted")
        if result["train_manifest_sha256"] != expected_train:
            raise ValueError(f"{cell} train manifest hash drifted")
        if result["dev_manifest_sha256"] != expected_dev:
            raise ValueError(f"{cell} dev manifest hash drifted")
        if len(metrics) != 5:
            raise ValueError(f"{cell} does not contain five metric rows")
        for epoch in metrics:
            if epoch["train"]["applicable_counts"]["localization_token_bce"] != 6681:
                raise ValueError(f"{cell} train sparse-token count drifted")
            if epoch["validation"]["applicable_counts"]["localization_token_bce"] != 2451:
                raise ValueError(f"{cell} dev sparse-token count drifted")
            for section in ("train", "checkpoint_train_evaluation", "validation"):
                if any(
                    not isinstance(value, (int, float)) or not math.isfinite(float(value))
                    for value in epoch[section]["losses"].values()
                ):
                    raise ValueError(f"{cell} contains non-finite loss")
        rows = read_jsonl(root / "localization_dev_scored.jsonl")
        if file_sha256(root / "localization_dev_scored.jsonl") != result["dev_scored_sha256"]:
            raise ValueError(f"{cell} scored-dev hash drifted")
        fixed = result["fixed_dev_metrics"]
        span = fixed["explicit_claim_span_tokens"]
        values = value_diagnostics(rows)
        evaluator_values = fixed["token_value_shaping"]
        for key, diagnostic_key in (
            ("mean_pre_onset", "mean_pre_onset"),
            ("mean_tail", "mean_tail"),
            ("mean_clean", "mean_clean"),
        ):
            if not math.isclose(
                float(evaluator_values[key]),
                float(values["token_value_populations"][diagnostic_key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{cell} value diagnostic disagrees with evaluator")
        cells[cell] = {
            "description": protocol["cells"][cell]["description"],
            "tail_weight": protocol["cells"][cell]["tail_weight"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "metrics_sha256": result["metrics_sha256"],
            "dev_scored_sha256": result["dev_scored_sha256"],
            "cell_result_sha256": file_sha256(result_path),
            "span_token_roc_auc": span["roc_auc"],
            "span_token_average_precision": span["average_precision"],
            "path_roc_auc": fixed["path_noisy_or_log_space"]["roc_auc"],
            "path_average_precision": fixed["path_noisy_or_log_space"][
                "average_precision"
            ],
            "fixed_onset_mae": fixed["onset"][
                "mean_absolute_error_with_miss_as_length"
            ],
            "fixed_onset_within_5": fixed["onset"]["within_5"]["count"],
            "tail_margin_violation_rate": evaluator_values[
                "tail_margin_violation_rate"
            ],
            "value_diagnostics": values,
        }
        code_commits.add(str(run["code"]["commit"]))
        dev_rows[cell] = rows
    if len(code_commits) != 1:
        raise ValueError("Tail comparison cells do not share one code commit")

    control = cells["t0_span_only"]
    gates: dict[str, Any] = {}
    bootstraps: dict[str, Any] = {}
    for cell in CELL_ORDER[1:]:
        gates[cell] = tail_gate(control, cells[cell])
        bootstraps[cell] = paired_bootstrap(
            dev_rows["t0_span_only"],
            dev_rows[cell],
            samples=int(protocol["evaluation"]["paired_query_bootstrap_samples"]),
            seed=int(protocol["evaluation"]["paired_query_bootstrap_seed"]),
        )
    retained = [cell for cell, gate in gates.items() if gate["all_pilot_guards_passed"]]
    status = (
        "completed_retain_tail_for_larger_validation"
        if retained
        else "completed_deprioritize_current_full_tail_not_permanently_reject"
    )
    report = {
        "schema_version": "clir-hallucination-tail-comparison-result-v2b",
        "status": status,
        "evidence_tier": "pipeline_pilot",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": expected_protocol,
        },
        "hypothesis_audit": protocol["inputs"]["tail_hypothesis_audit"],
        "data": {
            "train_rows": 48,
            "dev_rows": 16,
            "dev_hallucinated_rows": 6,
            "train_sha256": expected_train,
            "dev_sha256": expected_dev,
            "query_overlap": 0,
        },
        "execution_gate": {
            "all_three_cells_completed_epoch_5": True,
            "all_losses_finite": True,
            "identical_data_seed_architecture_budget_commit": True,
            "sparse_train_tokens_per_epoch": 6681,
            "sparse_dev_tokens_per_epoch": 2451,
            "passed": True,
        },
        "cells": cells,
        "predeclared_guards": gates,
        "paired_query_bootstrap_diagnostic": bootstraps,
        "decision": {
            "retained_tail_cells_for_larger_validation": retained,
            "authorize_tail_for_larger_validation": bool(retained),
            "authorize_tail_for_mixed_training": False,
            "permanently_reject_tail": False,
            "allowed_conclusion": (
                "At least one full-tail shaping strength passed all point-estimate pilot guards; "
                "retain it only for larger validation and multiple seeds."
                if retained
                else "Neither tested full-tail shaping strength passed every point-estimate pilot "
                "guard; deprioritize this implementation, but do not claim the tail hypothesis is "
                "permanently disproved."
            ),
        },
        "limitations": [
            "One training seed and sixteen development trajectories.",
            "Paired bootstrap intervals are diagnostics, not formal evidence.",
            "Full tail penalizes supported and unreviewed post-onset tokens by design.",
            "No pilot or final test and no Best-of-N evaluation were used."
        ],
        "code": {
            "training_commit": next(iter(code_commits)),
            "summary_commit": state,
        },
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
