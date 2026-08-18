#!/usr/bin/env python3
"""Audit and summarize the frozen multi-seed, out-of-fold tail comparison."""

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

from scripts.summarize_hallucination_tail_comparison_v2b import (  # noqa: E402
    _interval,
    span_probability_ap,
    tail_gate,
    value_diagnostics,
)
from src.clir_data import read_jsonl  # noqa: E402
from src.clir_localization_evaluation import evaluate_localization_rows  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/hallucination_localization_v2/tail_cv_protocol_v2c.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/hallucination_localization_v2/tail_cv_result_v2c.json"
)
CONTROL = "t0_span_only"
CANDIDATE = "t2_span_tail_historical"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric sequence")
    return sum(values) / len(values)


def metric_delta(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    control_values = control["value_diagnostics"]
    candidate_values = candidate["value_diagnostics"]
    return {
        "tail_minus_pre_gap": (
            candidate_values["token_value_populations"]["tail_mean_minus_pre_onset_mean"]
            - control_values["token_value_populations"]["tail_mean_minus_pre_onset_mean"]
        ),
        "tail_minus_clean_gap": (
            candidate_values["token_value_populations"]["tail_mean_minus_clean_mean"]
            - control_values["token_value_populations"]["tail_mean_minus_clean_mean"]
        ),
        "explicit_token_value_risk_average_precision": (
            candidate_values["explicit_token_value_localization"]["average_precision"]
            - control_values["explicit_token_value_localization"]["average_precision"]
        ),
        "span_hallucination_probability_average_precision": (
            candidate["span_token_average_precision"]
            - control["span_token_average_precision"]
        ),
        "reward_score_correctness_roc_auc": (
            candidate_values["reward_score_correctness"]["roc_auc"]
            - control_values["reward_score_correctness"]["roc_auc"]
        ),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluation = evaluate_localization_rows(
        rows,
        threshold=0.5,
        negative_tail_margin=0.5,
    )
    explicit = evaluation["explicit_claim_span_tokens"]
    return {
        "rows": len(rows),
        "span_token_roc_auc": explicit["roc_auc"],
        "span_token_average_precision": explicit["average_precision"],
        "path_roc_auc": evaluation["path_noisy_or_log_space"]["roc_auc"],
        "path_average_precision": evaluation["path_noisy_or_log_space"]["average_precision"],
        "fixed_onset_mae": evaluation["onset"][
            "mean_absolute_error_with_miss_as_length"
        ],
        "fixed_onset_within_5": evaluation["onset"]["within_5"]["count"],
        "fixed_onset_positive_rows": evaluation["onset"]["positive_rows"],
        "tail_margin_violation_rate": evaluation["token_value_shaping"][
            "tail_margin_violation_rate"
        ],
        "value_diagnostics": value_diagnostics(rows),
    }


def adoption_decision(
    seed_gates: Mapping[str, Mapping[str, Any]],
    seed_deltas: Mapping[str, Mapping[str, float]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rule = protocol["evaluation"]["adoption_rule"]
    passing = [seed for seed, gate in seed_gates.items() if gate["all_pilot_guards_passed"]]
    metric_names = next(iter(seed_deltas.values())).keys()
    means = {
        metric: mean([float(values[metric]) for values in seed_deltas.values()])
        for metric in metric_names
    }
    mean_guards = {
        "relative_tail_vs_pre": means["tail_minus_pre_gap"] < 0.0,
        "relative_tail_vs_clean": means["tail_minus_clean_gap"] < 0.0,
        "explicit_value_risk": means["explicit_token_value_risk_average_precision"]
        >= float(rule["mean_explicit_value_risk_ap_delta_minimum"]),
        "span": means["span_hallucination_probability_average_precision"]
        >= float(rule["mean_span_ap_delta_minimum"]),
        "correctness": means["reward_score_correctness_roc_auc"]
        >= float(rule["mean_correctness_roc_auc_delta_minimum"]),
    }
    catastrophic = {
        seed: {
            "span": float(values["span_hallucination_probability_average_precision"])
            < float(rule["per_seed_catastrophic_span_ap_delta_floor"]),
            "correctness": float(values["reward_score_correctness_roc_auc"])
            < float(rule["per_seed_catastrophic_correctness_roc_auc_delta_floor"]),
        }
        for seed, values in seed_deltas.items()
    }
    no_catastrophic_seed = not any(
        flag for values in catastrophic.values() for flag in values.values()
    )
    passed = (
        len(passing) >= int(rule["minimum_passing_confirmatory_seeds"])
        and all(mean_guards.values())
        and no_catastrophic_seed
    )
    return {
        "passed": passed,
        "passing_confirmatory_seeds": passing,
        "passing_seed_count": len(passing),
        "required_passing_seed_count": int(rule["minimum_passing_confirmatory_seeds"]),
        "mean_seed_deltas": means,
        "mean_guards": mean_guards,
        "catastrophic_seed_audit": catastrophic,
        "no_catastrophic_seed": no_catastrophic_seed,
        "selected_cell": CANDIDATE if passed else CONTROL,
        "allowed_conclusion": (
            rule["passing_conclusion"] if passed else rule["failing_conclusion"]
        ),
    }


def multiseed_bootstrap(
    control_by_seed: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_by_seed: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    control = {
        training_seed: {str(row["id"]): row for row in rows}
        for training_seed, rows in control_by_seed.items()
    }
    candidate = {
        training_seed: {str(row["id"]): row for row in rows}
        for training_seed, rows in candidate_by_seed.items()
    }
    if set(control) != set(candidate):
        raise ValueError("Tail CV bootstrap seed sets differ")
    identifiers: set[str] | None = None
    for training_seed in control:
        if set(control[training_seed]) != set(candidate[training_seed]):
            raise ValueError("Tail CV bootstrap T0/T2 row ids differ")
        if identifiers is None:
            identifiers = set(control[training_seed])
        elif identifiers != set(control[training_seed]):
            raise ValueError("Tail CV bootstrap row ids differ across training seeds")
    if not identifiers:
        raise ValueError("Tail CV bootstrap has no rows")
    ordered_ids = sorted(identifiers)
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {
        "tail_minus_pre_gap": [],
        "tail_minus_clean_gap": [],
        "explicit_token_value_risk_average_precision": [],
        "span_hallucination_probability_average_precision": [],
        "reward_score_correctness_roc_auc": [],
    }
    for _ in range(samples):
        selected = [ordered_ids[rng.randrange(len(ordered_ids))] for _ in ordered_ids]
        seed_differences: list[dict[str, float]] = []
        valid = True
        for training_seed in sorted(control):
            left_rows = [control[training_seed][identifier] for identifier in selected]
            right_rows = [candidate[training_seed][identifier] for identifier in selected]
            left = {
                "span_token_average_precision": span_probability_ap(left_rows),
                "value_diagnostics": value_diagnostics(left_rows),
            }
            right = {
                "span_token_average_precision": span_probability_ap(right_rows),
                "value_diagnostics": value_diagnostics(right_rows),
            }
            try:
                seed_differences.append(metric_delta(left, right))
            except TypeError:
                valid = False
                break
        if not valid:
            continue
        for metric in distributions:
            distributions[metric].append(
                mean([values[metric] for values in seed_differences])
            )
    if any(not values for values in distributions.values()):
        raise ValueError("Tail CV bootstrap produced an empty metric distribution")
    return {
        "unit": "query_trajectory_row",
        "training_seed_aggregation": "arithmetic_mean_of_per_seed_metric_differences",
        "requested_resamples": samples,
        "seed": seed,
        "valid_resamples": min(len(values) for values in distributions.values()),
        "candidate_minus_control": {
            metric: _interval(values) for metric, values in distributions.items()
        },
    }


def result_root(
    protocol: Mapping[str, Any], fold: int, seed: int, cell: str
) -> tuple[Path, str, bool]:
    if fold == 0 and seed == 42:
        reuse = protocol["reuse"]["fold_0_seed_42"]
        return (
            resolve(reuse["output_root"]) / cell,
            str(reuse["source_protocol"]["sha256"]),
            True,
        )
    return (
        resolve(protocol["execution"]["output_root"])
        / f"fold_{fold}"
        / f"seed_{seed}"
        / cell,
        "self",
        False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Tail CV summary requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite tail CV result: {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != "clir-hallucination-tail-cross-validation-v2c":
        raise ValueError("Unknown tail CV protocol")
    protocol_sha256 = file_sha256(protocol_path)
    for spec in protocol["inputs"].values():
        if file_sha256(resolve(spec["path"])) != spec["sha256"]:
            raise ValueError("Tail CV frozen input hash drifted")
    fold_audit = json.loads(
        resolve(protocol["inputs"]["fold_audit"]["path"]).read_text(encoding="utf-8")
    )
    if fold_audit.get("status") != "passed":
        raise ValueError("Tail CV fold audit did not pass")

    folds = [int(fold) for fold in protocol["cross_validation"]["folds"]]
    seeds = [int(seed) for seed in protocol["matched_training"]["seeds"]]
    cells = [CONTROL, CANDIDATE]
    rows_by_key: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    cell_reports: dict[str, Any] = {}
    new_commits: set[str] = set()
    checkpoint_hashes: set[str] = set()
    for fold in folds:
        fold_spec = protocol["cross_validation"]["folds"][str(fold)]
        expected_train_tokens = int(fold_spec["train"]["supervised_tokens"])
        expected_dev_tokens = int(fold_spec["dev"]["supervised_tokens"])
        expected_dev_ids = [
            str(row["id"]) for row in read_jsonl(resolve(fold_spec["dev"]["path"]))
        ]
        for training_seed in seeds:
            pair_commits: set[str] = set()
            for cell in cells:
                root, expected_protocol, reused = result_root(
                    protocol, fold, training_seed, cell
                )
                if expected_protocol == "self":
                    expected_protocol = protocol_sha256
                result_path = root / "cell_result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                run = json.loads((root / "run.json").read_text(encoding="utf-8"))
                metrics = read_jsonl(root / "metrics.jsonl")
                if run.get("status") != "completed" or run.get("completed_epoch") != 5:
                    raise ValueError(f"fold {fold} seed {training_seed} {cell} incomplete")
                if run["code"]["dirty"] is not False or result["code"]["dirty"] is not False:
                    raise ValueError("Tail CV cell used a dirty worktree")
                if result["protocol_sha256"] != expected_protocol:
                    raise ValueError("Tail CV cell protocol hash drifted")
                if result["train_manifest_sha256"] != fold_spec["train"]["sha256"]:
                    raise ValueError("Tail CV train manifest hash drifted")
                if result["dev_manifest_sha256"] != fold_spec["dev"]["sha256"]:
                    raise ValueError("Tail CV dev manifest hash drifted")
                if not reused:
                    if int(result.get("training_seed", -1)) != training_seed:
                        raise ValueError("Tail CV result training seed drifted")
                    if int(result.get("cross_validation_fold", -1)) != fold:
                        raise ValueError("Tail CV result fold drifted")
                    new_commits.add(str(run["code"]["commit"]))
                if len(metrics) != 5:
                    raise ValueError("Tail CV cell must contain five metric rows")
                for epoch in metrics:
                    if int(epoch["train"]["applicable_counts"]["localization_token_bce"]) != expected_train_tokens:
                        raise ValueError("Tail CV train sparse-token count drifted")
                    if int(epoch["validation"]["applicable_counts"]["localization_token_bce"]) != expected_dev_tokens:
                        raise ValueError("Tail CV dev sparse-token count drifted")
                    for section in ("train", "checkpoint_train_evaluation", "validation"):
                        if any(
                            not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            for value in epoch[section]["losses"].values()
                        ):
                            raise ValueError("Tail CV cell contains non-finite loss")
                scored_path = root / "localization_dev_scored.jsonl"
                if file_sha256(scored_path) != result["dev_scored_sha256"]:
                    raise ValueError("Tail CV scored-dev hash drifted")
                rows = read_jsonl(scored_path)
                if [str(row["id"]) for row in rows] != expected_dev_ids:
                    raise ValueError("Tail CV scored-dev row order drifted")
                rows_by_key[(fold, training_seed, cell)] = rows
                summary = summarize_rows(rows)
                cell_key = f"fold_{fold}/seed_{training_seed}/{cell}"
                cell_reports[cell_key] = {
                    "fold": fold,
                    "training_seed": training_seed,
                    "cell": cell,
                    "reused_from_v2b": reused,
                    "training_commit": run["code"]["commit"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "dev_scored_sha256": result["dev_scored_sha256"],
                    "cell_result_sha256": file_sha256(result_path),
                    **summary,
                }
                pair_commits.add(str(run["code"]["commit"]))
                if result["checkpoint_sha256"] in checkpoint_hashes:
                    raise ValueError("Tail CV unexpectedly reused a checkpoint")
                checkpoint_hashes.add(result["checkpoint_sha256"])
            if len(pair_commits) != 1:
                raise ValueError("Tail CV matched T0/T2 pair used different commits")
    if len(new_commits) != 1:
        raise ValueError("New tail CV cells did not share one clean training commit")

    confirmatory_folds = [
        int(fold) for fold in protocol["cross_validation"]["primary_confirmatory_folds"]
    ]
    pooled_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(seed): {cell: [] for cell in cells} for seed in seeds
    }
    all_oof_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(seed): {cell: [] for cell in cells} for seed in seeds
    }
    for training_seed in seeds:
        for cell in cells:
            for fold in folds:
                all_oof_rows[str(training_seed)][cell].extend(
                    rows_by_key[(fold, training_seed, cell)]
                )
                if fold in confirmatory_folds:
                    pooled_rows[str(training_seed)][cell].extend(
                        rows_by_key[(fold, training_seed, cell)]
                    )
            if len(pooled_rows[str(training_seed)][cell]) != 48:
                raise ValueError("Tail CV confirmatory pool must contain 48 rows")
            if len(all_oof_rows[str(training_seed)][cell]) != 64:
                raise ValueError("Tail CV all-fold pool must contain 64 rows")
            if len({row["id"] for row in pooled_rows[str(training_seed)][cell]}) != 48:
                raise ValueError("Tail CV confirmatory rows are not unique")

    confirmatory: dict[str, Any] = {}
    seed_gates: dict[str, Any] = {}
    seed_deltas: dict[str, Any] = {}
    all_oof: dict[str, Any] = {}
    for training_seed in map(str, seeds):
        control = summarize_rows(pooled_rows[training_seed][CONTROL])
        candidate = summarize_rows(pooled_rows[training_seed][CANDIDATE])
        seed_gates[training_seed] = tail_gate(control, candidate)
        seed_deltas[training_seed] = metric_delta(control, candidate)
        confirmatory[training_seed] = {
            CONTROL: control,
            CANDIDATE: candidate,
            "candidate_minus_control": seed_deltas[training_seed],
            "guard": seed_gates[training_seed],
        }
        all_control = summarize_rows(all_oof_rows[training_seed][CONTROL])
        all_candidate = summarize_rows(all_oof_rows[training_seed][CANDIDATE])
        all_oof[training_seed] = {
            CONTROL: all_control,
            CANDIDATE: all_candidate,
            "candidate_minus_control": metric_delta(all_control, all_candidate),
            "guard": tail_gate(all_control, all_candidate),
        }

    decision = adoption_decision(seed_gates, seed_deltas, protocol)
    bootstrap = multiseed_bootstrap(
        {seed: rows[CONTROL] for seed, rows in pooled_rows.items()},
        {seed: rows[CANDIDATE] for seed, rows in pooled_rows.items()},
        samples=int(protocol["evaluation"]["paired_query_bootstrap_samples"]),
        seed=int(protocol["evaluation"]["paired_query_bootstrap_seed"]),
    )
    report = {
        "schema_version": "clir-hallucination-tail-cv-result-v2c",
        "status": (
            "completed_adopt_t2_as_pipeline_default"
            if decision["passed"]
            else "completed_keep_t0_defer_tail"
        ),
        "evidence_tier": "pipeline_pilot",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": protocol_sha256,
        },
        "data": {
            "fold_audit": protocol["inputs"]["fold_audit"],
            "folds": 4,
            "training_seeds": seeds,
            "total_cells": 24,
            "reused_cells": 2,
            "new_cells": 22,
            "primary_confirmatory_folds": confirmatory_folds,
            "primary_confirmatory_rows": 48,
            "all_out_of_fold_rows": 64,
        },
        "execution_gate": {
            "all_24_cells_complete_epoch_5": True,
            "all_losses_finite": True,
            "matched_pair_parity": True,
            "scored_manifest_and_checkpoint_hashes_verified": True,
            "new_training_commit": next(iter(new_commits)),
            "reused_training_commit": protocol["reuse"]["fold_0_seed_42"][
                "training_commit"
            ],
            "passed": True,
        },
        "confirmatory_48_rows_by_seed": confirmatory,
        "all_64_out_of_fold_diagnostic_by_seed": all_oof,
        "fold_seed_cells": cell_reports,
        "paired_query_multiseed_bootstrap": bootstrap,
        "decision": {
            **decision,
            "adopt_tail_as_pipeline_default": decision["passed"],
            "authorize_standalone_localization_with_selected_cell": True,
            "authorize_mixed_training": False,
            "ready_to_enter_dual_prior_module": True,
            "exact_onset_gate_passed": False,
        },
        "limitations": [
            "The 64 trajectories retain Silver pipeline-pilot labels.",
            "Fold 0 is selection-exposed and excluded from the adoption gate.",
            "Folds 1-3 are out-of-fold but come from the earlier training pool, not a newly annotated external validation set.",
            "No pilot/final test or Best-of-N evaluation was used.",
            "Exact onset remains a separate unresolved boundary-decoding problem."
        ],
        "code": {
            "summary_commit": state,
        },
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
