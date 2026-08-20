#!/usr/bin/env python3
"""Summarize the frozen seed-42 JPH/JPC joint-training drop-one diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_joint_training import validate_joint_protocol  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/joint_training_drop_one_v1/training_protocol_v1.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_spec(spec: Mapping[str, Any], *, name: str) -> Path:
    path = resolve(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Frozen {name} is missing: {path}")
    observed = file_sha256(path)
    if observed != spec["sha256"]:
        raise ValueError(
            f"Frozen {name} hash drifted: expected {spec['sha256']}, got {observed}"
        )
    return path


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("Paired ranking arrays have incompatible shapes")
    delta = right_array - left_array
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, delta.size, size=(replicates, delta.size))
    means = delta[samples].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "query_count": int(delta.size),
        "mean_delta": float(delta.mean()),
        "bootstrap_ci": [
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1.0 - alpha)),
        ],
        "replicates": replicates,
        "confidence_level": confidence_level,
        "bootstrap_unit": "query",
        "right_minus_left": True,
    }


def ranking_values(
    reports: Mapping[str, Mapping[str, Any]], *, k: str
) -> tuple[list[str], dict[str, list[float]]]:
    reference_cell = next(iter(reports))
    reference_rows = reports[reference_cell]["per_query"]
    query_ids = [str(row["query_id"]) for row in reference_rows]
    reference_baselines = [
        (
            row["k"][k]["random_expected"],
            row["k"][k]["oracle"],
            row["k"][k]["correct_candidates"],
        )
        for row in reference_rows
    ]
    values: dict[str, list[float]] = {}
    for cell, report in reports.items():
        rows = report["per_query"]
        if [str(row["query_id"]) for row in rows] != query_ids:
            raise ValueError("Ranking query order differs across drop-one cells/controls")
        baselines = [
            (
                row["k"][k]["random_expected"],
                row["k"][k]["oracle"],
                row["k"][k]["correct_candidates"],
            )
            for row in rows
        ]
        if baselines != reference_baselines:
            raise ValueError("Ranking candidate outcomes differ across cells/controls")
        values[cell] = [float(row["k"][k]["reward"]) for row in rows]
    return query_ids, values


def classify_key_attribution(
    *, jph_reproduces_drop: bool, jpc_reproduces_drop: bool
) -> str:
    if jph_reproduces_drop and jpc_reproduces_drop:
        return "both_auxiliaries_individually_sufficient_at_seed42"
    if jph_reproduces_drop:
        return "hallucination_individually_sufficient_at_seed42"
    if jpc_reproduces_drop:
        return "consistency_individually_sufficient_at_seed42"
    return "joint_interaction_only_at_frozen_threshold_seed42"


def cell_summary(
    result: Mapping[str, Any], *, result_sha256: str, primary_k: str
) -> dict[str, Any]:
    return {
        "result_sha256": result_sha256,
        "bon_at_primary_k": float(
            result["ranking_metrics"][primary_k]["reward_bon_accuracy"]["value"]
        ),
        "pairwise_accuracy": float(result["ranking_health"]["pairwise_accuracy"]),
        "score_population_std": float(
            result["ranking_score_distribution"]["population_std"]
        ),
        "hallucination_span_token_ap": float(
            result["hallucination_dev"]["span_tokens"]["average_precision"]
        ),
        "hallucination_claim_mean_ap": float(
            result["hallucination_dev"]["claim_mean"]["average_precision"]
        ),
        "key_unit_ap": float(
            result["prior_dev"]["key_unit"]["average_precision_micro"]
        ),
        "complete_unit_ap": float(
            result["prior_dev"]["complete_unit"]["average_precision_micro"]
        ),
        "consistency_mean_cosine_gap": float(
            result["consistency_train_diagnostic"]["mean_cosine_gap"]
        ),
        "code": result["code"],
    }


def load_cell_result(
    path: Path,
    *,
    expected_cell: str,
    expected_seed: int,
    expected_protocol_sha256: str,
    expected_code_commit: str | None = None,
) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != "clir-joint-training-cell-result-v1"
        or result.get("cell") != expected_cell
        or int(result.get("seed", -1)) != expected_seed
        or result.get("protocol_sha256") != expected_protocol_sha256
    ):
        raise ValueError(f"Joint cell result provenance drifted: {expected_cell}")
    code = result.get("code")
    if not isinstance(code, Mapping) or code.get("dirty"):
        raise ValueError(f"Joint cell was not trained from clean code: {expected_cell}")
    if expected_code_commit is not None and code.get("commit") != expected_code_commit:
        raise ValueError(f"Frozen control code commit drifted: {expected_cell}")
    if result.get("pilot_test_accessed") or result.get("final_test_accessed"):
        raise ValueError(f"Forbidden test access recorded by {expected_cell}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_joint_protocol(protocol)
    protocol_sha = file_sha256(protocol_path)
    output = resolve(protocol["execution"]["result"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite drop-one summary: {output}")

    for name, spec in protocol["parent_experiment"].items():
        verify_spec(spec, name=f"parent {name}")
    parent_protocol_sha = protocol["parent_experiment"]["protocol"]["sha256"]
    seed = int(protocol["matched_training"]["seeds"][0])
    primary_k = str(protocol["evaluation"]["primary_k"])
    output_root = resolve(protocol["execution"]["output_root"])

    results: dict[str, Mapping[str, Any]] = {}
    result_hashes: dict[str, str] = {}
    ranking_reports: dict[str, Mapping[str, Any]] = {}
    artifact_roles: dict[str, str] = {}

    for cell in protocol["cells"]:
        cell_root = output_root / f"seed_{seed}" / cell
        result_path = cell_root / "cell_result.json"
        ranking_path = cell_root / "ranking_evaluation.json"
        if not result_path.is_file() or not ranking_path.is_file():
            raise FileNotFoundError(f"Drop-one cell is incomplete: {cell}/seed {seed}")
        result = load_cell_result(
            result_path,
            expected_cell=cell,
            expected_seed=seed,
            expected_protocol_sha256=protocol_sha,
        )
        results[cell] = result
        result_hashes[cell] = file_sha256(result_path)
        ranking_reports[cell] = json.loads(ranking_path.read_text(encoding="utf-8"))
        artifact_roles[cell] = "new_drop_one_cell"

    controls = protocol["frozen_controls"]
    if int(controls["seed"]) != seed:
        raise ValueError("Frozen control seed differs from the drop-one seed")
    for cell, specs in controls["cells"].items():
        result_path = verify_spec(
            specs["cell_result"], name=f"frozen control {cell} result"
        )
        ranking_path = verify_spec(
            specs["ranking_evaluation"], name=f"frozen control {cell} ranking"
        )
        result = load_cell_result(
            result_path,
            expected_cell=cell,
            expected_seed=seed,
            expected_protocol_sha256=parent_protocol_sha,
            expected_code_commit=str(controls["training_code_commit"]),
        )
        results[cell] = result
        result_hashes[cell] = file_sha256(result_path)
        ranking_reports[cell] = json.loads(ranking_path.read_text(encoding="utf-8"))
        artifact_roles[cell] = "immutable_parent_control"

    expected_cells = {
        "jph_prior_plus_hallucination",
        "jpc_prior_plus_consistency",
        "jp_original_prior",
        "jall_full_retained",
    }
    if set(results) != expected_cells:
        raise ValueError("Drop-one summary cell set drifted")

    query_ids, per_query = ranking_values(ranking_reports, k=primary_k)
    bootstrap = protocol["evaluation"]
    comparisons: dict[str, Any] = {}
    for left, right, name in (
        ("jp_original_prior", "jph_prior_plus_hallucination", "jph_minus_jp"),
        ("jp_original_prior", "jpc_prior_plus_consistency", "jpc_minus_jp"),
        ("jph_prior_plus_hallucination", "jall_full_retained", "jall_minus_jph"),
        ("jpc_prior_plus_consistency", "jall_full_retained", "jall_minus_jpc"),
    ):
        comparisons[name] = paired_bootstrap(
            per_query[left],
            per_query[right],
            replicates=int(bootstrap["bootstrap_replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=seed,
        )

    summaries = {
        cell: cell_summary(
            results[cell], result_sha256=result_hashes[cell], primary_k=primary_k
        )
        for cell in results
    }
    jp = summaries["jp_original_prior"]
    jall = summaries["jall_full_retained"]
    jph = summaries["jph_prior_plus_hallucination"]
    jpc = summaries["jpc_prior_plus_consistency"]
    rules = protocol["drop_one_decision_rules"]
    key_threshold = float(rules["key_ap_drop_threshold_vs_jp"])
    jall_key_delta = float(jall["key_unit_ap"] - jp["key_unit_ap"])
    if not jall_key_delta < -key_threshold:
        raise ValueError("The frozen parent no longer crosses its key-drop threshold")
    jph_key_delta = float(jph["key_unit_ap"] - jp["key_unit_ap"])
    jpc_key_delta = float(jpc["key_unit_ap"] - jp["key_unit_ap"])
    jph_reproduces = jph_key_delta < -key_threshold
    jpc_reproduces = jpc_key_delta < -key_threshold
    key_classification = classify_key_attribution(
        jph_reproduces_drop=jph_reproduces,
        jpc_reproduces_drop=jpc_reproduces,
    )

    span_baseline = float(rules["hallucination_span_token_ap_min_exclusive"])
    claim_baseline = float(rules["hallucination_claim_ap_min_exclusive"])
    jph_span_pass = float(jph["hallucination_span_token_ap"]) > span_baseline
    jph_claim_pass = float(jph["hallucination_claim_mean_ap"]) > claim_baseline
    h_recovered = jph_span_pass and jph_claim_pass
    h_classification = (
        "consistency_implicated_in_jall_localization_failure_seed42"
        if h_recovered
        else "failure_persists_without_consistency_mixing_or_prior_unresolved_seed42"
    )

    ranking_tolerance = float(rules["ranking_max_absolute_regression_vs_jp"])
    jph_bon_delta = float(jph["bon_at_primary_k"] - jp["bon_at_primary_k"])
    jpc_bon_delta = float(jpc["bon_at_primary_k"] - jp["bon_at_primary_k"])
    ranking_checks = {
        "jph_within_regression_tolerance_vs_jp": jph_bon_delta >= -ranking_tolerance,
        "jpc_within_regression_tolerance_vs_jp": jpc_bon_delta >= -ranking_tolerance,
    }

    new_commits = {str(results[cell]["code"]["commit"]) for cell in protocol["cells"]}
    if len(new_commits) != 1:
        raise ValueError("New drop-one cells were trained from different commits")

    report = {
        "schema_version": "clir-joint-training-drop-one-result-v1",
        "status": "completed_seed42_drop_one_diagnosis",
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "cells": summaries,
        "artifact_roles": artifact_roles,
        "primary_ranking": {
            "k": int(primary_k),
            "query_count": len(query_ids),
            "paired_comparisons": comparisons,
        },
        "key_prior_attribution": {
            "drop_threshold_vs_jp": key_threshold,
            "strict_definition": "cell_key_AP - JP_key_AP < -threshold",
            "jall_minus_jp": jall_key_delta,
            "jph_minus_jp": jph_key_delta,
            "jpc_minus_jp": jpc_key_delta,
            "jph_reproduces_drop": jph_reproduces,
            "jpc_reproduces_drop": jpc_reproduces,
            "classification": key_classification,
            "scope": "conditional seed-42 attribution under the frozen single-stream protocol; not a general causal or efficacy claim",
        },
        "hallucination_attribution": {
            "span_position_baseline": span_baseline,
            "claim_position_baseline": claim_baseline,
            "jph_span_pass": jph_span_pass,
            "jph_claim_pass": jph_claim_pass,
            "jph_recovers_both": h_recovered,
            "jph_minus_jall_span_ap": float(
                jph["hallucination_span_token_ap"]
                - jall["hallucination_span_token_ap"]
            ),
            "jph_minus_jall_claim_ap": float(
                jph["hallucination_claim_mean_ap"]
                - jall["hallucination_claim_mean_ap"]
            ),
            "classification": h_classification,
            "scope": "conditional seed-42 attribution; JPH still differs from standalone S1 in correctness/prior mixing and batch packing",
        },
        "ranking_protection": {
            "maximum_regression_vs_jp": ranking_tolerance,
            "jph_minus_jp": jph_bon_delta,
            "jpc_minus_jp": jpc_bon_delta,
            "checks": ranking_checks,
        },
        "automatic_loss_weight_tuning_performed": False,
        "automatic_sampler_or_stream_change_performed": False,
        "expand_to_additional_seeds_authorized": False,
        "new_training_commit": next(iter(new_commits)),
        "summary_code": git_state(ROOT),
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
        "next_action_requires_user_approval": True,
    }
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
