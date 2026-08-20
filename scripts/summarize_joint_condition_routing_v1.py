#!/usr/bin/env python3
"""Summarize the frozen seed-42 JPH hallucination-condition routing cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_joint_training_drop_one_v1 import (  # noqa: E402
    cell_summary,
    load_cell_result,
    paired_bootstrap,
    ranking_values,
    verify_spec,
)
from src.clir_joint_training import validate_joint_protocol  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/joint_condition_routing_v1/training_protocol_v1.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def classify_condition_routing_outcome(
    *,
    key_gates: bool,
    hallucination_gates: bool,
    complete_guard: bool,
    ranking_guard: bool,
) -> str:
    preservation = complete_guard and ranking_guard
    if preservation and key_gates and hallucination_gates:
        return "condition_route_supported_at_seed42_followup_required"
    if preservation and (key_gates ^ hallucination_gates):
        return "condition_route_partially_supported_at_seed42"
    return "condition_route_not_supported_at_frozen_gates_seed42"


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
        raise FileExistsError(f"Refusing to overwrite routing summary: {output}")

    for section in ("parent_experiment", "preceding_route_audit"):
        for name, spec in protocol[section].items():
            path = verify_spec(spec, name=f"{section} {name}")
            if section == "preceding_route_audit" and name == "result":
                audit = json.loads(path.read_text(encoding="utf-8"))
                if not audit.get("passed") or not audit.get("no_parameter_update"):
                    raise ValueError("Preceding condition-routing audit did not pass")

    seed = int(protocol["matched_training"]["seeds"][0])
    primary_k = str(protocol["evaluation"]["primary_k"])
    candidate_cell = "jph_h_condition_stopgrad"
    candidate_root = (
        resolve(protocol["execution"]["output_root"])
        / f"seed_{seed}"
        / candidate_cell
    )
    candidate_result_path = candidate_root / "cell_result.json"
    candidate_ranking_path = candidate_root / "ranking_evaluation.json"
    if not candidate_result_path.is_file() or not candidate_ranking_path.is_file():
        raise FileNotFoundError("Condition-routing JPH cell is incomplete")
    candidate_result = load_cell_result(
        candidate_result_path,
        expected_cell=candidate_cell,
        expected_seed=seed,
        expected_protocol_sha256=protocol_sha,
    )

    results: dict[str, Mapping[str, Any]] = {candidate_cell: candidate_result}
    result_hashes = {candidate_cell: file_sha256(candidate_result_path)}
    ranking_reports: dict[str, Mapping[str, Any]] = {
        candidate_cell: json.loads(
            candidate_ranking_path.read_text(encoding="utf-8")
        )
    }
    for cell, specs in protocol["frozen_controls"]["cells"].items():
        result_path = verify_spec(specs["cell_result"], name=f"control {cell} result")
        ranking_path = verify_spec(
            specs["ranking_evaluation"], name=f"control {cell} ranking"
        )
        result = load_cell_result(
            result_path,
            expected_cell=cell,
            expected_seed=seed,
            expected_protocol_sha256=str(specs["protocol_sha256"]),
            expected_code_commit=str(specs["training_code_commit"]),
        )
        results[cell] = result
        result_hashes[cell] = file_sha256(result_path)
        ranking_reports[cell] = json.loads(
            ranking_path.read_text(encoding="utf-8")
        )

    expected_cells = {
        candidate_cell,
        "jph_prior_plus_hallucination",
        "jp_original_prior",
    }
    if set(results) != expected_cells:
        raise ValueError("Condition-routing summary cell set drifted")

    query_ids, per_query = ranking_values(ranking_reports, k=primary_k)
    bootstrap = protocol["evaluation"]
    comparisons = {
        "candidate_minus_jph": paired_bootstrap(
            per_query["jph_prior_plus_hallucination"],
            per_query[candidate_cell],
            replicates=int(bootstrap["bootstrap_replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=seed,
        ),
        "candidate_minus_jp": paired_bootstrap(
            per_query["jp_original_prior"],
            per_query[candidate_cell],
            replicates=int(bootstrap["bootstrap_replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=seed,
        ),
    }
    summaries = {
        cell: cell_summary(
            results[cell], result_sha256=result_hashes[cell], primary_k=primary_k
        )
        for cell in results
    }
    candidate = summaries[candidate_cell]
    jph = summaries["jph_prior_plus_hallucination"]
    jp = summaries["jp_original_prior"]
    rules = protocol["condition_routing_decision_rules"]

    key_delta_jph = float(candidate["key_unit_ap"] - jph["key_unit_ap"])
    key_delta_jp = float(candidate["key_unit_ap"] - jp["key_unit_ap"])
    key_improves = key_delta_jph >= float(rules["key_ap_min_recovery_vs_jph"])
    key_near_jp = key_delta_jp >= -float(rules["key_ap_max_drop_vs_jp"])
    key_gates = key_improves and key_near_jp

    complete_delta_jph = float(
        candidate["complete_unit_ap"] - jph["complete_unit_ap"]
    )
    complete_guard = complete_delta_jph >= -float(
        rules["complete_ap_max_drop_vs_jph"]
    )

    span_delta_jph = float(
        candidate["hallucination_span_token_ap"]
        - jph["hallucination_span_token_ap"]
    )
    claim_delta_jph = float(
        candidate["hallucination_claim_mean_ap"]
        - jph["hallucination_claim_mean_ap"]
    )
    span_pass = float(candidate["hallucination_span_token_ap"]) > float(
        rules["hallucination_span_token_ap_min_exclusive"]
    )
    claim_pass = float(candidate["hallucination_claim_mean_ap"]) > float(
        rules["hallucination_claim_ap_min_exclusive"]
    )
    hallucination_gates = span_pass and claim_pass

    bon_delta_jp = float(candidate["bon_at_primary_k"] - jp["bon_at_primary_k"])
    ranking_guard = bon_delta_jp >= -float(
        rules["ranking_max_absolute_regression_vs_jp"]
    )
    classification = classify_condition_routing_outcome(
        key_gates=key_gates,
        hallucination_gates=hallucination_gates,
        complete_guard=complete_guard,
        ranking_guard=ranking_guard,
    )

    report = {
        "schema_version": "clir-joint-training-condition-routing-result-v1",
        "status": "completed_seed42_condition_routing_diagnostic",
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "cells": summaries,
        "artifact_roles": {
            candidate_cell: "new_single_factor_cell",
            "jph_prior_plus_hallucination": "immutable_direct_control",
            "jp_original_prior": "immutable_prior_only_reference",
        },
        "primary_ranking": {
            "k": int(primary_k),
            "query_count": len(query_ids),
            "paired_comparisons": comparisons,
        },
        "decision": {
            "classification": classification,
            "key": {
                "candidate_minus_jph": key_delta_jph,
                "candidate_minus_jp": key_delta_jp,
                "minimum_recovery_vs_jph": rules["key_ap_min_recovery_vs_jph"],
                "maximum_drop_vs_jp": rules["key_ap_max_drop_vs_jp"],
                "improves_vs_jph": key_improves,
                "near_jp": key_near_jp,
                "both_gates_pass": key_gates,
            },
            "complete": {
                "candidate_minus_jph": complete_delta_jph,
                "maximum_drop_vs_jph": rules["complete_ap_max_drop_vs_jph"],
                "guard_pass": complete_guard,
            },
            "hallucination": {
                "candidate_minus_jph_span_ap": span_delta_jph,
                "candidate_minus_jph_claim_ap": claim_delta_jph,
                "span_position_baseline": rules[
                    "hallucination_span_token_ap_min_exclusive"
                ],
                "claim_position_baseline": rules[
                    "hallucination_claim_ap_min_exclusive"
                ],
                "span_pass": span_pass,
                "claim_pass": claim_pass,
                "both_gates_pass": hallucination_gates,
            },
            "ranking": {
                "candidate_minus_jp_bon": bon_delta_jp,
                "maximum_regression_vs_jp": rules[
                    "ranking_max_absolute_regression_vs_jp"
                ],
                "guard_pass": ranking_guard,
            },
            "scope": "one predeclared seed-42 single-factor diagnostic; point estimates and paired validation bootstrap are not a formal efficacy claim",
        },
        "route_engineering": {
            "hallucination_condition_stop_gradient": True,
            "ordinary_jph_sampler_preserved": True,
            "original_dual_prior_preserved": True,
            "bidirectional_mutual_distillation_preserved": True,
            "shared_gradient_reward_gate_preserved": True,
            "loss_formulas_or_weights_changed": False,
        },
        "automatic_loss_weight_tuning_performed": False,
        "automatic_gradient_surgery_performed": False,
        "automatic_sampler_or_stream_change_performed": False,
        "expand_to_additional_seeds_authorized": False,
        "new_training_commit": candidate_result["code"]["commit"],
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
