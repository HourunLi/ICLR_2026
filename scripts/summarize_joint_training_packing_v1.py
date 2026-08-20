#!/usr/bin/env python3
"""Summarize the frozen seed-42 JPH supervision-packing diagnostic."""

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


DEFAULT_PROTOCOL = ROOT / "configs/joint_training_packing_v1/training_protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def classify_packing_outcome(
    *,
    key_gates: bool,
    hallucination_gates: bool,
    ranking_gate: bool,
) -> str:
    if key_gates and hallucination_gates and ranking_gate:
        return "packing_schedule_supported_at_seed42_followup_required"
    if ranking_gate and (key_gates or hallucination_gates):
        return "packing_schedule_partially_supported_at_seed42"
    return "packing_schedule_not_supported_at_frozen_gates_seed42"


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
        raise FileExistsError(f"Refusing to overwrite packing summary: {output}")
    for section in ("parent_experiment", "preceding_gradient_audit"):
        for name, spec in protocol[section].items():
            verify_spec(spec, name=f"{section} {name}")

    seed = int(protocol["matched_training"]["seeds"][0])
    primary_k = str(protocol["evaluation"]["primary_k"])
    packed_cell = "jph_supervision_packed"
    packed_root = resolve(protocol["execution"]["output_root"]) / f"seed_{seed}" / packed_cell
    packed_result_path = packed_root / "cell_result.json"
    packed_ranking_path = packed_root / "ranking_evaluation.json"
    if not packed_result_path.is_file() or not packed_ranking_path.is_file():
        raise FileNotFoundError("Packed JPH cell is incomplete")
    packed_result = load_cell_result(
        packed_result_path,
        expected_cell=packed_cell,
        expected_seed=seed,
        expected_protocol_sha256=protocol_sha,
    )

    results: dict[str, Mapping[str, Any]] = {packed_cell: packed_result}
    result_hashes = {packed_cell: file_sha256(packed_result_path)}
    ranking_reports: dict[str, Mapping[str, Any]] = {
        packed_cell: json.loads(packed_ranking_path.read_text(encoding="utf-8"))
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
        ranking_reports[cell] = json.loads(ranking_path.read_text(encoding="utf-8"))

    expected_cells = {
        packed_cell,
        "jph_prior_plus_hallucination",
        "jp_original_prior",
    }
    if set(results) != expected_cells:
        raise ValueError("Packing summary cell set drifted")
    query_ids, per_query = ranking_values(ranking_reports, k=primary_k)
    bootstrap = protocol["evaluation"]
    comparisons = {
        "packed_minus_jph": paired_bootstrap(
            per_query["jph_prior_plus_hallucination"],
            per_query[packed_cell],
            replicates=int(bootstrap["bootstrap_replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=seed,
        ),
        "packed_minus_jp": paired_bootstrap(
            per_query["jp_original_prior"],
            per_query[packed_cell],
            replicates=int(bootstrap["bootstrap_replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=seed,
        ),
    }
    summaries = {
        cell: cell_summary(
            results[cell],
            result_sha256=result_hashes[cell],
            primary_k=primary_k,
        )
        for cell in results
    }
    packed = summaries[packed_cell]
    jph = summaries["jph_prior_plus_hallucination"]
    jp = summaries["jp_original_prior"]
    rules = protocol["packing_decision_rules"]

    key_delta_jph = float(packed["key_unit_ap"] - jph["key_unit_ap"])
    key_delta_jp = float(packed["key_unit_ap"] - jp["key_unit_ap"])
    key_improves = key_delta_jph >= float(rules["key_ap_min_recovery_vs_jph"])
    key_near_jp = key_delta_jp >= -float(rules["key_ap_max_drop_vs_jp"])
    key_gates = key_improves and key_near_jp
    span_pass = float(packed["hallucination_span_token_ap"]) > float(
        rules["hallucination_span_token_ap_min_exclusive"]
    )
    claim_pass = float(packed["hallucination_claim_mean_ap"]) > float(
        rules["hallucination_claim_ap_min_exclusive"]
    )
    hallucination_gates = span_pass and claim_pass
    bon_delta_jp = float(packed["bon_at_primary_k"] - jp["bon_at_primary_k"])
    ranking_pass = bon_delta_jp >= -float(
        rules["ranking_max_absolute_regression_vs_jp"]
    )
    classification = classify_packing_outcome(
        key_gates=key_gates,
        hallucination_gates=hallucination_gates,
        ranking_gate=ranking_pass,
    )

    final_train = packed_result["training_engineering"]["final_epoch_train"]
    active_batches = final_train["active_batches"]
    expected_active = protocol["engineering_gates"]["per_epoch_active_batches"]
    if any(int(active_batches.get(name, -1)) != int(value) for name, value in expected_active.items()):
        raise ValueError("Published packed cell failed its active-batch gate")

    report = {
        "schema_version": "clir-joint-training-packing-result-v1",
        "status": "completed_seed42_packing_schedule_diagnostic",
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "cells": summaries,
        "primary_ranking": {
            "k": int(primary_k),
            "query_count": len(query_ids),
            "paired_comparisons": comparisons,
        },
        "packing_engineering": {
            "final_epoch_active_batches": active_batches,
            "expected_active_batches": expected_active,
            "row_exposure_unchanged": True,
            "total_optimizer_steps_unchanged": True,
            "effective_auxiliary_active_steps_changed": True,
            "loss_formulas_or_configured_weights_changed": False,
            "semantic_metadata_changed": False,
        },
        "decision": {
            "classification": classification,
            "key": {
                "packed_minus_jph": key_delta_jph,
                "packed_minus_jp": key_delta_jp,
                "minimum_recovery_vs_jph": rules["key_ap_min_recovery_vs_jph"],
                "maximum_drop_vs_jp": rules["key_ap_max_drop_vs_jp"],
                "improves_vs_jph": key_improves,
                "near_jp": key_near_jp,
                "both_gates_pass": key_gates,
            },
            "hallucination": {
                "packed_minus_jph_span_ap": float(
                    packed["hallucination_span_token_ap"]
                    - jph["hallucination_span_token_ap"]
                ),
                "packed_minus_jph_claim_ap": float(
                    packed["hallucination_claim_mean_ap"]
                    - jph["hallucination_claim_mean_ap"]
                ),
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
                "packed_minus_jp_bon": bon_delta_jp,
                "maximum_regression_vs_jp": rules[
                    "ranking_max_absolute_regression_vs_jp"
                ],
                "gate_pass": ranking_pass,
            },
            "scope": "one seed and a packing/effective-budget compound intervention; not formal efficacy or a pure causal packing estimate",
        },
        "automatic_loss_weight_tuning_performed": False,
        "automatic_gradient_surgery_performed": False,
        "automatic_stream_switch_performed": False,
        "expand_to_additional_seeds_authorized": False,
        "new_training_commit": packed_result["code"]["commit"],
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
