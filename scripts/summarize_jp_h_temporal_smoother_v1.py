#!/usr/bin/env python3
"""Audit the 12 paired cells and summarize the fixed temporal smoother."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import read_jsonl  # noqa: E402
from src.clir_frozen_h_probe import (  # noqa: E402
    localization_metric_summary,
    validate_probe_protocol,
)
from src.clir_h_temporal_smoother import (  # noqa: E402
    mean_sample_std,
    metric_delta,
    validate_smoother_protocol,
)
from src.clir_localization_evaluation import evaluate_localization_rows  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/jp_h_temporal_smoother_v1/protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "configs/jp_h_temporal_smoother_v1/result_v1.json"
H_CHANGED_FIELDS = {
    "clir_token_hallucination_probs",
    "clir_path_no_hallucination_log_prob",
    "clir_path_hallucination_prob",
    "clir_pseudo_onset",
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {name}: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen {name} hash drifted: expected {expected_sha256}, got {observed}"
        )


def pool_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = evaluate_localization_rows(rows, threshold=0.5, negative_tail_margin=0.5)
    return localization_metric_summary(metrics)


def assert_non_h_fields_identical(
    raw: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    if set(candidate) != set(raw) | {"frozen_h_temporal_smoother_provenance"}:
        raise ValueError("Temporal-smoother scored row key set drifted")
    for name, value in raw.items():
        if name not in H_CHANGED_FIELDS and candidate[name] != value:
            raise ValueError(f"Temporal smoother changed protected row field: {name}")


def assert_position_baseline(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, name: str
) -> None:
    fields = (
        "rows",
        "span_supervised_tokens",
        "span_positive_tokens",
        "span_strongest_position_average_precision",
        "claims",
        "positive_claims",
        "claim_strongest_position_average_precision",
    )
    actual = {field: observed[field] for field in fields}
    if actual != expected:
        raise ValueError(f"{name} same-row position baseline drifted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Temporal-smoother summary requires a clean worktree")
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite smoother result {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_smoother_protocol(protocol)
    protocol_sha256 = file_sha256(protocol_path)
    for name, spec in protocol["inputs"].items():
        verify(resolve(spec["path"]), spec["sha256"], name=name)

    source_protocol_path = resolve(protocol["inputs"]["linear_probe_protocol"]["path"])
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    validate_probe_protocol(source_protocol)
    source_result_path = resolve(protocol["inputs"]["linear_probe_result"]["path"])
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    source_root = resolve(source_protocol["execution"]["output_root"])
    output_root = resolve(protocol["execution"]["output_root"])
    matrix_path = output_root / "matrix_run.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("status") != "completed" or matrix.get("completed_cells") != 12:
        raise ValueError("Temporal-smoother matrix is incomplete")
    if matrix.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Temporal-smoother matrix protocol hash drifted")
    if matrix.get("code", {}).get("dirty") is not False:
        raise ValueError("Temporal-smoother matrix used a dirty worktree")
    if (
        matrix.get("trainable_parameter_count") != 0
        or matrix.get("head_retrained") is not False
    ):
        raise ValueError("Temporal-smoother matrix changed trainable state")

    folds = [int(value) for value in protocol["evaluation"]["folds"]]
    seeds = [int(value) for value in protocol["evaluation"]["seeds"]]
    primary_folds = [
        int(value) for value in protocol["evaluation"]["primary_confirmatory_folds"]
    ]
    rows_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    raw_rows_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    execution_commits: set[str] = set()
    cell_reports: dict[str, Any] = {}
    protected_row_views = 0

    for fold in folds:
        fold_spec = source_protocol["cross_validation"]["folds"][str(fold)]
        dev_manifest_path = resolve(fold_spec["dev"]["path"])
        expected_ids = [str(row["id"]) for row in read_jsonl(dev_manifest_path)]
        for seed in seeds:
            key = f"fold_{fold}/seed_{seed}"
            source_report = source_result["cell_reports"][key]
            source_cell_root = source_root / f"fold_{fold}" / f"seed_{seed}"
            source_scored_path = source_cell_root / "dev_scored.jsonl"
            verify(
                source_scored_path,
                source_report["dev_scored_sha256"],
                name=f"{key} source dev scoring",
            )
            source_probe_path = source_cell_root / "probe.pt"
            verify(
                source_probe_path,
                source_report["probe_checkpoint_sha256"],
                name=f"{key} source probe checkpoint",
            )
            source_cell_result_path = source_cell_root / "cell_result.json"
            verify(
                source_cell_result_path,
                source_report["cell_result_sha256"],
                name=f"{key} source cell result",
            )

            cell_root = output_root / f"fold_{fold}" / f"seed_{seed}"
            cell_result_path = cell_root / "cell_result.json"
            matrix_cell = matrix["cell_index"][key]
            verify(
                cell_result_path,
                matrix_cell["cell_result_sha256"],
                name=f"{key} smoother cell result",
            )
            cell_result = json.loads(cell_result_path.read_text(encoding="utf-8"))
            if cell_result.get("status") != "completed":
                raise ValueError(f"{key} temporal-smoother cell is incomplete")
            if cell_result.get("protocol_sha256") != protocol_sha256:
                raise ValueError(f"{key} temporal-smoother protocol hash drifted")
            if cell_result.get("code", {}).get("dirty") is not False:
                raise ValueError(f"{key} temporal-smoother cell used dirty code")
            if cell_result.get("trainable_parameter_count") != 0:
                raise ValueError(f"{key} introduced trainable parameters")
            if cell_result.get("head_retrained") is not False:
                raise ValueError(f"{key} retrained the linear head")
            execution_commits.add(str(cell_result["code"]["commit"]))

            candidate_path = cell_root / "dev_scored.jsonl"
            evaluation_path = cell_root / "evaluation.json"
            verify(candidate_path, cell_result["dev_scored_sha256"], name=key)
            verify(
                evaluation_path,
                cell_result["evaluation_sha256"],
                name=f"{key} evaluation",
            )
            raw_rows = read_jsonl(source_scored_path)
            candidate_rows = read_jsonl(candidate_path)
            if [str(row["id"]) for row in raw_rows] != expected_ids:
                raise ValueError(f"{key} source row order drifted")
            if [str(row["id"]) for row in candidate_rows] != expected_ids:
                raise ValueError(f"{key} candidate row order drifted")
            for raw, candidate in zip(raw_rows, candidate_rows):
                assert_non_h_fields_identical(raw, candidate)
                provenance = candidate["frozen_h_temporal_smoother_provenance"]
                if (
                    provenance["source_dev_scored_sha256"]
                    != source_report["dev_scored_sha256"]
                ):
                    raise ValueError(f"{key} source prediction provenance drifted")
                if (
                    provenance["source_probe_checkpoint_sha256"]
                    != source_report["probe_checkpoint_sha256"]
                ):
                    raise ValueError(f"{key} source head provenance drifted")
                protected_row_views += 1
            observed_metrics = pool_summary(candidate_rows)
            if observed_metrics != cell_result["dev_metrics"]:
                raise ValueError(f"{key} persisted smoother metrics drifted")
            rows_by_cell[(fold, seed)] = candidate_rows
            raw_rows_by_cell[(fold, seed)] = raw_rows
            cell_reports[key] = {
                "fold": fold,
                "seed": seed,
                "selection_exposed": bool(fold_spec["selection_exposed"]),
                "rows": len(candidate_rows),
                "source_probe_checkpoint_sha256": source_report[
                    "probe_checkpoint_sha256"
                ],
                "source_dev_scored_sha256": source_report["dev_scored_sha256"],
                "candidate_dev_scored_sha256": cell_result["dev_scored_sha256"],
                "cell_result_sha256": file_sha256(cell_result_path),
                "raw_metrics": source_report["dev_metrics"],
                "candidate_metrics": observed_metrics,
            }

    if len(execution_commits) != 1:
        raise ValueError("Temporal-smoother cells did not share one clean commit")
    if protected_row_views != 192:
        raise ValueError("Temporal-smoother protected-row audit count drifted")

    primary: dict[str, Any] = {}
    all_oof: dict[str, Any] = {}
    passing_seeds: list[int] = []
    delta_values = {
        "span_over_position": [],
        "claim_over_position": [],
        "span_over_raw_linear": [],
        "claim_over_raw_linear": [],
    }
    candidate_span_aps: list[float] = []
    candidate_claim_aps: list[float] = []
    for seed in seeds:
        candidate_primary_rows = [
            row for fold in primary_folds for row in rows_by_cell[(fold, seed)]
        ]
        raw_primary_rows = [
            row for fold in primary_folds for row in raw_rows_by_cell[(fold, seed)]
        ]
        candidate_all_rows = [
            row for fold in folds for row in rows_by_cell[(fold, seed)]
        ]
        raw_all_rows = [row for fold in folds for row in raw_rows_by_cell[(fold, seed)]]
        if (
            len(candidate_primary_rows) != 48
            or len({str(row["id"]) for row in candidate_primary_rows}) != 48
        ):
            raise ValueError("Temporal-smoother primary pool is not 48 unique rows")
        if (
            len(candidate_all_rows) != 64
            or len({str(row["id"]) for row in candidate_all_rows}) != 64
        ):
            raise ValueError("Temporal-smoother all-OOF pool is not 64 unique rows")

        candidate_primary = pool_summary(candidate_primary_rows)
        raw_primary = pool_summary(raw_primary_rows)
        candidate_all = pool_summary(candidate_all_rows)
        raw_all = pool_summary(raw_all_rows)
        expected_raw_primary = source_result["confirmatory_48_rows_by_seed"][str(seed)]
        expected_raw_all = source_result["all_64_out_of_fold_diagnostic_by_seed"][
            str(seed)
        ]
        for field, value in raw_primary.items():
            if field in expected_raw_primary and value != expected_raw_primary[field]:
                raise ValueError("Paired raw primary metric reproduction failed")
        if raw_all != expected_raw_all:
            raise ValueError("Paired raw all-OOF metric reproduction failed")
        assert_position_baseline(
            candidate_primary,
            source_protocol["position_baselines"]["confirmatory_48"],
            name="primary-48",
        )
        assert_position_baseline(
            candidate_all,
            source_protocol["position_baselines"]["all_oof_64"],
            name="all-OOF-64",
        )
        deltas = metric_delta(candidate_primary, raw_primary)
        gate = {name: value > 0.0 for name, value in deltas.items()}
        gate["all_four_passed"] = all(gate.values())
        if gate["all_four_passed"]:
            passing_seeds.append(seed)
        primary[str(seed)] = {
            "raw_linear": raw_primary,
            "fixed_logit_smoother": candidate_primary,
            "paired_advantages": deltas,
            "engineering_gate": gate,
        }
        all_oof[str(seed)] = {
            "raw_linear": raw_all,
            "fixed_logit_smoother": candidate_all,
            "paired_advantages": metric_delta(candidate_all, raw_all),
        }
        for name, value in deltas.items():
            delta_values[name].append(value)
        candidate_span_aps.append(float(candidate_primary["span_average_precision"]))
        candidate_claim_aps.append(
            float(candidate_primary["claim_mean_average_precision"])
        )

    mean_deltas = {
        name: sum(values) / len(values) for name, values in delta_values.items()
    }
    rules = protocol["engineering_decision_rules"]
    engineering_supported = len(passing_seeds) >= int(
        rules["minimum_passing_seeds"]
    ) and all(value > 0.0 for value in mean_deltas.values())
    decision = {
        "engineering_signal_supported": engineering_supported,
        "passing_seeds": passing_seeds,
        "passing_seed_count": len(passing_seeds),
        "required_passing_seed_count": int(rules["minimum_passing_seeds"]),
        "mean_paired_advantages": mean_deltas,
        "all_mean_advantages_strictly_positive": all(
            value > 0.0 for value in mean_deltas.values()
        ),
        "method_adopted": False,
        "score_coupling_authorized": False,
        "reason_not_adopted": (
            "The fixed 3-token window was selected after inspecting these 64 rows; "
            "this run is an engineering check, not independent validation."
        ),
        "next_required_gate": (
            "new_blind_localization_validation_with_independent_secondary_annotation"
            if engineering_supported
            else "reconsider_localizer_or_expand_training_labels_before_validation"
        ),
        "automatic_next_experiment_authorized": False,
    }

    report = {
        "schema_version": "clir-jp-h-temporal-smoother-result-v1",
        "status": (
            "completed_engineering_signal_supported"
            if engineering_supported
            else "completed_engineering_signal_not_supported"
        ),
        "evidence_tier": "small-scale real",
        "selection_status": "post_selection_exploratory",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": protocol_sha256,
        },
        "execution_gate": {
            "matrix_completed_cells": 12,
            "execution_commit": next(iter(execution_commits)),
            "all_cells_share_clean_commit": True,
            "source_prediction_files_hash_verified": 12,
            "source_probe_checkpoints_hash_verified": 12,
            "source_prediction_row_views_reused": 192,
            "protected_non_h_row_views_bit_identical": protected_row_views,
            "trainable_parameter_count": 0,
            "head_retrained": False,
            "passed": True,
        },
        "protected_original_method": {
            "jp_checkpoint_changed": False,
            "linear_probe_checkpoints_changed": False,
            "base_reward_fields_changed": False,
            "dual_prior_or_mutual_distillation_changed": False,
            "hallucination_tail_enabled": False,
            "smoother_enters_reward_score": False,
        },
        "data": {
            "folds": 4,
            "seeds": seeds,
            "cells": 12,
            "primary_folds": primary_folds,
            "primary_rows": 48,
            "all_oof_rows": 64,
            "fold_0_selection_exposed": True,
            "pilot_test_accessed": False,
            "final_test_accessed": False,
        },
        "smoother": protocol["smoother"],
        "primary_48_rows_by_seed": primary,
        "all_64_out_of_fold_diagnostic_by_seed": all_oof,
        "multi_seed_summary": {
            "candidate_span_average_precision": mean_sample_std(candidate_span_aps),
            "candidate_claim_mean_average_precision": mean_sample_std(
                candidate_claim_aps
            ),
            **{
                f"paired_{name}": mean_sample_std(values)
                for name, values in delta_values.items()
            },
        },
        "cell_reports": cell_reports,
        "decision": decision,
        "limitations": protocol["limitations"],
        "pilot_test_accessed": False,
        "final_test_accessed": False,
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
