#!/usr/bin/env python3
"""Apply the frozen 3-token logit smoother to all 12 OOF H-probe cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import read_jsonl  # noqa: E402
from src.clir_frozen_h_probe import (  # noqa: E402
    localization_metric_summary,
    validate_probe_protocol,
)
from src.clir_h_temporal_smoother import (  # noqa: E402
    build_smoothed_scored_row,
    smooth_persisted_probabilities,
    validate_smoother_protocol,
)
from src.clir_localization_evaluation import evaluate_localization_rows  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/jp_h_temporal_smoother_v1/protocol_v1.json"


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


def verify_source(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    for name, spec in protocol["inputs"].items():
        verify(resolve(spec["path"]), spec["sha256"], name=name)

    source_protocol_path = resolve(protocol["inputs"]["linear_probe_protocol"]["path"])
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    validate_probe_protocol(source_protocol)
    source_result_path = resolve(protocol["inputs"]["linear_probe_result"]["path"])
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source_result.get("status") != "completed_frozen_linear_probe_not_supported":
        raise ValueError("Source frozen linear-probe result status drifted")
    if source_result.get("protocol", {}).get("sha256") != file_sha256(
        source_protocol_path
    ):
        raise ValueError("Source probe result no longer binds its protocol")
    if source_result.get("base_invariance", {}).get("passed") is not True:
        raise ValueError("Source probe did not preserve the frozen JP base")
    if source_result.get("execution_gate", {}).get("matrix_completed_cells") != 12:
        raise ValueError("Source probe matrix is incomplete")
    return (
        source_protocol,
        source_result,
        resolve(source_protocol["execution"]["output_root"]),
    )


def unchanged_base_fields(raw: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    return (
        candidate["reward_score"] == raw["reward_score"]
        and candidate["clir_score"] == raw["clir_score"]
        and candidate["clir_token_values"] == raw["clir_token_values"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Temporal-smoother run requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_smoother_protocol(protocol)
    protocol_sha256 = file_sha256(protocol_path)
    source_protocol, source_result, source_root = verify_source(protocol)

    output_root = resolve(protocol["execution"]["output_root"])
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporal-smoother run {output_root}"
        )
    output_root.mkdir(parents=True)

    folds = [int(value) for value in protocol["evaluation"]["folds"]]
    seeds = [int(value) for value in protocol["evaluation"]["seeds"]]
    completed_cells = 0
    source_rows_verified = 0
    base_rows_verified = 0
    cell_index: dict[str, Any] = {}

    for fold in folds:
        fold_spec = source_protocol["cross_validation"]["folds"][str(fold)]
        dev_manifest_path = resolve(fold_spec["dev"]["path"])
        verify(
            dev_manifest_path,
            fold_spec["dev"]["sha256"],
            name=f"fold {fold} dev manifest",
        )
        expected_ids = [str(row["id"]) for row in read_jsonl(dev_manifest_path)]
        for seed in seeds:
            key = f"fold_{fold}/seed_{seed}"
            source_cell_report = source_result["cell_reports"][key]
            source_cell_root = source_root / f"fold_{fold}" / f"seed_{seed}"
            source_result_path = source_cell_root / "cell_result.json"
            source_probe_path = source_cell_root / "probe.pt"
            source_scored_path = source_cell_root / "dev_scored.jsonl"
            verify(
                source_result_path,
                source_cell_report["cell_result_sha256"],
                name=f"{key} source cell result",
            )
            verify(
                source_probe_path,
                source_cell_report["probe_checkpoint_sha256"],
                name=f"{key} source probe checkpoint",
            )
            verify(
                source_scored_path,
                source_cell_report["dev_scored_sha256"],
                name=f"{key} source dev scoring",
            )
            source_cell = json.loads(source_result_path.read_text(encoding="utf-8"))
            if source_cell.get("probe_parameter_count") != 769:
                raise ValueError(f"{key} source probe parameter count drifted")
            if source_cell.get("base_parameters_in_optimizer") != 0:
                raise ValueError(f"{key} source probe optimized base parameters")

            raw_rows = read_jsonl(source_scored_path)
            if [str(row["id"]) for row in raw_rows] != expected_ids:
                raise ValueError(f"{key} raw OOF row order drifted")
            candidate_rows: list[dict[str, Any]] = []
            for raw_row in raw_rows:
                raw_probabilities = raw_row["clir_token_hallucination_probs"]
                length = len(raw_row["output_token_ids"])
                if len(raw_probabilities) != length:
                    raise ValueError(f"{key} raw H probabilities are misaligned")
                smoothed_logits, smoothed_probabilities = (
                    smooth_persisted_probabilities(
                        raw_probabilities,
                        window_tokens=protocol["smoother"]["window_tokens"],
                    )
                )
                candidate = build_smoothed_scored_row(
                    raw_row,
                    smoothed_logits,
                    smoothed_probabilities,
                    fold=fold,
                    seed=seed,
                    source_dev_scored_sha256=source_cell_report["dev_scored_sha256"],
                    source_probe_checkpoint_sha256=source_cell_report[
                        "probe_checkpoint_sha256"
                    ],
                )
                if not unchanged_base_fields(raw_row, candidate):
                    raise AssertionError(f"{key} temporal smoother changed JP scores")
                candidate_rows.append(candidate)
                source_rows_verified += 1
                base_rows_verified += 1

            cell_root = output_root / f"fold_{fold}" / f"seed_{seed}"
            cell_root.mkdir(parents=True)
            scored_path = cell_root / "dev_scored.jsonl"
            atomic_write_jsonl(scored_path, candidate_rows)
            evaluation = evaluate_localization_rows(
                candidate_rows, threshold=0.5, negative_tail_margin=0.5
            )
            evaluation_path = cell_root / "evaluation.json"
            atomic_write_json(evaluation_path, evaluation)
            metrics = localization_metric_summary(evaluation)
            cell_result = {
                "schema_version": "clir-jp-h-temporal-smoother-cell-result-v1",
                "status": "completed",
                "evidence_tier": "small-scale real",
                "selection_status": "post_selection_exploratory",
                "fold": fold,
                "seed": seed,
                "protocol_sha256": protocol_sha256,
                "source_probe_checkpoint_sha256": source_cell_report[
                    "probe_checkpoint_sha256"
                ],
                "source_dev_scored_sha256": source_cell_report["dev_scored_sha256"],
                "source_cell_result_sha256": source_cell_report["cell_result_sha256"],
                "dev_manifest_sha256": fold_spec["dev"]["sha256"],
                "rows": len(candidate_rows),
                "source_prediction_rows_verified": len(raw_rows),
                "base_reward_rows_bit_identical": len(candidate_rows),
                "trainable_parameter_count": 0,
                "head_retrained": False,
                "window_tokens": protocol["smoother"]["window_tokens"],
                "dev_scored_sha256": file_sha256(scored_path),
                "evaluation_sha256": file_sha256(evaluation_path),
                "dev_metrics": metrics,
                "code": state,
                "pilot_test_accessed": False,
                "final_test_accessed": False,
                "formal_mechanism_claim_allowed": False,
            }
            cell_result_path = cell_root / "cell_result.json"
            atomic_write_json(cell_result_path, cell_result)
            cell_index[key] = {
                "source_probe_checkpoint_sha256": source_cell_report[
                    "probe_checkpoint_sha256"
                ],
                "source_dev_scored_sha256": source_cell_report["dev_scored_sha256"],
                "dev_scored_sha256": cell_result["dev_scored_sha256"],
                "evaluation_sha256": cell_result["evaluation_sha256"],
                "cell_result_sha256": file_sha256(cell_result_path),
            }
            completed_cells += 1

    matrix = {
        "schema_version": "clir-jp-h-temporal-smoother-matrix-run-v1",
        "status": "completed",
        "evidence_tier": "small-scale real",
        "selection_status": "post_selection_exploratory",
        "protocol_sha256": protocol_sha256,
        "code": state,
        "cells": len(folds) * len(seeds),
        "completed_cells": completed_cells,
        "source_prediction_rows_verified": source_rows_verified,
        "base_reward_rows_bit_identical": base_rows_verified,
        "trainable_parameter_count": 0,
        "head_retrained": False,
        "window_tokens": protocol["smoother"]["window_tokens"],
        "cell_index": cell_index,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
    }
    atomic_write_json(output_root / "matrix_run.json", matrix)
    print(json.dumps(matrix, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
