#!/usr/bin/env python3
"""Evaluate dual-prior localization plus reward-gate integration diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_dual_prior_predictions_v1 import (  # noqa: E402
    HEAD_FIELDS,
    collect_head_arrays,
    correctness_metrics,
    evaluate_head,
    indexed,
    prior_collaboration_metrics,
    separation_metrics,
)
from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = (
    ROOT / "configs/dual_prior_reward_gate_v1/training_protocol_v1.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _distribution(row: Mapping[str, Any], field: str) -> np.ndarray:
    values = np.asarray(row.get(field), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{row.get('id')}/{field}: expected a finite non-empty vector")
    if np.any(values < 0.0) or not np.isclose(
        values.sum(), 1.0, rtol=1e-5, atol=1e-6
    ):
        raise ValueError(f"{row.get('id')}/{field}: expected normalized nonnegative mass")
    return values


def _normalized_entropy(distribution: np.ndarray) -> tuple[float, float]:
    entropy = float(
        -np.sum(
            distribution
            * np.log(np.clip(distribution, np.finfo(np.float64).tiny, None))
        )
    )
    if distribution.size == 1:
        return 0.0, 1.0
    return entropy / math.log(distribution.size), math.exp(entropy) / distribution.size


def _population_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Gate evaluation requires a finite non-empty metric vector")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def gate_integration_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    fusion_alpha: float,
) -> dict[str, Any]:
    if not 0.0 <= fusion_alpha <= 1.0:
        raise ValueError("fusion_alpha must lie in [0, 1]")
    objective_mse: list[float] = []
    l1_distance: list[float] = []
    overlap_mass: list[float] = []
    cosine_similarity: list[float] = []
    reported_dot_product: list[float] = []
    gate_entropy: list[float] = []
    gate_effective_fraction: list[float] = []
    gate_uniform_squared_l2: list[float] = []
    gate_max_mass: list[float] = []
    fused_entropy: list[float] = []
    fused_effective_fraction: list[float] = []
    fused_uniform_squared_l2: list[float] = []
    fused_max_mass: list[float] = []
    raw_mean_gate: list[float] = []
    scores: list[float] = []
    token_scores: list[float] = []
    residual_scores: list[float] = []
    token_absolute_fraction: list[float] = []
    for row in rows:
        gate = _distribution(row, "clir_gate_attention")
        key = _distribution(row, "clir_key_prior")
        complete = _distribution(row, "clir_complete_prior")
        if gate.shape != key.shape or gate.shape != complete.shape:
            raise ValueError(f"{row.get('id')}: gate and priors use different token axes")
        fused = fusion_alpha * key + (1.0 - fusion_alpha) * complete
        fused = fused / fused.sum()
        delta = gate - fused
        objective_mse.append(float(np.sum(delta * delta)))
        l1_distance.append(float(np.sum(np.abs(delta))))
        overlap_mass.append(float(np.minimum(gate, fused).sum()))
        denominator = float(np.linalg.norm(gate) * np.linalg.norm(fused))
        cosine_similarity.append(float(np.dot(gate, fused) / denominator))
        dot_product = float(np.dot(gate, fused))
        reported = float(row["clir_prior_gate_alignment"])
        if not math.isfinite(reported) or not math.isclose(
            dot_product, reported, rel_tol=2e-4, abs_tol=2e-7
        ):
            raise ValueError(f"{row.get('id')}: published gate alignment drifted")
        reported_dot_product.append(reported)

        gate_h, gate_eff = _normalized_entropy(gate)
        fused_h, fused_eff = _normalized_entropy(fused)
        uniform = np.full(gate.shape, 1.0 / gate.size, dtype=np.float64)
        gate_entropy.append(gate_h)
        gate_effective_fraction.append(gate_eff)
        gate_uniform_squared_l2.append(float(np.sum((gate - uniform) ** 2)))
        gate_max_mass.append(float(gate.max()))
        fused_entropy.append(fused_h)
        fused_effective_fraction.append(fused_eff)
        fused_uniform_squared_l2.append(float(np.sum((fused - uniform) ** 2)))
        fused_max_mass.append(float(fused.max()))

        mean_gate = float(row["clir_mean_gate"])
        score = float(row["clir_score"])
        token_values = np.asarray(row["clir_token_values"], dtype=np.float64)
        if (
            token_values.shape != gate.shape
            or not np.isfinite(token_values).all()
            or not math.isfinite(mean_gate)
            or not 0.0 <= mean_gate <= 1.0
            or not math.isfinite(score)
        ):
            raise ValueError(f"{row.get('id')}: invalid gate/score diagnostics")
        token_score = float(np.dot(gate, token_values))
        residual = score - token_score
        scale = abs(token_score) + abs(residual)
        raw_mean_gate.append(mean_gate)
        scores.append(score)
        token_scores.append(token_score)
        residual_scores.append(residual)
        token_absolute_fraction.append(abs(token_score) / scale if scale else 0.0)

    return {
        "rows": len(rows),
        "fusion_alpha": fusion_alpha,
        "training_objective_mse": float(np.mean(objective_mse)),
        "mean_l1_distance": float(np.mean(l1_distance)),
        "mean_overlap_mass": float(np.mean(overlap_mass)),
        "mean_cosine_similarity": float(np.mean(cosine_similarity)),
        "mean_reported_dot_product": float(np.mean(reported_dot_product)),
        "gate_normalized_entropy": float(np.mean(gate_entropy)),
        "gate_effective_token_fraction": float(np.mean(gate_effective_fraction)),
        "gate_uniform_squared_l2": float(np.mean(gate_uniform_squared_l2)),
        "gate_max_token_mass": float(np.mean(gate_max_mass)),
        "fused_prior_normalized_entropy": float(np.mean(fused_entropy)),
        "fused_prior_effective_token_fraction": float(
            np.mean(fused_effective_fraction)
        ),
        "fused_prior_uniform_squared_l2": float(
            np.mean(fused_uniform_squared_l2)
        ),
        "fused_prior_max_token_mass": float(np.mean(fused_max_mass)),
        "raw_mean_gate": _population_summary(raw_mean_gate),
        "score": _population_summary(scores),
        "token_score": _population_summary(token_scores),
        "score_residual": _population_summary(residual_scores),
        "mean_token_absolute_score_fraction": float(
            np.mean(token_absolute_fraction)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-scored", type=Path, required=True)
    parser.add_argument("--dev-scored", type=Path, required=True)
    parser.add_argument("--expected-train-scored-sha256", required=True)
    parser.add_argument("--expected-dev-scored-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-dual-prior-reward-gate-integration-training-protocol-v1"
    ):
        raise ValueError("Unexpected reward-gate protocol schema")
    if args.cell not in protocol["cells"]:
        raise ValueError(f"Unknown reward-gate cell: {args.cell}")
    if args.seed not in [int(value) for value in protocol["matched_training"]["seeds"]]:
        raise ValueError(f"Seed {args.seed} is not frozen in the protocol")
    train_scored_path = args.train_scored.resolve()
    dev_scored_path = args.dev_scored.resolve()
    if file_sha256(train_scored_path) != args.expected_train_scored_sha256:
        raise ValueError("Train scored predictions hash drifted")
    if file_sha256(dev_scored_path) != args.expected_dev_scored_sha256:
        raise ValueError("Dev scored predictions hash drifted")
    gold_path = resolve(protocol["inputs"]["gold_labels"]["path"])
    if file_sha256(gold_path) != protocol["inputs"]["gold_labels"]["sha256"]:
        raise ValueError("Gold labels hash drifted")

    gold_rows = read_jsonl(gold_path)
    gold_by_id = indexed(gold_rows, "id", name="dual-prior gold")
    train_rows = read_jsonl(train_scored_path)
    dev_rows = read_jsonl(dev_scored_path)
    cell = protocol["cells"][args.cell]
    expected_train_ids = [
        row["id"] for row in read_jsonl(resolve(cell["train"]["path"]))
    ]
    expected_dev_ids = [
        row["id"] for row in read_jsonl(resolve(cell["dev"]["path"]))
    ]
    if [row["id"] for row in train_rows] != expected_train_ids:
        raise ValueError("Train scored row identities/order differ from the frozen cell")
    if [row["id"] for row in dev_rows] != expected_dev_ids:
        raise ValueError("Dev scored row identities/order differ from the frozen cell")

    heads: dict[str, Any] = {}
    for head in HEAD_FIELDS:
        heads[head] = evaluate_head(
            collect_head_arrays(train_rows, gold_by_id, head),
            collect_head_arrays(dev_rows, gold_by_id, head),
        )
    key_threshold = heads["key"]["token"]["train_calibration"][
        "neural_threshold"
    ]
    complete_threshold = heads["complete"]["token"]["train_calibration"][
        "neural_threshold"
    ]
    fusion_alpha = float(protocol["method"]["prior_fusion_alpha"])
    report = {
        "schema_version": "clir-dual-prior-gate-cell-evaluation-v1",
        "evidence_tier": protocol["evidence_tier"],
        "cell": args.cell,
        "seed": args.seed,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "gold_labels_sha256": file_sha256(gold_path),
        "train_scored": str(train_scored_path),
        "train_scored_sha256": file_sha256(train_scored_path),
        "dev_scored": str(dev_scored_path),
        "dev_scored_sha256": file_sha256(dev_scored_path),
        "heads": heads,
        "correctness": {
            "train": correctness_metrics(train_rows),
            "dev": correctness_metrics(dev_rows),
        },
        "head_separation": {
            "train": separation_metrics(
                train_rows,
                key_threshold=key_threshold,
                complete_threshold=complete_threshold,
            ),
            "dev": separation_metrics(
                dev_rows,
                key_threshold=key_threshold,
                complete_threshold=complete_threshold,
            ),
        },
        "prior_collaboration": {
            "train": prior_collaboration_metrics(train_rows),
            "dev": prior_collaboration_metrics(dev_rows),
        },
        "gate_integration": {
            "train": gate_integration_metrics(
                train_rows, fusion_alpha=fusion_alpha
            ),
            "dev": gate_integration_metrics(dev_rows, fusion_alpha=fusion_alpha),
        },
        "position_only_baseline_included": True,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(args.output_json.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
