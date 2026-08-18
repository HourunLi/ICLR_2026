#!/usr/bin/env python3
"""Calibrate path/token/onset thresholds on dense train and apply once to dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import read_jsonl  # noqa: E402
from src.clir_localization_evaluation import (  # noqa: E402
    evaluate_localization_rows,
    select_binary_threshold,
    select_onset_threshold,
)
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


def token_targets(rows: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        probabilities = [float(value) for value in row["clir_token_hallucination_probs"]]
        onset = int(row["hallucination_onset"])
        labels.extend(
            int(onset >= 0 and position >= onset)
            for position in range(len(probabilities))
        )
        scores.extend(probabilities)
    return labels, scores


def constant_onset_baselines(
    train: Sequence[Mapping[str, Any]],
    dev: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_positive = [row for row in train if int(row["hallucination_onset"]) >= 0]
    dev_positive = [row for row in dev if int(row["hallucination_onset"]) >= 0]
    absolute_median = float(median([int(row["hallucination_onset"]) for row in train_positive]))
    normalized_median = float(
        median(
            int(row["hallucination_onset"])
            / max(len(row["clir_token_hallucination_probs"]) - 1, 1)
            for row in train_positive
        )
    )

    def evaluate(mode: str) -> dict[str, Any]:
        errors: list[int] = []
        for row in dev_positive:
            length = len(row["clir_token_hallucination_probs"])
            predicted = (
                min(round(absolute_median), length - 1)
                if mode == "absolute"
                else round(normalized_median * (length - 1))
            )
            errors.append(abs(predicted - int(row["hallucination_onset"])))
        return {
            "positive_rows": len(errors),
            "mean_absolute_error": sum(errors) / len(errors) if errors else None,
            "median_absolute_error": median(errors) if errors else None,
            "within_5_rate": (
                sum(error <= 5 for error in errors) / len(errors) if errors else None
            ),
        }

    return {
        "selected_on_train": {
            "absolute_token_median": absolute_median,
            "normalized_position_median": normalized_median,
        },
        "dev_absolute_median": evaluate("absolute"),
        "dev_normalized_median": evaluate("normalized"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=["h0", "h1", "h2", "h3"])
    parser.add_argument("--train-scored", required=True)
    parser.add_argument("--dev-scored", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-train-sha256", default=None)
    parser.add_argument("--expected-dev-sha256", default=None)
    parser.add_argument("--negative-tail-margin", type=float, default=0.5)
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Threshold calibration requires a clean committed worktree")
    train_path = Path(args.train_scored).resolve()
    dev_path = Path(args.dev_scored).resolve()
    output_path = Path(args.output_json).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite threshold calibration: {output_path}")
    train_sha256 = file_sha256(train_path)
    dev_sha256 = file_sha256(dev_path)
    if args.expected_train_sha256 and train_sha256 != args.expected_train_sha256:
        raise ValueError("Train scored SHA256 drifted")
    if args.expected_dev_sha256 and dev_sha256 != args.expected_dev_sha256:
        raise ValueError("Dev scored SHA256 drifted")
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)

    path_labels = [int(row["path_hallucinated"]) for row in train]
    path_scores = [
        -float(row["clir_path_no_hallucination_log_prob"])
        for row in train
    ]
    token_labels, token_scores = token_targets(train)
    path_calibration = select_binary_threshold(
        path_labels,
        path_scores,
        objective="balanced_accuracy",
    )
    token_calibration = select_binary_threshold(
        token_labels,
        token_scores,
        objective="f1",
    )
    onset_calibration = select_onset_threshold(train)
    calibrated_kwargs: dict[str, Any] = {
        "path_log_threshold": path_calibration["threshold"],
        "token_threshold": token_calibration["threshold"],
        "onset_threshold": onset_calibration["threshold"],
        "negative_tail_margin": args.negative_tail_margin,
    }
    report = {
        "schema_version": "clir-hallucination-threshold-calibration-v1",
        "evidence_tier": "posthoc_pipeline_diagnostic",
        "cell": args.cell,
        "selection_data": "dense_train_only",
        "development_labels_used_for_threshold_selection": False,
        "train_scored": str(train_path),
        "train_scored_sha256": train_sha256,
        "dev_scored": str(dev_path),
        "dev_scored_sha256": dev_sha256,
        "selected_thresholds": {
            "path_negative_log_survival": path_calibration,
            "contaminated_tail_token_probability": token_calibration,
            "onset_token_probability": onset_calibration,
        },
        "train_fixed_0_5": evaluate_localization_rows(
            train,
            negative_tail_margin=args.negative_tail_margin,
        ),
        "train_calibrated": evaluate_localization_rows(train, **calibrated_kwargs),
        "dev_fixed_0_5": evaluate_localization_rows(
            dev,
            negative_tail_margin=args.negative_tail_margin,
        ),
        "dev_calibrated_once": evaluate_localization_rows(dev, **calibrated_kwargs),
        "constant_onset_shortcut_baselines": constant_onset_baselines(train, dev),
        "code": code,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
