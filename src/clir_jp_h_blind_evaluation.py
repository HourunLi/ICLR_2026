"""Frozen post-unseal metrics for the JP-to-H mixed-domain validation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .clir_localization_evaluation import binary_average_precision


POSITIVE_STATUSES = frozenset({"contradicted", "unsupported"})
NEGATIVE_STATUSES = frozenset({"supported", "non_claim"})
MASKED_STATUSES = frozenset({"uncertain"})
KNOWN_STATUSES = POSITIVE_STATUSES | NEGATIVE_STATUSES | MASKED_STATUSES


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty score span")
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("Score span contains a non-finite value")
    return sum(normalized) / len(normalized)


def derive_sparse_claim_targets(
    label: Mapping[str, Any], *, token_count: int
) -> dict[str, Any]:
    if token_count <= 0:
        raise ValueError("Sparse claim targets require a positive token count")
    assignments: list[int | None] = [None] * token_count
    claims: list[dict[str, int]] = []
    reviews = label.get("claim_reviews")
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("Mapped label requires non-empty claim_reviews")
    for claim in reviews:
        status = str(claim.get("status"))
        if status not in KNOWN_STATUSES:
            raise ValueError(f"Unknown claim status {status!r}")
        start = claim.get("token_start")
        end = claim.get("token_end_exclusive")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end <= token_count
        ):
            raise ValueError("Claim span is outside the frozen token sequence")
        if status in MASKED_STATUSES:
            continue
        target = int(status in POSITIVE_STATUSES)
        for position in range(start, end):
            prior = assignments[position]
            if prior is not None and prior != target:
                raise ValueError("Overlapping claim spans assign conflicting targets")
            assignments[position] = target
        claims.append(
            {
                "claim_index": int(claim["claim_index"]),
                "token_start": start,
                "token_end_exclusive": end,
                "target": target,
            }
        )
    positions = [index for index, value in enumerate(assignments) if value is not None]
    targets = [int(assignments[index]) for index in positions]
    if not positions:
        raise ValueError("Label has no explicitly supervised claim tokens")
    positive_positions = [
        position for position, target in zip(positions, targets) if target == 1
    ]
    status = str(label.get("path_status"))
    if status == "hallucinated":
        if not positive_positions:
            raise ValueError("Hallucinated path has no explicit positive token")
        if int(label.get("hallucination_onset", -1)) != min(positive_positions):
            raise ValueError("Mapped onset differs from first explicit positive token")
    elif status == "clean":
        if positive_positions or int(label.get("hallucination_onset", -1)) != -1:
            raise ValueError("Clean path has a positive target or nonnegative onset")
    elif status == "uncertain":
        raise ValueError("Common uncertain rows are excluded before frozen evaluation")
    else:
        raise ValueError("Unknown path status")
    return {
        "positions": positions,
        "targets": targets,
        "claims": claims,
        "supervised_tokens": len(positions),
        "positive_tokens": sum(targets),
        "negative_tokens": len(targets) - sum(targets),
        "positive_claims": sum(claim["target"] for claim in claims),
        "negative_claims": len(claims) - sum(claim["target"] for claim in claims),
    }


def strongest_position(
    absolute_ap: float | None, normalized_ap: float | None
) -> tuple[str | None, float | None]:
    available = [
        ("absolute", absolute_ap),
        ("normalized", normalized_ap),
    ]
    finite = [(name, value) for name, value in available if value is not None]
    if not finite:
        return None, None
    # Stable order deliberately chooses absolute if the two AP values tie.
    name, value = max(finite, key=lambda pair: pair[1])
    return name, float(value)


def seed_gate(
    *,
    raw_token_ap: float | None,
    smoothed_token_ap: float | None,
    position_token_ap: float | None,
    raw_claim_ap: float | None,
    smoothed_claim_ap: float | None,
    position_claim_ap: float | None,
) -> dict[str, Any]:
    values = (
        raw_token_ap,
        smoothed_token_ap,
        position_token_ap,
        raw_claim_ap,
        smoothed_claim_ap,
        position_claim_ap,
    )
    if any(value is None for value in values):
        return {
            "advantages": None,
            "mean_four_advantages": None,
            "all_four_strictly_positive": False,
            "passed": False,
            "reason": "metric_undefined",
        }
    advantages = {
        "smoothed_token_minus_raw": float(smoothed_token_ap - raw_token_ap),
        "smoothed_token_minus_position": float(
            smoothed_token_ap - position_token_ap
        ),
        "smoothed_claim_minus_raw": float(smoothed_claim_ap - raw_claim_ap),
        "smoothed_claim_minus_position": float(
            smoothed_claim_ap - position_claim_ap
        ),
    }
    average = sum(advantages.values()) / 4.0
    all_positive = all(value > 0.0 for value in advantages.values())
    return {
        "advantages": advantages,
        "mean_four_advantages": average,
        "all_four_strictly_positive": all_positive,
        "passed": all_positive and average > 0.0,
        "reason": "passed" if all_positive and average > 0.0 else "strict_gate_failed",
    }


def evaluate_rows(
    labels: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    domains: Sequence[str] | None = None,
    include_domain: str | None = None,
) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("Evaluation labels and predictions have different row counts")
    if domains is not None and len(domains) != len(labels):
        raise ValueError("Domain vector differs from evaluation row count")

    token_labels: list[int] = []
    claim_labels: list[int] = []
    absolute_token_scores: list[float] = []
    normalized_token_scores: list[float] = []
    absolute_claim_scores: list[float] = []
    normalized_claim_scores: list[float] = []
    raw_token_scores = {int(seed): [] for seed in seeds}
    smooth_token_scores = {int(seed): [] for seed in seeds}
    raw_claim_scores = {int(seed): [] for seed in seeds}
    smooth_claim_scores = {int(seed): [] for seed in seeds}
    row_count = 0

    for row_index, (label, prediction) in enumerate(zip(labels, predictions)):
        if include_domain is not None:
            if domains is None or domains[row_index] != include_domain:
                continue
        if label["item_id"] != prediction["item_id"]:
            raise ValueError("Evaluation item order or identity drifted")
        if label["query_id"] != prediction["query_id"]:
            raise ValueError("Evaluation query identity drifted")
        if label["output_token_ids_sha256"] != prediction[
            "output_token_ids_sha256"
        ]:
            raise ValueError("Evaluation output-token identity drifted")
        token_count = int(prediction["token_count"])
        if token_count != len(prediction["output_token_ids"]):
            raise ValueError("Prediction token count drifted")
        target = derive_sparse_claim_targets(label, token_count=token_count)
        positions = target["positions"]
        token_labels.extend(target["targets"])
        claims = target["claims"]
        claim_labels.extend(int(claim["target"]) for claim in claims)

        absolute = [float(value) for value in prediction["absolute_position_scores"]]
        normalized = [float(value) for value in prediction["normalized_position_scores"]]
        if len(absolute) != token_count or len(normalized) != token_count:
            raise ValueError("Position baseline is not exact-token aligned")
        absolute_token_scores.extend(absolute[position] for position in positions)
        normalized_token_scores.extend(normalized[position] for position in positions)
        for claim in claims:
            start = claim["token_start"]
            end = claim["token_end_exclusive"]
            absolute_claim_scores.append(mean(absolute[start:end]))
            normalized_claim_scores.append(mean(normalized[start:end]))

        seed_rows = {
            int(seed_row["seed"]): seed_row
            for seed_row in prediction["seed_predictions"]
        }
        if list(seed_rows) != [int(seed) for seed in seeds]:
            raise ValueError("Frozen prediction seed order drifted")
        for seed in seeds:
            seed = int(seed)
            seed_row = seed_rows[seed]
            raw = [float(value) for value in seed_row["raw_probabilities"]]
            smooth = [
                float(value) for value in seed_row["smoothed_probabilities"]
            ]
            if len(raw) != token_count or len(smooth) != token_count:
                raise ValueError("Seed prediction is not exact-token aligned")
            raw_token_scores[seed].extend(raw[position] for position in positions)
            smooth_token_scores[seed].extend(
                smooth[position] for position in positions
            )
            for claim in claims:
                start = claim["token_start"]
                end = claim["token_end_exclusive"]
                raw_claim_scores[seed].append(mean(raw[start:end]))
                smooth_claim_scores[seed].append(mean(smooth[start:end]))
        row_count += 1

    if row_count == 0:
        raise ValueError("Evaluation subset has no rows")
    absolute_token_ap = binary_average_precision(token_labels, absolute_token_scores)
    normalized_token_ap = binary_average_precision(token_labels, normalized_token_scores)
    absolute_claim_ap = binary_average_precision(claim_labels, absolute_claim_scores)
    normalized_claim_ap = binary_average_precision(claim_labels, normalized_claim_scores)
    token_position_name, token_position_ap = strongest_position(
        absolute_token_ap, normalized_token_ap
    )
    claim_position_name, claim_position_ap = strongest_position(
        absolute_claim_ap, normalized_claim_ap
    )

    seed_results: dict[str, Any] = {}
    passing = 0
    for seed in seeds:
        seed = int(seed)
        raw_token_ap = binary_average_precision(
            token_labels, raw_token_scores[seed]
        )
        smoothed_token_ap = binary_average_precision(
            token_labels, smooth_token_scores[seed]
        )
        raw_claim_ap = binary_average_precision(
            claim_labels, raw_claim_scores[seed]
        )
        smoothed_claim_ap = binary_average_precision(
            claim_labels, smooth_claim_scores[seed]
        )
        gate = seed_gate(
            raw_token_ap=raw_token_ap,
            smoothed_token_ap=smoothed_token_ap,
            position_token_ap=token_position_ap,
            raw_claim_ap=raw_claim_ap,
            smoothed_claim_ap=smoothed_claim_ap,
            position_claim_ap=claim_position_ap,
        )
        passing += int(gate["passed"])
        seed_results[str(seed)] = {
            "token_average_precision": {
                "raw": raw_token_ap,
                "smoothed": smoothed_token_ap,
            },
            "claim_average_precision": {
                "raw": raw_claim_ap,
                "smoothed": smoothed_claim_ap,
            },
            "gate": gate,
        }
    return {
        "rows": row_count,
        "token_samples": len(token_labels),
        "positive_tokens": sum(token_labels),
        "negative_tokens": len(token_labels) - sum(token_labels),
        "claim_samples": len(claim_labels),
        "positive_claims": sum(claim_labels),
        "negative_claims": len(claim_labels) - sum(claim_labels),
        "position_baseline": {
            "token_average_precision": {
                "absolute": absolute_token_ap,
                "normalized": normalized_token_ap,
                "strongest_name": token_position_name,
                "strongest": token_position_ap,
            },
            "claim_average_precision": {
                "absolute": absolute_claim_ap,
                "normalized": normalized_claim_ap,
                "strongest_name": claim_position_name,
                "strongest": claim_position_ap,
            },
        },
        "seeds": seed_results,
        "passing_seeds": passing,
    }


__all__ = [
    "derive_sparse_claim_targets",
    "evaluate_rows",
    "seed_gate",
    "strongest_position",
]
