#!/usr/bin/env python3
"""Compare repaired Route-A pair geometry at init and after matched training.

This is an in-training-set pipeline diagnostic.  It must not be interpreted as
held-out evidence that consistency generalizes to unseen equivalence pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import (  # noqa: E402
    CLIRTrajectoryDataset,
    clir_collate,
    move_batch_to_device,
)
from src.consistency_localized_reward import (  # noqa: E402
    RewardConfig,
    build_reward_model,
)


DEFAULT_MANIFEST = (
    ROOT
    / "run_artifacts/on_policy_pilot0_v1a/data_v1/train_mixed_precomputed_v1.jsonl"
)
DEFAULT_BASELINE = (
    ROOT / "run_artifacts/on_policy_pilot0_v1a/train_baseline_v1/model.pt"
)
DEFAULT_CONSISTENCY = (
    ROOT / "run_artifacts/on_policy_pilot0_v1a/train_consistency_v1/model.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "run_artifacts/on_policy_pilot0_v1a/representation_diagnostic_v1/report.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "mean": mean(ordered),
        "p75": quantile(0.75),
        "max": ordered[-1],
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().unsqueeze(0), right.float().unsqueeze(0)
        ).item()
    )


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--consistency", type=Path, default=DEFAULT_CONSISTENCY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    manifest = args.manifest.resolve()
    baseline_path = args.baseline.resolve()
    consistency_path = args.consistency.resolve()
    output = args.output.resolve()
    if output in {manifest, baseline_path, consistency_path}:
        raise ValueError("Diagnostic output must not overwrite an input")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    dataset = CLIRTrajectoryDataset(
        manifest,
        check_finite=False,
        require_correctness=True,
        hidden_state_source="precomputed",
    )
    selected_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if row.get("semantic_id") is not None and row.get("style_id") is not None
    ]
    selected_rows = [dataset.rows[index] for index in selected_indices]
    if len(selected_rows) != 54:
        raise ValueError(f"Expected 54 consistency rows, found {len(selected_rows)}")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    styles: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected_rows:
        groups.setdefault(str(row["semantic_id"]), []).append(row)
        styles.setdefault(str(row["style_id"]), []).append(row)
    if len(groups) != 27 or any(len(group) != 2 for group in groups.values()):
        raise ValueError("Expected exactly 27 two-row semantic groups")
    if sorted(len(rows) for rows in styles.values()) != [27, 27]:
        raise ValueError("Expected two styles with 27 rows each")

    baseline_checkpoint = load_checkpoint(baseline_path)
    consistency_checkpoint = load_checkpoint(consistency_path)
    baseline_config = RewardConfig(**baseline_checkpoint["config"])
    consistency_config = RewardConfig(**consistency_checkpoint["config"])
    architecture_keys = [
        key
        for key in baseline_checkpoint["config"]
        if key
        not in {
            "consistency_weight",
            "negative_consistency_weight",
            "score_consistency_weight",
        }
    ]
    architecture_mismatches = {
        key: {
            "baseline": baseline_checkpoint["config"][key],
            "consistency": consistency_checkpoint["config"][key],
        }
        for key in architecture_keys
        if baseline_checkpoint["config"][key]
        != consistency_checkpoint["config"][key]
    }
    if architecture_mismatches:
        raise ValueError(
            f"Matched checkpoints differ outside consistency weights: {architecture_mismatches}"
        )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    models = {
        "random_init_seed42": build_reward_model(baseline_config),
        "baseline": build_reward_model(baseline_config),
        "consistency": build_reward_model(consistency_config),
    }
    models["baseline"].load_state_dict(baseline_checkpoint["state_dict"])
    models["consistency"].load_state_dict(consistency_checkpoint["state_dict"])
    for model in models.values():
        model.to(device).eval()

    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    features: dict[str, dict[str, dict[str, torch.Tensor | float]]] = {
        name: {} for name in models
    }
    raw_features: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for batch in loader:
            ids = [str(value) for value in batch["ids"]]
            batch = move_batch_to_device(batch, device)
            for key in ("hidden_states", "condition_states", "condition_embedding"):
                if key in batch:
                    batch[key] = batch[key].float()
            mask = batch["mask"].bool()
            for row_index, row_id in enumerate(ids):
                raw_features[row_id] = (
                    batch["hidden_states"][row_index][mask[row_index]]
                    .mean(dim=0)
                    .detach()
                    .cpu()
                )
            for model_name, model in models.items():
                encoded, _ = model.input_encoder(batch["hidden_states"])
                values = model(
                    batch["hidden_states"],
                    mask=batch["mask"],
                    condition_states=batch.get("condition_states"),
                    condition_mask=batch.get("condition_mask"),
                    condition_embedding=batch.get("condition_embedding"),
                    condition_embedding_mask=batch.get("condition_embedding_mask"),
                )
                for row_index, row_id in enumerate(ids):
                    features[model_name][row_id] = {
                        "encoder_mean": (
                            encoded[row_index][mask[row_index]]
                            .mean(dim=0)
                            .detach()
                            .cpu()
                        ),
                        "conditioned_pooled": values["pooled_features"][row_index]
                        .detach()
                        .cpu(),
                        "projected_representation": values["representations"][row_index]
                        .detach()
                        .cpu(),
                        "score": float(values["scores"][row_index].cpu()),
                    }

    positive_pairs = [
        (str(group[0]["id"]), str(group[1]["id"]))
        for group in groups.values()
    ]
    negative_pairs = []
    for style_rows in styles.values():
        ordered = sorted(style_rows, key=lambda row: str(row["semantic_id"]))
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                negative_pairs.append(
                    (str(ordered[left_index]["id"]), str(ordered[right_index]["id"]))
                )
    if len(positive_pairs) != 27 or len(negative_pairs) != 702:
        raise ValueError("Pair count drifted")

    distributions: dict[str, Any] = {}
    raw_positive = [cosine(raw_features[a], raw_features[b]) for a, b in positive_pairs]
    raw_negative = [cosine(raw_features[a], raw_features[b]) for a, b in negative_pairs]
    distributions["phi_raw_mean"] = {
        "same_semantic_different_style_cosine": summarize(raw_positive),
        "different_semantic_same_style_cosine": summarize(raw_negative),
        "mean_cosine_gap": mean(raw_positive) - mean(raw_negative),
    }

    pair_rows: list[dict[str, Any]] = []
    for model_name, model_features in features.items():
        model_report: dict[str, Any] = {}
        for feature_name in (
            "encoder_mean",
            "conditioned_pooled",
            "projected_representation",
        ):
            positive = [
                cosine(
                    model_features[left][feature_name],
                    model_features[right][feature_name],
                )
                for left, right in positive_pairs
            ]
            negative = [
                cosine(
                    model_features[left][feature_name],
                    model_features[right][feature_name],
                )
                for left, right in negative_pairs
            ]
            model_report[feature_name] = {
                "same_semantic_different_style_cosine": summarize(positive),
                "different_semantic_same_style_cosine": summarize(negative),
                "mean_cosine_gap": mean(positive) - mean(negative),
            }
        score_deltas = [
            abs(
                float(model_features[left]["score"])
                - float(model_features[right]["score"])
            )
            for left, right in positive_pairs
        ]
        model_report["same_semantic_absolute_score_delta"] = summarize(score_deltas)
        distributions[model_name] = model_report

    for semantic_id, group in sorted(groups.items()):
        left = str(group[0]["id"])
        right = str(group[1]["id"])
        row: dict[str, Any] = {
            "semantic_id": semantic_id,
            "left_id": left,
            "right_id": right,
            "correctness": int(group[0]["correctness"]),
        }
        for model_name, model_features in features.items():
            row[model_name] = {
                "projected_cosine": cosine(
                    model_features[left]["projected_representation"],
                    model_features[right]["projected_representation"],
                ),
                "absolute_score_delta": abs(
                    float(model_features[left]["score"])
                    - float(model_features[right]["score"])
                ),
            }
        pair_rows.append(row)

    report = {
        "schema_version": "clir-on-policy-representation-diagnostic-v1a",
        "evidence_tier": "pipeline_pilot",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "rows": len(selected_rows),
        "semantic_groups": len(groups),
        "styles": {key: len(value) for key, value in sorted(styles.items())},
        "positive_pair_count": len(positive_pairs),
        "potential_negative_pair_count": len(negative_pairs),
        "checkpoints": {
            "baseline": {
                "path": str(baseline_path),
                "sha256": file_sha256(baseline_path),
            },
            "consistency": {
                "path": str(consistency_path),
                "sha256": file_sha256(consistency_path),
            },
        },
        "random_initialization": {
            "seed": args.seed,
            "contract": "same model construction seed as both matched training cells",
        },
        "distributions": distributions,
        "pairs": pair_rows,
        "interpretation_allowed": "in_training_relation_geometry_diagnostic_only",
        "formal_mechanism_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({key: report[key] for key in (
        "schema_version",
        "rows",
        "semantic_groups",
        "positive_pair_count",
        "potential_negative_pair_count",
        "distributions",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
