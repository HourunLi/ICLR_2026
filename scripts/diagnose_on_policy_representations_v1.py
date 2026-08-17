#!/usr/bin/env python3
"""Locate where Route A native-pair representation similarity saturates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

from src.clir_data import read_jsonl
from src.consistency_localized_reward import RewardConfig, build_reward_model
from src.clir_real_data import forward_all_layer_features


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "run_artifacts/on_policy_pilot0_v1/published_silver_v1/train_online.jsonl"
SOURCE = ROOT / "run_artifacts/stage1b_v3/labels/train_extracted.v5.jsonl"
CHECKPOINTS = {
    "no_consistency": ROOT / "run_artifacts/on_policy_pilot0_v1/train_no_consistency_v1/model.pt",
    "positive_only": ROOT / "run_artifacts/on_policy_pilot0_v1/train_positive_only_v1/model.pt",
    "negative_0.05": ROOT / "run_artifacts/on_policy_pilot0_v1/train_negative_005_v1/model.pt",
    "negative_0.20": ROOT / "run_artifacts/on_policy_pilot0_v1/train_negative_020_v1/model.pt",
    "negative_0.50": ROOT / "run_artifacts/on_policy_pilot0_v1/train_negative_050_v1/model.pt",
    "negative_0.50_score_1.00": ROOT / "run_artifacts/on_policy_pilot0_v1/train_negative_050_score_100_v1/model.pt",
    "consistency": ROOT / "run_artifacts/on_policy_pilot0_v1/train_consistency_v1/model.pt",
}
OUTPUT = ROOT / "run_artifacts/on_policy_pilot0_v1/representation_diagnostic_v1/report.json"


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.float().unsqueeze(0), right.float().unsqueeze(0)).item())


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight
    return {
        "count": len(ordered), "min": ordered[0], "p25": quantile(0.25),
        "median": quantile(0.5), "mean": mean(ordered), "p75": quantile(0.75),
        "max": ordered[-1],
    }


def pair_distributions(features: dict[str, dict[str, torch.Tensor]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_semantic: dict[str, list[dict[str, Any]]] = {}
    by_style: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_semantic.setdefault(row["semantic_id"], []).append(row)
        by_style.setdefault(row["style_id"], []).append(row)
    positive = []
    for group in by_semantic.values():
        if len(group) != 2:
            raise ValueError("Every published semantic group must contain two views")
        positive.append((group[0]["id"], group[1]["id"]))
    cross = []
    for style in sorted(by_style):
        ordered = sorted(by_style[style], key=lambda row: (row["query_id"], row["id"]))
        cross.extend((ordered[index]["id"], ordered[index + 1]["id"]) for index in range(0, len(ordered) - 1, 2))
    output = {}
    names = next(iter(features.values())).keys()
    for name in names:
        positive_values = [cosine(features[a][name], features[b][name]) for a, b in positive]
        cross_values = [cosine(features[a][name], features[b][name]) for a, b in cross]
        output[name] = {
            "same_semantic_native_pair": summarize(positive_values),
            "different_query_same_view_role": summarize(cross_values),
            "mean_gap": mean(positive_values) - mean(cross_values),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default="/prodcpfs/user/panzhixin/hf_cache")
    args = parser.parse_args()
    rows = read_jsonl(PUBLISHED)
    checkpoints = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in CHECKPOINTS.items()}
    config = RewardConfig(**checkpoints["no_consistency"]["config"])
    models = {}
    torch.manual_seed(42)
    models["random_init_seed42"] = build_reward_model(config).to(args.device).eval()
    for name, checkpoint in checkpoints.items():
        model = build_reward_model(RewardConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        models[name] = model.to(args.device).eval()

    from transformers import AutoModelForCausalLM
    phi = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        revision="2fe192450127e6a83f7441aef6e3ca586c338b77",
        cache_dir=args.cache_dir, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    phi.requires_grad_(False)

    def phi_features(token_ids: list[int]) -> torch.Tensor:
        values, layer_count, per_layer_size = forward_all_layer_features(
            phi, token_ids, device=args.device,
        )
        if (layer_count, per_layer_size, values.shape[-1]) != (33, 3072, 101376):
            raise ValueError("Phi feature layout drifted")
        return values
    features: dict[str, dict[str, dict[str, torch.Tensor]]] = {name: {} for name in models}
    raw: dict[str, dict[str, torch.Tensor]] = {}
    score_rows = []
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            all_states = phi_features([*row["prompt_token_ids"], *row["output_token_ids"]])
            trajectory = all_states[len(row["prompt_token_ids"]):].unsqueeze(0)
            condition = phi_features(row["prompt_token_ids"]).unsqueeze(0)
            mask = torch.ones((1, trajectory.shape[1]), device=args.device)
            condition_mask = torch.ones((1, condition.shape[1]), device=args.device)
            raw[row["id"]] = {"phi_raw_mean": trajectory[0].float().mean(dim=0).cpu()}
            for model_name, model in models.items():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    encoded, _ = model.input_encoder(trajectory)
                    output = model(
                        trajectory, mask=mask, condition_states=condition,
                        condition_mask=condition_mask,
                    )
                features[model_name][row["id"]] = {
                    "encoder_mean": encoded[0].float().mean(dim=0).cpu(),
                    "conditioned_pooled": output["pooled_features"][0].float().cpu(),
                    "projected_representation": output["representations"][0].float().cpu(),
                }
                score_rows.append({"id": row["id"], "semantic_id": row["semantic_id"], "style_id": row["style_id"], "model": model_name, "score": float(output["scores"][0].float().item())})
            print(json.dumps({"row": index, "total": len(rows), "id": row["id"]}))

    distributions = {"phi_raw": pair_distributions(raw, rows)}
    distributions.update({name: pair_distributions(values, rows) for name, values in features.items()})
    score_by = {(row["model"], row["id"]): row["score"] for row in score_rows}
    score_deltas = {}
    by_semantic: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_semantic.setdefault(row["semantic_id"], []).append(row)
    for model_name in models:
        values = [abs(score_by[(model_name, group[0]["id"])] - score_by[(model_name, group[1]["id"])]) for group in by_semantic.values()]
        score_deltas[model_name] = summarize(values)
    report = {
        "schema_version": "clir-on-policy-representation-diagnostic-v1",
        "evidence_tier": "pipeline_pilot", "rows": len(rows), "semantic_groups": len(by_semantic),
        "published_manifest_sha256": hashlib.sha256(PUBLISHED.read_bytes()).hexdigest(),
        "checkpoint_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in CHECKPOINTS.items()},
        "distributions": distributions, "same_semantic_absolute_score_delta": score_deltas,
        "interpretation_allowed": "representation_geometry_diagnostic_only",
        "formal_mechanism_claim_allowed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
