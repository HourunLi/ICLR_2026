#!/usr/bin/env python3
"""Evaluate in-training Route-A relation geometry for one joint pilot cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Mapping, Sequence

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
from src.clir_joint_training import (  # noqa: E402
    reward_config_from_protocol,
    validate_joint_protocol,
)
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json  # noqa: E402
from src.consistency_localized_reward import build_reward_model  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/joint_training_pilot_v1/training_protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
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
        "minimum": ordered[0],
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "mean": mean(ordered),
        "p75": quantile(0.75),
        "maximum": ordered[-1],
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float()[None, :], right.float()[None, :]).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_joint_protocol(protocol)
    if args.cell not in protocol["cells"]:
        raise ValueError(f"Unknown joint-training cell {args.cell!r}")
    if args.seed not in [int(value) for value in protocol["matched_training"]["seeds"]]:
        raise ValueError("Seed is not frozen in the joint-training protocol")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid diagnostic loader configuration")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite consistency diagnostic: {output}")
    manifest_spec = protocol["manifests"]["train"]
    manifest = resolve(manifest_spec["path"])
    if file_sha256(manifest) != manifest_spec["sha256"]:
        raise ValueError("Joint train manifest hash drifted")
    model_path = args.model.resolve()
    checkpoint_sha = file_sha256(model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    expected_config = reward_config_from_protocol(protocol, args.cell)
    if checkpoint.get("config") != expected_config.__dict__:
        raise ValueError("Checkpoint config differs from the frozen joint cell")
    expected_protocol_state = {
        "path": str(protocol_path),
        "sha256": file_sha256(protocol_path),
        "schema_version": protocol["schema_version"],
    }
    if checkpoint.get("experiment_protocol") != expected_protocol_state:
        raise ValueError("Checkpoint experiment protocol provenance drifted")

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
    groups: dict[str, list[Mapping[str, Any]]] = {}
    styles: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected_rows:
        groups.setdefault(str(row["semantic_id"]), []).append(row)
        styles.setdefault(str(row["style_id"]), []).append(row)
    if len(selected_rows) != 54 or len(groups) != 27:
        raise ValueError("Expected the frozen 54 rows / 27 semantic groups")
    if any(
        len(rows) != 2
        or {str(row["style_id"]) for row in rows}
        != {"native_compact", "native_expanded"}
        for rows in groups.values()
    ):
        raise ValueError("Consistency semantic groups are incomplete")
    if sorted(len(rows) for rows in styles.values()) != [27, 27]:
        raise ValueError("Consistency style populations drifted")

    device = torch.device(args.device)
    model = build_reward_model(expected_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    features: dict[str, dict[str, torch.Tensor | float]] = {}
    with torch.inference_mode():
        for batch in loader:
            ids = [str(value) for value in batch["ids"]]
            batch = move_batch_to_device(batch, device)
            for key in ("hidden_states", "condition_states", "condition_embedding"):
                if key in batch:
                    batch[key] = batch[key].float()
            outputs = model(
                batch["hidden_states"],
                mask=batch["mask"],
                condition_states=batch.get("condition_states"),
                condition_mask=batch.get("condition_mask"),
                condition_embedding=batch.get("condition_embedding"),
                condition_embedding_mask=batch.get("condition_embedding_mask"),
            )
            for position, row_id in enumerate(ids):
                features[row_id] = {
                    "representation": outputs["representations"][position].detach().cpu(),
                    "score": float(outputs["scores"][position].detach().cpu()),
                }

    positive_pairs = [
        (str(rows[0]["id"]), str(rows[1]["id"])) for rows in groups.values()
    ]
    negative_pairs: list[tuple[str, str]] = []
    for style_rows in styles.values():
        ordered = sorted(style_rows, key=lambda row: str(row["semantic_id"]))
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                negative_pairs.append(
                    (str(ordered[left]["id"]), str(ordered[right]["id"]))
                )
    if len(positive_pairs) != 27 or len(negative_pairs) != 702:
        raise ValueError("Consistency pair counts drifted")
    positive_cosines = [
        cosine(
            features[left]["representation"],  # type: ignore[arg-type]
            features[right]["representation"],  # type: ignore[arg-type]
        )
        for left, right in positive_pairs
    ]
    negative_cosines = [
        cosine(
            features[left]["representation"],  # type: ignore[arg-type]
            features[right]["representation"],  # type: ignore[arg-type]
        )
        for left, right in negative_pairs
    ]
    score_deltas = [
        abs(float(features[left]["score"]) - float(features[right]["score"]))
        for left, right in positive_pairs
    ]
    positive_summary = summarize(positive_cosines)
    negative_summary = summarize(negative_cosines)
    gap = float(positive_summary["mean"]) - float(negative_summary["mean"])
    report = {
        "schema_version": "clir-joint-consistency-diagnostic-v1",
        "evidence_tier": protocol["evidence_tier"],
        "cell": args.cell,
        "seed": args.seed,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": file_sha256(manifest),
        "checkpoint": str(model_path),
        "checkpoint_sha256": checkpoint_sha,
        "rows": len(selected_rows),
        "semantic_groups": len(groups),
        "positive_pair_count": len(positive_pairs),
        "potential_negative_pair_count": len(negative_pairs),
        "projected_representation": {
            "same_semantic_different_style_cosine": positive_summary,
            "different_semantic_same_style_cosine": negative_summary,
            "mean_cosine_gap": gap,
            "gate_passed": gap > 0.0,
        },
        "same_semantic_absolute_score_delta": summarize(score_deltas),
        "interpretation_allowed": "in_training_relation_geometry_diagnostic_only",
        "held_out_consistency_evidence": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
