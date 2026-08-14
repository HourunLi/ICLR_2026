"""Train the self-contained CLIR reward model with auditable Stage A state."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.clir_data import (
    CLIRTrajectoryDataset,
    SemanticGroupBatchSampler,
    clir_collate,
    move_batch_to_device,
)
from src.clir_real_data import file_sha256
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state
from src.consistency_localized_reward import (
    RewardConfig,
    build_reward_model,
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIR on pre-extracted hidden states.")
    parser.add_argument("--train_jsonl", required=True, help="Training JSONL file.")
    parser.add_argument("--feature_root", default=None, help="Base directory for training feature paths.")
    parser.add_argument("--val_jsonl", default=None, help="Explicit query-disjoint validation JSONL.")
    parser.add_argument("--val_feature_root", default=None, help="Base directory for validation feature paths.")
    parser.add_argument("--output_model", required=True, help="Atomic latest/full-state checkpoint path.")
    parser.add_argument("--resume_from", default=None, help="Full-state checkpoint to resume.")
    parser.add_argument("--metrics_jsonl", default=None)
    parser.add_argument("--run_json", default=None)
    parser.add_argument("--hidden_dim", type=int, required=True,
                        help="Raw input feature width; 101376 for frozen Phi all-layer features.")
    parser.add_argument("--model_variant", default="clir",
                        choices=["strict_swift", "encoded_swift", "clir"])
    parser.add_argument("--encoder_type", default="identity",
                        choices=["identity", "flat_linear", "layer_transformer"])
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--num_feature_layers", type=int, default=1)
    parser.add_argument("--per_layer_dim", type=int, default=None)
    parser.add_argument("--layer_encoder_dim", type=int, default=256)
    parser.add_argument("--layer_encoder_blocks", type=int, default=2)
    parser.add_argument("--layer_encoder_heads", type=int, default=8)
    parser.add_argument("--layer_pool_queries", type=int, default=4)
    parser.add_argument("--encoder_dropout", type=float, default=0.0)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--epochs", type=int, default=5, help="Total target epochs, including resumed epochs.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Clip the global gradient norm when > 0; zero disables clipping.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.0,
                        help="Legacy row split only; formal Stage A runs must use --val_jsonl.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp_dtype", default="none", choices=["none", "bfloat16"])
    parser.add_argument(
        "--skip_feature_finite_check",
        action="store_true",
        help="Use only after extraction marker/checksum validation; non-finite total loss still fails training.",
    )
    parser.add_argument("--group_by_semantic_id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior_phase_mode", default="alternate",
                        choices=["joint", "alternate", "key", "complete"])

    parser.add_argument("--final_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--score_consistency_weight", type=float, default=0.1)
    parser.add_argument("--hallucination_weight", type=float, default=1.0)
    parser.add_argument("--mil_weight", type=float, default=0.25)
    parser.add_argument("--token_reward_weight", type=float, default=0.5)
    parser.add_argument("--tail_weight", type=float, default=0.5)
    parser.add_argument("--pseudo_tail_weight", type=float, default=0.1)
    parser.add_argument("--progress_weight", type=float, default=0.25)
    parser.add_argument("--prior_weight", type=float, default=0.25)
    parser.add_argument("--negative_tail_margin", type=float, default=0.5)
    parser.add_argument("--pseudo_onset_threshold", type=float, default=0.5)
    parser.add_argument("--condition_attention_temperature", type=float, default=1.0)
    parser.add_argument("--progress_score_weight", type=float, default=0.5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def make_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        hidden_dim=args.hidden_dim,
        model_variant=args.model_variant,
        encoder_type=args.encoder_type,
        model_dim=args.model_dim,
        num_feature_layers=args.num_feature_layers,
        per_layer_dim=args.per_layer_dim,
        layer_encoder_dim=args.layer_encoder_dim,
        layer_encoder_blocks=args.layer_encoder_blocks,
        layer_encoder_heads=args.layer_encoder_heads,
        layer_pool_queries=args.layer_pool_queries,
        encoder_dropout=args.encoder_dropout,
        projection_dim=args.projection_dim,
        final_weight=args.final_weight,
        consistency_weight=args.consistency_weight,
        score_consistency_weight=args.score_consistency_weight,
        hallucination_weight=args.hallucination_weight,
        mil_weight=args.mil_weight,
        token_reward_weight=args.token_reward_weight,
        tail_weight=args.tail_weight,
        pseudo_tail_weight=args.pseudo_tail_weight,
        progress_weight=args.progress_weight,
        prior_weight=args.prior_weight,
        negative_tail_margin=args.negative_tail_margin,
        pseudo_onset_threshold=args.pseudo_onset_threshold,
        condition_attention_temperature=args.condition_attention_temperature,
        progress_score_weight=args.progress_score_weight,
    )


def validate_dataset_feature_contract(
    dataset: CLIRTrajectoryDataset,
    config: RewardConfig,
    split_name: str,
) -> None:
    """Cross-check CLI architecture values against extracted feature metadata."""

    metadata_rows = [row.get("feature_metadata") for row in dataset.rows]
    present = [metadata for metadata in metadata_rows if isinstance(metadata, Mapping)]
    if not present:
        return
    if len(present) != len(metadata_rows):
        raise ValueError(f"{split_name} mixes rows with and without feature_metadata")
    contracts = {
        (
            int(metadata.get("feature_dim", -1)),
            int(metadata.get("layer_count", -1)),
            int(metadata.get("per_layer_hidden_size", -1)),
        )
        for metadata in present
    }
    if len(contracts) != 1:
        raise ValueError(f"{split_name} has non-uniform feature metadata: {sorted(contracts)}")
    feature_dim, layer_count, per_layer_dim = next(iter(contracts))
    if feature_dim != config.hidden_dim:
        raise ValueError(
            f"{split_name} feature_dim={feature_dim} but --hidden_dim={config.hidden_dim}"
        )
    if config.encoder_type == "layer_transformer" and (
        layer_count != config.num_feature_layers
        or per_layer_dim != config.per_layer_dim
    ):
        raise ValueError(
            f"{split_name} layer contract is {layer_count} x {per_layer_dim}, but model "
            f"configuration is {config.num_feature_layers} x {config.per_layer_dim}"
        )


def _row_query_id(row: Mapping[str, Any], index: int) -> str:
    return str(
        row.get(
            "query_id",
            row.get("candidate_group_id", row.get("prompt_id", index)),
        )
    )


def split_indices(dataset: CLIRTrajectoryDataset, val_fraction: float, seed: int):
    indices = list(range(len(dataset)))
    if val_fraction <= 0.0:
        return indices, None

    query_to_indices: Dict[str, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        query_to_indices.setdefault(_row_query_id(row, index), []).append(index)
    query_ids = list(query_to_indices)
    if len(query_ids) < 2:
        raise ValueError("val_fraction requires at least two distinct query groups")

    val_query_count = max(1, int(len(query_ids) * val_fraction))
    val_query_count = min(val_query_count, len(query_ids) - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(query_ids), generator=generator).tolist()
    val_queries = {query_ids[position] for position in permutation[:val_query_count]}
    train_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if _row_query_id(row, index) not in val_queries
    ]
    val_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if _row_query_id(row, index) in val_queries
    ]
    return train_indices, val_indices


def make_loader(
    dataset: CLIRTrajectoryDataset,
    batch_size: int,
    indices: list[int],
    shuffle: bool,
    group_by_semantic_id: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if group_by_semantic_id and shuffle:
        sampler = SemanticGroupBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            seed=seed,
            indices=indices,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=clir_collate,
            **loader_kwargs,
        )
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=clir_collate,
        **loader_kwargs,
    )


def prior_phase_for_epoch(mode: str, epoch: int) -> str:
    if mode == "alternate":
        return "key" if epoch % 2 == 1 else "complete"
    return mode


def _component_counts(batch: Mapping[str, Any], losses: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    batch_size = int(batch["hidden_states"].shape[0])
    mask = batch["mask"].bool()
    counts: Dict[str, int] = {"total": batch_size}
    if "final" in losses:
        if "correctness_mask" in batch:
            counts["final"] = int(batch["correctness_mask"].sum().item())
        else:
            counts["final"] = int(batch.get("correctness", torch.empty(0)).numel())
    consistency_count = 0
    if "consistency_mask" in batch:
        valid = batch["consistency_mask"].bool()
        semantic = batch["semantic_ids"]
        style = batch["style_ids"]
        upper = torch.triu(torch.ones((batch_size, batch_size), dtype=torch.bool, device=mask.device), diagonal=1)
        pairs = upper & valid[:, None] & valid[None, :]
        consistency_count = int((pairs & ((semantic[:, None] == semantic[None, :]) | (style[:, None] == style[None, :]))).sum().item())
    for key in losses:
        if key.startswith("consistency_"):
            counts[key] = consistency_count
    onset_tokens = int((batch.get("onset_label_mask", torch.zeros(batch_size, dtype=torch.bool, device=mask.device))[:, None] & mask).sum().item())
    for key in losses:
        if key.startswith("localization_"):
            counts[key] = onset_tokens
    if "hallucination_mil" in losses:
        counts["hallucination_mil"] = int(batch.get("path_label_mask", torch.zeros(batch_size, dtype=torch.bool, device=mask.device)).sum().item())
    if "pseudo_tail" in losses:
        counts["pseudo_tail"] = int(batch.get("path_label_mask", torch.zeros(batch_size, dtype=torch.bool, device=mask.device)).sum().item())
    if "progress" in losses:
        counts["progress"] = int(batch.get("progress_mask", torch.zeros_like(mask)).sum().item())
    key_count = int(batch.get("key_prior_mask", torch.zeros_like(mask)).sum().item())
    complete_count = int(batch.get("complete_prior_mask", torch.zeros_like(mask)).sum().item())
    reconstruction_count = int(batch.get("complete_reconstruction_target_mask", torch.zeros(batch_size, dtype=torch.bool, device=mask.device)).sum().item())
    for key in losses:
        if key == "prior_key":
            counts[key] = key_count
        elif key == "prior_complete":
            counts[key] = complete_count
        elif key in {"prior_distill", "prior_gate", "prior_total"}:
            counts[key] = min(key_count, complete_count)
        elif key == "prior_reconstruction":
            counts[key] = reconstruction_count
    return counts


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    prior_phase: str,
    amp_dtype: str = "none",
    max_grad_norm: float = 0.0,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    applicable: Dict[str, int] = {}
    examples = 0
    batches = 0
    amp_enabled = amp_dtype == "bfloat16"
    if amp_enabled and device.type != "cuda":
        raise ValueError("--amp_dtype bfloat16 currently requires CUDA")

    grad_context = nullcontext() if training else torch.no_grad()
    with grad_context:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if not amp_enabled:
                parameter_dtype = next(model.parameters()).dtype
                for key in ("hidden_states", "condition_states", "condition_embedding"):
                    if key in batch:
                        batch[key] = batch[key].to(dtype=parameter_dtype)
            if training:
                optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
            with autocast:
                _, losses = model.training_step(batch, prior_phase=prior_phase)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite total loss in {prior_phase} phase")
            if training:
                losses["total"].backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            batch_size = int(batch["hidden_states"].shape[0])
            examples += batch_size
            batches += 1
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            for key, value in _component_counts(batch, losses).items():
                applicable[key] = applicable.get(key, 0) + value

    return {
        "losses": {key: value / max(examples, 1) for key, value in totals.items()},
        "applicable_counts": applicable,
        "examples": examples,
        "batches": batches,
    }


def _query_ids(dataset: CLIRTrajectoryDataset, indices: list[int] | None = None) -> set[str]:
    selected = indices if indices is not None else list(range(len(dataset.rows)))
    return {_row_query_id(dataset.rows[index], index) for index in selected}


def _rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.val_jsonl and args.val_fraction > 0:
        raise ValueError("Use explicit --val_jsonl or legacy --val_fraction, not both")
    if args.val_feature_root and not args.val_jsonl:
        raise ValueError("--val_feature_root requires --val_jsonl")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output_model).resolve()
    metrics_path = Path(args.metrics_jsonl).resolve() if args.metrics_jsonl else output.with_suffix(output.suffix + ".metrics.jsonl")
    run_path = Path(args.run_json).resolve() if args.run_json else output.with_suffix(output.suffix + ".run.json")

    train_dataset = CLIRTrajectoryDataset(
        args.train_jsonl,
        feature_root=args.feature_root,
        check_finite=not args.skip_feature_finite_check,
        require_correctness=True,
    )
    if args.val_jsonl:
        val_dataset = CLIRTrajectoryDataset(
            args.val_jsonl,
            feature_root=args.val_feature_root,
            check_finite=not args.skip_feature_finite_check,
            require_correctness=True,
        )
        train_indices = list(range(len(train_dataset)))
        val_indices = list(range(len(val_dataset)))
    else:
        val_dataset = train_dataset
        train_indices, val_indices = split_indices(train_dataset, args.val_fraction, args.seed)

    train_queries = _query_ids(train_dataset, train_indices)
    val_queries = _query_ids(val_dataset, val_indices) if val_indices is not None else set()
    overlap = train_queries & val_queries
    if overlap:
        raise ValueError(f"Train/validation query leakage detected: {sorted(overlap)[:5]}")

    config = make_config(args)
    validate_dataset_feature_contract(train_dataset, config, "train")
    if val_indices is not None:
        validate_dataset_feature_contract(val_dataset, config, "validation")

    train_loader = make_loader(
        train_dataset, args.batch_size, train_indices, shuffle=True,
        group_by_semantic_id=args.group_by_semantic_id, seed=args.seed,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    val_loader = (
        make_loader(val_dataset, args.batch_size, val_indices, shuffle=False,
                    group_by_semantic_id=False, seed=args.seed,
                    num_workers=args.num_workers, pin_memory=args.pin_memory,
                    persistent_workers=args.persistent_workers)
        if val_indices is not None else None
    )

    model = build_reward_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data_state = {
        "train_jsonl": str(Path(args.train_jsonl).resolve()),
        "train_sha256": file_sha256(args.train_jsonl),
        "train_rows": len(train_indices),
        "train_queries": len(train_queries),
        "val_jsonl": str(Path(args.val_jsonl).resolve()) if args.val_jsonl else None,
        "val_sha256": file_sha256(args.val_jsonl) if args.val_jsonl else None,
        "val_rows": len(val_indices) if val_indices is not None else 0,
        "val_queries": len(val_queries),
    }
    start_epoch = 0
    metric_rows: list[Dict[str, Any]] = []
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        if checkpoint.get("config") != config.__dict__:
            raise ValueError("Resume checkpoint model config differs from current CLI config")
        if checkpoint.get("data_state") != data_state:
            raise ValueError("Resume checkpoint data files/hashes differ from current run")
        if "optimizer_state_dict" not in checkpoint or "rng_state" not in checkpoint:
            raise ValueError("Resume checkpoint lacks full optimizer/RNG state")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["completed_epoch"])
        _restore_rng_state(checkpoint["rng_state"])
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                metric_rows = [json.loads(line) for line in handle if line.strip()]
        if len(metric_rows) != start_epoch:
            raise ValueError("Metrics history length does not match completed_epoch")
    elif metrics_path.exists() or run_path.exists() or output.exists():
        raise FileExistsError("Training outputs already exist; use --resume_from with the full checkpoint")

    if isinstance(getattr(train_loader, "batch_sampler", None), SemanticGroupBatchSampler):
        train_loader.batch_sampler.epoch = start_epoch
    if start_epoch >= args.epochs:
        raise ValueError(f"Checkpoint already completed {start_epoch} epochs; target is {args.epochs}")

    run_record: Dict[str, Any] = {
        "schema_version": "clir-training-run-v1",
        "status": "running",
        "output_model": str(output),
        "metrics_jsonl": str(metrics_path),
        "target_epochs": args.epochs,
        "completed_epoch": start_epoch,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "model_variant": config.model_variant,
        "encoder_type": config.encoder_type,
        "input_dim": config.hidden_dim,
        "model_dim": config.model_dim,
        "trainable_parameters": count_trainable_parameters(model),
        "data_state": data_state,
        "code": git_state(Path(__file__).resolve().parent),
    }
    atomic_write_json(run_path, run_record)
    print(
        f"model_variant={config.model_variant} encoder_type={config.encoder_type} "
        f"input_dim={config.hidden_dim} model_dim={config.model_dim} "
        f"trainable_parameters={count_trainable_parameters(model)} resume_epoch={start_epoch}"
    )

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            prior_phase = prior_phase_for_epoch(args.prior_phase_mode, epoch)
            train_metrics = run_epoch(
                model, train_loader, device, optimizer, prior_phase=prior_phase,
                amp_dtype=args.amp_dtype, max_grad_norm=args.max_grad_norm,
            )
            val_metrics = None
            if val_loader is not None:
                val_metrics = run_epoch(
                    model, val_loader, device, optimizer=None, prior_phase="joint", amp_dtype=args.amp_dtype
                )
            metric_record = {
                "epoch": epoch,
                "prior_phase": prior_phase,
                "train": train_metrics,
                "validation": val_metrics,
            }
            metric_rows.append(metric_record)
            atomic_write_jsonl(metrics_path, metric_rows)
            checkpoint = {
                "schema_version": "clir-full-checkpoint-v1",
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_epoch": epoch,
                "rng_state": _rng_state(),
                "config": config.__dict__,
                "data_state": data_state,
                "training_args": vars(args),
                "metrics": metric_rows,
                "code": git_state(Path(__file__).resolve().parent),
            }
            _atomic_torch_save(checkpoint, output)
            run_record["completed_epoch"] = epoch
            atomic_write_json(run_path, run_record)
            message = f"epoch={epoch} prior_phase={prior_phase} train_total={train_metrics['losses'].get('total', 0.0):.4f}"
            if val_metrics is not None:
                message += f" val_total={val_metrics['losses'].get('total', 0.0):.4f}"
            print(message)
    except BaseException as exc:
        run_record["status"] = "failed"
        run_record["error_type"] = type(exc).__name__
        run_record["error"] = str(exc)
        atomic_write_json(run_path, run_record)
        raise

    run_record["status"] = "completed"
    atomic_write_json(run_path, run_record)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
