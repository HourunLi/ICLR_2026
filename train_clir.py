"""Train the self-contained CLIR reward model with auditable Stage A state."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.clir_data import (
    CLIRTrajectoryDataset,
    EpochRandomSampler,
    SemanticGroupBatchSampler,
    clir_collate,
    move_batch_to_device,
)
from src.clir_real_data import file_sha256
from src.clir_hidden_states import (
    OnlineHiddenStateExtractor,
    add_hidden_state_source_arguments,
    load_online_hidden_state_extractor,
    online_hidden_state_config_from_args,
    validate_online_rows,
)
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state
from src.consistency_localized_reward import (
    RewardConfig,
    build_reward_model,
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLIR on pre-extracted or exact-token online hidden states."
    )
    parser.add_argument("--train_jsonl", required=True, help="Training JSONL file.")
    parser.add_argument("--feature_root", default=None, help="Base directory for training feature paths.")
    parser.add_argument("--val_jsonl", default=None, help="Explicit query-disjoint validation JSONL.")
    parser.add_argument("--val_feature_root", default=None, help="Base directory for validation feature paths.")
    add_hidden_state_source_arguments(parser)
    parser.add_argument("--output_model", required=True, help="Atomic latest/full-state checkpoint path.")
    parser.add_argument("--resume_from", default=None, help="Full-state checkpoint to resume.")
    parser.add_argument(
        "--experiment_protocol_config",
        default=None,
        help="Optional frozen experiment protocol recorded in the run and checkpoint.",
    )
    parser.add_argument(
        "--epoch_checkpoint_dir",
        default=None,
        help="Optional directory in which to retain one immutable full checkpoint per epoch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Restart only a failed zero-epoch run record when no checkpoint or metrics exist. "
            "Completed/partial training artifacts are never overwritten."
        ),
    )
    parser.add_argument("--metrics_jsonl", default=None)
    parser.add_argument("--run_json", default=None)
    parser.add_argument("--expected_train_sha256", default=None)
    parser.add_argument("--expected_val_sha256", default=None)
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
    parser.add_argument(
        "--val_every_n_epochs",
        type=int,
        default=1,
        help="Run validation at this interval and always at the final target epoch.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Clip the global gradient norm when > 0; zero disables clipping.",
    )
    parser.add_argument(
        "--prior_collapse_tolerance",
        type=float,
        default=0.0,
        help=(
            "Require train correctness BCE to improve on the constant class-prior entropy "
            "by at least this relative fraction; zero disables the gate."
        ),
    )
    parser.add_argument(
        "--fail_on_prior_collapse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero after saving a checkpoint when the preregistered prior-collapse gate fails.",
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
    parser.add_argument("--negative_consistency_weight", type=float, default=1.0)
    parser.add_argument("--score_consistency_weight", type=float, default=0.1)
    parser.add_argument("--hallucination_weight", type=float, default=1.0)
    parser.add_argument("--mil_weight", type=float, default=0.25)
    parser.add_argument("--token_reward_weight", type=float, default=0.5)
    parser.add_argument("--tail_weight", type=float, default=0.5)
    parser.add_argument("--pseudo_tail_weight", type=float, default=0.1)
    parser.add_argument("--progress_weight", type=float, default=0.25)
    parser.add_argument("--prior_weight", type=float, default=0.25)
    parser.add_argument("--negative_tail_margin", type=float, default=0.5)
    parser.add_argument(
        "--hallucination_target_mode",
        default="auto",
        choices=["auto", "onset_tail", "explicit"],
    )
    parser.add_argument("--hallucination_positive_weight", type=float, default=1.0)
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
        negative_consistency_weight=args.negative_consistency_weight,
        score_consistency_weight=args.score_consistency_weight,
        hallucination_weight=args.hallucination_weight,
        mil_weight=args.mil_weight,
        token_reward_weight=args.token_reward_weight,
        tail_weight=args.tail_weight,
        pseudo_tail_weight=args.pseudo_tail_weight,
        progress_weight=args.progress_weight,
        prior_weight=args.prior_weight,
        negative_tail_margin=args.negative_tail_margin,
        hallucination_target_mode=args.hallucination_target_mode,
        hallucination_positive_weight=args.hallucination_positive_weight,
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

    if dataset.hidden_state_source == "online":
        # OnlineHiddenStateConfig is authoritative. Some rewrite manifests
        # retain historical metadata only on the original view; requiring
        # every regenerated view to carry a precomputed-payload record would
        # reintroduce the storage-layer coupling this mode removes.
        return
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
    # Never let DataLoader worker/base-seed bookkeeping consume the reward
    # model's global Torch RNG stream.  This generator is deliberately
    # separate from the epoch-indexed sampler below.
    worker_generator = torch.Generator().manual_seed(seed + 1_000_003)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "generator": worker_generator,
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
    if shuffle:
        return DataLoader(
            subset,
            batch_size=batch_size,
            sampler=EpochRandomSampler(subset, seed=seed),
            collate_fn=clir_collate,
            **loader_kwargs,
        )
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        **loader_kwargs,
    )


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Set a one-based training epoch without depending on prior iterations."""

    if epoch <= 0:
        raise ValueError("Loader epoch must be positive")
    zero_based_epoch = epoch - 1
    batch_sampler = getattr(loader, "batch_sampler", None)
    if isinstance(batch_sampler, SemanticGroupBatchSampler):
        batch_sampler.epoch = zero_based_epoch
        return
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, EpochRandomSampler):
        sampler.epoch = zero_based_epoch


def prior_phase_for_epoch(mode: str, epoch: int) -> str:
    if mode == "alternate":
        return "key" if epoch % 2 == 1 else "complete"
    return mode


def _component_counts(
    batch: Mapping[str, Any],
    losses: Mapping[str, torch.Tensor],
    *,
    hallucination_target_mode: str = "auto",
) -> Dict[str, int]:
    batch_size = int(batch["hidden_states"].shape[0])
    mask = batch["mask"].bool()
    counts: Dict[str, int] = {"total": batch_size}
    if "final" in losses:
        if "correctness_mask" in batch:
            counts["final"] = int(batch["correctness_mask"].sum().item())
        else:
            counts["final"] = int(batch.get("correctness", torch.empty(0)).numel())
    positive_consistency_count = 0
    negative_consistency_count = 0
    if "consistency_mask" in batch:
        valid = batch["consistency_mask"].bool()
        semantic = batch["semantic_ids"]
        style = batch["style_ids"]
        upper = torch.triu(torch.ones((batch_size, batch_size), dtype=torch.bool, device=mask.device), diagonal=1)
        pairs = upper & valid[:, None] & valid[None, :]
        semantic_eq = semantic[:, None] == semantic[None, :]
        style_eq = style[:, None] == style[None, :]
        positive_consistency_count = int((pairs & semantic_eq & ~style_eq).sum().item())
        negative_consistency_count = int((pairs & ~semantic_eq & style_eq).sum().item())
    for key in losses:
        if key in {"consistency_positive", "consistency_score"}:
            counts[key] = positive_consistency_count
        elif key == "consistency_negative":
            counts[key] = negative_consistency_count
        elif key == "consistency_total":
            counts[key] = positive_consistency_count + negative_consistency_count

    onset_mask = batch.get(
        "onset_label_mask",
        torch.zeros(batch_size, dtype=torch.bool, device=mask.device),
    ).bool()
    use_explicit_targets = (
        "token_hallucination_target" in batch
        and hallucination_target_mode != "onset_tail"
    )
    if use_explicit_targets:
        supervised_tokens = batch["token_hallucination_mask"].bool() & mask
    else:
        supervised_tokens = onset_mask[:, None] & mask
    onset = batch.get("hallucination_onset", torch.full((batch_size,), -1, device=mask.device)).long()
    positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
    tail = onset_mask[:, None] & (onset[:, None] >= 0) & (positions >= onset[:, None]) & mask
    if "token_advantage_mask" in batch:
        reward_mask = batch["token_advantage_mask"].bool() & mask
    elif "token_advantage" in batch:
        reward_mask = mask
    else:
        reward_mask = tail
    reward_mask = reward_mask | tail
    localization_counts = {
        "localization_token_bce": int(supervised_tokens.sum().item()),
        "localization_token_reward": int(reward_mask.sum().item()),
        "localization_tail_margin": int(tail.sum().item()),
    }
    for key, value in localization_counts.items():
        if key in losses:
            counts[key] = value
    if "hallucination_mil" in losses:
        counts["hallucination_mil"] = int(batch.get("path_label_mask", torch.zeros(batch_size, dtype=torch.bool, device=mask.device)).sum().item())
    if "pseudo_tail" in losses:
        path_mask = batch.get(
            "path_label_mask",
            torch.zeros(batch_size, dtype=torch.bool, device=mask.device),
        ).bool()
        path_positive = batch.get(
            "path_hallucinated",
            torch.zeros(batch_size, device=mask.device),
        ).bool()
        counts["pseudo_tail"] = int((path_mask & path_positive & ~onset_mask).sum().item())
    if "progress" in losses:
        counts["progress"] = int(batch.get("progress_mask", torch.zeros_like(mask)).sum().item())
    key_mask = batch.get("key_prior_mask", torch.zeros_like(mask)).bool() & mask
    complete_mask = batch.get("complete_prior_mask", torch.zeros_like(mask)).bool() & mask
    key_count = int(key_mask.sum().item())
    complete_count = int(complete_mask.sum().item())
    joint_prior_count = int((key_mask & complete_mask).sum().item())
    reconstruction_count = int(batch.get("complete_reconstruction_target_mask", torch.zeros(batch_size, dtype=torch.bool, device=mask.device)).sum().item())
    for key in losses:
        if key == "prior_key":
            counts[key] = key_count
        elif key == "prior_complete":
            counts[key] = complete_count
        elif key in {"prior_distill", "prior_gate", "prior_total"}:
            counts[key] = joint_prior_count
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
    hidden_state_extractor: OnlineHiddenStateExtractor | None = None,
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
            if hidden_state_extractor is not None:
                batch = hidden_state_extractor.materialize(
                    batch,
                    include_condition=model.config.model_variant == "clir",
                )
            elif "hidden_states" not in batch:
                raise ValueError(
                    "Token-only batch requires an online hidden-state extractor"
                )
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
                non_finite_gradients = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                ]
                if non_finite_gradients:
                    optimizer.zero_grad(set_to_none=True)
                    preview = ", ".join(non_finite_gradients[:5])
                    raise FloatingPointError(f"Non-finite gradients in: {preview}")
                if max_grad_norm > 0:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if not torch.isfinite(gradient_norm):
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("Non-finite gradient norm")
                optimizer.step()

            batch_size = int(batch["hidden_states"].shape[0])
            examples += batch_size
            batches += 1
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            for key, value in _component_counts(
                batch,
                losses,
                hallucination_target_mode=model.config.hallucination_target_mode,
            ).items():
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
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        saved_cuda_states = list(state["cuda"])
        if not saved_cuda_states:
            raise ValueError("Checkpoint has no CUDA RNG state for a CUDA resume")
        # A checkpoint may have been written while more GPUs were visible than
        # are exposed to the resumed single-device process.  Restore every
        # overlapping device explicitly instead of letting set_rng_state_all
        # index past the current device set.  Device 0, which this trainer uses,
        # remains bit-exact.
        for device_index, cuda_state in enumerate(
            saved_cuda_states[: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device_index)


RESUME_PINNED_ARGS = (
    "batch_size",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "lr",
    "weight_decay",
    "max_grad_norm",
    "val_fraction",
    "seed",
    "device",
    "amp_dtype",
    "skip_feature_finite_check",
    "group_by_semantic_id",
    "prior_phase_mode",
    "val_every_n_epochs",
    "prior_collapse_tolerance",
    "fail_on_prior_collapse",
    "hidden_state_source",
    "extractor_model_id",
    "extractor_model_revision",
    "extractor_tokenizer_revision",
    "extractor_torch_dtype",
    "extractor_trust_remote_code",
    "extractor_layer_count",
    "extractor_per_layer_hidden_size",
)


LEGACY_RESUME_DEFAULTS: Dict[str, Any] = {
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
    "max_grad_norm": 0.0,
    "val_every_n_epochs": 1,
    "prior_collapse_tolerance": 0.0,
    "fail_on_prior_collapse": False,
    "hidden_state_source": "precomputed",
    "extractor_model_id": None,
    "extractor_model_revision": None,
    "extractor_tokenizer_revision": None,
    "extractor_torch_dtype": "bfloat16",
    "extractor_trust_remote_code": False,
    "extractor_layer_count": None,
    "extractor_per_layer_hidden_size": None,
}


def _validate_resume_training_args(
    checkpoint_args: Mapping[str, Any],
    current_args: argparse.Namespace,
) -> None:
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key in RESUME_PINNED_ARGS:
        if key in checkpoint_args:
            checkpoint_value = checkpoint_args[key]
        elif key in LEGACY_RESUME_DEFAULTS:
            checkpoint_value = LEGACY_RESUME_DEFAULTS[key]
        else:
            checkpoint_value = "<missing>"
        current_value = getattr(current_args, key)
        if checkpoint_value != current_value:
            mismatches[key] = {
                "checkpoint": checkpoint_value,
                "current": current_value,
            }
    if mismatches:
        raise ValueError(f"Resume training arguments differ: {json.dumps(mismatches, sort_keys=True)}")


def _load_force_restart_record(
    output: Path,
    metrics_path: Path,
    run_path: Path,
) -> Dict[str, Any]:
    """Validate and return the sole failed zero-epoch artifact for ``--force``."""

    if output.exists() or metrics_path.exists():
        raise FileExistsError(
            "--force never overwrites a checkpoint or metrics history; use --resume_from "
            "or a new output directory"
        )
    if not run_path.exists():
        return {}
    try:
        with run_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot validate existing run record {run_path}: {exc}") from exc
    if previous.get("status") != "failed" or int(previous.get("completed_epoch", -1)) != 0:
        raise FileExistsError(
            "--force only restarts a run record with status=failed and completed_epoch=0"
        )
    return previous


def _read_metric_rows(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _reconcile_metrics_history(
    metrics_path: Path,
    checkpoint_metrics: Any,
    completed_epoch: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any] | None]:
    """Recover an interrupted metrics/checkpoint publication from checkpoint truth."""

    if not isinstance(checkpoint_metrics, list) or len(checkpoint_metrics) != completed_epoch:
        raise ValueError("Checkpoint metrics history does not match completed_epoch")
    expected = [dict(row) for row in checkpoint_metrics]
    if not metrics_path.exists():
        atomic_write_jsonl(metrics_path, expected)
        return expected, {"action": "restored_missing_metrics_from_checkpoint"}

    recorded = _read_metric_rows(metrics_path)
    common = min(len(recorded), len(expected))
    if recorded[:common] != expected[:common]:
        raise ValueError("Metrics history disagrees with checkpoint-embedded metrics")
    if recorded == expected:
        return expected, None

    atomic_write_jsonl(metrics_path, expected)
    return expected, {
        "action": "reconciled_metrics_to_checkpoint",
        "recorded_rows": len(recorded),
        "checkpoint_rows": len(expected),
    }


def _constant_prior_health(
    *,
    positive_count: int,
    example_count: int,
    final_correctness_bce: float,
    tolerance: float,
) -> Dict[str, Any]:
    if example_count <= 0 or not 0 <= positive_count <= example_count:
        raise ValueError("Invalid training correctness counts for health gate")
    prevalence = positive_count / example_count
    if prevalence in {0.0, 1.0}:
        prior_entropy = 0.0
        relative_distance = None
        relative_improvement = None
        passed = tolerance <= 0.0
        reason = "degenerate_class_distribution"
    else:
        prior_entropy = -(
            prevalence * math.log(prevalence)
            + (1.0 - prevalence) * math.log(1.0 - prevalence)
        )
        relative_distance = abs(final_correctness_bce - prior_entropy) / prior_entropy
        relative_improvement = (prior_entropy - final_correctness_bce) / prior_entropy
        passed = tolerance <= 0.0 or relative_improvement >= tolerance
        if tolerance <= 0.0:
            reason = "disabled"
        elif passed:
            reason = "minimum_improvement_over_prior_met"
        elif relative_improvement <= 0.0:
            reason = "not_better_than_constant_prior"
        else:
            reason = "insufficient_improvement_over_prior"
    return {
        "schema_version": "clir-training-health-v2",
        "gate": "constant_class_prior_bce",
        "enabled": tolerance > 0.0,
        "passed": passed,
        "reason": reason,
        "positive_count": positive_count,
        "example_count": example_count,
        "positive_prevalence": prevalence,
        "constant_prior_bce": prior_entropy,
        "observed_train_correctness_bce": final_correctness_bce,
        "relative_distance_from_prior_bce": relative_distance,
        "relative_improvement_over_prior_bce": relative_improvement,
        "minimum_relative_improvement": tolerance,
    }


def _checkpoint_training_health(
    checkpoint_train_metrics: Mapping[str, Any],
    *,
    epoch: int,
    positive_count: int,
    example_count: int,
    tolerance: float,
) -> Dict[str, Any]:
    """Build the health gate from a no-grad full-train checkpoint evaluation."""

    losses = checkpoint_train_metrics.get("losses")
    if not isinstance(losses, Mapping) or "final" not in losses:
        raise ValueError("Checkpoint train evaluation lacks final correctness BCE")
    observed_examples = int(checkpoint_train_metrics.get("examples", -1))
    if observed_examples != example_count:
        raise ValueError(
            "Checkpoint train evaluation did not cover the complete train split: "
            f"expected {example_count}, observed {observed_examples}"
        )
    health = _constant_prior_health(
        positive_count=positive_count,
        example_count=example_count,
        final_correctness_bce=float(losses["final"]),
        tolerance=tolerance,
    )
    health.update(
        {
            "schema_version": "clir-training-health-v3",
            "measurement": "checkpoint_full_train_split_no_grad_eval",
            "checkpoint_epoch": int(epoch),
            "model_mode": "eval",
            "shuffle_order": "explicit_seed_plus_epoch_complete_coverage",
        }
    )
    return health


def _checkpoint_execution_device(checkpoint: Mapping[str, Any]) -> str | None:
    recorded = checkpoint.get("execution_device")
    if isinstance(recorded, str) and recorded:
        return recorded
    segments = checkpoint.get("training_segments")
    if isinstance(segments, list) and segments:
        last = segments[-1]
        if isinstance(last, Mapping) and isinstance(last.get("device"), str):
            return str(last["device"])
    training_args = checkpoint.get("training_args")
    if isinstance(training_args, Mapping):
        declared = training_args.get("device")
        if declared in {"cpu", "cuda", "mps"}:
            return str(declared)
        if declared == "auto":
            rng_state = checkpoint.get("rng_state")
            if isinstance(rng_state, Mapping) and rng_state.get("cuda") is not None:
                return "cuda"
    return None


def _validate_resume_device(
    checkpoint: Mapping[str, Any],
    current_device: torch.device,
) -> str:
    recorded = _checkpoint_execution_device(checkpoint)
    if recorded is None:
        raise ValueError(
            "Resume checkpoint lacks an auditable resolved execution device; "
            "start a new run instead of cross-device or ambiguous resume"
        )
    if torch.device(recorded).type != current_device.type:
        raise ValueError(
            "Cross-device resume is not reproducible: checkpoint used "
            f"{recorded!r}, current run resolved to {str(current_device)!r}"
        )
    return recorded


def _final_training_health(
    metric_rows: list[Mapping[str, Any]],
    *,
    positive_count: int,
    example_count: int,
    tolerance: float,
) -> Dict[str, Any]:
    if not metric_rows:
        raise ValueError("Cannot publish a completed run without checkpoint metrics")
    recorded = metric_rows[-1].get("training_health")
    if isinstance(recorded, Mapping):
        return dict(recorded)
    train = metric_rows[-1].get("train")
    if not isinstance(train, Mapping) or not isinstance(train.get("losses"), Mapping):
        raise ValueError("Final checkpoint lacks train losses for health reconstruction")
    return _constant_prior_health(
        positive_count=positive_count,
        example_count=example_count,
        final_correctness_bce=float(train["losses"]["final"]),
        tolerance=tolerance,
    )


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _experiment_protocol_state(path: str | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    protocol_path = Path(path).resolve()
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    schema_version = protocol.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("Experiment protocol requires a non-empty schema_version")
    return {
        "path": str(protocol_path),
        "sha256": file_sha256(protocol_path),
        "schema_version": schema_version,
    }


def main() -> None:
    args = parse_args()
    if args.force and args.resume_from:
        raise ValueError("--force and --resume_from are mutually exclusive")
    if args.val_jsonl and args.val_fraction > 0:
        raise ValueError("Use explicit --val_jsonl or legacy --val_fraction, not both")
    if args.val_feature_root and not args.val_jsonl:
        raise ValueError("--val_feature_root requires --val_jsonl")
    if args.hidden_state_source == "online" and (
        args.feature_root is not None or args.val_feature_root is not None
    ):
        raise ValueError(
            "Online hidden-state extraction does not consume --feature_root/--val_feature_root"
        )
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.val_every_n_epochs <= 0:
        raise ValueError("val_every_n_epochs must be positive")
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative")
    if not 0.0 <= args.prior_collapse_tolerance < 1.0:
        raise ValueError("prior_collapse_tolerance must be in [0, 1)")
    if args.fail_on_prior_collapse and args.prior_collapse_tolerance <= 0.0:
        raise ValueError("fail_on_prior_collapse requires a positive prior_collapse_tolerance")
    set_seed(args.seed)
    device = resolve_device(args.device)
    online_hidden_state_config = online_hidden_state_config_from_args(
        args,
        feature_dim=args.hidden_dim,
    )
    output = Path(args.output_model).resolve()
    metrics_path = Path(args.metrics_jsonl).resolve() if args.metrics_jsonl else output.with_suffix(output.suffix + ".metrics.jsonl")
    run_path = Path(args.run_json).resolve() if args.run_json else output.with_suffix(output.suffix + ".run.json")
    epoch_checkpoint_dir = (
        Path(args.epoch_checkpoint_dir).resolve() if args.epoch_checkpoint_dir else None
    )
    experiment_protocol = _experiment_protocol_state(args.experiment_protocol_config)
    restarted_from: Dict[str, Any] | None = None
    if not args.resume_from and (metrics_path.exists() or run_path.exists() or output.exists()):
        if not args.force:
            raise FileExistsError(
                "Training outputs already exist; use --resume_from, --force for a failed "
                "zero-epoch record, or a new output directory"
            )
        restarted_from = _load_force_restart_record(output, metrics_path, run_path)

    train_dataset = CLIRTrajectoryDataset(
        args.train_jsonl,
        feature_root=args.feature_root,
        check_finite=not args.skip_feature_finite_check,
        require_correctness=True,
        load_condition=args.model_variant == "clir",
        hidden_state_source=args.hidden_state_source,
    )
    if args.val_jsonl:
        val_dataset = CLIRTrajectoryDataset(
            args.val_jsonl,
            feature_root=args.val_feature_root,
            check_finite=not args.skip_feature_finite_check,
            require_correctness=True,
            load_condition=args.model_variant == "clir",
            hidden_state_source=args.hidden_state_source,
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
    train_positive_count = sum(int(train_dataset.rows[index]["correctness"]) for index in train_indices)

    config = make_config(args)
    if online_hidden_state_config is not None:
        validate_online_rows(train_dataset.rows, online_hidden_state_config)
        if val_dataset is not train_dataset:
            validate_online_rows(val_dataset.rows, online_hidden_state_config)
        if config.encoder_type == "layer_transformer" and (
            config.num_feature_layers != online_hidden_state_config.layer_count
            or config.per_layer_dim
            != online_hidden_state_config.per_layer_hidden_size
        ):
            raise ValueError(
                "Online extractor layer layout differs from the reward encoder: "
                f"extractor={online_hidden_state_config.layer_count}x"
                f"{online_hidden_state_config.per_layer_hidden_size}, "
                f"encoder={config.num_feature_layers}x{config.per_layer_dim}"
            )
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
    hidden_state_extractor = None
    train_sha256 = file_sha256(args.train_jsonl)
    val_sha256 = file_sha256(args.val_jsonl) if args.val_jsonl else None
    if args.expected_train_sha256 and train_sha256 != args.expected_train_sha256:
        raise ValueError(
            f"Train manifest SHA256 mismatch: expected {args.expected_train_sha256}, got {train_sha256}"
        )
    if args.expected_val_sha256 and val_sha256 != args.expected_val_sha256:
        raise ValueError(
            f"Validation manifest SHA256 mismatch: expected {args.expected_val_sha256}, got {val_sha256}"
        )
    data_state = {
        "train_jsonl": str(Path(args.train_jsonl).resolve()),
        "train_sha256": train_sha256,
        "train_rows": len(train_indices),
        "train_queries": len(train_queries),
        "val_jsonl": str(Path(args.val_jsonl).resolve()) if args.val_jsonl else None,
        "val_sha256": val_sha256,
        "val_rows": len(val_indices) if val_indices is not None else 0,
        "val_queries": len(val_queries),
    }
    if online_hidden_state_config is not None:
        data_state["hidden_state_source"] = "online"
        data_state["online_hidden_state_config"] = online_hidden_state_config.to_dict()
    start_epoch = 0
    metric_rows: list[Dict[str, Any]] = []
    metrics_recovery: Dict[str, Any] | None = None
    resumed_from: Dict[str, Any] | None = None
    training_segments: list[Dict[str, Any]] = []
    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        resume_sha256 = file_sha256(resume_path)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config") != config.__dict__:
            raise ValueError("Resume checkpoint model config differs from current CLI config")
        if checkpoint.get("data_state") != data_state:
            raise ValueError("Resume checkpoint data files/hashes differ from current run")
        if "optimizer_state_dict" not in checkpoint or "rng_state" not in checkpoint:
            raise ValueError("Resume checkpoint lacks full optimizer/RNG state")
        if not isinstance(checkpoint.get("training_args"), Mapping):
            raise ValueError("Resume checkpoint lacks recorded training arguments")
        if checkpoint.get("experiment_protocol") != experiment_protocol:
            raise ValueError("Resume checkpoint experiment protocol differs from current CLI config")
        _validate_resume_training_args(checkpoint["training_args"], args)
        checkpoint_device = _validate_resume_device(checkpoint, device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["completed_epoch"])
        recorded_segments = checkpoint.get("training_segments")
        if isinstance(recorded_segments, list) and all(
            isinstance(segment, Mapping) for segment in recorded_segments
        ):
            training_segments = [dict(segment) for segment in recorded_segments]
        elif start_epoch > 0:
            training_segments = [{
                "start_epoch": 1,
                "target_epoch": start_epoch,
                "completed_epoch": start_epoch,
                "device": checkpoint_device,
                "legacy_inferred": True,
            }]
        resumed_from = {
            "path": str(resume_path),
            "sha256": resume_sha256,
            "completed_epoch": start_epoch,
            "device": checkpoint_device,
        }
        _restore_rng_state(checkpoint["rng_state"])
        metric_rows, metrics_recovery = _reconcile_metrics_history(
            metrics_path,
            checkpoint.get("metrics"),
            start_epoch,
        )

    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint completed {start_epoch} epochs; target is only {args.epochs}")
    if epoch_checkpoint_dir is not None:
        conflicting_snapshots = [
            epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt"
            for epoch in range(start_epoch + 1, args.epochs + 1)
            if (epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt").exists()
        ]
        if conflicting_snapshots:
            raise FileExistsError(
                "Future epoch checkpoint already exists: "
                f"{conflicting_snapshots[0]}"
            )

    active_segment: Dict[str, Any] | None = None
    if start_epoch < args.epochs:
        active_segment = {
            "start_epoch": start_epoch + 1,
            "target_epoch": args.epochs,
            "completed_epoch": start_epoch,
            "device": str(device),
            "resumed_from_sha256": resumed_from["sha256"] if resumed_from else None,
        }
        training_segments.append(active_segment)

    run_record: Dict[str, Any] = {
        "schema_version": "clir-training-run-v2",
        "status": "running",
        "output_model": str(output),
        "metrics_jsonl": str(metrics_path),
        "target_epochs": args.epochs,
        "start_epoch": start_epoch,
        "completed_epoch": start_epoch,
        "device": str(device),
        "resumed_from": resumed_from,
        "training_segments": training_segments,
        "amp_dtype": args.amp_dtype,
        "model_variant": config.model_variant,
        "encoder_type": config.encoder_type,
        "input_dim": config.hidden_dim,
        "model_dim": config.model_dim,
        "hidden_state_source": args.hidden_state_source,
        "online_hidden_state_config": (
            online_hidden_state_config.to_dict()
            if online_hidden_state_config is not None
            else None
        ),
        "trainable_parameters": count_trainable_parameters(model),
        "data_state": data_state,
        "experiment_protocol": experiment_protocol,
        "epoch_checkpoint_dir": str(epoch_checkpoint_dir) if epoch_checkpoint_dir else None,
        "epoch_checkpoints": (
            [
                str(epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt")
                for epoch in range(1, start_epoch + 1)
                if (epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt").is_file()
            ]
            if epoch_checkpoint_dir is not None
            else []
        ),
        "code": git_state(Path(__file__).resolve().parent),
    }
    if restarted_from:
        run_record["restarted_from_failed_zero_epoch"] = restarted_from
    if metrics_recovery:
        run_record["metrics_recovery"] = metrics_recovery
    atomic_write_json(run_path, run_record)
    print(
        f"model_variant={config.model_variant} encoder_type={config.encoder_type} "
        f"input_dim={config.hidden_dim} model_dim={config.model_dim} "
        f"trainable_parameters={count_trainable_parameters(model)} resume_epoch={start_epoch}"
    )

    # The checkpoint is published before metrics/run JSON.  If a process died
    # between those writes at the final epoch, an idempotent resume at the same
    # target must reconstruct the secondary artifacts instead of declaring the
    # authoritative checkpoint unusable.
    if start_epoch == args.epochs:
        run_record["completed_epoch"] = start_epoch
        run_record["health_gate"] = _final_training_health(
            metric_rows,
            positive_count=train_positive_count,
            example_count=len(train_indices),
            tolerance=args.prior_collapse_tolerance,
        )
        run_record["recovered_completed_checkpoint"] = True
        run_record["status"] = "completed"
        if args.fail_on_prior_collapse and not run_record["health_gate"]["passed"]:
            run_record["status"] = "health_gate_failed"
        atomic_write_json(run_path, run_record)
        if run_record["status"] == "health_gate_failed":
            raise RuntimeError(
                "Recovered training checkpoint failed the preregistered constant-prior collapse gate"
            )
        print(f"recovered completed checkpoint {output}")
        return

    try:
        if online_hidden_state_config is not None:
            # Loading a frozen Hugging Face model may touch global RNG state.
            # Keep reward initialization/resume and subsequent training
            # randomness identical to the precomputed path for the same seed.
            training_rng_state = _rng_state()
            try:
                hidden_state_extractor = load_online_hidden_state_extractor(
                    online_hidden_state_config,
                    device=device,
                    cache_dir=args.extractor_cache_dir,
                    local_files_only=args.extractor_local_files_only,
                    check_finite=not args.skip_feature_finite_check,
                )
            finally:
                _restore_rng_state(training_rng_state)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            prior_phase = prior_phase_for_epoch(args.prior_phase_mode, epoch)
            set_loader_epoch(train_loader, epoch)
            train_metrics = run_epoch(
                model, train_loader, device, optimizer, prior_phase=prior_phase,
                amp_dtype=args.amp_dtype, max_grad_norm=args.max_grad_norm,
                hidden_state_extractor=hidden_state_extractor,
            )
            # The health gate is about the published checkpoint, not the
            # average of losses observed while that checkpoint was changing.
            # Reuse the complete train loader in eval/no-grad mode, with an
            # explicit epoch order, and isolate the evaluation from training
            # RNG state so resume remains exact.
            checkpoint_eval_rng_state = _rng_state()
            try:
                set_loader_epoch(train_loader, epoch)
                checkpoint_train_metrics = run_epoch(
                    model,
                    train_loader,
                    device,
                    optimizer=None,
                    prior_phase="joint",
                    amp_dtype=args.amp_dtype,
                    hidden_state_extractor=hidden_state_extractor,
                )
            finally:
                _restore_rng_state(checkpoint_eval_rng_state)
            val_metrics = None
            should_validate = (
                val_loader is not None
                and (epoch == args.epochs or epoch % args.val_every_n_epochs == 0)
            )
            if should_validate:
                val_metrics = run_epoch(
                    model, val_loader, device, optimizer=None, prior_phase="joint",
                    amp_dtype=args.amp_dtype,
                    hidden_state_extractor=hidden_state_extractor,
                )
            metric_record = {
                "epoch": epoch,
                "prior_phase": prior_phase,
                "train": train_metrics,
                "checkpoint_train_evaluation": checkpoint_train_metrics,
                "validation": val_metrics,
            }
            metric_record["training_health"] = _checkpoint_training_health(
                checkpoint_train_metrics,
                epoch=epoch,
                positive_count=train_positive_count,
                example_count=len(train_indices),
                tolerance=args.prior_collapse_tolerance,
            )
            metric_rows.append(metric_record)
            if active_segment is not None:
                active_segment["completed_epoch"] = epoch
            checkpoint = {
                "schema_version": "clir-full-checkpoint-v2",
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_epoch": epoch,
                "rng_state": _rng_state(),
                "config": config.__dict__,
                "data_state": data_state,
                "training_args": vars(args),
                "metrics": metric_rows,
                "experiment_protocol": experiment_protocol,
                "execution_device": str(device),
                "resumed_from": resumed_from,
                "training_segments": training_segments,
                "code": git_state(Path(__file__).resolve().parent),
            }
            _atomic_torch_save(checkpoint, output)
            if epoch_checkpoint_dir is not None:
                epoch_checkpoint = epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt"
                _atomic_torch_save(checkpoint, epoch_checkpoint)
                run_record["epoch_checkpoints"].append(str(epoch_checkpoint))
            atomic_write_jsonl(metrics_path, metric_rows)
            run_record["completed_epoch"] = epoch
            atomic_write_json(run_path, run_record)
            message = f"epoch={epoch} prior_phase={prior_phase} train_total={train_metrics['losses'].get('total', 0.0):.4f}"
            message += (
                " checkpoint_train_bce="
                f"{checkpoint_train_metrics['losses'].get('final', float('nan')):.4f}"
            )
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
    run_record["health_gate"] = _final_training_health(
        metric_rows,
        positive_count=train_positive_count,
        example_count=len(train_indices),
        tolerance=args.prior_collapse_tolerance,
    )
    if args.fail_on_prior_collapse and not run_record["health_gate"]["passed"]:
        run_record["status"] = "health_gate_failed"
        atomic_write_json(run_path, run_record)
        raise RuntimeError(
            "Training completed but failed the preregistered constant-prior collapse gate"
        )
    atomic_write_json(run_path, run_record)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
