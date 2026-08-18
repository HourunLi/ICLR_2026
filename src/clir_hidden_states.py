"""On-demand hidden-state materialization for CLIR token manifests.

The canonical input is the exact ``prompt_token_ids`` / ``output_token_ids``
pair saved during generation.  Human-readable prompt, question, and response
fields remain audit metadata; they are never re-tokenized for feature
extraction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor

from .clir_real_data import forward_all_layer_features, validate_rollout_row


SUPPORTED_EXTRACTOR_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def add_hidden_state_source_arguments(parser: Any) -> None:
    """Register the shared train/score feature-source CLI contract."""

    parser.add_argument(
        "--hidden_state_source",
        default="precomputed",
        choices=["precomputed", "online"],
        help=(
            "Load feature payloads, or reproduce them batch-locally from exact saved token IDs."
        ),
    )
    parser.add_argument("--extractor_model_id", default=None)
    parser.add_argument("--extractor_model_revision", default=None)
    parser.add_argument("--extractor_tokenizer_revision", default=None)
    parser.add_argument(
        "--extractor_torch_dtype",
        default="bfloat16",
        choices=sorted(SUPPORTED_EXTRACTOR_DTYPES),
    )
    parser.add_argument(
        "--extractor_trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--extractor_layer_count", type=int, default=None)
    parser.add_argument("--extractor_per_layer_hidden_size", type=int, default=None)
    parser.add_argument("--extractor_cache_dir", default=None)
    parser.add_argument(
        "--extractor_local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )


@dataclass(frozen=True)
class OnlineHiddenStateConfig:
    """Frozen provenance and shape contract for a hidden-state extractor."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    torch_dtype: str
    trust_remote_code: bool
    layer_count: int
    per_layer_hidden_size: int
    feature_dim: int

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_revision or not self.tokenizer_revision:
            raise ValueError("Online extractor model/tokenizer identity must be fully pinned")
        if self.torch_dtype not in SUPPORTED_EXTRACTOR_DTYPES:
            raise ValueError(
                f"Unsupported extractor dtype {self.torch_dtype!r}; "
                f"expected one of {sorted(SUPPORTED_EXTRACTOR_DTYPES)}"
            )
        if self.layer_count <= 0 or self.per_layer_hidden_size <= 0:
            raise ValueError("Online extractor layer dimensions must be positive")
        expected_dim = self.layer_count * self.per_layer_hidden_size
        if self.feature_dim != expected_dim:
            raise ValueError(
                f"Online extractor feature_dim={self.feature_dim}, expected "
                f"{self.layer_count} * {self.per_layer_hidden_size} = {expected_dim}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def online_hidden_state_config_from_args(
    args: Any,
    *,
    feature_dim: int,
) -> OnlineHiddenStateConfig | None:
    """Resolve and validate the shared train/score online extractor flags."""

    if args.hidden_state_source == "precomputed":
        return None
    required = {
        "extractor_model_id": args.extractor_model_id,
        "extractor_model_revision": args.extractor_model_revision,
        "extractor_tokenizer_revision": args.extractor_tokenizer_revision,
        "extractor_layer_count": args.extractor_layer_count,
        "extractor_per_layer_hidden_size": args.extractor_per_layer_hidden_size,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Online hidden-state extraction requires: " + ", ".join(missing)
        )
    return OnlineHiddenStateConfig(
        model_id=args.extractor_model_id,
        model_revision=args.extractor_model_revision,
        tokenizer_revision=args.extractor_tokenizer_revision,
        torch_dtype=args.extractor_torch_dtype,
        trust_remote_code=args.extractor_trust_remote_code,
        layer_count=args.extractor_layer_count,
        per_layer_hidden_size=args.extractor_per_layer_hidden_size,
        feature_dim=feature_dim,
    )


def validate_online_rows(
    rows: Sequence[Mapping[str, Any]],
    config: OnlineHiddenStateConfig,
) -> None:
    """Validate exact-token and model provenance before loading the task LLM."""

    if not rows:
        raise ValueError("Online hidden-state extraction requires at least one row")
    expected_provenance = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
    }
    for row_index, row in enumerate(rows):
        validate_rollout_row(row)
        provenance = row["provenance"]
        mismatches = {
            key: {"expected": expected, "observed": provenance.get(key)}
            for key, expected in expected_provenance.items()
            if provenance.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Online extractor provenance mismatch in row {row_index}: {mismatches}"
            )

        metadata = row.get("feature_metadata")
        if isinstance(metadata, Mapping):
            observed = {
                "layer_count": metadata.get("layer_count"),
                "per_layer_hidden_size": metadata.get("per_layer_hidden_size"),
                "feature_dim": metadata.get("feature_dim"),
            }
            expected = {
                "layer_count": config.layer_count,
                "per_layer_hidden_size": config.per_layer_hidden_size,
                "feature_dim": config.feature_dim,
            }
            if observed != expected:
                raise ValueError(
                    f"Online extractor feature metadata mismatch in row {row_index}: "
                    f"expected {expected}, observed {observed}"
                )


class OnlineHiddenStateExtractor:
    """Materialize exact-token all-layer features directly on the train device.

    Each trajectory uses the historical extraction semantics: one unpadded
    teacher-forced forward over ``prompt_ids + output_ids`` and an exact output
    slice.  Conditions use separate prompt-only forwards, deduplicated within
    the current batch.  No full-width feature survives beyond the batch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: OnlineHiddenStateConfig,
        *,
        device: str | torch.device,
        check_finite: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.config = config
        self.check_finite = check_finite
        self.model = model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def _forward_features(self, token_ids: Sequence[int]) -> Tensor:
        features, layer_count, per_layer_hidden_size = forward_all_layer_features(
            self.model,
            token_ids,
            device=self.device,
        )
        if layer_count != self.config.layer_count:
            raise ValueError(
                f"Online extractor returned {layer_count} layers, "
                f"expected {self.config.layer_count}"
            )
        if per_layer_hidden_size != self.config.per_layer_hidden_size:
            raise ValueError(
                "Online extractor per-layer width is "
                f"{per_layer_hidden_size}, expected "
                f"{self.config.per_layer_hidden_size}"
            )

        if features.shape[-1] != self.config.feature_dim:
            raise AssertionError("Internal online extractor feature-width error")
        expected_dtype = SUPPORTED_EXTRACTOR_DTYPES[self.config.torch_dtype]
        if features.dtype != expected_dtype:
            raise ValueError(
                f"Online extractor returned dtype {features.dtype}, expected {expected_dtype}"
            )
        if self.check_finite and not torch.isfinite(features).all():
            raise ValueError("Online extractor produced NaN or Inf")
        return features

    def materialize(
        self,
        batch: Dict[str, Any],
        *,
        include_condition: bool,
    ) -> Dict[str, Any]:
        """Add ``hidden_states`` and optional conditions to a token batch."""

        if "hidden_states" in batch:
            raise ValueError("Online extractor received a batch that already has hidden_states")
        prompt_rows = batch.get("prompt_token_ids")
        output_rows = batch.get("output_token_ids")
        if not isinstance(prompt_rows, list) or not isinstance(output_rows, list):
            raise ValueError("Online extraction requires collated exact token-id lists")
        if len(prompt_rows) != len(output_rows) or not prompt_rows:
            raise ValueError("Online extraction received an invalid token batch")

        batch_size = len(output_rows)
        output_lengths = [len(values) for values in output_rows]
        max_output_length = max(output_lengths)
        trajectory_batch: Tensor | None = None

        for row_index, (prompt_ids, output_ids) in enumerate(zip(prompt_rows, output_rows)):
            all_features = self._forward_features([*prompt_ids, *output_ids])
            trajectory = all_features[len(prompt_ids) :].clone()
            del all_features
            if trajectory.shape[0] != len(output_ids):
                raise AssertionError("Internal online output-token slicing error")
            if trajectory_batch is None:
                trajectory_batch = torch.zeros(
                    batch_size,
                    max_output_length,
                    self.config.feature_dim,
                    dtype=trajectory.dtype,
                    device=self.device,
                )
            elif trajectory.dtype != trajectory_batch.dtype:
                raise ValueError("Online extractor returned inconsistent trajectory dtypes")
            trajectory_batch[row_index, : len(output_ids)] = trajectory

        if trajectory_batch is None:
            raise AssertionError("Internal empty online trajectory batch")
        mask = torch.zeros(
            batch_size,
            max_output_length,
            dtype=torch.float32,
            device=self.device,
        )
        for row_index, length in enumerate(output_lengths):
            mask[row_index, :length] = 1.0
        existing_mask = batch.get("mask")
        if existing_mask is not None and not torch.equal(existing_mask.to(self.device), mask):
            raise ValueError("Online extractor output mask differs from token collator mask")
        batch["hidden_states"] = trajectory_batch
        batch["mask"] = mask

        if include_condition:
            max_prompt_length = max(len(values) for values in prompt_rows)
            condition_batch = torch.zeros(
                batch_size,
                max_prompt_length,
                self.config.feature_dim,
                dtype=trajectory_batch.dtype,
                device=self.device,
            )
            condition_mask = torch.zeros(
                batch_size,
                max_prompt_length,
                dtype=torch.float32,
                device=self.device,
            )
            prompt_groups: Dict[tuple[int, ...], list[int]] = {}
            for row_index, prompt_ids in enumerate(prompt_rows):
                prompt_key = tuple(prompt_ids)
                prompt_groups.setdefault(prompt_key, []).append(row_index)

            for prompt_key, row_indices in prompt_groups.items():
                condition = self._forward_features(prompt_key)
                if condition.dtype != condition_batch.dtype:
                    raise ValueError("Online extractor returned inconsistent condition dtype")
                for row_index in row_indices:
                    condition_batch[row_index, : len(prompt_key)] = condition
                    condition_mask[row_index, : len(prompt_key)] = 1.0
            batch["condition_states"] = condition_batch
            batch["condition_mask"] = condition_mask

        return batch


def load_online_hidden_state_extractor(
    config: OnlineHiddenStateConfig,
    *,
    device: str | torch.device,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    check_finite: bool = True,
) -> OnlineHiddenStateExtractor:
    """Load a frozen Hugging Face causal LM for batch-local extraction."""

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - environment failure path
        raise RuntimeError("transformers is required for online hidden-state extraction") from exc

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=SUPPORTED_EXTRACTOR_DTYPES[config.torch_dtype],
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    return OnlineHiddenStateExtractor(
        model,
        config,
        device=device,
        check_finite=check_finite,
    )


__all__ = [
    "add_hidden_state_source_arguments",
    "OnlineHiddenStateConfig",
    "OnlineHiddenStateExtractor",
    "load_online_hidden_state_extractor",
    "online_hidden_state_config_from_args",
    "validate_online_rows",
]
