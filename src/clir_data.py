"""Data utilities for CLIR JSONL training and scoring.

Each JSONL row represents one generated trajectory. Pre-extracted mode uses:

{
  "id": "sample-0-cand-0",
  "query_id": "sample-0",
  "hidden_states_path": "features/sample-0-cand-0.pt",
  "correctness": 1,
  "semantic_id": "sample-0",
  "style_id": "direct"
}

Online mode instead consumes exact `prompt_token_ids` and `output_token_ids`
plus frozen model provenance. Human-readable prompt/question/response fields
are audit metadata and are never re-tokenized. Optional prior targets,
hallucination labels, and token advantage targets follow the model names.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import BatchSampler, Dataset, Sampler

from .clir_real_data import (
    TOKEN_LABEL_ALIASES,
    validate_extracted_row,
    validate_rollout_row,
)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class CLIRTrajectoryDataset(Dataset):
    """JSONL trajectories backed by precomputed features or exact token IDs."""

    def __init__(
        self,
        jsonl_path: str | Path,
        feature_root: Optional[str | Path] = None,
        *,
        check_finite: bool = True,
        require_correctness: bool = False,
        load_condition: bool = True,
        hidden_state_source: str = "precomputed",
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.feature_root = Path(feature_root) if feature_root is not None else self.jsonl_path.parent
        self.check_finite = check_finite
        self.require_correctness = require_correctness
        self.load_condition = load_condition
        if hidden_state_source not in {"precomputed", "online"}:
            raise ValueError(
                "hidden_state_source must be either 'precomputed' or 'online'"
            )
        self.hidden_state_source = hidden_state_source
        self.rows = read_jsonl(self.jsonl_path)
        if not self.rows:
            raise ValueError(f"No rows found in {self.jsonl_path}")
        if self.require_correctness:
            for row_index, row in enumerate(self.rows):
                value = first_present(row, ("correctness", "label", "final_correct"))
                if value is None:
                    raise ValueError(
                        f"Training row {row_index} is missing required correctness label"
                    )
                if isinstance(value, bool) or value not in (0, 1, 0.0, 1.0):
                    raise ValueError(
                        f"Training row {row_index} correctness must be numeric 0 or 1, got {value!r}"
                    )
        if self.hidden_state_source == "online":
            for row in self.rows:
                validate_rollout_row(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = dict(self.rows[index])
        if self.hidden_state_source == "online":
            output_token_ids = list(row["output_token_ids"])
            item: Dict[str, Any] = {
                "row_index": index,
                "id": row.get("id", str(index)),
                "query_id": row.get(
                    "query_id",
                    row.get("candidate_group_id", row.get("prompt_id", str(index))),
                ),
                "prompt_token_ids": list(row["prompt_token_ids"]),
                "output_token_ids": output_token_ids,
            }
            item.update(extract_metadata(row, len(output_token_ids)))
            return item

        hidden_states = load_tensor_field(row, "hidden_states", "hidden_states_path", self.feature_root)
        if hidden_states.ndim != 2:
            raise ValueError("Each hidden state item must have shape [time, hidden_dim]")

        item: Dict[str, Any] = {
            "row_index": index,
            "id": row.get("id", str(index)),
            "query_id": row.get("query_id", row.get("candidate_group_id", row.get("prompt_id", str(index)))),
            # Preserve extracted BF16/FP16 storage on CPU. CUDA autocast can
            # consume it directly, avoiding a 2x host-memory expansion and a
            # full 101376-wide conversion for every real trajectory.
            "hidden_states": hidden_states if hidden_states.is_floating_point() else hidden_states.float(),
        }

        condition_states = (
            maybe_load_tensor_field(
                row,
                "condition_states",
                "condition_states_path",
                self.feature_root,
            )
            if self.load_condition
            else None
        )
        if condition_states is not None:
            if condition_states.ndim != 2:
                raise ValueError("condition_states must have shape [condition_time, hidden_dim]")
            item["condition_states"] = (
                condition_states if condition_states.is_floating_point() else condition_states.float()
            )

        # Real-data manifests carry the exact generated token ids.  Their
        # presence switches on the strict contract before legacy toy padding
        # or trimming can hide a token/feature mismatch.
        if "output_token_ids" in row:
            validate_extracted_row(
                row,
                hidden_states,
                condition_states,
                check_finite=self.check_finite,
                require_correctness=self.require_correctness,
            )

        condition_embedding = (
            maybe_load_tensor_field(
                row,
                "condition_embedding",
                "condition_embedding_path",
                self.feature_root,
            )
            if self.load_condition
            else None
        )
        if condition_embedding is not None:
            if condition_embedding.ndim != 1:
                raise ValueError("condition_embedding must have shape [hidden_dim]")
            item["condition_embedding"] = condition_embedding.float()

        item.update(extract_metadata(row, hidden_states.shape[0]))
        return item


def load_tensor_field(
    row: Dict[str, Any],
    inline_key: str,
    path_key: str,
    feature_root: Path,
) -> Tensor:
    tensor = maybe_load_tensor_field(row, inline_key, path_key, feature_root)
    if tensor is None:
        raise KeyError(f"Row must contain `{inline_key}` or `{path_key}`")
    return tensor


def maybe_load_tensor_field(
    row: Dict[str, Any],
    inline_key: str,
    path_key: str,
    feature_root: Path,
) -> Optional[Tensor]:
    if inline_key in row and row[inline_key] is not None:
        return torch.as_tensor(row[inline_key])
    if path_key not in row or row[path_key] is None:
        return None

    path = Path(row[path_key])
    if not path.is_absolute():
        path = feature_root / path
    if path.suffix == ".pt" or path.suffix == ".pth":
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ("hidden_states", "features", "tensor", "states"):
                if key in value:
                    value = value[key]
                    break
        return torch.as_tensor(value)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    if path.suffix == ".json":
        return torch.as_tensor(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported tensor file suffix: {path}")


def extract_metadata(row: Dict[str, Any], time: int) -> Dict[str, Any]:
    item: Dict[str, Any] = {}

    scalar_fields = {
        "correctness": ("correctness", "label", "final_correct"),
        "semantic_id": ("semantic_id", "semantic_ids", "augmentation_group", "augmentation_group_id", "group_id"),
        "style_id": ("style_id", "style_ids", "augmentation_style", "rewrite_style", "domain_id", "domain", "style"),
        "hallucination_onset": ("hallucination_onset", "hallucination_start", "onset"),
        "path_hallucinated": ("path_hallucinated", "hallucinated", "hallucination"),
    }
    for output_key, aliases in scalar_fields.items():
        value = first_present(row, aliases)
        if value is not None:
            item[output_key] = value

    for output_key, aliases in TOKEN_LABEL_ALIASES.items():
        value = first_present(row, aliases)
        if value is not None:
            item[output_key] = exact_length_1d(value, time, output_key)

    reconstruction_value = first_present(row, ("complete_reconstruction_target", "csr_target"))
    if reconstruction_value is not None:
        item["complete_reconstruction_target"] = torch.as_tensor(reconstruction_value, dtype=torch.float32).flatten()

    return item


def first_present(row: Dict[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        if key in row:
            return row[key]
    return None


def exact_length_1d(values: Any, length: int, field: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
    if tensor.numel() != length:
        raise ValueError(
            f"Token label `{field}` length mismatch: expected {length}, got {tensor.numel()}"
        )
    return tensor


def clir_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    has_hidden = ["hidden_states" in item for item in batch]
    has_tokens = [
        "prompt_token_ids" in item and "output_token_ids" in item
        for item in batch
    ]
    if all(has_hidden):
        source = "precomputed"
        max_time = max(item["hidden_states"].shape[0] for item in batch)
    elif all(has_tokens) and not any(has_hidden):
        source = "online"
        max_time = max(len(item["output_token_ids"]) for item in batch)
    else:
        raise ValueError(
            "A CLIR batch must contain either all precomputed features or all exact token IDs"
        )

    mask = torch.zeros(len(batch), max_time, dtype=torch.float32)
    for row, item in enumerate(batch):
        length = (
            item["hidden_states"].shape[0]
            if source == "precomputed"
            else len(item["output_token_ids"])
        )
        mask[row, :length] = 1.0

    output: Dict[str, Any] = {
        "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        "ids": [item["id"] for item in batch],
        "query_ids_raw": [item["query_id"] for item in batch],
        "mask": mask,
    }

    if source == "precomputed":
        hidden_dim = batch[0]["hidden_states"].shape[1]
        hidden_dtype = batch[0]["hidden_states"].dtype
        hidden_states = torch.zeros(len(batch), max_time, hidden_dim, dtype=hidden_dtype)
        for row, item in enumerate(batch):
            states = item["hidden_states"]
            hidden_states[row, : states.shape[0]] = states
        output["hidden_states"] = hidden_states
    else:
        output["prompt_token_ids"] = [item["prompt_token_ids"] for item in batch]
        output["output_token_ids"] = [item["output_token_ids"] for item in batch]

    if source == "precomputed" and any("condition_states" in item for item in batch):
        max_condition_time = max(item.get("condition_states", torch.empty(0, hidden_dim)).shape[0] for item in batch)
        condition_states = torch.zeros(len(batch), max_condition_time, hidden_dim, dtype=hidden_dtype)
        condition_mask = torch.zeros(len(batch), max_condition_time, dtype=torch.float32)
        for row, item in enumerate(batch):
            states = item.get("condition_states")
            if states is None:
                continue
            length = states.shape[0]
            condition_states[row, :length] = states
            condition_mask[row, :length] = 1.0
        output["condition_states"] = condition_states
        output["condition_mask"] = condition_mask

    if source == "precomputed" and any("condition_embedding" in item for item in batch):
        condition_embedding = torch.zeros(len(batch), hidden_dim, dtype=hidden_dtype)
        condition_embedding_mask = torch.zeros(len(batch), dtype=torch.float32)
        for row, item in enumerate(batch):
            if "condition_embedding" in item:
                condition_embedding[row] = item["condition_embedding"]
                condition_embedding_mask[row] = 1.0
        output["condition_embedding"] = condition_embedding
        output["condition_embedding_mask"] = condition_embedding_mask

    add_optional_float(output, batch, "correctness", mask_key="correctness_mask")
    add_encoded_ids(output, batch, "semantic_id", "semantic_ids", "consistency_mask_semantic")
    add_encoded_ids(output, batch, "style_id", "style_ids", "consistency_mask_style")
    if "semantic_ids" in output and "style_ids" in output:
        output["consistency_mask"] = output.pop("consistency_mask_semantic") & output.pop("consistency_mask_style")

    add_optional_onset(output, batch)
    add_optional_float(output, batch, "path_hallucinated", mask_key="path_label_mask")
    add_optional_sequence(output, batch, "token_advantage", max_time, mask_key="token_advantage_mask")
    add_optional_sequence(output, batch, "progress_targets", max_time, mask_key="progress_mask")
    add_optional_sequence(output, batch, "key_prior_target", max_time, mask_key="key_prior_mask")
    add_optional_sequence(output, batch, "complete_prior_target", max_time, mask_key="complete_prior_mask")
    # Reconstruction targets live in reward-model space, which may differ from
    # the raw all-layer input width. Infer and validate their own fixed width;
    # the model performs the final model_dim contract check.
    add_optional_vector(output, batch, "complete_reconstruction_target")

    output["query_ids"] = torch.tensor(encode_raw_ids(output["query_ids_raw"])[0], dtype=torch.long)
    return output


def add_optional_float(
    output: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
    key: str,
    mask_key: Optional[str] = None,
) -> None:
    values: List[float] = []
    mask: List[bool] = []
    for item in batch:
        if key in item and item[key] is not None:
            values.append(float(item[key]))
            mask.append(True)
        else:
            values.append(0.0)
            mask.append(False)
    if any(mask):
        output[key] = torch.tensor(values, dtype=torch.float32)
        if mask_key is not None:
            output[mask_key] = torch.tensor(mask, dtype=torch.bool)


def add_optional_onset(output: Dict[str, Any], batch: Sequence[Dict[str, Any]]) -> None:
    values: List[int] = []
    mask: List[bool] = []
    for item in batch:
        if "hallucination_onset" in item and item["hallucination_onset"] is not None:
            values.append(int(item["hallucination_onset"]))
            mask.append(True)
        else:
            values.append(-1)
            mask.append(False)
    if any(mask):
        output["hallucination_onset"] = torch.tensor(values, dtype=torch.long)
        output["onset_label_mask"] = torch.tensor(mask, dtype=torch.bool)


def add_encoded_ids(
    output: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
    input_key: str,
    output_key: str,
    mask_key: str,
) -> None:
    values = [item.get(input_key) for item in batch]
    encoded, mask = encode_raw_ids(values)
    if any(mask):
        output[output_key] = torch.tensor(encoded, dtype=torch.long)
        output[mask_key] = torch.tensor(mask, dtype=torch.bool)


def encode_raw_ids(values: Sequence[Any]) -> Tuple[List[int], List[bool]]:
    mapping: Dict[str, int] = {}
    encoded: List[int] = []
    mask: List[bool] = []
    for value in values:
        if value is None:
            encoded.append(0)
            mask.append(False)
            continue
        key = repr(value)
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        encoded.append(mapping[key])
        mask.append(True)
    return encoded, mask


def add_optional_sequence(
    output: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
    key: str,
    max_time: int,
    mask_key: str,
) -> None:
    values = torch.zeros(len(batch), max_time, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_time, dtype=torch.bool)
    has_any = False
    for row, item in enumerate(batch):
        if key not in item:
            continue
        tensor = item[key].float().flatten()
        trajectory_length = int(output["mask"][row].sum().item())
        if tensor.numel() != trajectory_length:
            raise ValueError(
                f"Token label `{key}` length mismatch during collate: "
                f"expected {trajectory_length}, got {tensor.numel()}"
            )
        length = trajectory_length
        if length > 0:
            values[row, :length] = tensor[:length]
            mask[row, :length] = True
            has_any = True
    if has_any:
        output[key] = values
        output[mask_key] = mask


def add_optional_vector(
    output: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
    key: str,
) -> None:
    present = [item[key].float().flatten() for item in batch if key in item]
    if not present:
        return
    vector_dim = present[0].numel()
    if vector_dim == 0:
        raise ValueError(f"{key} must not be empty")

    values = torch.zeros(len(batch), vector_dim, dtype=torch.float32)
    mask: List[bool] = []
    for row, item in enumerate(batch):
        if key not in item:
            mask.append(False)
            continue
        tensor = item[key].float().flatten()
        if tensor.numel() != vector_dim:
            raise ValueError(f"{key} must have one fixed length {vector_dim}, got {tensor.numel()}")
        values[row] = tensor
        mask.append(True)
    if any(mask):
        output[key] = values
        output[f"{key}_mask"] = torch.tensor(mask, dtype=torch.bool)


class SemanticGroupBatchSampler(BatchSampler):
    """Batch sampler that keeps LLM rewrites from the same semantic id together.

    Each batch packs small chunks from multiple semantic groups when possible.
    This makes PRISM-style positive and negative pairs likely in real training,
    instead of relying on random shuffle to place augmentations together.
    """

    def __init__(
        self,
        dataset: CLIRTrajectoryDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        if batch_size < 2:
            raise ValueError("SemanticGroupBatchSampler requires batch_size >= 2")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.indices = list(indices) if indices is not None else list(range(len(dataset.rows)))
        self.groups: Dict[str, List[int]] = {}
        for row_index in self.indices:
            row = dataset.rows[row_index]
            semantic_id = first_present(
                row,
                ("semantic_id", "semantic_ids", "augmentation_group", "augmentation_group_id", "group_id"),
            )
            key = repr(semantic_id) if semantic_id is not None else f"__row_{row_index}"
            self.groups.setdefault(key, []).append(row_index)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        yield from self._build_batches(rng if self.shuffle else None)

    def __len__(self) -> int:
        return len(self._build_batches(None))

    def _build_batches(self, rng: Optional[random.Random]) -> List[List[int]]:
        group_items = [list(indices) for indices in self.groups.values()]
        if rng is not None:
            rng.shuffle(group_items)
            for indices in group_items:
                rng.shuffle(indices)

        chunks: List[List[int]] = []
        leftovers: List[int] = []
        max_group_chunk = max(2, self.batch_size // 2)
        for indices in group_items:
            start = 0
            while start < len(indices):
                remaining = len(indices) - start
                if remaining == 1:
                    leftovers.append(indices[start])
                    break
                chunk = indices[start : start + min(max_group_chunk, remaining)]
                start += len(chunk)
                if len(chunk) >= 2:
                    chunks.append(chunk)
                else:
                    leftovers.extend(chunk)

        chunks.sort(key=len, reverse=True)
        if rng is not None:
            rng.shuffle(leftovers)

        batches: List[List[int]] = []
        current: List[int] = []
        for chunk in chunks:
            if len(current) + len(chunk) > self.batch_size:
                while len(current) < self.batch_size and leftovers:
                    current.append(leftovers.pop())
                if len(current) == self.batch_size or (current and not self.drop_last):
                    batches.append(current)
                current = []
            if len(chunk) == self.batch_size:
                batches.append(chunk)
            else:
                current.extend(chunk)

        while len(current) < self.batch_size and leftovers:
            current.append(leftovers.pop())
        if len(current) == self.batch_size or (current and not self.drop_last):
            batches.append(current)

        while leftovers:
            batch = leftovers[: self.batch_size]
            leftovers = leftovers[self.batch_size :]
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                batches.append(batch)

        if rng is not None:
            rng.shuffle(batches)
        return batches


class EpochRandomSampler(Sampler[int]):
    """Shuffle a finite dataset from an explicit ``(seed, epoch)`` stream.

    PyTorch's default ``RandomSampler`` can share the global/worker generator
    used by ``DataLoader``.  With persistent workers, iterator construction
    consumes that stream differently after a process restart.  Keeping sample
    order in this dedicated sampler makes interrupted and uninterrupted CLIR
    runs use exactly the same row order for every epoch.
    """

    def __init__(self, data_source: Dataset, *, seed: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


__all__ = [
    "CLIRTrajectoryDataset",
    "EpochRandomSampler",
    "SemanticGroupBatchSampler",
    "clir_collate",
    "move_batch_to_device",
    "read_jsonl",
    "write_jsonl",
]
