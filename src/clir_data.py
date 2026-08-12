"""Data utilities for CLIR JSONL training and scoring.

Each JSONL row represents one generated trajectory. Minimal fields:

{
  "id": "sample-0-cand-0",
  "query_id": "sample-0",
  "hidden_states_path": "features/sample-0-cand-0.pt",
  "correctness": 1,
  "semantic_id": "sample-0",
  "style_id": "direct"
}

Hidden states can be stored inline as `hidden_states`, or by path in
`hidden_states_path`. Optional `condition_states`, prior targets, hallucination
labels, and token advantage targets follow the names used by the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


TEXT_ID_FIELDS = {
    "id",
    "query_id",
    "semantic_id",
    "style_id",
    "domain_id",
    "source",
    "prompt",
    "context",
    "trajectory",
    "answer",
}

TOKEN_SEQUENCE_FIELDS = {
    "token_advantage",
    "token_advantages",
    "advantages",
    "progress_targets",
    "key_prior_target",
    "complete_prior_target",
    "key_prior",
    "complete_prior",
}


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
    """JSONL dataset for pre-extracted hidden-state trajectories."""

    def __init__(self, jsonl_path: str | Path, feature_root: Optional[str | Path] = None) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.feature_root = Path(feature_root) if feature_root is not None else self.jsonl_path.parent
        self.rows = read_jsonl(self.jsonl_path)
        if not self.rows:
            raise ValueError(f"No rows found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = dict(self.rows[index])
        hidden_states = load_tensor_field(row, "hidden_states", "hidden_states_path", self.feature_root)
        if hidden_states.ndim != 2:
            raise ValueError("Each hidden state item must have shape [time, hidden_dim]")

        item: Dict[str, Any] = {
            "row_index": index,
            "id": row.get("id", str(index)),
            "query_id": row.get("query_id", row.get("group_id", row.get("semantic_id", str(index)))),
            "hidden_states": hidden_states.float(),
        }

        condition_states = maybe_load_tensor_field(row, "condition_states", "condition_states_path", self.feature_root)
        if condition_states is not None:
            if condition_states.ndim != 2:
                raise ValueError("condition_states must have shape [condition_time, hidden_dim]")
            item["condition_states"] = condition_states.float()

        condition_embedding = maybe_load_tensor_field(
            row,
            "condition_embedding",
            "condition_embedding_path",
            self.feature_root,
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

    sequence_aliases = {
        "token_advantage": ("token_advantage", "token_advantages", "advantages"),
        "progress_targets": ("progress_targets", "progress", "progress_target"),
        "key_prior_target": ("key_prior_target", "key_prior"),
        "complete_prior_target": ("complete_prior_target", "complete_prior"),
    }
    for output_key, aliases in sequence_aliases.items():
        value = first_present(row, aliases)
        if value is not None:
            item[output_key] = pad_or_trim_1d(value, time)

    return item


def first_present(row: Dict[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        if key in row:
            return row[key]
    return None


def pad_or_trim_1d(values: Any, length: int) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
    output = torch.zeros(length, dtype=torch.float32)
    if tensor.numel() > 0:
        output[: min(length, tensor.numel())] = tensor[:length]
    return output


def clir_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    max_time = max(item["hidden_states"].shape[0] for item in batch)
    hidden_dim = batch[0]["hidden_states"].shape[1]
    hidden_states = torch.zeros(len(batch), max_time, hidden_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_time, dtype=torch.float32)

    for row, item in enumerate(batch):
        states = item["hidden_states"]
        length = states.shape[0]
        hidden_states[row, :length] = states
        mask[row, :length] = 1.0

    output: Dict[str, Any] = {
        "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        "ids": [item["id"] for item in batch],
        "query_ids_raw": [item["query_id"] for item in batch],
        "hidden_states": hidden_states,
        "mask": mask,
    }

    if any("condition_states" in item for item in batch):
        max_condition_time = max(item.get("condition_states", torch.empty(0, hidden_dim)).shape[0] for item in batch)
        condition_states = torch.zeros(len(batch), max_condition_time, hidden_dim, dtype=torch.float32)
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

    if any("condition_embedding" in item for item in batch):
        condition_embedding = torch.zeros(len(batch), hidden_dim, dtype=torch.float32)
        for row, item in enumerate(batch):
            if "condition_embedding" in item:
                condition_embedding[row] = item["condition_embedding"]
        output["condition_embedding"] = condition_embedding

    add_optional_float(output, batch, "correctness")
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
        length = min(tensor.numel(), int(output["mask"][row].sum().item()), max_time)
        if length > 0:
            values[row, :length] = tensor[:length]
            mask[row, :length] = True
            has_any = True
    if has_any:
        output[key] = values
        output[mask_key] = mask


def move_batch_to_device(batch: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


__all__ = [
    "CLIRTrajectoryDataset",
    "clir_collate",
    "move_batch_to_device",
    "read_jsonl",
    "write_jsonl",
]
