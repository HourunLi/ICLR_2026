#!/usr/bin/env python
"""Extract all-layer CLIR features from exact saved prompt/output token ids."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import torch

# Make direct execution (`python /path/to/script.py`) independent of PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl, write_jsonl
from src.clir_real_data import (
    artifact_stem,
    canonical_json_sha256,
    extract_aligned_hidden_states,
    file_sha256,
    load_protocol,
    validate_extracted_row,
    validate_rollout_row,
    validate_uniform_extraction,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phi35_gsm8k_pilot_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--storage-dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _git_state() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {"commit": commit, "dirty": bool(dirty_output)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _assert_target_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Hidden-state extraction dependencies are missing. Activate SWIFT and install the updated requirements.txt."
        ) from exc

    protocol_path = Path(args.protocol_config).resolve()
    protocol = load_protocol(protocol_path)
    protocol_hash = canonical_json_sha256(protocol)
    model_cfg = protocol["model"]
    hidden_cfg = protocol["hidden_states"]
    storage_dtype = args.storage_dtype or hidden_cfg["storage_dtype"]
    output_jsonl = Path(args.output_jsonl).resolve()
    feature_root = Path(args.feature_dir).resolve()
    extraction_manifest_path = output_jsonl.with_suffix(output_jsonl.suffix + ".extraction.json")
    _assert_target_available(output_jsonl, args.overwrite)
    _assert_target_available(extraction_manifest_path, args.overwrite)

    rows = read_jsonl(args.input_jsonl)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("max-rows must be > 0")
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("Input rollout manifest is empty")
    for row in rows:
        validate_rollout_row(row)
        provenance = row["provenance"]
        if provenance["model_id"] != model_cfg["repo_id"]:
            raise ValueError(f"Rollout/model id mismatch for {row['id']}")
        if provenance["model_revision"] != model_cfg["revision"]:
            raise ValueError(f"Rollout/model revision mismatch for {row['id']}")
        if provenance["tokenizer_revision"] != model_cfg["tokenizer_revision"]:
            raise ValueError(f"Rollout/tokenizer revision mismatch for {row['id']}")
        if provenance.get("protocol_version") != protocol["protocol_version"]:
            raise ValueError(f"Rollout/protocol version mismatch for {row['id']}")
        if provenance.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"Rollout/protocol hash mismatch for {row['id']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["repo_id"],
        revision=model_cfg["tokenizer_revision"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        cache_dir=args.cache_dir,
    )
    torch_dtype = getattr(torch, model_cfg["torch_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["repo_id"],
        revision=model_cfg["revision"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )
    model.eval()

    trajectories_dir = feature_root / "trajectories"
    conditions_dir = feature_root / "conditions"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)
    condition_cache: Dict[str, tuple[Path, torch.Tensor, tuple[int, ...]]] = {}
    extracted_rows = []
    total_feature_bytes = 0
    total_output_tokens = 0
    expected_layer_count = int(model.config.num_hidden_layers) + 1

    for row in rows:
        decoded_response = tokenizer.decode(
            row["output_token_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded_response != row["response"]:
            raise ValueError(
                f"Saved response does not decode from output_token_ids for {row['id']!r}; "
                "refusing to extract misaligned features"
            )

        query_id = str(row["query_id"])
        cached = condition_cache.get(query_id)
        canonical_condition = None
        if cached is not None:
            _, canonical_condition, cached_prompt_ids = cached
            if cached_prompt_ids != tuple(row["prompt_token_ids"]):
                raise ValueError(f"Prompt token IDs changed within query {query_id!r}")

        aligned = extract_aligned_hidden_states(
            model,
            row["prompt_token_ids"],
            row["output_token_ids"],
            storage_dtype=storage_dtype,
            canonical_condition=canonical_condition,
        )
        if aligned.layer_count != expected_layer_count:
            raise ValueError(
                f"Expected embedding + {model.config.num_hidden_layers} blocks "
                f"({expected_layer_count} states), got {aligned.layer_count}"
            )
        validate_extracted_row(row, aligned.trajectory, aligned.condition)

        trajectory_path = trajectories_dir / f"{artifact_stem(row['id'])}.pt"
        _assert_target_available(trajectory_path, args.overwrite)
        common_feature_metadata = {
            "layer_policy": hidden_cfg["layer_policy"],
            "layer_count": aligned.layer_count,
            "per_layer_hidden_size": aligned.per_layer_hidden_size,
            "feature_dim": aligned.feature_dim,
            "apply_final_norm": False,
            "storage_dtype": storage_dtype,
        }
        trajectory_metadata = {
            **common_feature_metadata,
            "feature_role": "trajectory",
            "trajectory_tokens": len(row["output_token_ids"]),
            "condition_tokens": len(row["prompt_token_ids"]),
        }
        _atomic_torch_save(
            {"hidden_states": aligned.trajectory, "metadata": trajectory_metadata},
            trajectory_path,
        )

        if cached is not None:
            condition_path, cached_condition, _ = cached
            if cached_condition.data_ptr() != aligned.condition.data_ptr():
                raise AssertionError("Canonical condition cache was not reused")
        else:
            condition_path = conditions_dir / f"{artifact_stem(query_id)}.pt"
            _assert_target_available(condition_path, args.overwrite)
            condition_metadata = {
                **common_feature_metadata,
                "feature_role": "condition",
                "condition_tokens": len(row["prompt_token_ids"]),
                "condition_forward": hidden_cfg["condition_forward"],
            }
            _atomic_torch_save(
                {"hidden_states": aligned.condition, "metadata": condition_metadata},
                condition_path,
            )
            condition_cache[query_id] = (
                condition_path,
                aligned.condition,
                tuple(row["prompt_token_ids"]),
            )
            total_feature_bytes += condition_path.stat().st_size

        output_row = dict(row)
        output_row["hidden_states_path"] = str(trajectory_path)
        output_row["condition_states_path"] = str(condition_path)
        output_row["feature_metadata"] = trajectory_metadata
        output_row["feature_sha256"] = file_sha256(trajectory_path)
        output_row["condition_sha256"] = file_sha256(condition_path)
        extracted_rows.append(output_row)
        total_feature_bytes += trajectory_path.stat().st_size
        total_output_tokens += len(row["output_token_ids"])

    uniform = validate_uniform_extraction(extracted_rows)
    write_jsonl(output_jsonl, extracted_rows)
    report = {
        "protocol_version": protocol["protocol_version"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "input_sha256": file_sha256(args.input_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_sha256": file_sha256(output_jsonl),
        "feature_root": str(feature_root),
        "rows": len(extracted_rows),
        "queries": len(condition_cache),
        "total_output_tokens": total_output_tokens,
        "feature_bytes": total_feature_bytes,
        "bytes_per_output_token_including_conditions": total_feature_bytes / max(total_output_tokens, 1),
        **uniform,
        "model": {
            "repo_id": model_cfg["repo_id"],
            "revision": model_cfg["revision"],
            "tokenizer_revision": model_cfg["tokenizer_revision"],
            "torch_dtype": model_cfg["torch_dtype"],
        },
        "library_versions": {
            "torch": torch.__version__,
            "transformers": _version("transformers"),
            "accelerate": _version("accelerate"),
        },
        "code": _git_state(),
    }
    extraction_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    extraction_manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
