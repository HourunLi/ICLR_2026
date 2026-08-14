#!/usr/bin/env python
"""Extract all-layer CLIR features from exact saved prompt/output token ids."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl
from src.clir_real_data import (
    artifact_stem,
    extract_aligned_hidden_states,
    file_sha256,
    load_protocol,
    protocol_hashes,
    validate_extracted_row,
    validate_protocol_reference,
    validate_rollout_row,
    validate_uniform_extraction,
)
from src.clir_stage_a import (
    atomic_write_json,
    atomic_write_jsonl,
    build_payload_record,
    candidate_count_for_membership,
    git_state,
    load_split_manifest,
    membership_entries,
    publish_completion_marker,
    query_shard_dir,
    validate_completion_marker,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phi35_gsm8k_pilot_v4.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--input-jsonl", default=None,
                        help="Legacy single-file input; mutually exclusive with --shard-root.")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--membership", default=None)
    parser.add_argument("--shard-root", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--n-rollouts", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Legacy mode only; shard mode is query atomic.")
    parser.add_argument("--storage-dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _assert_target_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def _load_feature(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "hidden_states" in payload:
        payload = payload["hidden_states"]
    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"Feature payload is not a tensor: {path}")
    return payload


def _validate_extracted_payload_row(row: Mapping[str, Any]) -> None:
    trajectory_path = Path(str(row.get("hidden_states_path", "")))
    condition_path = Path(str(row.get("condition_states_path", "")))
    if not trajectory_path.is_file() or not condition_path.is_file():
        raise ValueError(f"Extracted feature path is missing for {row.get('id')}")
    if file_sha256(trajectory_path) != row.get("feature_sha256"):
        raise ValueError(f"Trajectory checksum mismatch for {row.get('id')}")
    if file_sha256(condition_path) != row.get("condition_sha256"):
        raise ValueError(f"Condition checksum mismatch for {row.get('id')}")
    validate_extracted_row(row, _load_feature(trajectory_path), _load_feature(condition_path))


def _validate_rollout_provenance(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> None:
    model_cfg = protocol["model"]
    expected_candidate_policy = protocol.get("generation", {}).get("candidate_index_policy")
    for row in rows:
        validate_rollout_row(row)
        provenance = row["provenance"]
        expected = {
            "model_id": model_cfg["repo_id"],
            "model_revision": model_cfg["revision"],
            "tokenizer_revision": model_cfg["tokenizer_revision"],
            "protocol_version": protocol["protocol_version"],
        }
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise ValueError(f"Rollout/{key} mismatch for {row['id']}")
        recorded_acquisition = provenance.get("acquisition_protocol_sha256")
        if recorded_acquisition is not None:
            if recorded_acquisition != hashes["acquisition_protocol_sha256"]:
                raise ValueError(f"Rollout/acquisition_protocol_sha256 mismatch for {row['id']}")
            if provenance.get("label_protocol_sha256") != hashes["label_protocol_sha256"]:
                raise ValueError(f"Rollout/label_protocol_sha256 mismatch for {row['id']}")
        elif provenance.get("protocol_sha256") != hashes["protocol_sha256"]:
            raise ValueError(f"Rollout/protocol_sha256 mismatch for {row['id']}")
        if (
            expected_candidate_policy is not None
            and row.get("generation", {}).get("candidate_index_policy")
            != expected_candidate_policy
        ):
            raise ValueError(f"Rollout/candidate_index_policy mismatch for {row['id']}")


def _extract_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    hidden_cfg: Mapping[str, Any],
    storage_dtype: str,
    feature_root: Path,
    overwrite: bool,
) -> tuple[list[Dict[str, Any]], Dict[str, Any], list[Path]]:
    trajectories_dir = feature_root / "trajectories"
    conditions_dir = feature_root / "conditions"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)
    condition_cache: Dict[str, tuple[Path, torch.Tensor, tuple[int, ...]]] = {}
    extracted_rows: list[Dict[str, Any]] = []
    payload_paths: list[Path] = []
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
        _assert_target_available(trajectory_path, overwrite)
        common_feature_metadata = {
            "layer_policy": hidden_cfg["layer_policy"],
            "layer_count": aligned.layer_count,
            "per_layer_hidden_size": aligned.per_layer_hidden_size,
            "feature_dim": aligned.feature_dim,
            "hidden_states_source": "model_outputs.hidden_states_as_returned",
            "extractor_applied_additional_final_norm": False,
            "last_returned_state_norm_semantics": hidden_cfg.get(
                "last_returned_state_norm_semantics",
                "model_architecture_dependent",
            ),
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
        payload_paths.append(trajectory_path)

        if cached is not None:
            condition_path, cached_condition, _ = cached
            if cached_condition.data_ptr() != aligned.condition.data_ptr():
                raise AssertionError("Canonical condition cache was not reused")
        else:
            condition_path = conditions_dir / f"{artifact_stem(query_id)}.pt"
            _assert_target_available(condition_path, overwrite)
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
            payload_paths.append(condition_path)
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
    stats = {
        "rows": len(extracted_rows),
        "queries": len(condition_cache),
        "total_output_tokens": total_output_tokens,
        "feature_bytes": total_feature_bytes,
        "bytes_per_output_token_including_conditions": total_feature_bytes / max(total_output_tokens, 1),
        **uniform,
    }
    return extracted_rows, stats, payload_paths


def main() -> None:
    args = parse_args()
    shard_mode = args.shard_root is not None
    if shard_mode:
        if any(value is not None for value in (args.input_jsonl, args.output_jsonl, args.feature_dir)):
            raise ValueError("Legacy input/output/feature arguments are mutually exclusive with --shard-root")
        if args.split_manifest is None or args.membership is None:
            raise ValueError("Shard mode requires --split-manifest and --membership")
        if args.max_rows is not None:
            raise ValueError("Shard mode is query atomic; use --max-queries, not --max-rows")
    else:
        if not all((args.input_jsonl, args.output_jsonl, args.feature_dir)):
            raise ValueError("Legacy mode requires --input-jsonl, --output-jsonl, and --feature-dir")
        if args.split_manifest is not None or args.membership is not None:
            raise ValueError("--split-manifest/--membership require --shard-root")

    protocol_path = Path(args.protocol_config).resolve()
    protocol = load_protocol(protocol_path)
    hashes = protocol_hashes(protocol)
    protocol_hash = hashes["protocol_sha256"]
    acquisition_hash = hashes["acquisition_protocol_sha256"]
    model_cfg = protocol["model"]
    hidden_cfg = protocol["hidden_states"]
    storage_dtype = args.storage_dtype or hidden_cfg["storage_dtype"]
    generation_cfg = protocol["generation"]

    pending: list[tuple[Dict[str, Any], Path, list[Dict[str, Any]]]] = []
    completed_query_ids: list[str] = []
    split_manifest_hash = None
    if shard_mode:
        split_manifest = load_split_manifest(args.split_manifest)
        split_manifest_hash = split_manifest["manifest_sha256"]
        validate_protocol_reference(split_manifest, protocol)
        entries = membership_entries(
            split_manifest,
            args.membership,
            max_queries=args.max_queries,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
        )
        default_n = candidate_count_for_membership(generation_cfg, args.membership)
        n_rollouts = args.n_rollouts or default_n
        for entry in entries:
            query_dir = query_shard_dir(args.shard_root, entry["query_id"])
            rollout_marker = validate_completion_marker(
                query_dir,
                "_ROLLOUT_SUCCESS.json",
                stage="rollout",
                query_id=entry["query_id"],
                protocol_sha256=protocol_hash,
                acquisition_protocol_sha256=acquisition_hash,
                label_protocol_sha256=hashes["label_protocol_sha256"],
                split_manifest_sha256=split_manifest_hash,
                expected_candidate_count=n_rollouts,
                rows_loader=read_jsonl,
                row_validator=validate_rollout_row,
            )
            extraction_marker_path = query_dir / "_EXTRACTION_SUCCESS.json"
            if extraction_marker_path.exists() and not args.resume and not args.overwrite:
                raise FileExistsError(
                    f"Completed extraction shard exists; use --resume or --overwrite: {query_dir}"
                )
            if args.resume and not args.overwrite:
                try:
                    validate_completion_marker(
                        query_dir,
                        "_EXTRACTION_SUCCESS.json",
                        stage="extraction",
                        query_id=entry["query_id"],
                        protocol_sha256=protocol_hash,
                        acquisition_protocol_sha256=acquisition_hash,
                        label_protocol_sha256=hashes["label_protocol_sha256"],
                        split_manifest_sha256=split_manifest_hash,
                        expected_candidate_count=n_rollouts,
                        rows_loader=read_jsonl,
                        row_validator=_validate_extracted_payload_row,
                    )
                    completed_query_ids.append(entry["query_id"])
                    continue
                except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                    pass
            query_rows = read_jsonl(rollout_marker["_rows_path"])
            _validate_rollout_provenance(query_rows, protocol, hashes)
            pending.append((entry, query_dir, query_rows))
        if not pending:
            print(json.dumps({
                "status": "all_complete",
                "membership": args.membership,
                "shard_id": args.shard_id,
                "skipped_queries": len(completed_query_ids),
            }, indent=2))
            return
    else:
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
        _validate_rollout_provenance(rows, protocol, hashes)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Hidden-state extraction dependencies are missing. Activate SWIFT and install the updated requirements.txt."
        ) from exc

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

    if shard_mode:
        aggregate_stats: list[Dict[str, Any]] = []
        for entry, query_dir, rows in pending:
            feature_root = query_dir / "features"
            extracted_rows, stats, payload_paths = _extract_rows(
                rows,
                model=model,
                tokenizer=tokenizer,
                hidden_cfg=hidden_cfg,
                storage_dtype=storage_dtype,
                feature_root=feature_root,
                overwrite=True,
            )
            output_jsonl = query_dir / "extracted.jsonl"
            atomic_write_jsonl(output_jsonl, extracted_rows)
            payloads = [build_payload_record(output_jsonl, role="rows", root=query_dir)]
            payloads.extend(
                build_payload_record(path, role="feature", root=query_dir)
                for path in payload_paths
            )
            marker = {
                "schema_version": "clir-query-shard-v1",
                "stage": "extraction",
                "query_id": entry["query_id"],
                "source_split": entry["source_split"],
                "source_index": entry["source_index"],
                "protocol_version": protocol["protocol_version"],
                "protocol_sha256": protocol_hash,
                "acquisition_protocol_sha256": acquisition_hash,
                "label_protocol_sha256": hashes["label_protocol_sha256"],
                "split_manifest_sha256": split_manifest_hash,
                "split_membership": args.membership,
                "candidate_count": len(extracted_rows),
                "payloads": payloads,
                "stats": stats,
                "code": git_state(PROJECT_ROOT),
            }
            publish_completion_marker(query_dir, "_EXTRACTION_SUCCESS.json", marker)
            aggregate_stats.append(stats)
        print(json.dumps({
            "status": "completed",
            "shard_root": str(Path(args.shard_root).resolve()),
            "membership": args.membership,
            "shard_id": args.shard_id,
            "skipped_queries": len(completed_query_ids),
            "queries": len(pending),
            "rows": sum(item["rows"] for item in aggregate_stats),
            "feature_bytes": sum(item["feature_bytes"] for item in aggregate_stats),
            "protocol_sha256": protocol_hash,
            "acquisition_protocol_sha256": acquisition_hash,
            "label_protocol_sha256": hashes["label_protocol_sha256"],
        }, indent=2))
        return

    extracted_rows, stats, _ = _extract_rows(
        rows,
        model=model,
        tokenizer=tokenizer,
        hidden_cfg=hidden_cfg,
        storage_dtype=storage_dtype,
        feature_root=feature_root,
        overwrite=args.overwrite,
    )
    atomic_write_jsonl(output_jsonl, extracted_rows)
    report = {
        "protocol_version": protocol["protocol_version"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "acquisition_protocol_sha256": acquisition_hash,
        "label_protocol_sha256": hashes["label_protocol_sha256"],
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "input_sha256": file_sha256(args.input_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_sha256": file_sha256(output_jsonl),
        "feature_root": str(feature_root),
        **stats,
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
        "code": git_state(PROJECT_ROOT),
    }
    atomic_write_json(extraction_manifest_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
