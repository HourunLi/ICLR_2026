#!/usr/bin/env python
"""Generate GSM8K candidates while preserving the backend's exact token ids."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch

# Make direct execution (`python /path/to/script.py`) independent of PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl, write_jsonl
from src.clir_real_data import (
    build_gsm8k_prompt,
    canonical_json_sha256,
    check_gsm8k_response,
    load_protocol,
    protocol_hashes,
    validate_protocol_reference,
    validate_rollout_row,
)
from src.clir_stage_a import (
    atomic_write_jsonl,
    build_payload_record,
    candidate_count_for_membership,
    git_state,
    load_split_manifest,
    membership_entries,
    publish_completion_marker,
    query_shard_dir,
    sha256_text,
    validate_completion_marker,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phi35_gsm8k_pilot_v3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-jsonl", default=None,
                        help="Legacy single-file output; mutually exclusive with --shard-root.")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--membership", default=None)
    parser.add_argument("--shard-root", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1,
        help="Safety default for the alignment gate; increase only after validating the first artifact.",
    )
    parser.add_argument("--n-rollouts", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=0, help="0 means all visible CUDA devices")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def ordered_vllm_candidates(request_output: Any, expected_count: int) -> list[Any]:
    """Recover vLLM's original sample order from ``CompletionOutput.index``.

    vLLM 0.5.x exposes ``request_output.outputs`` in cumulative-logprob order,
    not generation order. ``CompletionOutput.index`` is the retained original
    sequence index and is therefore the only valid Best-of-N prefix key.
    """

    candidates = list(request_output.outputs)
    indices = [int(candidate.index) for candidate in candidates]
    if sorted(indices) != list(range(expected_count)):
        raise ValueError(
            "vLLM candidate indices must be unique and contiguous: "
            f"expected 0..{expected_count - 1}, got {sorted(indices)}"
        )
    return sorted(candidates, key=lambda candidate: int(candidate.index))


def main() -> None:
    args = parse_args()
    if args.start_index < 0 or args.max_queries <= 0:
        raise ValueError("start-index must be >= 0 and max-queries must be > 0")
    shard_mode = args.shard_root is not None
    if shard_mode:
        if args.output_jsonl is not None:
            raise ValueError("--output-jsonl and --shard-root are mutually exclusive")
        if args.split_manifest is None or args.membership is None:
            raise ValueError("Shard mode requires --split-manifest and --membership")
        if args.start_index != 0:
            raise ValueError("Shard mode selects frozen membership IDs and does not accept --start-index")
        output_path = None
    else:
        if args.output_jsonl is None:
            raise ValueError("Provide --output-jsonl or enable --shard-root mode")
        if args.split_manifest is not None or args.membership is not None:
            raise ValueError("--split-manifest/--membership require --shard-root")
        output_path = Path(args.output_jsonl).resolve()
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")

    protocol = load_protocol(args.protocol_config)
    model_cfg = protocol["model"]
    dataset_cfg = protocol["dataset"]
    prompt_cfg = protocol["prompt"]
    generation_cfg = protocol["generation"]
    n_rollouts = args.n_rollouts
    if n_rollouts is None:
        n_rollouts = (
            candidate_count_for_membership(generation_cfg, args.membership)
            if shard_mode
            else generation_cfg[
                "train_candidates" if args.split == "train" else "pilot_eval_candidates"
            ]
        )
    if n_rollouts <= 0:
        raise ValueError("n-rollouts must be > 0")

    hashes = protocol_hashes(protocol)
    protocol_hash = hashes["protocol_sha256"]
    acquisition_hash = hashes["acquisition_protocol_sha256"]
    label_hash = hashes["label_protocol_sha256"]
    split_manifest = None
    split_manifest_hash = None
    frozen_entries = None
    completed_query_ids: list[str] = []
    if shard_mode:
        split_manifest = load_split_manifest(args.split_manifest)
        split_manifest_hash = split_manifest["manifest_sha256"]
        validate_protocol_reference(split_manifest, protocol)
        frozen_entries = membership_entries(
            split_manifest,
            args.membership,
            max_queries=args.max_queries,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
        )
        if not frozen_entries:
            print(json.dumps({"status": "empty_shard", "shard_id": args.shard_id}, indent=2))
            return

        resume_rejections: Dict[str, str] = {}
        for entry in frozen_entries:
            if entry["source_split"] != args.split:
                raise ValueError(
                    f"Membership {args.membership} contains {entry['source_split']} query "
                    f"but --split is {args.split}"
                )
            query_dir = query_shard_dir(args.shard_root, entry["query_id"])
            marker_path = query_dir / "_ROLLOUT_SUCCESS.json"
            if marker_path.exists() and not args.resume and not args.overwrite:
                raise FileExistsError(
                    f"Completed query shard exists; use --resume or --overwrite: {query_dir}"
                )
            if args.resume and not args.overwrite:
                try:
                    validate_completion_marker(
                        query_dir,
                        "_ROLLOUT_SUCCESS.json",
                        stage="rollout",
                        query_id=entry["query_id"],
                        protocol_sha256=protocol_hash,
                        acquisition_protocol_sha256=acquisition_hash,
                        label_protocol_sha256=label_hash,
                        split_manifest_sha256=split_manifest_hash,
                        expected_candidate_count=n_rollouts,
                        rows_loader=read_jsonl,
                        row_validator=validate_rollout_row,
                    )
                    completed_query_ids.append(entry["query_id"])
                except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                    resume_rejections[entry["query_id"]] = f"{type(exc).__name__}: {exc}"
        frozen_entries = [
            entry for entry in frozen_entries if entry["query_id"] not in set(completed_query_ids)
        ]
        if not frozen_entries:
            print(
                json.dumps(
                    {
                        "status": "all_complete",
                        "membership": args.membership,
                        "shard_id": args.shard_id,
                        "skipped_queries": len(completed_query_ids),
                        "resume_rejections": resume_rejections,
                    },
                    indent=2,
                )
            )
            return

    try:
        from datasets import load_dataset
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Real generation dependencies are missing. Activate SWIFT and install the updated requirements.txt."
        ) from exc

    dataset = load_dataset(
        dataset_cfg["repo_id"],
        dataset_cfg["subset"],
        split=args.split,
        revision=dataset_cfg["revision"],
        cache_dir=args.cache_dir,
    )
    if shard_mode:
        source_indices = [int(entry["source_index"]) for entry in frozen_entries]
        if any(index < 0 or index >= len(dataset) for index in source_indices):
            raise ValueError("Split manifest source index is outside the loaded dataset")
        selected = dataset.select(source_indices)
    else:
        stop = min(args.start_index + args.max_queries, len(dataset))
        if args.start_index >= stop:
            raise ValueError(f"Requested slice starts beyond the {args.split} split")
        source_indices = list(range(args.start_index, stop))
        selected = dataset.select(source_indices)

    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()
    if tensor_parallel_size <= 0:
        raise RuntimeError("vLLM generation requires at least one visible CUDA device")
    llm = LLM(
        model=model_cfg["repo_id"],
        revision=model_cfg["revision"],
        tokenizer_revision=model_cfg["tokenizer_revision"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        dtype=model_cfg["torch_dtype"],
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=int(generation_cfg["max_model_length"]),
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=int(generation_cfg["seed"]),
        download_dir=args.cache_dir,
    )
    tokenizer = llm.get_tokenizer()
    prompts = []
    query_metadata = []
    for offset, sample in enumerate(selected):
        source_index = source_indices[offset]
        question = sample["question"]
        frozen_entry = frozen_entries[offset] if shard_mode else None
        if frozen_entry is not None:
            if frozen_entry["query_id"] != f"gsm8k-{args.split}-{source_index:05d}":
                raise ValueError("Frozen query ID/source index mismatch")
            if frozen_entry["question_sha256"] != sha256_text(question):
                raise ValueError(f"Frozen question hash mismatch for {frozen_entry['query_id']}")
        prompt = build_gsm8k_prompt(question, prompt_cfg["template"])
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(formatted)
        query_metadata.append(
            {
                "source_index": source_index,
                "query_id": (
                    frozen_entry["query_id"]
                    if frozen_entry is not None
                    else f"gsm8k-{args.split}-{source_index:05d}"
                ),
                "question": question,
                "prompt": prompt,
                "raw_reference": sample["answer"],
            }
        )

    sampling = SamplingParams(
        n=n_rollouts,
        temperature=float(generation_cfg["temperature"]),
        top_p=float(generation_cfg["top_p"]),
        max_tokens=int(generation_cfg["max_new_tokens"]),
        stop=list(generation_cfg.get("stop_sequences", [])) or None,
        seed=int(generation_cfg["seed"]),
    )
    request_outputs = llm.generate(prompts, sampling, use_tqdm=True)
    chat_template_hash = canonical_json_sha256(tokenizer.chat_template or "")
    common_provenance = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_hash,
        "acquisition_protocol_sha256": acquisition_hash,
        "label_protocol_sha256": label_hash,
        "model_id": model_cfg["repo_id"],
        "model_revision": model_cfg["revision"],
        "tokenizer_revision": model_cfg["tokenizer_revision"],
        "chat_template_sha256": chat_template_hash,
        "dataset_id": dataset_cfg["repo_id"],
        "dataset_subset": dataset_cfg["subset"],
        "dataset_revision": dataset_cfg["revision"],
        "generation_backend": "vllm",
        "library_versions": {
            "torch": torch.__version__,
            "vllm": _version("vllm"),
            "transformers": _version("transformers"),
            "datasets": _version("datasets"),
        },
        "code": git_state(PROJECT_ROOT),
    }
    if shard_mode:
        common_provenance["split_manifest_sha256"] = split_manifest_hash
        common_provenance["split_membership"] = args.membership

    rows = []
    rows_by_query: Dict[str, list[Dict[str, Any]]] = {}
    for query, request_output in zip(query_metadata, request_outputs):
        prompt_ids = [int(value) for value in request_output.prompt_token_ids]
        query_rows: list[Dict[str, Any]] = []
        for candidate in ordered_vllm_candidates(request_output, n_rollouts):
            candidate_index = int(candidate.index)
            output_ids = [int(value) for value in candidate.token_ids]
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            checker = check_gsm8k_response(
                response,
                query["raw_reference"],
                checker_version=protocol["correctness"]["checker"],
            )
            row = {
                "id": f"{query['query_id']}-cand-{candidate_index:03d}",
                "query_id": query["query_id"],
                "candidate_index": candidate_index,
                "source": "gsm8k",
                "split": args.split,
                "source_index": query["source_index"],
                "question": query["question"],
                "prompt": query["prompt"],
                "prompt_token_ids": prompt_ids,
                "output_token_ids": output_ids,
                "response": response,
                "trajectory": response,
                "backend_response_text": candidate.text,
                "decode_matches_backend_text": response == candidate.text,
                "raw_reference": query["raw_reference"],
                **checker,
                "generation": {
                    "candidate_count": n_rollouts,
                    "temperature": generation_cfg["temperature"],
                    "top_p": generation_cfg["top_p"],
                    "max_new_tokens": generation_cfg["max_new_tokens"],
                    "max_model_length": generation_cfg["max_model_length"],
                    "seed": generation_cfg["seed"],
                    "terminal_token_policy": generation_cfg["terminal_token_policy"],
                    "candidate_index_policy": "vllm_completion_output_index",
                    "cumulative_logprob": float(candidate.cumulative_logprob),
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "stop_reason": getattr(candidate, "stop_reason", None),
                },
                "provenance": dict(common_provenance),
            }
            validate_rollout_row(row)
            query_rows.append(row)
        rows.extend(query_rows)
        rows_by_query[query["query_id"]] = query_rows

    if shard_mode:
        for query in query_metadata:
            query_id = query["query_id"]
            query_dir = query_shard_dir(args.shard_root, query_id)
            rows_path = query_dir / "rollouts.jsonl"
            query_rows = rows_by_query[query_id]
            atomic_write_jsonl(rows_path, query_rows)
            marker = {
                "schema_version": "clir-query-shard-v1",
                "stage": "rollout",
                "query_id": query_id,
                "source_split": args.split,
                "source_index": query["source_index"],
                "protocol_version": protocol["protocol_version"],
                "protocol_sha256": protocol_hash,
                "acquisition_protocol_sha256": acquisition_hash,
                "label_protocol_sha256": label_hash,
                "split_manifest_sha256": split_manifest_hash,
                "split_membership": args.membership,
                "candidate_count": n_rollouts,
                "payloads": [build_payload_record(rows_path, role="rows", root=query_dir)],
                "stats": {
                    "correct": sum(int(row["correctness"]) for row in query_rows),
                    "decode_mismatches": sum(
                        not row["decode_matches_backend_text"] for row in query_rows
                    ),
                    "output_tokens": sum(len(row["output_token_ids"]) for row in query_rows),
                },
                "code": git_state(PROJECT_ROOT),
            }
            publish_completion_marker(query_dir, "_ROLLOUT_SUCCESS.json", marker)
    else:
        write_jsonl(output_path, rows)
    print(
        json.dumps(
            {
                "output_jsonl": str(output_path) if output_path is not None else None,
                "shard_root": str(Path(args.shard_root).resolve()) if shard_mode else None,
                "membership": args.membership if shard_mode else None,
                "shard_id": args.shard_id if shard_mode else None,
                "skipped_queries": len(completed_query_ids),
                "queries": len(query_metadata),
                "rows": len(rows),
                "correct": sum(int(row["correctness"]) for row in rows),
                "decode_mismatches": sum(not row["decode_matches_backend_text"] for row in rows),
                "protocol_sha256": protocol_hash,
                "acquisition_protocol_sha256": acquisition_hash,
                "label_protocol_sha256": label_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
