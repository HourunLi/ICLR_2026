#!/usr/bin/env python
"""Generate GSM8K candidates while preserving the backend's exact token ids."""

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

from src.clir_data import write_jsonl
from src.clir_real_data import (
    build_gsm8k_prompt,
    canonical_json_sha256,
    check_gsm8k_response,
    load_protocol,
    validate_rollout_row,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phi35_gsm8k_pilot_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-jsonl", required=True)
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


def _git_state() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> None:
    args = parse_args()
    if args.start_index < 0 or args.max_queries <= 0:
        raise ValueError("start-index must be >= 0 and max-queries must be > 0")
    output_path = Path(args.output_jsonl).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")

    try:
        from datasets import load_dataset
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Real generation dependencies are missing. Activate SWIFT and install the updated requirements.txt."
        ) from exc

    protocol = load_protocol(args.protocol_config)
    model_cfg = protocol["model"]
    dataset_cfg = protocol["dataset"]
    prompt_cfg = protocol["prompt"]
    generation_cfg = protocol["generation"]
    n_rollouts = args.n_rollouts
    if n_rollouts is None:
        n_rollouts = generation_cfg["train_candidates" if args.split == "train" else "pilot_eval_candidates"]
    if n_rollouts <= 0:
        raise ValueError("n-rollouts must be > 0")

    dataset = load_dataset(
        dataset_cfg["repo_id"],
        dataset_cfg["subset"],
        split=args.split,
        revision=dataset_cfg["revision"],
        cache_dir=args.cache_dir,
    )
    stop = min(args.start_index + args.max_queries, len(dataset))
    if args.start_index >= stop:
        raise ValueError(f"Requested slice starts beyond the {args.split} split")
    selected = dataset.select(range(args.start_index, stop))

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
        source_index = args.start_index + offset
        question = sample["question"]
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
                "query_id": f"gsm8k-{args.split}-{source_index:05d}",
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
    )
    request_outputs = llm.generate(prompts, sampling, use_tqdm=True)
    protocol_hash = canonical_json_sha256(protocol)
    chat_template_hash = canonical_json_sha256(tokenizer.chat_template or "")
    common_provenance = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_hash,
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
        "code": _git_state(),
    }

    rows = []
    for query, request_output in zip(query_metadata, request_outputs):
        prompt_ids = [int(value) for value in request_output.prompt_token_ids]
        for candidate_index, candidate in enumerate(request_output.outputs):
            output_ids = [int(value) for value in candidate.token_ids]
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            checker = check_gsm8k_response(response, query["raw_reference"])
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
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "stop_reason": getattr(candidate, "stop_reason", None),
                },
                "provenance": dict(common_provenance),
            }
            validate_rollout_row(row)
            rows.append(row)

    write_jsonl(output_path, rows)
    print(
        json.dumps(
            {
                "output_jsonl": str(output_path),
                "queries": len(query_metadata),
                "rows": len(rows),
                "correct": sum(int(row["correctness"]) for row in rows),
                "decode_mismatches": sum(not row["decode_matches_backend_text"] for row in rows),
                "protocol_sha256": protocol_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
