#!/usr/bin/env python3
"""Generate exact-token Phi trajectories for the mixed-domain blind set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_gsm8k_rollouts import (  # noqa: E402
    derive_query_sampling_seed,
    ordered_vllm_candidates,
)
from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_jsonl,
    canonical_sha256,
    file_sha256,
    read_jsonl,
)
from src.clir_real_data import check_gsm8k_response  # noqa: E402
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/jp_h_blind_validation_v1/protocol_v1.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def parse_governed_label(response: str, valid_labels: list[str]) -> str | None:
    matches = list(re.finditer(r"final\s+answer\s*:\s*([^\n\r]+)", response, re.IGNORECASE))
    if not matches:
        return None
    suffix = matches[-1].group(1)
    labels = sorted({label.strip() for label in valid_labels}, key=len, reverse=True)
    for label in labels:
        match = re.search(
            rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])",
            suffix,
            re.IGNORECASE,
        )
        if match:
            return label
    return None


def parse_yes_no(response: str) -> str | None:
    matches = list(re.finditer(r"final\s+answer\s*:\s*([^\n\r]+)", response, re.IGNORECASE))
    if not matches:
        return None
    suffix = matches[-1].group(1)
    match = re.search(r"\b(yes|no|true|false)\b", suffix, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).lower()
    return {"true": "yes", "false": "no"}.get(value, value)


def correctness_record(source: Mapping[str, Any], response: str) -> dict[str, Any]:
    answer_type = source["answer_type"]
    reference = source["raw_reference"]
    if answer_type == "gsm8k_numeric":
        result = check_gsm8k_response(
            response,
            str(reference),
            checker_version="clir_gsm8k_numeric_v5",
        )
        return {
            "correctness": int(result["correctness"]),
            "parsed_answer": result.get("parsed_answer"),
            "reference_answer": result.get("reference_answer"),
            "answer_parse_valid": result.get("parsed_answer") is not None,
            "checker_status": result["checker_status"],
            "checker_version": result["checker_version"],
        }
    if answer_type == "choice_label":
        labels = [str(option["label"]) for option in source["options"]]
        parsed = parse_governed_label(response, labels)
        reference_text = str(reference)
        return {
            "correctness": int(parsed is not None and parsed.casefold() == reference_text.casefold()),
            "parsed_answer": parsed,
            "reference_answer": reference_text,
            "answer_parse_valid": parsed is not None,
            "checker_status": "parsed" if parsed is not None else "parse_failed",
            "checker_version": "clir_governed_choice_label_v1",
        }
    if answer_type == "yes_no":
        parsed = parse_yes_no(response)
        reference_text = str(reference).lower()
        return {
            "correctness": int(parsed is not None and parsed == reference_text),
            "parsed_answer": parsed,
            "reference_answer": reference_text,
            "answer_parse_valid": parsed is not None,
            "checker_status": "parsed" if parsed is not None else "parse_failed",
            "checker_version": "clir_governed_yes_no_v1",
        }
    raise ValueError(f"Unknown answer type: {answer_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--cache-dir")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema_version"] != "clir-jp-h-blind-validation-protocol-v1":
        raise ValueError("Unexpected blind-validation protocol schema")
    protocol_sha256 = file_sha256(protocol_path)
    generation = protocol["generation"]
    sources_path = args.sources.resolve() if args.sources else resolve(protocol["outputs"]["source_questions"])
    output_path = args.output.resolve() if args.output else resolve(protocol["outputs"]["candidate_pool"])
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite frozen candidate pool: {output_path}")
    sources = read_jsonl(sources_path)
    expected_sources = (
        len(protocol["source_sampling"]["domains"])
        * int(protocol["source_sampling"]["queries_per_domain"])
    )
    if len(sources) != expected_sources:
        raise ValueError(f"Expected {expected_sources} source questions, found {len(sources)}")
    if len({row["query_id"] for row in sources}) != len(sources):
        raise ValueError("Source query IDs are not unique")
    if any(row["protocol_sha256"] != protocol_sha256 for row in sources):
        raise ValueError("Source questions were built from a different protocol")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("Activate the SWIFT environment with vLLM installed") from exc

    model_reference = str(args.model_path.resolve()) if args.model_path else generation["model_id"]
    llm_kwargs: dict[str, Any] = {
        "model": model_reference,
        "trust_remote_code": bool(generation["trust_remote_code"]),
        "dtype": generation["dtype"],
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "max_model_len": int(generation["max_model_length"]),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "seed": int(generation["base_seed"]),
        "download_dir": args.cache_dir,
    }
    if args.model_path is None:
        llm_kwargs["revision"] = generation["model_revision"]
        llm_kwargs["tokenizer_revision"] = generation["tokenizer_revision"]
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    formatted_prompts: list[str] = []
    samplings: list[Any] = []
    sampling_seeds: list[int] = []
    for source in sources:
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": source["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        sampling_seed = derive_query_sampling_seed(
            int(generation["base_seed"]),
            str(source["query_id"]),
        )
        sampling_seeds.append(sampling_seed)
        samplings.append(
            SamplingParams(
                n=int(generation["candidates_per_query"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=int(generation["max_new_tokens"]),
                seed=sampling_seed,
            )
        )

    request_outputs = llm.generate(formatted_prompts, samplings, use_tqdm=True)
    if len(request_outputs) != len(sources):
        raise RuntimeError("vLLM request count drifted")
    common = {
        "protocol_sha256": protocol_sha256,
        "source_questions_sha256": file_sha256(sources_path),
        "model_id": generation["model_id"],
        "model_revision": generation["model_revision"],
        "tokenizer_revision": generation["tokenizer_revision"],
        "chat_template_sha256": canonical_sha256(tokenizer.chat_template or ""),
        "generation_backend": "vllm",
        "library_versions": {
            "torch": torch.__version__,
            "vllm": package_version("vllm"),
            "transformers": package_version("transformers"),
        },
        "code": git_state(ROOT),
    }
    rows: list[dict[str, Any]] = []
    for source, sampling_seed, request_output in zip(sources, sampling_seeds, request_outputs):
        prompt_ids = [int(value) for value in request_output.prompt_token_ids]
        ordered = ordered_vllm_candidates(
            request_output,
            int(generation["candidates_per_query"]),
        )
        for candidate in ordered:
            candidate_index = int(candidate.index)
            output_ids = [int(value) for value in candidate.token_ids]
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            check = correctness_record(source, response)
            finish_reason = getattr(candidate, "finish_reason", None)
            rows.append(
                {
                    "schema_version": "clir-jp-h-blind-candidate-v1",
                    "id": f"{source['query_id']}-cand-{candidate_index:03d}",
                    "query_id": source["query_id"],
                    "candidate_index": candidate_index,
                    "domain": source["domain"],
                    "source_split": source["source_split"],
                    "source_index": int(source["source_index"]),
                    "problem": source["problem"],
                    "prompt": source["prompt"],
                    "prompt_token_ids": prompt_ids,
                    "output_token_ids": output_ids,
                    "response": response,
                    "backend_response_text": candidate.text,
                    "decode_matches_backend_text": response == candidate.text,
                    "raw_reference": source["raw_reference"],
                    "answer_type": source["answer_type"],
                    **check,
                    "generation": {
                        "candidate_count": int(generation["candidates_per_query"]),
                        "temperature": float(generation["temperature"]),
                        "top_p": float(generation["top_p"]),
                        "max_new_tokens": int(generation["max_new_tokens"]),
                        "max_model_length": int(generation["max_model_length"]),
                        "base_seed": int(generation["base_seed"]),
                        "sampling_seed": sampling_seed,
                        "seed_scope": generation["seed_scope"],
                        "candidate_index_policy": generation["candidate_index_policy"],
                        "finish_reason": finish_reason,
                        "stop_reason": getattr(candidate, "stop_reason", None),
                        "cumulative_logprob": float(candidate.cumulative_logprob),
                    },
                    "provenance": common,
                }
            )
    expected_rows = expected_sources * int(generation["candidates_per_query"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} candidates, got {len(rows)}")
    if any(not row["output_token_ids"] for row in rows):
        raise ValueError("Empty exact-token trajectory in candidate pool")
    if any(not row["decode_matches_backend_text"] for row in rows):
        raise ValueError("Tokenizer decode differs from vLLM backend text")
    atomic_write_jsonl(output_path, rows)
    per_domain = {
        domain: {
            "rows": sum(row["domain"] == domain for row in rows),
            "parsed": sum(row["domain"] == domain and row["answer_parse_valid"] for row in rows),
            "correct": sum(row["domain"] == domain and row["correctness"] == 1 for row in rows),
            "incorrect_parsed": sum(
                row["domain"] == domain
                and row["answer_parse_valid"]
                and row["correctness"] == 0
                for row in rows
            ),
        }
        for domain in protocol["source_sampling"]["domains"]
    }
    report = {
        "schema_version": "clir-jp-h-blind-generation-report-v1",
        "status": "passed",
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "rows": len(rows),
        "queries": len(sources),
        "per_domain": per_domain,
        "decode_mismatches": 0,
        "protocol_sha256": protocol_sha256,
        "source_questions_sha256": file_sha256(sources_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
