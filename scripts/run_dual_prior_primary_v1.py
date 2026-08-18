#!/usr/bin/env python3
"""Run/finalize the frozen 24B primary key/complete evidence annotation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_dual_prior_annotation import validate_annotation, validate_item  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    file_sha256,
    read_jsonl,
)
from src.clir_reasoning_rewrite import parse_strict_json_object  # noqa: E402
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_annotation(text: str) -> tuple[dict[str, Any], str]:
    try:
        return parse_strict_json_object(text), "strict_json"
    except ValueError as strict_error:
        payload = text.strip()
        if not payload.startswith("```json\n") or not payload.endswith("\n```"):
            raise strict_error
        inner = payload[len("```json\n") : -len("\n```")]
        return parse_strict_json_object(inner), "single_json_fence"


def prompt_for(item: Mapping[str, Any], guide: str) -> list[dict[str, str]]:
    system = (
        "You are a conservative reasoning-evidence annotator. Follow the supplied "
        "guide exactly. Select only fixed unit indices. Do not output a correctness "
        "verdict, rewrite text, or use hidden labels. Return exactly one strict JSON "
        "object with no Markdown or commentary."
    )
    user = (
        "ANNOTATION GUIDE\n<guide>\n"
        + guide
        + "\n</guide>\n\nBLIND ITEM\n<item>\n"
        + canonical_json(dict(item))
        + "\n</item>\n\nReturn one JSON object. Copy item_id exactly."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_context(protocol_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-dual-prior-evidence-protocol-v1":
        raise ValueError("Unexpected dual-prior protocol schema")
    outputs = protocol["outputs"]
    items_path = resolve(outputs["annotation_items"])
    guide_path = resolve(protocol["annotation"]["guide"])
    package_path = resolve(outputs["package_report"])
    package = json.loads(package_path.read_text(encoding="utf-8"))
    items = read_jsonl(items_path)
    if len(items) != int(protocol["source"]["rows"]):
        raise ValueError("Frozen dual-prior item count drifted")
    for item in items:
        validate_item(item)
    if file_sha256(items_path) != package["annotation_items_sha256"]:
        raise ValueError("Frozen dual-prior items hash drifted")
    guide = guide_path.read_text(encoding="utf-8")
    if file_sha256(guide_path) != package["annotation_guide_sha256"]:
        raise ValueError("Frozen dual-prior guide hash drifted")
    primary = protocol["primary_annotator"]
    frozen = {
        "schema_version": "clir-dual-prior-primary-run-contract-v1",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "items_sha256": file_sha256(items_path),
        "guide_sha256": file_sha256(guide_path),
        "model_id": primary["model_id"],
        "model_revision": primary["model_revision"],
        "decoding": dict(primary["decoding"]),
    }
    return protocol, items, guide, frozen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dry-run", "run-shard", "finalize"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--retry-invalid", action="store_true")
    return parser.parse_args()


def run_shard(args: argparse.Namespace) -> None:
    if args.model_path is None or not args.model_path.is_dir():
        raise ValueError("run-shard requires an existing --model-path directory")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    protocol_path = args.protocol.resolve()
    protocol, items, guide, frozen = load_context(protocol_path)
    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Primary annotation requires a clean committed worktree")
    selected = [
        (index, item)
        for index, item in enumerate(items)
        if index % args.num_shards == args.shard_index
    ]
    output_root = resolve(protocol["outputs"]["primary_run_root"])
    worker_path = output_root / "workers" / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    worker_contract = {
        **frozen,
        "model_path": str(args.model_path.resolve()),
        "device": args.device,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "item_indices": [index for index, _ in selected],
    }
    if worker_path.exists():
        prior_worker = json.loads(worker_path.read_text(encoding="utf-8"))
        if prior_worker.get("frozen") != worker_contract or prior_worker.get("code") != code:
            raise ValueError("Existing worker provenance drifted")
    atomic_write_json(
        worker_path,
        {
            "schema_version": "clir-dual-prior-primary-worker-v1",
            "status": "running",
            "frozen": worker_contract,
            "code": code,
            "runtime": {"python": platform.python_version()},
        },
    )

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    ).eval()
    decoding = frozen["decoding"]
    parse_counts: Counter[str] = Counter()
    started_all = time.perf_counter()
    for local_number, (global_index, item) in enumerate(selected, start=1):
        item_id = str(item["item_id"])
        item_sha = hashlib.sha256(canonical_json(item).encode()).hexdigest()
        shard_path = shard_root / f"{item_id}.json"
        previous_attempts: list[dict[str, Any]] = []
        if shard_path.exists():
            existing = json.loads(shard_path.read_text(encoding="utf-8"))
            if existing.get("item_sha256") != item_sha:
                raise ValueError(f"Existing shard identity drifted for {item_id}")
            if existing.get("run_contract") != frozen or existing.get("code") != code:
                raise ValueError(f"Existing shard provenance drifted for {item_id}")
            if existing.get("parse_valid") or not args.retry_invalid:
                parse_counts["valid" if existing.get("parse_valid") else "invalid"] += 1
                print(json.dumps({"item": local_number, "total": len(selected), "item_id": item_id, "resumed": True, "parse_valid": existing.get("parse_valid")}))
                continue
            previous_attempts = list(existing.get("previous_attempts", []))
            previous_attempts.append(
                {
                    key: existing.get(key)
                    for key in (
                        "completion_text",
                        "completion_sha256",
                        "parse_valid",
                        "parse_mode",
                        "parse_error",
                    )
                }
            )
        messages = prompt_for(item, guide)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=bool(decoding["do_sample"]),
                num_beams=int(decoding["num_beams"]),
                max_new_tokens=int(decoding["max_new_tokens"]),
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        completion_ids = generated[0, input_length:]
        completion = tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        annotation = None
        parse_mode = None
        error = None
        try:
            parsed, parse_mode = parse_annotation(completion)
            annotation = validate_annotation(parsed, item)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "schema_version": "clir-dual-prior-primary-row-v1",
            "item_index": global_index,
            "item_id": item_id,
            "item_sha256": item_sha,
            "run_contract": frozen,
            "code": code,
            "messages_sha256": hashlib.sha256(canonical_json(messages).encode()).hexdigest(),
            "completion_text": completion,
            "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
            "input_tokens": input_length,
            "output_tokens": int(completion_ids.numel()),
            "elapsed_seconds": elapsed,
            "parse_valid": error is None,
            "parse_mode": parse_mode,
            "parse_error": error,
            "annotation": annotation,
            "previous_attempts": previous_attempts,
        }
        atomic_write_json(shard_path, row)
        parse_counts["valid" if error is None else "invalid"] += 1
        print(json.dumps({"item": local_number, "total": len(selected), "item_id": item_id, "parse_valid": error is None, "eligibility": annotation.get("eligibility") if annotation else None, "elapsed_seconds": elapsed, "parse_error": error}))

    atomic_write_json(
        worker_path,
        {
            "schema_version": "clir-dual-prior-primary-worker-v1",
            "status": "complete",
            "frozen": worker_contract,
            "code": code,
            "runtime": {
                "python": platform.python_version(),
                "transformers": transformers.__version__,
                "torch": torch.__version__,
            },
            "items": len(selected),
            "parse_counts": dict(sorted(parse_counts.items())),
            "elapsed_seconds": time.perf_counter() - started_all,
        },
    )


def finalize(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol, items, _guide, frozen = load_context(protocol_path)
    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Primary annotation finalization requires a clean committed worktree")
    output_root = resolve(protocol["outputs"]["primary_run_root"])
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []
    for index, item in enumerate(items):
        item_id = str(item["item_id"])
        shard_path = output_root / "shards" / f"{item_id}.json"
        if not shard_path.exists():
            missing.append(item_id)
            continue
        row = json.loads(shard_path.read_text(encoding="utf-8"))
        if row.get("item_index") != index or row.get("run_contract") != frozen or row.get("code") != code:
            raise ValueError(f"Primary shard provenance/order drifted for {item_id}")
        if not row.get("parse_valid") or row.get("annotation") is None:
            invalid.append(item_id)
        else:
            validate_annotation(row["annotation"], item)
        rows.append(row)
    if missing or invalid:
        raise ValueError(f"Cannot finalize primary annotation: missing={missing}, invalid={invalid}")
    results_path = output_root / "all_results.jsonl"
    atomic_write_jsonl(results_path, rows)
    eligibility = Counter(row["annotation"]["eligibility"] for row in rows)
    run_record = {
        "schema_version": "clir-dual-prior-primary-run-v1",
        "status": "complete",
        "frozen": frozen,
        "code": code,
        "rows": len(rows),
        "parse_valid": len(rows),
        "eligibility_counts": dict(sorted(eligibility.items())),
        "results_sha256": file_sha256(results_path),
        "workers": sorted(path.name for path in (output_root / "workers").glob("*.json")),
    }
    atomic_write_json(output_root / "run.json", run_record)
    print(json.dumps(run_record, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.stage == "dry-run":
        protocol, items, guide, frozen = load_context(args.protocol.resolve())
        print(json.dumps({"rows": len(items), "guide_characters": len(guide), "frozen": frozen, "first_messages": prompt_for(items[0], guide)}, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.stage == "run-shard":
        run_shard(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
