#!/usr/bin/env python3
"""Run the frozen 24B candidate primary annotator over 64 blind items."""

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

from src.clir_hallucination_annotation import (  # noqa: E402
    canonical_json,
    file_sha256,
    read_jsonl,
    validate_annotation,
)
from src.clir_reasoning_rewrite import parse_strict_json_object  # noqa: E402
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/hallucination_localization_v1/protocol_v1.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


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
        "You are a conservative hallucination-localization annotator. Follow the "
        "supplied guide exactly. Do not infer a hidden correctness label. Audit claims "
        "from the problem and trajectory, copy exact trajectory substrings, and return "
        "exactly one strict JSON object with no Markdown or commentary."
    )
    user = (
        "ANNOTATION GUIDE\n<guide>\n"
        + guide
        + "\n</guide>\n\nBLIND ITEM\n<item>\n"
        + canonical_json(dict(item))
        + "\n</item>\n\nReturn one JSON object. Copy item_id exactly."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_context(
    protocol_path: Path,
    *,
    model_path: Path | None,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    outputs = protocol["outputs"]
    items_path = resolve(outputs["annotation_items"])
    guide_path = resolve(protocol["annotation"]["guide"])
    items = read_jsonl(items_path)
    if len(items) != int(protocol["selection"]["rows"]):
        raise ValueError("Frozen blind-item count drifted")
    guide = guide_path.read_text(encoding="utf-8")
    primary = protocol["primary_annotator"]
    frozen = {
        "schema_version": "clir-hallucination-primary-run-contract-v1",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "items_sha256": file_sha256(items_path),
        "guide_sha256": file_sha256(guide_path),
        "model_id": primary["model_id"],
        "model_revision": primary["model_revision"],
        "model_path": str(model_path.resolve()) if model_path is not None else None,
        "decoding": dict(primary["decoding"]),
        "device": device,
    }
    return protocol, items, guide, frozen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dry-run", "run"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.model_path is None or not args.model_path.is_dir():
        raise ValueError("run requires an existing --model-path directory")
    protocol_path = args.protocol.resolve()
    protocol, items, guide, frozen = load_context(
        protocol_path,
        model_path=args.model_path,
        device=args.device,
    )
    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Primary annotation requires a clean committed worktree")
    output_root = resolve(protocol["outputs"]["primary_run_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_path = output_root / "run.json"
    if run_path.exists():
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if record["frozen"] != frozen or record["code"] != code:
            raise ValueError("Existing primary run provenance drifted")
    else:
        record = {
            "schema_version": "clir-hallucination-primary-run-v1",
            "status": "running",
            "frozen": frozen,
            "code": code,
            "runtime": {"python": platform.python_version()},
        }
        temporary = run_path.with_name(f".{run_path.name}.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(run_path)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    ).eval()
    decoding = frozen["decoding"]
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for item_index, item in enumerate(items, start=1):
        item_id = str(item["item_id"])
        item_sha = hashlib.sha256(canonical_json(item).encode("utf-8")).hexdigest()
        shard = output_root / "shards" / f"{item_id}.json"
        shard.parent.mkdir(parents=True, exist_ok=True)
        if shard.exists():
            row = json.loads(shard.read_text(encoding="utf-8"))
            if row["item_sha256"] != item_sha:
                raise ValueError(f"Existing shard identity drifted for {item_id}")
            rows.append(row)
            print(
                json.dumps(
                    {
                        "item": item_index,
                        "total": len(items),
                        "item_id": item_id,
                        "resumed": True,
                        "parse_valid": row["parse_valid"],
                    }
                )
            )
            continue
        messages = prompt_for(item, guide)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
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
            annotation, parse_mode = parse_annotation(completion)
            validate_annotation(annotation, item)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "schema_version": "clir-hallucination-primary-row-v1",
            "item_id": item_id,
            "item_sha256": item_sha,
            "messages_sha256": hashlib.sha256(
                canonical_json(messages).encode("utf-8")
            ).hexdigest(),
            "completion_text": completion,
            "completion_sha256": hashlib.sha256(
                completion.encode("utf-8")
            ).hexdigest(),
            "input_tokens": input_length,
            "output_tokens": int(completion_ids.numel()),
            "elapsed_seconds": elapsed,
            "parse_valid": error is None,
            "parse_mode": parse_mode,
            "parse_error": error,
            "annotation": annotation if error is None else None,
        }
        temporary = shard.with_name(f".{shard.name}.tmp")
        temporary.write_text(
            json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(shard)
        rows.append(row)
        print(
            json.dumps(
                {
                    "item": item_index,
                    "total": len(items),
                    "item_id": item_id,
                    "parse_valid": error is None,
                    "path_status": (
                        annotation["path_status"] if error is None else None
                    ),
                    "elapsed_seconds": elapsed,
                    "parse_error": error,
                }
            )
        )

    results_path = output_root / "all_results.jsonl"
    results_payload = "".join(canonical_json(row) + "\n" for row in rows)
    temporary = results_path.with_name(f".{results_path.name}.tmp")
    temporary.write_text(results_payload, encoding="utf-8")
    temporary.replace(results_path)
    path_counts = Counter(
        row["annotation"]["path_status"]
        for row in rows
        if row["parse_valid"]
    )
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record.update(
        {
            "status": "complete",
            "items": len(rows),
            "parse_valid": sum(int(row["parse_valid"]) for row in rows),
            "path_status_counts": dict(sorted(path_counts.items())),
            "elapsed_seconds": time.perf_counter() - started_all,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "results_sha256": file_sha256(results_path),
        }
    )
    temporary = run_path.with_name(f".{run_path.name}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(run_path)
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    if args.stage == "dry-run":
        protocol, items, guide, frozen = load_context(
            protocol_path,
            model_path=args.model_path,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "items": len(items),
                    "guide_characters": len(guide),
                    "frozen": frozen,
                    "output_root": protocol["outputs"]["primary_run_root"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    run(args)


if __name__ == "__main__":
    main()
