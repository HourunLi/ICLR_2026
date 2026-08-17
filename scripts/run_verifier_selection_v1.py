#!/usr/bin/env python3
"""Run and score one frozen verifier candidate on selection v1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

from scripts.audit_verifier_selection_v1 import load_jsonl, validate_labels
from src.clir_reasoning_rewrite import parse_strict_json_object


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs" / "verifier_selection_v1"
SCHEMA_KEYS = {
    "item_id", "decision", "same_task_and_goal", "same_core_premises",
    "same_reasoning_method", "same_key_inferences", "same_intermediate_conclusions",
    "same_final_conclusion", "entailed_elaboration_present",
    "entailed_elaboration_valid", "introduced_new_premise_or_evidence",
    "omitted_essential_claim", "contradicted_source_claim",
    "replaced_with_different_solution", "introduced_new_error",
    "error_alignment_applicable", "same_error_mechanism",
    "same_semantic_error_location", "same_downstream_effect", "target_style",
    "style_satisfied", "confidence", "reason",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_candidate_annotation(text: str) -> tuple[dict[str, Any], str]:
    """Accept strict JSON or one exact JSON fence; never extract arbitrary text."""

    try:
        return parse_strict_json_object(text), "strict_json"
    except ValueError as strict_error:
        payload = text.strip()
        if not payload.startswith("```json\n") or not payload.endswith("\n```"):
            raise strict_error
        inner = payload[len("```json\n") : -len("\n```")]
        return parse_strict_json_object(inner), "single_json_fence"


def prompt_for(item: dict[str, Any], guide: str) -> list[dict[str, str]]:
    model_guide = guide.split("## Independence requirements", 1)[0].rstrip()
    system = (
        "You are a conservative reasoning-equivalence auditor. Follow the supplied "
        "annotation guide exactly. Compare the source and rewrite; do not solve the "
        "problem again or repair an incorrect source. Return exactly one strict JSON "
        "object with all required fields and no Markdown or commentary."
    )
    user = (
        "ANNOTATION GUIDE\n<guide>\n" + model_guide + "\n</guide>\n\n"
        "ITEM\n<item>\n" + canonical(item) + "\n</item>\n\n"
        "Return one JSON object for this item. Copy item_id and target_style exactly."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dry-run", "run", "score"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", default="mistralai/Mistral-Small-24B-Instruct-2501")
    parser.add_argument("--model-revision", default="9527884be6e5616bdd54de542f9ae13384489724")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def context(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    items_path = DATA / "verifier_selection_items_v1.jsonl"
    guide_path = DATA / "verifier_selection_annotation_guide_v1.md"
    items = load_jsonl(items_path)
    guide = guide_path.read_text(encoding="utf-8")
    frozen = {
        "schema_version": "clir-verifier-selection-run-v1",
        "items_sha256": sha256(items_path),
        "guide_sha256": sha256(guide_path),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_path": str(args.model_path.resolve()) if args.model_path else None,
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
        "device": args.device,
    }
    return items, guide, frozen


def run(args: argparse.Namespace) -> None:
    if args.model_path is None or not args.model_path.is_dir():
        raise ValueError("run requires an existing --model-path")
    items, guide, frozen = context(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_path = args.output_root / "run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["frozen"] != frozen:
            raise ValueError("Existing verifier selection run provenance drifted")
    else:
        run_path.write_text(
            json.dumps(
                {
                    "frozen": frozen,
                    "runtime": {"python": platform.python_version()},
                    "status": "running",
                }, indent=2, sort_keys=True
            ) + "\n", encoding="utf-8"
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
    rows = []
    valid = 0
    started_all = time.perf_counter()
    for item in items:
        shard = args.output_root / "shards" / f"{item['item_id']}.json"
        shard.parent.mkdir(parents=True, exist_ok=True)
        if shard.exists():
            row = json.loads(shard.read_text(encoding="utf-8"))
            if row["item_sha256"] != hashlib.sha256(canonical(item).encode()).hexdigest():
                raise ValueError(f"Existing shard identity drifted for {item['item_id']}")
            rows.append(row)
            valid += int(row["parse_valid"])
            continue
        messages = prompt_for(item, guide)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        completion_ids = output[0, input_length:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        parsed = None
        error = None
        parse_mode = None
        try:
            parsed, parse_mode = parse_candidate_annotation(text)
            validate_labels([item], [parsed], primary=False)
        except Exception as exc:  # Persist exact model failures for audit.
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "schema_version": "clir-verifier-selection-candidate-row-v1",
            "item_id": item["item_id"],
            "item_sha256": hashlib.sha256(canonical(item).encode()).hexdigest(),
            "messages_sha256": hashlib.sha256(canonical(messages).encode()).hexdigest(),
            "completion_text": text,
            "completion_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "input_tokens": int(input_length),
            "output_tokens": int(completion_ids.numel()),
            "elapsed_seconds": elapsed,
            "parse_valid": error is None,
            "parse_mode": parse_mode,
            "parse_error": error,
            "annotation": parsed if error is None else None,
        }
        temporary = shard.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(shard)
        rows.append(row)
        valid += int(error is None)
        print(json.dumps({"item_id": item["item_id"], "parse_valid": error is None, "elapsed": elapsed}))
    (args.output_root / "all_results.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in rows), encoding="utf-8"
    )
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record.update({
        "status": "complete", "items": len(rows), "parse_valid": valid,
        "elapsed_seconds": time.perf_counter() - started_all,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
    })
    run_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score(args: argparse.Namespace) -> None:
    items, _, frozen = context(args)
    run_record = json.loads((args.output_root / "run.json").read_text(encoding="utf-8"))
    if run_record["frozen"] != frozen or run_record["status"] != "complete":
        raise ValueError("Cannot score an incomplete or provenance-drifted run")
    rows = load_jsonl(args.output_root / "all_results.jsonl")
    gold = load_jsonl(DATA / "verifier_selection_gold_v1.jsonl")
    if len(rows) != len(items) or len(gold) != len(items):
        raise ValueError("Selection scoring row count drifted")
    confusion: Counter[str] = Counter()
    valid = correct = 0
    accept_tp = accept_fp = 0
    records = []
    for item, row, target in zip(items, rows, gold):
        if item["item_id"] != row["item_id"] or row["item_id"] != target["item_id"]:
            raise ValueError("Selection scoring order drifted")
        prediction = row["annotation"]["decision"] if row["parse_valid"] else "invalid"
        truth = target["decision"]
        confusion[f"{truth}->{prediction}"] += 1
        valid += int(row["parse_valid"])
        correct += int(prediction == truth)
        accept_tp += int(prediction == "accept" and truth == "accept")
        accept_fp += int(prediction == "accept" and truth != "accept")
        records.append({"item_id": item["item_id"], "domain": item["domain"], "gold": truth, "prediction": prediction})
    report = {
        "schema_version": "clir-verifier-selection-score-v1",
        "model_id": frozen["model_id"], "model_revision": frozen["model_revision"],
        "items": len(items), "parse_valid": valid,
        "parse_valid_rate": valid / len(items), "decision_accuracy": correct / len(items),
        "auto_accept_precision": accept_tp / (accept_tp + accept_fp) if accept_tp + accept_fp else None,
        "auto_accept_count": accept_tp + accept_fp,
        "confusion": dict(sorted(confusion.items())),
        "gold_sha256": sha256(DATA / "verifier_selection_gold_v1.jsonl"),
    }
    (args.output_root / "score_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "scored_records.jsonl").write_text("".join(canonical(row) + "\n" for row in records), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.stage == "dry-run":
        items, guide, frozen = context(args)
        print(json.dumps({"items": len(items), "guide_chars": len(guide), "frozen": frozen}, indent=2, sort_keys=True))
    elif args.stage == "run":
        run(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
