#!/usr/bin/env python3
"""Run the selected Silver verifier over frozen Route A Pilot-0 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

from scripts.run_verifier_selection_v1 import parse_candidate_annotation
from src.clir_data import read_jsonl
from src.clir_on_policy_pairs import canonical_json


ROOT = Path(__file__).resolve().parents[1]
LABEL_KEYS = {
    "item_id", "decision", "same_task_and_goal", "same_core_premises",
    "same_reasoning_method", "same_key_inferences", "same_intermediate_conclusions",
    "same_final_conclusion", "entailed_elaboration_present", "entailed_elaboration_valid",
    "introduced_new_premise_or_evidence", "omitted_essential_claim",
    "contradicted_source_claim", "replaced_with_different_solution", "introduced_new_error",
    "error_alignment_applicable", "same_error_mechanism", "same_semantic_error_location",
    "same_downstream_effect", "target_style", "style_satisfied", "confidence", "reason",
}
BOOL_KEYS = {
    "same_task_and_goal", "same_core_premises", "same_reasoning_method",
    "same_key_inferences", "same_intermediate_conclusions", "same_final_conclusion",
    "entailed_elaboration_present", "introduced_new_premise_or_evidence",
    "omitted_essential_claim", "contradicted_source_claim", "replaced_with_different_solution",
    "introduced_new_error", "error_alignment_applicable", "style_satisfied",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_annotation(value: dict[str, Any], pair_id: str) -> None:
    if set(value) != LABEL_KEYS:
        raise ValueError("Verifier annotation keys differ from schema")
    if value["item_id"] != pair_id:
        raise ValueError("Verifier item_id drifted")
    if value["decision"] not in {"accept", "reject", "review"}:
        raise ValueError("Verifier decision is invalid")
    if value["target_style"] != "native_surface_difference":
        raise ValueError("Verifier target_style drifted")
    if value["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("Verifier confidence is invalid")
    if not isinstance(value["reason"], str) or len(value["reason"].strip()) < 20:
        raise ValueError("Verifier reason is not evidence-specific")
    if any(type(value[key]) is not bool for key in BOOL_KEYS):
        raise ValueError("Verifier Boolean field has invalid type")
    if value["entailed_elaboration_present"]:
        if type(value["entailed_elaboration_valid"]) is not bool:
            raise ValueError("Elaboration validity is required")
    elif value["entailed_elaboration_valid"] is not None:
        raise ValueError("Unused elaboration validity must be null")
    if value["error_alignment_applicable"]:
        if any(type(value[key]) is not bool for key in ("same_error_mechanism", "same_semantic_error_location", "same_downstream_effect")):
            raise ValueError("Error alignment values are incomplete")
    elif any(value[key] is not None for key in ("same_error_mechanism", "same_semantic_error_location", "same_downstream_effect")):
        raise ValueError("Inapplicable error alignment values must be null")


def item_projection(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the model-visible projection; omit correctness/outcome/provenance."""

    return {
        "item_id": candidate["pair_id"],
        "problem": candidate["problem"],
        "anchor_trajectory": candidate["anchor"]["response"],
        "alternate_trajectory": candidate["alternate"]["response"],
        "target_style": "native_surface_difference",
    }


def prompt_for(candidate: dict[str, Any], guide: str) -> list[dict[str, str]]:
    item = item_projection(candidate)
    system = (
        "You are a conservative reasoning-equivalence auditor. Follow the guide exactly. "
        "Judge the relationship between the supplied trajectories without solving the problem "
        "again. Return exactly one strict JSON object with no commentary."
    )
    user = f"GUIDE\n<guide>\n{guide}\n</guide>\n\nITEM\n<item>\n{canonical_json(item)}\n</item>"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def silver_status(annotation: dict[str, Any]) -> str:
    positive = all(annotation[key] for key in (
        "same_task_and_goal", "same_core_premises", "same_reasoning_method",
        "same_key_inferences", "same_intermediate_conclusions", "same_final_conclusion",
        "style_satisfied",
    ))
    prohibited = any(annotation[key] for key in (
        "introduced_new_premise_or_evidence", "omitted_essential_claim",
        "contradicted_source_claim", "replaced_with_different_solution", "introduced_new_error",
    ))
    elaboration_ok = not annotation["entailed_elaboration_present"] or annotation["entailed_elaboration_valid"] is True
    if annotation["decision"] == "reject" or prohibited:
        return "rejected"
    if annotation["decision"] == "accept" and annotation["confidence"] == "high" and positive and elaboration_ok:
        return "silver_accepted"
    return "review_required"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, default=ROOT / "run_artifacts/on_policy_pilot0_v1/candidate_build/candidates.jsonl")
    parser.add_argument("--guide", type=Path, default=ROOT / "configs/on_policy_pilot0_verifier_guide_v1.md")
    parser.add_argument("--output-root", type=Path, default=ROOT / "run_artifacts/on_policy_pilot0_v1/verifier_mistral24_v1")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_jsonl(args.candidate_manifest)
    guide = args.guide.read_text(encoding="utf-8")
    frozen = {
        "schema_version": "clir-on-policy-verifier-run-v1",
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "guide_sha256": sha256(args.guide),
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "model_revision": "9527884be6e5616bdd54de542f9ae13384489724",
        "model_path": str(args.model_path.resolve()), "device": args.device,
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_path = args.output_root / "run.json"
    if run_path.exists():
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if record["frozen"] != frozen:
            raise ValueError("Existing run provenance drifted")
    else:
        record = {"frozen": frozen, "runtime": {"python": platform.python_version()}, "status": "running"}
        run_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": args.device}, low_cpu_mem_usage=True,
    ).eval()
    rows = []
    started_all = time.perf_counter()
    for candidate in candidates:
        shard = args.output_root / "shards" / f"{candidate['pair_id']}.json"
        shard.parent.mkdir(parents=True, exist_ok=True)
        if shard.exists():
            rows.append(json.loads(shard.read_text(encoding="utf-8")))
            continue
        messages = prompt_for(candidate, guide)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded, do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        completion = tokenizer.decode(output[0, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        annotation = None
        parse_mode = None
        error = None
        try:
            annotation, parse_mode = parse_candidate_annotation(completion)
            validate_annotation(annotation, candidate["pair_id"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "schema_version": "clir-on-policy-verifier-row-v1", "pair_id": candidate["pair_id"],
            "candidate_sha256": hashlib.sha256(canonical_json(candidate).encode()).hexdigest(),
            "messages_sha256": hashlib.sha256(canonical_json(messages).encode()).hexdigest(),
            "completion_text": completion, "parse_valid": error is None, "parse_mode": parse_mode,
            "parse_error": error, "annotation": annotation if error is None else None,
            "acceptance_status": silver_status(annotation) if error is None else "invalid",
            "elapsed_seconds": elapsed, "input_tokens": int(input_length),
            "output_tokens": int(output[0, input_length:].numel()),
        }
        temporary = shard.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(shard)
        rows.append(row)
        print(json.dumps({"pair_id": candidate["pair_id"], "parse_valid": error is None, "status": row["acceptance_status"], "elapsed": elapsed}))

    output_path = args.output_root / "all_results.jsonl"
    output_path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["acceptance_status"]] = counts.get(row["acceptance_status"], 0) + 1
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record.update({
        "status": "complete", "items": len(rows), "acceptance_counts": counts,
        "parse_valid": sum(row["parse_valid"] for row in rows),
        "elapsed_seconds": time.perf_counter() - started_all,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "results_sha256": sha256(output_path),
    })
    run_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
