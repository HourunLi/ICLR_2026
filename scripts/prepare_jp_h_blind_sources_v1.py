#!/usr/bin/env python3
"""Freeze mixed-domain source questions for the JP-to-H blind validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_jsonl,
    canonical_json,
    file_sha256,
)


DEFAULT_PROTOCOL = ROOT / "configs/jp_h_blind_validation_v1/protocol_v1.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def priority(seed: int, domain: str, query_id: str) -> str:
    payload = f"clir-jp-h-blind-source-v1:{seed}:{domain}:{query_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_options(question: str, choices: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
    labels = [str(value).strip() for value in choices["label"]]
    texts = [str(value).strip() for value in choices["text"]]
    if not labels or len(labels) != len(texts) or len(set(labels)) != len(labels):
        raise ValueError("Multiple-choice labels/text are empty, duplicated, or misaligned")
    options = [{"label": label, "text": text} for label, text in zip(labels, texts)]
    rendered = "\n".join(f"{row['label']}. {row['text']}" for row in options)
    problem = f"Question: {question.strip()}\nOptions:\n{rendered}"
    return problem, options


def make_prompt(domain: str, problem: str) -> str:
    if domain == "gsm8k":
        return (
            "Solve the following math problem step-by-step.\n"
            "Simplify your answer as much as possible. Present your final answer as "
            "\\boxed{Your Answer}.\n"
            + problem
        )
    if domain in {"arc_challenge", "commonsense_qa"}:
        return (
            "Solve the following multiple-choice problem. Explain the factual and logical "
            "steps that support your choice. End with exactly `Final Answer: <option label>`.\n\n"
            + problem
        )
    if domain == "boolq":
        return (
            "Answer the question using the supplied passage. Explain the evidence and inference "
            "step by step. End with exactly `Final Answer: yes` or `Final Answer: no`.\n\n"
            + problem
        )
    raise ValueError(f"Unknown domain: {domain}")


def row_record(
    *,
    domain: str,
    query_id: str,
    source_index: int,
    split: str,
    problem: str,
    raw_reference: Any,
    answer_type: str,
    dataset: Mapping[str, Any],
    protocol_sha256: str,
    options: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    row = {
        "schema_version": "clir-jp-h-blind-source-question-v1",
        "domain": domain,
        "query_id": query_id,
        "source_split": split,
        "source_index": source_index,
        "problem": problem,
        "problem_sha256": text_sha256(problem),
        "prompt": make_prompt(domain, problem),
        "raw_reference": raw_reference,
        "answer_type": answer_type,
        "dataset_id": dataset["repo_id"],
        "dataset_subset": dataset.get("subset"),
        "dataset_revision": dataset["revision"],
        "protocol_sha256": protocol_sha256,
    }
    if options is not None:
        row["options"] = options
    return row


def load_domain_rows(
    domain: str,
    dataset_cfg: Mapping[str, Any],
    *,
    cache_dir: str | None,
    protocol_sha256: str,
    gsm8k_manifest: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_cfg["repo_id"],
        dataset_cfg.get("subset"),
        split=dataset_cfg["split"],
        revision=dataset_cfg["revision"],
        cache_dir=cache_dir,
    )
    if domain == "gsm8k":
        if gsm8k_manifest is None:
            raise ValueError("GSM8K requires the frozen split manifest")
        frozen_dataset = gsm8k_manifest["dataset"]
        for key in ("repo_id", "subset", "revision"):
            if frozen_dataset[key] != dataset_cfg[key]:
                raise ValueError(f"GSM8K split-manifest {key} drifted")
        membership = dataset_cfg["allowed_membership"]
        query_ids = list(gsm8k_manifest["memberships"][membership])
        rows: list[dict[str, Any]] = []
        for query_id in query_ids:
            frozen = gsm8k_manifest["queries"][query_id]
            if frozen["source_split"] != dataset_cfg["split"]:
                raise ValueError("GSM8K reserve membership escaped its frozen split")
            source_index = int(frozen["source_index"])
            sample = dataset[source_index]
            question = str(sample["question"]).strip()
            if text_sha256(question) != frozen["question_sha256"]:
                raise ValueError(f"GSM8K question hash drifted for {query_id}")
            rows.append(
                row_record(
                    domain=domain,
                    query_id=query_id,
                    source_index=source_index,
                    split=dataset_cfg["split"],
                    problem=question,
                    raw_reference=sample["answer"],
                    answer_type="gsm8k_numeric",
                    dataset=dataset_cfg,
                    protocol_sha256=protocol_sha256,
                )
            )
        return rows

    rows = []
    for source_index, sample in enumerate(dataset):
        if domain in {"arc_challenge", "commonsense_qa"}:
            problem, options = format_options(str(sample["question"]), sample["choices"])
            query_id = f"{domain}-validation-{source_index:05d}"
            raw_reference = str(sample["answerKey"]).strip()
            valid_labels = {option["label"] for option in options}
            if raw_reference not in valid_labels:
                raise ValueError(f"Reference label absent from choices for {query_id}")
            rows.append(
                row_record(
                    domain=domain,
                    query_id=query_id,
                    source_index=source_index,
                    split=dataset_cfg["split"],
                    problem=problem,
                    raw_reference=raw_reference,
                    answer_type="choice_label",
                    options=options,
                    dataset=dataset_cfg,
                    protocol_sha256=protocol_sha256,
                )
            )
        elif domain == "boolq":
            problem = (
                f"Passage: {str(sample['passage']).strip()}\n\n"
                f"Question: {str(sample['question']).strip()}"
            )
            query_id = f"boolq-validation-{source_index:05d}"
            rows.append(
                row_record(
                    domain=domain,
                    query_id=query_id,
                    source_index=source_index,
                    split=dataset_cfg["split"],
                    problem=problem,
                    raw_reference="yes" if bool(sample["answer"]) else "no",
                    answer_type="yes_no",
                    dataset=dataset_cfg,
                    protocol_sha256=protocol_sha256,
                )
            )
        else:
            raise ValueError(f"Unknown domain: {domain}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema_version"] != "clir-jp-h-blind-validation-protocol-v1":
        raise ValueError("Unexpected blind-validation protocol schema")
    protocol_sha256 = file_sha256(protocol_path)
    sampling = protocol["source_sampling"]
    seed = int(sampling["seed"])
    count = int(sampling["queries_per_domain"])
    datasets = protocol["datasets"]
    gsm8k_manifest_path = resolve(datasets["gsm8k"]["split_manifest"])
    gsm8k_manifest = json.loads(gsm8k_manifest_path.read_text(encoding="utf-8"))

    selected: list[dict[str, Any]] = []
    seen_problem_hashes: set[str] = set()
    per_domain: dict[str, int] = {}
    for domain in sampling["domains"]:
        candidates = load_domain_rows(
            domain,
            datasets[domain],
            cache_dir=args.cache_dir,
            protocol_sha256=protocol_sha256,
            gsm8k_manifest=gsm8k_manifest if domain == "gsm8k" else None,
        )
        candidates.sort(key=lambda row: priority(seed, domain, row["query_id"]))
        domain_rows: list[dict[str, Any]] = []
        for row in candidates:
            problem_hash = row["problem_sha256"]
            if problem_hash in seen_problem_hashes:
                continue
            record = dict(row)
            record["source_priority"] = priority(seed, domain, row["query_id"])
            domain_rows.append(record)
            seen_problem_hashes.add(problem_hash)
            if len(domain_rows) == count:
                break
        if len(domain_rows) != count:
            raise ValueError(f"Could freeze only {len(domain_rows)}/{count} {domain} questions")
        selected.extend(domain_rows)
        per_domain[domain] = len(domain_rows)

    output = args.output.resolve() if args.output else resolve(protocol["outputs"]["source_questions"])
    atomic_write_jsonl(output, selected)
    report = {
        "schema_version": "clir-jp-h-blind-source-report-v1",
        "status": "passed",
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "gsm8k_split_manifest_sha256": file_sha256(gsm8k_manifest_path),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "rows": len(selected),
        "per_domain": per_domain,
        "unique_query_ids": len({row["query_id"] for row in selected}),
        "unique_problem_sha256": len(seen_problem_hashes),
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "canonical_manifest_sha256": hashlib.sha256(
            canonical_json(selected).encode("utf-8")
        ).hexdigest(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
