#!/usr/bin/env python3
"""Map valid primary unit decisions to frozen Phi token targets and audit yield."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_dual_prior_annotation import validate_annotation, validate_item  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    char_span_to_token_span,
    content_token_offsets,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def unit_token_spans(
    trajectory: str,
    units: Sequence[Mapping[str, Any]],
    offsets: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for expected_index, unit in enumerate(units):
        index = int(unit["unit_index"])
        start = int(unit["char_start"])
        end = int(unit["char_end"])
        if index != expected_index or trajectory[start:end] != unit["text"]:
            raise ValueError("Private unit lineage differs from the exact trajectory")
        token_start, token_end = char_span_to_token_span((start, end), offsets)
        output.append(
            {
                **dict(unit),
                "token_start": token_start,
                "token_end_exclusive": token_end,
            }
        )
    return output


def target_for(
    selected: Sequence[int],
    spans: Sequence[Mapping[str, Any]],
    token_count: int,
) -> list[int]:
    target = [0] * token_count
    for unit_index in selected:
        span = spans[unit_index]
        for token_index in range(int(span["token_start"]), int(span["token_end_exclusive"])):
            target[token_index] = 1
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir", default="/prodcpfs/user/panzhixin/hf_cache")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    outputs = protocol["outputs"]
    items_path = resolve(outputs["annotation_items"])
    lineage_path = resolve(outputs["annotation_lineage"])
    source_path = resolve(protocol["source"]["trajectory_manifest"])
    run_root = resolve(outputs["primary_run_root"])
    results_path = run_root / "all_results.jsonl"
    run_path = run_root / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    if run_record.get("status") != "complete":
        raise ValueError("Primary annotation run is not complete")
    if file_sha256(results_path) != run_record["results_sha256"]:
        raise ValueError("Primary results hash differs from run record")

    items = read_jsonl(items_path)
    lineage = read_jsonl(lineage_path)
    results = read_jsonl(results_path)
    source_by_id = {str(row["id"]): row for row in read_jsonl(source_path)}
    if not (len(items) == len(lineage) == len(results) == int(protocol["source"]["rows"])):
        raise ValueError("Primary materialization row count drifted")

    from transformers import AutoTokenizer

    token_cfg = protocol["token_mapping"]
    tokenizer = AutoTokenizer.from_pretrained(
        token_cfg["model_id"],
        revision=token_cfg["tokenizer_revision"],
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_network,
        trust_remote_code=bool(token_cfg["trust_remote_code"]),
    )
    if not tokenizer.is_fast:
        raise ValueError("Exact dual-prior span mapping requires a fast tokenizer")

    labels: list[dict[str, Any]] = []
    eligibility: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    key_positive_counts: list[int] = []
    complete_positive_counts: list[int] = []
    token_counts: list[int] = []
    key_unit_counts: list[int] = []
    complete_unit_counts: list[int] = []
    strict_nested_rows = 0
    key_equals_complete_rows = 0
    key_only_last_unit_rows = 0
    for row_number, (item, identity, result) in enumerate(zip(items, lineage, results), start=1):
        if not (item["item_id"] == identity["item_id"] == result["item_id"]):
            raise ValueError(f"Primary row {row_number} identity/order drifted")
        validate_item(item)
        if not result.get("parse_valid") or result.get("annotation") is None:
            raise ValueError(f"Primary row {row_number} is structurally invalid")
        annotation = validate_annotation(result["annotation"], item)
        source = source_by_id[str(identity["source_id"])]
        trajectory = str(source["response"])
        if trajectory != item["trajectory"]:
            raise ValueError("Blind trajectory differs from frozen source")
        output_token_ids = [int(value) for value in source["output_token_ids"]]
        if canonical_sha256(output_token_ids) != identity["output_token_ids_sha256"]:
            raise ValueError("Output-token identity hash drifted")
        encoded = tokenizer(trajectory, add_special_tokens=False, return_offsets_mapping=True)
        encoded_ids = [int(value) for value in encoded["input_ids"]]
        trailing = output_token_ids[len(encoded_ids) :]
        offsets = content_token_offsets(
            response=trajectory,
            output_token_ids=output_token_ids,
            encoded_token_ids=encoded_ids,
            offsets=encoded["offset_mapping"],
            trailing_token_decodes_to_empty=[
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                == ""
                for token_id in trailing
            ],
        )
        mapped_units = unit_token_spans(trajectory, identity["units"], offsets)
        eligibility[annotation["eligibility"]] += 1
        confidence[annotation["confidence"]] += 1
        label: dict[str, Any] = {
            "schema_version": "clir-dual-prior-evidence-mapped-label-v1",
            **annotation,
            "id": str(identity["source_id"]),
            "query_id": str(identity["query_id"]),
            "membership": str(identity["membership"]),
            "output_token_ids_sha256": str(identity["output_token_ids_sha256"]),
            "mapped_units": mapped_units,
            "annotator": {
                "role": "candidate_primary",
                "model_id": protocol["primary_annotator"]["model_id"],
                "model_revision": protocol["primary_annotator"]["model_revision"],
                "run_results_sha256": run_record["results_sha256"],
                "completion_sha256": result["completion_sha256"],
            },
        }
        if annotation["eligibility"] == "usable":
            key = target_for(annotation["key_unit_indices"], mapped_units, len(output_token_ids))
            complete = target_for(
                annotation["complete_unit_indices"], mapped_units, len(output_token_ids)
            )
            if not any(key) or not any(complete) or any(k > c for k, c in zip(key, complete)):
                raise ValueError("Mapped token targets violate non-empty key subset complete")
            label["key_prior_target"] = key
            label["complete_prior_target"] = complete
            key_count = sum(key)
            complete_count = sum(complete)
            key_positive_counts.append(key_count)
            complete_positive_counts.append(complete_count)
            token_counts.append(len(output_token_ids))
            key_unit_counts.append(len(annotation["key_unit_indices"]))
            complete_unit_counts.append(len(annotation["complete_unit_indices"]))
            strict_nested_rows += int(set(annotation["key_unit_indices"]) < set(annotation["complete_unit_indices"]))
            key_equals_complete_rows += int(annotation["key_unit_indices"] == annotation["complete_unit_indices"])
            key_only_last_unit_rows += int(
                annotation["key_unit_indices"] == [len(item["units"]) - 1]
            )
        labels.append(label)

    labels_path = resolve(outputs["primary_labels"])
    atomic_write_jsonl(labels_path, labels)
    usable = eligibility["usable"]
    report = {
        "schema_version": "clir-dual-prior-evidence-primary-report-v1",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "run_results_sha256": run_record["results_sha256"],
        "labels_sha256": file_sha256(labels_path),
        "rows": len(labels),
        "eligibility_counts": dict(sorted(eligibility.items())),
        "confidence_counts": dict(sorted(confidence.items())),
        "selection_gate_minimum_usable_rows": protocol["selection_gate"]["minimum_usable_rows_per_annotator"],
        "usable_yield_gate_passed": usable >= int(protocol["selection_gate"]["minimum_usable_rows_per_annotator"]),
        "usable_target_audit": {
            "rows": usable,
            "strict_key_subset_complete_rows": strict_nested_rows,
            "key_equals_complete_rows": key_equals_complete_rows,
            "key_only_last_unit_rows": key_only_last_unit_rows,
            "key_positive_token_fraction_micro": sum(key_positive_counts) / sum(token_counts) if token_counts else None,
            "complete_positive_token_fraction_micro": sum(complete_positive_counts) / sum(token_counts) if token_counts else None,
            "key_positive_tokens_median": statistics.median(key_positive_counts) if key_positive_counts else None,
            "complete_positive_tokens_median": statistics.median(complete_positive_counts) if complete_positive_counts else None,
            "key_units_median": statistics.median(key_unit_counts) if key_unit_counts else None,
            "complete_units_median": statistics.median(complete_unit_counts) if complete_unit_counts else None,
        },
        "tokenizer": {
            "model_id": token_cfg["model_id"],
            "revision": token_cfg["tokenizer_revision"],
            "class": type(tokenizer).__name__,
            "is_fast": tokenizer.is_fast,
        },
    }
    report_path = resolve(outputs["primary_report"])
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
