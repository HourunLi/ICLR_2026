#!/usr/bin/env python3
"""Resolve primary-only quote/index drift without consulting labels or scores."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hallucination_primary_v1 import parse_annotation  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    file_sha256,
    locate_occurrence,
    read_jsonl,
    repair_annotation_contract,
    validate_annotation,
)
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/jp_h_blind_validation_v1/primary_annotation_protocol_v1.json"
)
DEFAULT_OVERRIDES = (
    ROOT
    / "configs/jp_h_blind_validation_v1/primary_contract_repair_overrides_v1.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def exact_occurrence_at(text: str, quote: str, start: int) -> int:
    starts: list[int] = []
    cursor = 0
    while True:
        position = text.find(quote, cursor)
        if position < 0:
            break
        starts.append(position)
        cursor = position + 1
    if start not in starts:
        raise ValueError("Resolved repair span is not an exact quote occurrence")
    return starts.index(start)


def claim_text_sha256(claim: Mapping[str, Any]) -> str:
    return hashlib.sha256(str(claim.get("claim_text", "")).encode("utf-8")).hexdigest()


def semantic_unique_signature(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Ignore location syntax and collapse only byte-identical repeated judgments."""

    unique_judgments = {
        canonical_sha256(
            {
                "status": claim.get("status"),
                "reason": claim.get("reason"),
            }
        )
        for claim in annotation.get("claim_reviews", [])
        if isinstance(claim, Mapping)
    }
    return {
        "item_id": annotation.get("item_id"),
        "path_status": annotation.get("path_status"),
        "confidence": annotation.get("confidence"),
        "summary": annotation.get("summary"),
        "unique_claim_judgment_sha256": sorted(unique_judgments),
    }


def status_reason_multiset(annotation: Mapping[str, Any]) -> Counter[str]:
    return Counter(
        canonical_sha256(
            {"status": claim.get("status"), "reason": claim.get("reason")}
        )
        for claim in annotation.get("claim_reviews", [])
        if isinstance(claim, Mapping)
    )


def apply_override_operations(
    annotation: Mapping[str, Any],
    item: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    repaired = deepcopy(dict(annotation))
    trajectory = str(item["trajectory"])
    applied: list[dict[str, Any]] = []
    exact_duplicate_drops = 0
    claims = repaired.get("claim_reviews")
    if not isinstance(claims, list):
        raise ValueError("Primary repair requires claim_reviews")

    for spec in operations:
        operation = str(spec.get("operation"))
        if operation == "stable_sort_claims_by_exact_character_start":
            expected_count = int(spec["expected_claim_count"])
            if len(claims) != expected_count:
                raise ValueError("Claim count drifted before stable sort")
            located = [
                (*locate_occurrence(trajectory, claim["claim_text"], claim["occurrence"]), index)
                for index, claim in enumerate(claims)
            ]
            order = [entry[2] for entry in sorted(located, key=lambda entry: (entry[0], entry[2]))]
            if order == list(range(len(claims))):
                raise ValueError("Stable-sort override no longer changes claim order")
            claims[:] = [claims[index] for index in order]
            applied.append(
                {
                    "operation": operation,
                    "original_claim_order": list(range(len(order))),
                    "resolved_claim_order": order,
                }
            )
            continue

        claim_index = int(spec["claim_index"])
        if not 0 <= claim_index < len(claims):
            raise ValueError(f"Repair claim index is out of range: {claim_index}")
        claim = claims[claim_index]
        if claim_text_sha256(claim) != spec["original_claim_text_sha256"]:
            raise ValueError(
                f"Repair source quote hash drifted for {item['item_id']}/{claim_index}"
            )
        original_quote = str(claim["claim_text"])
        original_occurrence = int(claim["occurrence"])

        if operation == "case_insensitive_unique_quote_alignment":
            matches = list(re.finditer(re.escape(original_quote), trajectory, re.IGNORECASE))
            if len(matches) != 1:
                raise ValueError(
                    f"Case-insensitive repair requires one match, got {len(matches)}"
                )
            match = matches[0]
            resolved_quote = trajectory[match.start() : match.end()]
            resolved_occurrence = exact_occurrence_at(
                trajectory, resolved_quote, match.start()
            )
            claim["claim_text"] = resolved_quote
            claim["occurrence"] = resolved_occurrence
            applied.append(
                {
                    "operation": operation,
                    "claim_index": claim_index,
                    "original_claim_text_sha256": spec["original_claim_text_sha256"],
                    "resolved_claim_text_sha256": hashlib.sha256(
                        resolved_quote.encode("utf-8")
                    ).hexdigest(),
                    "resolved_char_start": match.start(),
                    "resolved_char_end": match.end(),
                    "original_occurrence": original_occurrence,
                    "resolved_occurrence": resolved_occurrence,
                }
            )
            continue

        if operation == "unique_exact_quote_replacement":
            resolved_quote = str(spec["resolved_claim_text"])
            starts: list[int] = []
            cursor = 0
            while True:
                position = trajectory.find(resolved_quote, cursor)
                if position < 0:
                    break
                starts.append(position)
                cursor = position + 1
            if len(starts) != 1:
                raise ValueError(
                    f"Exact replacement requires one trajectory match, got {len(starts)}"
                )
            claim["claim_text"] = resolved_quote
            claim["occurrence"] = 0
            applied.append(
                {
                    "operation": operation,
                    "claim_index": claim_index,
                    "original_claim_text_sha256": spec["original_claim_text_sha256"],
                    "resolved_claim_text_sha256": hashlib.sha256(
                        resolved_quote.encode("utf-8")
                    ).hexdigest(),
                    "resolved_char_start": starts[0],
                    "resolved_char_end": starts[0] + len(resolved_quote),
                    "original_occurrence": original_occurrence,
                    "resolved_occurrence": 0,
                }
            )
            continue

        if operation == "set_exact_occurrence":
            resolved_occurrence = int(spec["resolved_occurrence"])
            char_span = locate_occurrence(
                trajectory, original_quote, resolved_occurrence
            )
            claim["occurrence"] = resolved_occurrence
            applied.append(
                {
                    "operation": operation,
                    "claim_index": claim_index,
                    "claim_text_sha256": spec["original_claim_text_sha256"],
                    "original_occurrence": original_occurrence,
                    "resolved_occurrence": resolved_occurrence,
                    "resolved_char_start": char_span[0],
                    "resolved_char_end": char_span[1],
                }
            )
            continue

        if operation == "drop_byte_identical_duplicate_claim":
            duplicate_of = int(spec["duplicate_of_claim_index"])
            if not 0 <= duplicate_of < len(claims) or duplicate_of == claim_index:
                raise ValueError("Duplicate target claim index is invalid")
            if claims[claim_index] != claims[duplicate_of]:
                raise ValueError("Only a byte-identical duplicate claim may be dropped")
            removed_sha256 = canonical_sha256(claims[claim_index])
            claims.pop(claim_index)
            exact_duplicate_drops += 1
            applied.append(
                {
                    "operation": operation,
                    "claim_index": claim_index,
                    "duplicate_of_claim_index": duplicate_of,
                    "dropped_claim_sha256": removed_sha256,
                    "retained_identical_claim_sha256": canonical_sha256(
                        claims[duplicate_of]
                    ),
                }
            )
            continue

        raise ValueError(f"Unknown primary contract-repair override: {operation}")
    return repaired, applied, exact_duplicate_drops


def resolve_invalid_annotation(
    annotation: Mapping[str, Any],
    item: Mapping[str, Any],
    override_operations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    before = deepcopy(dict(annotation))
    before_multiset = status_reason_multiset(before)
    repaired, operations, duplicate_drops = apply_override_operations(
        before, item, override_operations
    )
    try:
        validate_annotation(repaired, item)
    except ValueError:
        repaired, generic_operations = repair_annotation_contract(repaired, item)
        operations.extend(generic_operations)
    validate_annotation(repaired, item)

    if (
        repaired.get("item_id") != before.get("item_id")
        or repaired.get("path_status") != before.get("path_status")
        or repaired.get("confidence") != before.get("confidence")
        or repaired.get("summary") != before.get("summary")
    ):
        raise AssertionError("Primary repair changed a top-level semantic judgment")
    after_multiset = status_reason_multiset(repaired)
    if duplicate_drops == 0:
        if before_multiset != after_multiset:
            raise AssertionError("Primary repair changed claim status/reason judgments")
    else:
        if len(before["claim_reviews"]) - len(repaired["claim_reviews"]) != duplicate_drops:
            raise AssertionError("Duplicate-only claim-count change drifted")
        if semantic_unique_signature(before) != semantic_unique_signature(repaired):
            raise AssertionError("Duplicate-only repair changed unique semantic judgments")
    if not operations:
        raise ValueError("Invalid primary annotation received no repair operation")
    return repaired, operations, duplicate_drops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Blind primary contract repair requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    overrides_path = args.overrides.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    if (
        overrides.get("schema_version")
        != "clir-jp-h-blind-primary-contract-repair-overrides-v1"
        or overrides.get("status")
        != "frozen_primary_only_before_secondary_or_score_access"
        or overrides.get("primary_protocol_sha256") != file_sha256(protocol_path)
    ):
        raise ValueError("Primary contract-repair override protocol drifted")

    items_path = resolve(protocol["outputs"]["annotation_items"])
    run_root = resolve(protocol["outputs"]["primary_run_root"])
    raw_results_path = run_root / "all_results.jsonl"
    run_path = run_root / "run.json"
    resolved_path = run_root / "all_results_resolved_v1.jsonl"
    report_path = run_root / "contract_repair_report_v1.json"
    if resolved_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite a prior blind primary repair")
    if file_sha256(items_path) != overrides["blind_items_sha256"]:
        raise ValueError("Blind items changed after repair overrides were frozen")
    if file_sha256(raw_results_path) != overrides["raw_primary_results_sha256"]:
        raise ValueError("Raw primary results changed after repair overrides were frozen")
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        run_record.get("status") != "complete"
        or run_record.get("items") != 96
        or run_record.get("results_sha256") != file_sha256(raw_results_path)
        or run_record.get("frozen", {}).get("protocol_sha256")
        != file_sha256(protocol_path)
    ):
        raise ValueError("Raw primary run provenance drifted")

    items = read_jsonl(items_path)
    raw_results = read_jsonl(raw_results_path)
    if len(items) != 96 or len(raw_results) != 96:
        raise ValueError("Blind primary repair requires exactly 96 rows")
    expected_override_ids = set(overrides["overrides"])
    invalid_ids = {
        str(row["item_id"]) for row in raw_results if not row["parse_valid"]
    }
    if not expected_override_ids <= invalid_ids:
        raise ValueError("An override no longer targets a raw-invalid primary row")

    resolved_results: list[dict[str, Any]] = []
    repaired_items: list[dict[str, Any]] = []
    operation_counts: Counter[str] = Counter()
    duplicate_drops = 0
    for row_number, (item, raw) in enumerate(zip(items, raw_results), start=1):
        if item["item_id"] != raw["item_id"]:
            raise ValueError(f"Primary row {row_number} identity/order drifted")
        if raw["parse_valid"]:
            validate_annotation(raw["annotation"], item)
            resolved_results.append(raw)
            continue
        parsed, parse_mode = parse_annotation(raw["completion_text"])
        item_overrides = overrides["overrides"].get(str(item["item_id"]), [])
        repaired, operations, row_duplicate_drops = resolve_invalid_annotation(
            parsed, item, item_overrides
        )
        duplicate_drops += row_duplicate_drops
        for operation in operations:
            operation_counts[str(operation["operation"])] += 1
        repair_record = {
            "schema_version": "clir-jp-h-blind-primary-contract-repair-row-v1",
            "policy": "location_only_plus_one_byte_identical_duplicate_drop",
            "raw_parse_error": raw["parse_error"],
            "raw_completion_sha256": raw["completion_sha256"],
            "raw_semantic_unique_signature_sha256": canonical_sha256(
                semantic_unique_signature(parsed)
            ),
            "resolved_semantic_unique_signature_sha256": canonical_sha256(
                semantic_unique_signature(repaired)
            ),
            "operations": operations,
            "secondary_labels_accessed": False,
            "model_predictions_accessed": False,
        }
        resolved = {
            **raw,
            "parse_valid": True,
            "parse_mode": f"{parse_mode}+blind_primary_contract_repair_v1",
            "parse_error": None,
            "annotation": repaired,
            "contract_repair": repair_record,
        }
        resolved_results.append(resolved)
        repaired_items.append(
            {
                "row_number": row_number,
                "item_id": item["item_id"],
                "raw_parse_error": raw["parse_error"],
                "operations": operations,
                "semantic_unique_signature_preserved": (
                    repair_record["raw_semantic_unique_signature_sha256"]
                    == repair_record["resolved_semantic_unique_signature_sha256"]
                ),
            }
        )

    atomic_write_jsonl(resolved_path, resolved_results)
    persisted = read_jsonl(resolved_path)
    if len(persisted) != 96:
        raise AssertionError("Resolved blind primary row count drifted after write")
    for item, result in zip(items, persisted):
        if not result["parse_valid"] or result["annotation"] is None:
            raise AssertionError("Resolved blind primary result remains invalid")
        validate_annotation(result["annotation"], item)

    report = {
        "schema_version": "clir-jp-h-blind-primary-contract-repair-report-v1",
        "status": "passed",
        "evidence_tier": "blind_validation_primary_pre_unseal",
        "code": code,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "overrides": str(overrides_path.relative_to(ROOT)),
        "overrides_sha256": file_sha256(overrides_path),
        "raw_results": str(raw_results_path.relative_to(ROOT)),
        "raw_results_sha256": file_sha256(raw_results_path),
        "resolved_results": str(resolved_path.relative_to(ROOT)),
        "resolved_results_sha256": file_sha256(resolved_path),
        "rows": 96,
        "raw_parse_valid": sum(int(row["parse_valid"]) for row in raw_results),
        "repaired_rows": len(repaired_items),
        "resolved_parse_valid": 96,
        "exact_duplicate_claims_dropped": duplicate_drops,
        "operation_counts": dict(sorted(operation_counts.items())),
        "repaired_items": repaired_items,
        "top_level_semantic_judgments_changed": False,
        "claim_status_or_reason_changed": False,
        "secondary_labels_accessed": False,
        "model_predictions_accessed": False,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
