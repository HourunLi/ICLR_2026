#!/usr/bin/env python3
"""Materialize adjudicated dual-prior unit decisions on the frozen Phi token axis."""

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

from src.clir_dual_prior_adjudication import validate_adjudication  # noqa: E402
from src.clir_dual_prior_annotation import (  # noqa: E402
    ANNOTATION_SCHEMA,
    validate_annotation,
    validate_item,
)
from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    file_sha256,
    read_jsonl,
)
from src.clir_supervision import validate_supervision_annotation  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/dual_prior_evidence_v1/gold_materialization_protocol_v1.json"
)
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def indexed(rows: Sequence[Mapping[str, Any]], key: str, *, name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key, ""))
        if not identity or identity in output:
            raise ValueError(f"{name} has a missing or duplicate {key}: {identity!r}")
        output[identity] = dict(row)
    return output


def annotation_from(row: Mapping[str, Any], item_id: str) -> dict[str, Any]:
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "item_id": item_id,
        "eligibility": row["eligibility"],
        "key_unit_indices": row["key_unit_indices"],
        "complete_unit_indices": row["complete_unit_indices"],
        "confidence": row["confidence"],
        "rationale": row["rationale"],
    }


def target_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        annotation["eligibility"],
        tuple(annotation["key_unit_indices"]),
        tuple(annotation["complete_unit_indices"]),
    )


def lower_confidence(first: str, second: str) -> str:
    if first not in CONFIDENCE_ORDER or second not in CONFIDENCE_ORDER:
        raise ValueError("Unknown annotation confidence")
    return min((first, second), key=CONFIDENCE_ORDER.__getitem__)


def target_for(
    selected: Sequence[int],
    mapped_units: Sequence[Mapping[str, Any]],
    token_count: int,
) -> list[int]:
    target = [0] * token_count
    for unit_index in selected:
        unit = mapped_units[int(unit_index)]
        for token_index in range(
            int(unit["token_start"]), int(unit["token_end_exclusive"])
        ):
            target[token_index] = 1
    return target


def validate_mapping_carrier(
    *,
    item: Mapping[str, Any],
    lineage: Mapping[str, Any],
    primary: Mapping[str, Any],
    source: Mapping[str, Any],
    split: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    item_id = str(item["item_id"])
    source_id = str(lineage["source_id"])
    if not (
        item_id == str(primary["item_id"])
        and source_id == str(primary["id"])
        and source_id == str(source["id"])
        and source_id == str(split["source_id"])
    ):
        raise ValueError(f"{item_id}: mapping-carrier identity drift")
    if str(lineage["query_id"]) != str(primary["query_id"]) or str(
        lineage["query_id"]
    ) != str(source["query_id"]):
        raise ValueError(f"{item_id}: query identity drift")
    if str(lineage["query_id"]) != str(split["query_id"]):
        raise ValueError(f"{item_id}: split query identity drift")
    if str(lineage["membership"]) != str(primary["membership"]) or str(
        lineage["membership"]
    ) != str(split["membership"]):
        raise ValueError(f"{item_id}: membership drift")
    trajectory = str(item["trajectory"])
    if trajectory != str(source["response"]):
        raise ValueError(f"{item_id}: source response differs from blind trajectory")
    output_token_ids = source.get("output_token_ids")
    if not isinstance(output_token_ids, list) or not output_token_ids:
        raise ValueError(f"{item_id}: frozen output_token_ids are absent")
    token_hash = canonical_sha256([int(value) for value in output_token_ids])
    if not (
        token_hash == str(lineage["output_token_ids_sha256"])
        and token_hash == str(primary["output_token_ids_sha256"])
    ):
        raise ValueError(f"{item_id}: output-token identity hash drift")
    if int(lineage["output_token_count"]) != len(output_token_ids):
        raise ValueError(f"{item_id}: output-token count drift")

    private_units = lineage["units"]
    mapped_units = primary["mapped_units"]
    if len(item["units"]) != len(private_units) or len(private_units) != len(mapped_units):
        raise ValueError(f"{item_id}: unit count drift")
    normalized: list[dict[str, Any]] = []
    previous_token_start = -1
    for expected_index, (public, private, mapped) in enumerate(
        zip(item["units"], private_units, mapped_units)
    ):
        expected = {
            "unit_index": expected_index,
            "text": str(public["text"]),
            "char_start": int(private["char_start"]),
            "char_end": int(private["char_end"]),
        }
        for key, value in expected.items():
            if mapped.get(key) != value or private.get(key) != value:
                raise ValueError(f"{item_id}: mapped unit {expected_index} lineage drift")
        if trajectory[expected["char_start"] : expected["char_end"]] != expected["text"]:
            raise ValueError(f"{item_id}: mapped unit {expected_index} text drift")
        token_start = int(mapped["token_start"])
        token_end = int(mapped["token_end_exclusive"])
        if not 0 <= token_start < token_end <= len(output_token_ids):
            raise ValueError(f"{item_id}: mapped unit {expected_index} token span is invalid")
        if token_start < previous_token_start:
            raise ValueError(f"{item_id}: mapped token spans are not ordered")
        previous_token_start = token_start
        normalized.append(
            {
                **expected,
                "token_start": token_start,
                "token_end_exclusive": token_end,
            }
        )

    primary_annotation = validate_annotation(annotation_from(primary, item_id), item)
    if primary_annotation["eligibility"] != "usable":
        raise ValueError(f"{item_id}: primary mapping carrier unexpectedly lacks targets")
    remapped_key = target_for(
        primary_annotation["key_unit_indices"], normalized, len(output_token_ids)
    )
    remapped_complete = target_for(
        primary_annotation["complete_unit_indices"], normalized, len(output_token_ids)
    )
    if remapped_key != primary["key_prior_target"] or remapped_complete != primary[
        "complete_prior_target"
    ]:
        raise ValueError(f"{item_id}: primary mapping carrier no longer reproduces itself")
    return normalized, len(output_token_ids)


def verify_protocol_inputs(protocol: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(spec["path"])
        if not path.is_file():
            raise ValueError(f"Missing frozen input {name}: {path}")
        observed = file_sha256(path)
        if observed != spec["sha256"]:
            raise ValueError(
                f"Frozen input {name} hash drifted: expected {spec['sha256']}, got {observed}"
            )
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != (
        "clir-dual-prior-evidence-gold-materialization-protocol-v1"
    ):
        raise ValueError("Unexpected dual-prior gold materialization protocol schema")
    paths = verify_protocol_inputs(protocol)
    protocol_sha = file_sha256(protocol_path)

    items = read_jsonl(paths["annotation_items"])
    lineage_by_item = indexed(
        read_jsonl(paths["annotation_lineage"]), "item_id", name="annotation lineage"
    )
    primary_by_item = indexed(
        read_jsonl(paths["primary_mapped_labels"]), "item_id", name="primary labels"
    )
    secondary_by_item = indexed(
        read_jsonl(paths["secondary_labels"]), "item_id", name="secondary labels"
    )
    source_by_id = indexed(
        read_jsonl(paths["trajectory_manifest"]), "id", name="trajectory manifest"
    )
    split_by_id = indexed(
        read_jsonl(paths["training_split_manifest"]), "source_id", name="split manifest"
    )
    agreement = json.loads(paths["agreement_report"].read_text(encoding="utf-8"))
    diagnostic_by_item = indexed(
        agreement["row_diagnostics"], "item_id", name="agreement diagnostics"
    )
    packets_by_item = indexed(
        read_jsonl(paths["adjudication_items"]), "item_id", name="adjudication packet"
    )
    adjudications_by_item = indexed(
        read_jsonl(paths["adjudications"]), "item_id", name="adjudications"
    )
    adjudication_report = json.loads(
        paths["adjudication_report"].read_text(encoding="utf-8")
    )

    expected_rows = int(protocol["decision_rule"]["expected_rows"])
    aligned_sets = [
        len(items),
        len(lineage_by_item),
        len(primary_by_item),
        len(secondary_by_item),
        len(diagnostic_by_item),
        len(split_by_id),
    ]
    if any(count != expected_rows for count in aligned_sets):
        raise ValueError(f"The aligned 64-row artifacts drifted: {aligned_sets}")
    if not agreement.get("agreement_gate_passed"):
        raise ValueError("Agreement gate did not pass")
    if agreement.get("disagreement_rows") != len(adjudications_by_item):
        raise ValueError("Agreement disagreement count differs from adjudications")
    if not adjudication_report.get("all_disagreements_resolved"):
        raise ValueError("Adjudication report does not authorize gold materialization")

    gold_rows: list[dict[str, Any]] = []
    supervision_rows: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    membership_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    key_positive_counts: list[int] = []
    complete_positive_counts: list[int] = []
    token_counts: list[int] = []
    key_unit_counts: list[int] = []
    complete_unit_counts: list[int] = []
    selected_key_positions: list[float] = []
    rows_all_key_last_quarter = 0
    rows_all_key_first_quarter = 0
    strict_unit_nested_rows = 0
    equal_unit_rows = 0
    strict_token_nested_rows = 0
    equal_token_rows = 0
    queries_by_membership: dict[str, set[str]] = {
        "dense_train": set(),
        "localization_dev": set(),
    }

    for row_number, item in enumerate(items, start=1):
        validate_item(item)
        item_id = str(item["item_id"])
        if item_id not in lineage_by_item or item_id not in primary_by_item:
            raise ValueError(f"Row {row_number}: missing aligned identity for {item_id}")
        lineage = lineage_by_item[item_id]
        primary = primary_by_item[item_id]
        secondary = secondary_by_item[item_id]
        source_id = str(lineage["source_id"])
        if source_id not in source_by_id or source_id not in split_by_id:
            raise ValueError(f"{item_id}: source or split identity is missing")
        source = source_by_id[source_id]
        split = split_by_id[source_id]
        mapped_units, token_count = validate_mapping_carrier(
            item=item,
            lineage=lineage,
            primary=primary,
            source=source,
            split=split,
        )

        primary_annotation = validate_annotation(annotation_from(primary, item_id), item)
        secondary_annotation = validate_annotation(annotation_from(secondary, item_id), item)
        diagnostic = diagnostic_by_item[item_id]
        exact = bool(diagnostic["exact_target_agreement"])
        if exact:
            if target_signature(primary_annotation) != target_signature(secondary_annotation):
                raise ValueError(f"{item_id}: reported exact agreement is not exact")
            if item_id in adjudications_by_item or item_id in packets_by_item:
                raise ValueError(f"{item_id}: exact-agreement row entered adjudication")
            final_annotation = dict(primary_annotation)
            final_annotation["confidence"] = lower_confidence(
                primary_annotation["confidence"], secondary_annotation["confidence"]
            )
            final_annotation["rationale"] = (
                "Independent primary and secondary annotators selected identical "
                "eligibility, key, and complete target sets."
            )
            decision_source = "exact_independent_agreement"
            resolution = None
        else:
            if item_id not in adjudications_by_item or item_id not in packets_by_item:
                raise ValueError(f"{item_id}: disagreement lacks adjudication")
            adjudication = validate_adjudication(
                adjudications_by_item[item_id], packets_by_item[item_id]
            )
            final_annotation = validate_annotation(
                annotation_from(adjudication, item_id), item
            )
            decision_source = "role_blind_adjudication"
            resolution = str(adjudication["resolution"])
            resolution_counts[resolution] += 1

        decision_counts[decision_source] += 1
        confidence_counts[str(final_annotation["confidence"])] += 1
        membership = str(lineage["membership"])
        membership_counts[membership] += 1
        if membership not in queries_by_membership:
            raise ValueError(f"{item_id}: unexpected membership {membership}")
        query_id = str(lineage["query_id"])
        queries_by_membership[membership].add(query_id)
        if final_annotation["eligibility"] != "usable":
            raise ValueError(f"{item_id}: v1 readiness requires every final row usable")

        key_units = final_annotation["key_unit_indices"]
        complete_units = final_annotation["complete_unit_indices"]
        key_target = target_for(key_units, mapped_units, token_count)
        complete_target = target_for(complete_units, mapped_units, token_count)
        if not any(key_target) or not any(complete_target):
            raise ValueError(f"{item_id}: final token target is empty")
        if any(key > complete for key, complete in zip(key_target, complete_target)):
            raise ValueError(f"{item_id}: final key target is not nested in complete")

        key_set = set(key_units)
        complete_set = set(complete_units)
        strict_unit_nested_rows += int(key_set < complete_set)
        equal_unit_rows += int(key_set == complete_set)
        strict_token_nested_rows += int(
            any(key != complete for key, complete in zip(key_target, complete_target))
        )
        equal_token_rows += int(key_target == complete_target)
        key_count = sum(key_target)
        complete_count = sum(complete_target)
        key_positive_counts.append(key_count)
        complete_positive_counts.append(complete_count)
        token_counts.append(token_count)
        key_unit_counts.append(len(key_units))
        complete_unit_counts.append(len(complete_units))
        denominator = max(1, len(item["units"]) - 1)
        positions = [int(index) / denominator for index in key_units]
        selected_key_positions.extend(positions)
        rows_all_key_last_quarter += int(all(position >= 0.75 for position in positions))
        rows_all_key_first_quarter += int(all(position <= 0.25 for position in positions))

        semantic_decision = {
            "eligibility": final_annotation["eligibility"],
            "key_unit_indices": key_units,
            "complete_unit_indices": complete_units,
        }
        gold_row = {
            "schema_version": "clir-dual-prior-evidence-gold-label-v1",
            "item_id": item_id,
            "id": source_id,
            "query_id": query_id,
            "membership": membership,
            "eligibility": final_annotation["eligibility"],
            "key_unit_indices": key_units,
            "complete_unit_indices": complete_units,
            "confidence": final_annotation["confidence"],
            "rationale": final_annotation["rationale"],
            "key_prior_target": key_target,
            "complete_prior_target": complete_target,
            "output_token_ids_sha256": str(lineage["output_token_ids_sha256"]),
            "mapped_units": mapped_units,
            "gold_provenance": {
                "decision_source": decision_source,
                "resolution": resolution,
                "semantic_decision_sha256": canonical_sha256(semantic_decision),
                "agreement_report_sha256": protocol["inputs"]["agreement_report"]["sha256"],
                "adjudications_sha256": protocol["inputs"]["adjudications"]["sha256"],
                "mapping_carrier_sha256": protocol["inputs"]["primary_mapped_labels"]["sha256"],
                "unilateral_primary_semantic_target_used": False,
            },
        }
        gold_rows.append(gold_row)
        supervision = {
            "schema_version": "clir-supervision-annotation-v1",
            "id": source_id,
            "query_id": query_id,
            "output_token_ids_sha256": str(lineage["output_token_ids_sha256"]),
            "key_prior_target": key_target,
            "complete_prior_target": complete_target,
            "annotation_provenance": {
                "schema_version": "clir-supervision-provenance-v1",
                "protocol_sha256": protocol_sha,
                "annotation_source": "dual_prior_evidence_gold_v1",
                "semantic_decision_sha256": canonical_sha256(semantic_decision),
                "decision_source": decision_source,
            },
            "annotation_metadata": {
                "item_id": item_id,
                "membership": membership,
                "key_unit_indices": key_units,
                "complete_unit_indices": complete_units,
                "resolution": resolution,
            },
        }
        validate_supervision_annotation(supervision, source)
        supervision_rows.append(supervision)

    expected_exact = int(protocol["decision_rule"]["exact_agreement_rows"])
    expected_disagreements = int(
        protocol["decision_rule"]["adjudicated_disagreement_rows"]
    )
    if decision_counts != {
        "exact_independent_agreement": expected_exact,
        "role_blind_adjudication": expected_disagreements,
    }:
        raise ValueError(f"Semantic decision-source counts drifted: {decision_counts}")
    if set(packets_by_item) != set(adjudications_by_item):
        raise ValueError("Adjudication packet and decisions have different item sets")
    if queries_by_membership["dense_train"] & queries_by_membership["localization_dev"]:
        raise ValueError("dense_train and localization_dev are not query-disjoint")

    outputs = protocol["outputs"]
    labels_path = resolve(outputs["mapped_gold_labels"])
    supervision_path = resolve(outputs["supervision_all"])
    train_path = resolve(outputs["supervision_dense_train"])
    dev_path = resolve(outputs["supervision_localization_dev"])
    atomic_write_jsonl(labels_path, gold_rows)
    atomic_write_jsonl(supervision_path, supervision_rows)
    train_rows = [
        row
        for row, gold in zip(supervision_rows, gold_rows)
        if gold["membership"] == "dense_train"
    ]
    dev_rows = [
        row
        for row, gold in zip(supervision_rows, gold_rows)
        if gold["membership"] == "localization_dev"
    ]
    atomic_write_jsonl(train_path, train_rows)
    atomic_write_jsonl(dev_path, dev_rows)

    total_tokens = sum(token_counts)
    key_fraction = sum(key_positive_counts) / total_tokens
    complete_fraction = sum(complete_positive_counts) / total_tokens
    expected_membership = protocol["split_rule"]["expected_counts"]
    gates = {
        "all_rows_usable": len(gold_rows) == expected_rows,
        "nested_fraction": 1.0,
        "minimum_strict_key_subset_complete_rows": strict_unit_nested_rows
        >= int(protocol["readiness_gates"]["minimum_strict_key_subset_complete_rows"]),
        "key_positive_fraction_below_complete": key_fraction < complete_fraction,
        "membership_counts": dict(membership_counts) == expected_membership,
        "query_disjoint": not bool(
            queries_by_membership["dense_train"]
            & queries_by_membership["localization_dev"]
        ),
    }
    ready = all(gates.values())
    report = {
        "schema_version": "clir-dual-prior-evidence-gold-report-v1",
        "status": "ready_for_standalone_D0_D3_protocol" if ready else "gold_readiness_gate_failed",
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "input_hashes": {
            name: spec["sha256"] for name, spec in protocol["inputs"].items()
        },
        "rows": len(gold_rows),
        "decision_source_counts": dict(sorted(decision_counts.items())),
        "adjudication_resolution_counts": dict(sorted(resolution_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "membership_counts": dict(sorted(membership_counts.items())),
        "query_counts": {
            membership: len(queries)
            for membership, queries in sorted(queries_by_membership.items())
        },
        "target_audit": {
            "nested_rows": len(gold_rows),
            "strict_key_subset_complete_unit_rows": strict_unit_nested_rows,
            "key_equals_complete_unit_rows": equal_unit_rows,
            "strict_key_subset_complete_token_rows": strict_token_nested_rows,
            "key_equals_complete_token_rows": equal_token_rows,
            "key_positive_token_fraction_micro": key_fraction,
            "complete_positive_token_fraction_micro": complete_fraction,
            "key_positive_tokens_median": statistics.median(key_positive_counts),
            "complete_positive_tokens_median": statistics.median(complete_positive_counts),
            "key_units_median": statistics.median(key_unit_counts),
            "complete_units_median": statistics.median(complete_unit_counts),
        },
        "position_diagnostic": {
            "selected_key_unit_relative_position_mean": sum(selected_key_positions)
            / len(selected_key_positions),
            "selected_key_unit_relative_position_median": statistics.median(
                selected_key_positions
            ),
            "rows_with_every_key_unit_in_last_quarter": rows_all_key_last_quarter,
            "rows_with_every_key_unit_in_first_quarter": rows_all_key_first_quarter,
            "position_only_baseline_required": bool(
                protocol["readiness_gates"]["position_only_baseline_required"]
            ),
        },
        "mapping_audit": {
            "rows_reproducing_frozen_primary_mapping_carrier": len(gold_rows),
            "unilateral_primary_semantic_targets_used": False,
            "axis": "frozen Phi output_token_ids",
        },
        "readiness_gates": gates,
        "ready_for_standalone_D0_D3_protocol": ready,
        "outputs": {
            "mapped_gold_labels": str(labels_path.relative_to(ROOT)),
            "mapped_gold_labels_sha256": file_sha256(labels_path),
            "supervision_all": str(supervision_path.relative_to(ROOT)),
            "supervision_all_sha256": file_sha256(supervision_path),
            "supervision_dense_train": str(train_path.relative_to(ROOT)),
            "supervision_dense_train_sha256": file_sha256(train_path),
            "supervision_localization_dev": str(dev_path.relative_to(ROOT)),
            "supervision_localization_dev_sha256": file_sha256(dev_path),
        },
        "formal_claim_allowed": False,
        "allowed_conclusion": (
            protocol["allowed_conclusion_if_passed"]
            if ready
            else "The frozen readiness gate failed; do not start D0-D3 training."
        ),
    }
    report_path = resolve(outputs["report"])
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
