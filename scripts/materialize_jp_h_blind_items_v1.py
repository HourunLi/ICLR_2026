#!/usr/bin/env python3
"""Select balanced trajectories and publish only the blind annotation view."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = ROOT / "configs/jp_h_blind_validation_v1/protocol_v1.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def priority(seed: int, domain: str, correctness: int, length_bin: int, source_id: str) -> str:
    payload = (
        f"clir-jp-h-blind-select-v1:{seed}:{domain}:"
        f"{correctness}:{length_bin}:{source_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def item_id(source_id: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{source_id}".encode("utf-8")).hexdigest()
    return f"JPHB-{digest[:16]}"


def blind_priority(seed: int, value: str) -> str:
    payload = f"clir-jp-h-blind-order-v1:{seed}:{value}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def length_bins(rows: Sequence[Mapping[str, Any]], bins: int) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (len(row["output_token_ids"]), str(row["id"])))
    return {
        str(row["id"]): min(bins - 1, index * bins // len(ordered))
        for index, row in enumerate(ordered)
    }


def select_domain(
    domain_rows: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    seed = int(selection["seed"])
    bins = int(selection["length_bins_per_correctness_domain"])
    rows_per_cell = int(selection["rows_per_length_bin"])
    minimum_tokens = int(selection["minimum_output_tokens"])
    eligible = [
        row
        for row in domain_rows
        if bool(row["answer_parse_valid"])
        and len(row["output_token_ids"]) >= minimum_tokens
        and str(row["generation"].get("finish_reason")) != "length"
    ]
    by_cell: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for correctness in (0, 1):
        stratum = [row for row in eligible if int(row["correctness"]) == correctness]
        if not stratum:
            raise ValueError(f"No eligible correctness={correctness} candidates for {domain}")
        bins_by_id = length_bins(stratum, bins)
        for length_bin in range(bins):
            cell = [row for row in stratum if bins_by_id[str(row["id"])] == length_bin]
            if not cell:
                raise ValueError(
                    f"No candidates for {domain} correctness={correctness} bin={length_bin}"
                )
            cell.sort(
                key=lambda row: priority(
                    seed,
                    domain,
                    correctness,
                    length_bin,
                    str(row["id"]),
                )
            )
            by_cell[(correctness, length_bin)] = cell

    # Each cell needs three unique queries. Reduce candidate rows to the best row
    # per query/cell, then solve a deterministic bipartite matching between cell
    # slots and query IDs. This prevents a greedy choice in one stratum from
    # silently making another stratum impossible.
    best_by_cell_query: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = {}
    for cell_key, rows in by_cell.items():
        best: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            best.setdefault(str(row["query_id"]), row)
        best_by_cell_query[cell_key] = best
        if len(best) < rows_per_cell:
            raise ValueError(f"Cell {domain}/{cell_key} has too few unique queries")

    slots: list[tuple[int, int, int]] = []
    for correctness in (0, 1):
        for length_bin in range(bins):
            for slot_index in range(rows_per_cell):
                slots.append((correctness, length_bin, slot_index))
    slots.sort(
        key=lambda slot: (
            len(best_by_cell_query[(slot[0], slot[1])]),
            slot[0],
            slot[1],
            slot[2],
        )
    )
    query_to_slot: dict[str, tuple[int, int, int]] = {}

    def augment(slot: tuple[int, int, int], visited: set[str]) -> bool:
        cell_key = (slot[0], slot[1])
        candidates = best_by_cell_query[cell_key]
        ordered_queries = sorted(
            candidates,
            key=lambda query_id: priority(
                seed,
                domain,
                slot[0],
                slot[1],
                str(candidates[query_id]["id"]),
            ),
        )
        for query_id in ordered_queries:
            if query_id in visited:
                continue
            visited.add(query_id)
            previous = query_to_slot.get(query_id)
            if previous is None or augment(previous, visited):
                query_to_slot[query_id] = slot
                return True
        return False

    for slot in slots:
        if not augment(slot, set()):
            raise ValueError(f"Could not satisfy unique-query matching for {domain}, slot={slot}")
    slot_to_query = {slot: query_id for query_id, slot in query_to_slot.items()}
    if len(slot_to_query) != len(slots):
        raise AssertionError("Cell-slot matching cardinality drifted")

    selected: list[dict[str, Any]] = []
    for slot in sorted(slot_to_query):
        correctness, length_bin, _ = slot
        query_id = slot_to_query[slot]
        row = dict(best_by_cell_query[(correctness, length_bin)][query_id])
        row["selection_correctness"] = correctness
        row["selection_length_bin"] = length_bin
        row["selection_priority"] = priority(
            seed,
            domain,
            correctness,
            length_bin,
            str(row["id"]),
        )
        selected.append(row)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidate-pool", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema_version"] != "clir-jp-h-blind-validation-protocol-v1":
        raise ValueError("Unexpected blind-validation protocol schema")
    protocol_sha256 = file_sha256(protocol_path)
    candidate_path = (
        args.candidate_pool.resolve()
        if args.candidate_pool
        else resolve(protocol["outputs"]["candidate_pool"])
    )
    candidates = read_jsonl(candidate_path)
    if not candidates:
        raise ValueError("Candidate pool is empty")
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("Candidate IDs are not unique")
    if any(row["provenance"]["protocol_sha256"] != protocol_sha256 for row in candidates):
        raise ValueError("Candidate pool protocol provenance drifted")
    selection_cfg = protocol["selection"]
    domains = list(protocol["source_sampling"]["domains"])
    selected: list[dict[str, Any]] = []
    for domain in domains:
        domain_rows = [row for row in candidates if row["domain"] == domain]
        selected.extend(
            select_domain(domain_rows, domain=domain, selection=selection_cfg)
        )

    expected = int(selection_cfg["expected_rows"])
    if len(selected) != expected or len({row["query_id"] for row in selected}) != expected:
        raise ValueError("Selected row count or global query uniqueness drifted")
    per_domain = Counter(row["domain"] for row in selected)
    per_domain_correctness = Counter(
        (row["domain"], int(row["selection_correctness"])) for row in selected
    )
    for domain in domains:
        if per_domain[domain] != int(selection_cfg["rows_per_domain"]):
            raise ValueError(f"Selected count drifted for {domain}")
        for correctness in (0, 1):
            expected_stratum = (
                int(selection_cfg["correct_per_domain"])
                if correctness == 1
                else int(selection_cfg["incorrect_per_domain"])
            )
            if per_domain_correctness[(domain, correctness)] != expected_stratum:
                raise ValueError(f"Correctness balance drifted for {domain}/{correctness}")

    salt = str(selection_cfg["item_id_salt"])
    seed = int(selection_cfg["seed"])
    staged: list[tuple[str, dict[str, Any]]] = []
    for row in selected:
        value = item_id(str(row["id"]), salt)
        staged.append((blind_priority(seed, value), {**row, "item_id": value}))
    staged.sort(key=lambda pair: pair[0])
    selected = [row for _, row in staged]
    if len({row["item_id"] for row in selected}) != expected:
        raise ValueError("Blind item ID collision")

    private_selection: list[dict[str, Any]] = []
    private_lineage: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for order, row in enumerate(selected):
        private_selection.append(
            {
                "schema_version": "clir-jp-h-blind-private-selection-v1",
                "selection_order": order,
                "item_id": row["item_id"],
                "source_id": row["id"],
                "query_id": row["query_id"],
                "candidate_index": int(row["candidate_index"]),
                "domain": row["domain"],
                "correctness_stratum": int(row["selection_correctness"]),
                "length_bin": int(row["selection_length_bin"]),
                "output_token_count": len(row["output_token_ids"]),
                "selection_priority": row["selection_priority"],
            }
        )
        items.append(
            {
                "schema_version": "clir-hallucination-annotation-item-v1",
                "item_id": row["item_id"],
                "problem": row["problem"],
                "trajectory": row["response"],
            }
        )
        private_lineage.append(
            {
                "schema_version": "clir-jp-h-blind-private-lineage-v1",
                "selection_order": order,
                "item_id": row["item_id"],
                "source_id": row["id"],
                "query_id": row["query_id"],
                "candidate_index": int(row["candidate_index"]),
                "domain": row["domain"],
                "source_split": row["source_split"],
                "source_index": int(row["source_index"]),
                "correctness": int(row["correctness"]),
                "parsed_answer": row["parsed_answer"],
                "reference_answer": row["reference_answer"],
                "answer_type": row["answer_type"],
                "problem": row["problem"],
                "prompt": row["prompt"],
                "trajectory": row["response"],
                "prompt_token_ids": row["prompt_token_ids"],
                "output_token_ids": row["output_token_ids"],
                "prompt_token_ids_sha256": canonical_sha256(row["prompt_token_ids"]),
                "output_token_ids_sha256": canonical_sha256(row["output_token_ids"]),
                "problem_sha256": hashlib.sha256(row["problem"].encode("utf-8")).hexdigest(),
                "trajectory_sha256": hashlib.sha256(row["response"].encode("utf-8")).hexdigest(),
                "candidate_pool": str(candidate_path),
                "candidate_pool_sha256": file_sha256(candidate_path),
                "protocol_sha256": protocol_sha256,
                "generation": row["generation"],
                "provenance": row["provenance"],
            }
        )

    outputs = protocol["outputs"]
    selection_path = resolve(outputs["private_selection"])
    lineage_path = resolve(outputs["private_lineage"])
    items_path = resolve(outputs["annotation_items"])
    report_path = resolve(outputs["package_report"])
    atomic_write_jsonl(selection_path, private_selection)
    atomic_write_jsonl(lineage_path, private_lineage)
    atomic_write_jsonl(items_path, items)
    forbidden = set(protocol["blind_item_contract"]["forbidden_fields"])
    visible = set(protocol["blind_item_contract"]["visible_fields"])
    for item in items:
        if set(item) != visible or set(item) & forbidden:
            raise ValueError("Published blind item violates its visible-field contract")
    report = {
        "schema_version": "clir-jp-h-blind-package-report-v1",
        "status": "passed",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256,
        "candidate_pool_sha256": file_sha256(candidate_path),
        "private_selection": str(selection_path.relative_to(ROOT)),
        "private_selection_sha256": file_sha256(selection_path),
        "private_lineage": str(lineage_path.relative_to(ROOT)),
        "private_lineage_sha256": file_sha256(lineage_path),
        "annotation_items": str(items_path.relative_to(ROOT)),
        "annotation_items_sha256": file_sha256(items_path),
        "rows": len(items),
        "queries": len({row["query_id"] for row in private_selection}),
        "per_domain": dict(sorted(per_domain.items())),
        "per_domain_correctness": {
            domain: {
                "incorrect": per_domain_correctness[(domain, 0)],
                "correct": per_domain_correctness[(domain, 1)],
            }
            for domain in sorted(domains)
        },
        "non_math_rows": sum(row["domain"] != "gsm8k" for row in selected),
        "blind_visible_fields": sorted(visible),
        "forbidden_field_leaks": 0,
        "model_score_used_for_selection": False,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
