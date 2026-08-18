#!/usr/bin/env python3
"""Build the frozen 64-row hallucination-localization annotation package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    build_annotation_records,
    canonical_json,
    file_sha256,
    read_jsonl,
    select_stratified_rows,
)


DEFAULT_PROTOCOL = ROOT / "configs/hallucination_localization_v1/protocol_v1.json"
FORBIDDEN_BLIND_KEYS = {
    "correctness",
    "correctness_stratum",
    "source_id",
    "query_id",
    "candidate_index",
    "parsed_answer",
    "reference_answer",
    "primary_annotation",
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def publish_or_verify_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite drifted artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def publish_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite drifted artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-hallucination-localization-protocol-v1":
        raise ValueError("Unexpected hallucination-localization protocol schema")
    protocol_sha = file_sha256(protocol_path)
    source_path = resolve(protocol["source"]["manifest"])
    if file_sha256(source_path) != protocol["source"]["sha256"]:
        raise ValueError("Frozen source manifest SHA256 drifted")
    exclusion_path = resolve(protocol["exclusions"]["query_ids_from"])
    if file_sha256(exclusion_path) != protocol["exclusions"]["query_ids_from_sha256"]:
        raise ValueError("Frozen exclusion lineage SHA256 drifted")

    source_rows = read_jsonl(source_path)
    if len(source_rows) != int(protocol["source"]["rows"]):
        raise ValueError("Frozen source row count drifted")
    excluded_query_ids = {
        str(row["query_id"]) for row in read_jsonl(exclusion_path)
    }
    if len(excluded_query_ids) != 31:
        raise ValueError("Expected 31 unique Route-A exclusion queries")
    selection_cfg = protocol["selection"]
    selected = select_stratified_rows(
        source_rows,
        excluded_query_ids=excluded_query_ids,
        seed=int(selection_cfg["seed"]),
        per_correctness=int(selection_cfg["per_correctness"]),
        length_bins=int(selection_cfg["length_bins_per_correctness"]),
    )
    selection, items, lineage = build_annotation_records(
        selected,
        source_manifest=protocol["source"]["manifest"],
        source_manifest_sha256=protocol["source"]["sha256"],
        protocol_sha256=protocol_sha,
        item_salt=selection_cfg["item_id_salt"],
    )

    expected_visible = set(protocol["blind_item_contract"]["visible_fields"])
    for item in items:
        if set(item) != expected_visible:
            raise ValueError("Blind item visible-field contract drifted")
        if set(item) & FORBIDDEN_BLIND_KEYS:
            raise ValueError("Blind item leaked private outcome/identity fields")

    outputs = protocol["outputs"]
    selection_path = resolve(outputs["selection_manifest"])
    items_path = resolve(outputs["annotation_items"])
    lineage_path = resolve(outputs["annotation_lineage"])
    report_path = resolve(outputs["package_report"])
    publish_or_verify_jsonl(selection_path, selection)
    publish_or_verify_jsonl(items_path, items)
    publish_or_verify_jsonl(lineage_path, lineage)

    counts = Counter()
    lengths: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in selection:
        key = (int(row["correctness_stratum"]), int(row["length_bin"]))
        counts[f"correctness_{key[0]}__length_bin_{key[1]}"] += 1
        lengths[key].append(int(row["output_token_count"]))
    report = {
        "schema_version": "clir-hallucination-localization-package-report-v1",
        "evidence_tier": "pipeline_pilot",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "source_manifest_sha256": protocol["source"]["sha256"],
        "excluded_query_count": len(excluded_query_ids),
        "rows": len(items),
        "queries": len({row["query_id"] for row in selection}),
        "correctness_counts": dict(
            sorted(Counter(row["correctness_stratum"] for row in selection).items())
        ),
        "stratum_bin_counts": dict(sorted(counts.items())),
        "stratum_bin_length_ranges": {
            f"correctness_{key[0]}__length_bin_{key[1]}": {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for key, values in sorted(lengths.items())
        },
        "selection_manifest": str(selection_path.relative_to(ROOT)),
        "selection_manifest_sha256": file_sha256(selection_path),
        "annotation_items": str(items_path.relative_to(ROOT)),
        "annotation_items_sha256": file_sha256(items_path),
        "annotation_lineage": str(lineage_path.relative_to(ROOT)),
        "annotation_lineage_sha256": file_sha256(lineage_path),
        "blind_field_gate_passed": True,
        "correctness_used_for_selection_only": True,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    publish_or_verify_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
