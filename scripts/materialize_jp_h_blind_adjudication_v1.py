#!/usr/bin/env python3
"""Materialize score-blind JP-H adjudications into two resolved label views."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_jp_h_blind_annotations_v1 import visible_annotation  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    canonical_json,
    canonical_sha256,
    file_sha256,
    raw_annotation_from_visible,
    read_jsonl,
    validate_annotation,
)
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/jp_h_blind_validation_v1/adjudication_protocol_v1.json"
)
SELECTION_KEYS = {
    "schema_version",
    "item_id",
    "adjudicator",
    "relation_to_views",
    "rationale",
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen input hash drifted for {path}: expected {expected_sha256}, got {observed}"
        )


def jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def publish_or_verify(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite drifted artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def index_unique(
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in indexed:
            raise ValueError(f"{name} requires unique non-empty item_id values")
        indexed[item_id] = row
    return indexed


def validate_selection(
    selection: Mapping[str, Any],
    package_item: Mapping[str, Any],
    *,
    allowed_relations: set[str] | frozenset[str] = frozenset(
        {"annotation_a", "annotation_b", "revised", "unresolved"}
    ),
) -> None:
    if set(selection) != SELECTION_KEYS:
        raise ValueError("Adjudication selection keys differ from the frozen schema")
    if (
        selection["schema_version"]
        != "clir-jp-h-blind-adjudication-selection-v1"
    ):
        raise ValueError("Unknown JP-H adjudication-selection schema")
    if selection["item_id"] != package_item["item_id"]:
        raise ValueError("Adjudication selection item identity drifted")
    if selection["relation_to_views"] not in allowed_relations:
        raise ValueError("Unknown adjudication relation")
    if selection["relation_to_views"] not in {"annotation_a", "annotation_b"}:
        raise ValueError("This frozen selection file contains no revised annotations")
    if not isinstance(selection["adjudicator"], str) or not selection[
        "adjudicator"
    ].strip():
        raise ValueError("Adjudication selection requires an identity")
    if not isinstance(selection["rationale"], str) or len(
        selection["rationale"].strip()
    ) < 24:
        raise ValueError("Adjudication rationale is not evidence-specific")


def expand_selection(
    selection: Mapping[str, Any], package_item: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_selection(selection, package_item)
    relation = str(selection["relation_to_views"])
    final_annotation = raw_annotation_from_visible(
        package_item[relation], item_id=str(package_item["item_id"])
    )
    validate_annotation(final_annotation, package_item)
    resolution = {
        "schema_version": "clir-jp-h-blind-adjudication-decision-v1",
        "item_id": package_item["item_id"],
        "relation_to_views": relation,
        "final_annotation": final_annotation,
        "rationale": selection["rationale"],
    }
    return final_annotation, resolution


def claim_status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(claim["status"])
                for row in rows
                for claim in row["claim_reviews"]
            ).items()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Blind adjudication materialization requires a clean worktree")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-jp-h-blind-adjudication-protocol-v1":
        raise ValueError("Unknown JP-H blind adjudication protocol schema")
    if protocol.get("prediction_values_accessed") is not False:
        raise ValueError("Adjudication must precede prediction-value access")

    paths: dict[str, Path] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(spec["path"])
        verify(path, str(spec["sha256"]))
        paths[name] = path
    all_items = read_jsonl(paths["all_items"])
    primary = read_jsonl(paths["primary_mapped_labels"])
    secondary = read_jsonl(paths["secondary_mapped_labels"])
    comparisons = read_jsonl(paths["row_comparisons"])
    package = read_jsonl(paths["adjudication_items"])
    lineage = read_jsonl(paths["adjudication_lineage"])
    selections = read_jsonl(paths["adjudication_selections"])
    expected_all = int(protocol["resolution_policy"]["expected_all_rows"])
    expected_blocking = int(protocol["resolution_policy"]["expected_blocking_rows"])
    if not (
        len(all_items)
        == len(primary)
        == len(secondary)
        == len(comparisons)
        == expected_all
    ):
        raise ValueError("All-row adjudication inputs differ from the frozen row count")
    if not (len(package) == len(lineage) == len(selections) == expected_blocking):
        raise ValueError("Blocking-row adjudication inputs differ from the frozen count")

    item_by_id = index_unique(all_items, name="all_items")
    primary_by_id = index_unique(primary, name="primary_labels")
    secondary_by_id = index_unique(secondary, name="secondary_labels")
    comparison_by_id = index_unique(comparisons, name="row_comparisons")
    package_by_id = index_unique(package, name="adjudication_items")
    lineage_by_id = index_unique(lineage, name="adjudication_lineage")
    selection_by_id = index_unique(selections, name="adjudication_selections")
    all_ids = set(item_by_id)
    if not (
        all_ids
        == set(primary_by_id)
        == set(secondary_by_id)
        == set(comparison_by_id)
    ):
        raise ValueError("All-row adjudication item-ID sets differ")
    blocking_ids = {
        item_id
        for item_id, compared in comparison_by_id.items()
        if compared["requires_adjudication"] is True
    }
    if not (
        blocking_ids
        == set(package_by_id)
        == set(lineage_by_id)
        == set(selection_by_id)
    ):
        raise ValueError("Adjudication package does not exactly cover blocking rows")

    protocol_sha256 = file_sha256(protocol_path)
    selection_sha256 = file_sha256(paths["adjudication_selections"])
    final_by_id: dict[str, dict[str, Any]] = {}
    raw_resolutions: list[dict[str, Any]] = []
    selected_view_counts: Counter[str] = Counter()
    selected_source_counts: Counter[str] = Counter()
    selected_path_counts: Counter[str] = Counter()
    for package_item, role_lineage, selection in zip(package, lineage, selections):
        item_id = str(package_item["item_id"])
        if not (
            item_id == role_lineage["item_id"] == selection["item_id"]
        ):
            raise ValueError("Blocking adjudication order or identity drifted")
        if package_item["problem"] != item_by_id[item_id]["problem"] or package_item[
            "trajectory"
        ] != item_by_id[item_id]["trajectory"]:
            raise ValueError("Adjudication package text differs from all-row item")
        left = primary_by_id[item_id]
        right = secondary_by_id[item_id]
        if canonical_sha256(left) != role_lineage["primary_label_sha256"]:
            raise ValueError("Primary adjudication-lineage hash drifted")
        if canonical_sha256(right) != role_lineage["secondary_label_sha256"]:
            raise ValueError("Secondary adjudication-lineage hash drifted")
        by_role = {"primary": left, "secondary": right}
        for view in ("annotation_a", "annotation_b"):
            role = str(role_lineage[f"{view}_role"])
            if role not in by_role or package_item[view] != visible_annotation(
                by_role[role]
            ):
                raise ValueError(f"{view} does not match its frozen hidden role")
        final_raw, resolution = expand_selection(selection, package_item)
        relation = str(selection["relation_to_views"])
        selected_role = str(role_lineage[f"{relation}_role"])
        selected = deepcopy(dict(by_role[selected_role]))
        if visible_annotation(selected) != package_item[relation]:
            raise AssertionError("Selected mapped source differs from visible view")
        selected["annotator"] = {
            "role": "score_blind_internal_adjudication",
            "identity": selection["adjudicator"],
            "selected_view": relation,
            "selected_source_role": selected_role,
            "selection_sha256": canonical_sha256(selection),
            "selection_file_sha256": selection_sha256,
            "adjudication_protocol_sha256": protocol_sha256,
            "selected_source_label_sha256": canonical_sha256(by_role[selected_role]),
            "claim_level_gold": False,
        }
        validate_annotation(final_raw, item_by_id[item_id])
        final_by_id[item_id] = selected
        raw_resolutions.append(resolution)
        selected_view_counts[relation] += 1
        selected_source_counts[selected_role] += 1
        selected_path_counts[str(selected["path_status"])] += 1

    resolved_primary: list[dict[str, Any]] = []
    resolved_secondary: list[dict[str, Any]] = []
    replaced = 0
    for item, left, right, compared in zip(
        all_items, primary, secondary, comparisons
    ):
        item_id = str(item["item_id"])
        if not (
            item_id
            == left["item_id"]
            == right["item_id"]
            == compared["item_id"]
        ):
            raise ValueError("Resolved-view all-row order drifted")
        if item_id in final_by_id:
            final = final_by_id[item_id]
            resolved_primary.append(deepcopy(final))
            resolved_secondary.append(deepcopy(final))
            replaced += 1
        else:
            if compared["requires_adjudication"] is not False:
                raise AssertionError("Blocking row escaped adjudication")
            resolved_primary.append(deepcopy(dict(left)))
            resolved_secondary.append(deepcopy(dict(right)))
    if replaced != expected_blocking:
        raise AssertionError("Resolved-view replacement count drifted")
    for left, right in zip(resolved_primary, resolved_secondary):
        if left["output_token_ids_sha256"] != right["output_token_ids_sha256"]:
            raise ValueError("Resolved evaluation views differ in token identity")
        if left["path_status"] != right["path_status"]:
            raise ValueError("Resolved evaluation views retain a path disagreement")
        left_positive = {
            position
            for claim in left["claim_reviews"]
            if claim["status"] in {"contradicted", "unsupported"}
            for position in range(claim["token_start"], claim["token_end_exclusive"])
        }
        right_positive = {
            position
            for claim in right["claim_reviews"]
            if claim["status"] in {"contradicted", "unsupported"}
            for position in range(claim["token_start"], claim["token_end_exclusive"])
        }
        if left_positive != right_positive:
            raise ValueError("Resolved views retain a positive-token disagreement")

    outputs = {name: resolve(path) for name, path in protocol["outputs"].items()}
    publish_or_verify(
        outputs["adjudication_resolutions_raw"], jsonl_payload(raw_resolutions)
    )
    publish_or_verify(
        outputs["resolved_primary_labels"], jsonl_payload(resolved_primary)
    )
    publish_or_verify(
        outputs["resolved_secondary_labels"], jsonl_payload(resolved_secondary)
    )

    resolved_path_counts = Counter(str(row["path_status"]) for row in resolved_primary)
    report = {
        "schema_version": "clir-jp-h-blind-adjudication-report-v1",
        "status": "adjudication_complete_predictions_still_sealed",
        "evidence_tier": "blind_validation_annotation_audit",
        "code": code,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256,
        "all_rows": expected_all,
        "blocking_rows_resolved": replaced,
        "nonblocking_rows_retaining_native_views": expected_all - replaced,
        "selected_view_counts": dict(sorted(selected_view_counts.items())),
        "selected_hidden_source_counts": dict(sorted(selected_source_counts.items())),
        "selected_blocking_path_counts": dict(sorted(selected_path_counts.items())),
        "resolved_path_status_counts": dict(sorted(resolved_path_counts.items())),
        "resolved_claim_status_counts": {
            "primary_view": claim_status_counts(resolved_primary),
            "secondary_view": claim_status_counts(resolved_secondary),
        },
        "zero_remaining_path_disagreements": True,
        "zero_remaining_positive_token_disagreements": True,
        "native_negative_claim_coverage_preserved": True,
        "claim_level_gold": False,
        "adjudicator_disclosure": protocol["adjudicator_disclosure"],
        "artifacts": {
            "adjudication_resolutions_raw": str(
                outputs["adjudication_resolutions_raw"].relative_to(ROOT)
            ),
            "adjudication_resolutions_raw_sha256": file_sha256(
                outputs["adjudication_resolutions_raw"]
            ),
            "resolved_primary_labels": str(
                outputs["resolved_primary_labels"].relative_to(ROOT)
            ),
            "resolved_primary_labels_sha256": file_sha256(
                outputs["resolved_primary_labels"]
            ),
            "resolved_secondary_labels": str(
                outputs["resolved_secondary_labels"].relative_to(ROOT)
            ),
            "resolved_secondary_labels_sha256": file_sha256(
                outputs["resolved_secondary_labels"]
            ),
        },
        "prediction_values_accessed": False,
        "adoption_gate_run": False,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "automatic_score_coupling": False,
    }
    publish_or_verify(
        outputs["adjudication_report"],
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
