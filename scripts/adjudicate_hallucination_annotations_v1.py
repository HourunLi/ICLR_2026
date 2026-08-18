#!/usr/bin/env python3
"""Materialize blind adjudication decisions and publish localization supervision."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    adjudication_decision_annotation,
    canonical_json,
    canonical_sha256,
    content_token_offsets,
    file_sha256,
    map_annotation,
    mapped_earliest_problem_claim,
    raw_annotation_from_visible,
    read_jsonl,
)
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/hallucination_localization_v1/adjudication_protocol_v1.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify_input(path: Path, expected_sha256: str) -> None:
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


def index_unique(rows: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in indexed:
            raise ValueError(f"{name} requires unique non-empty item_id values")
        indexed[item_id] = row
    return indexed


def visible_annotation(label: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path_status": label["path_status"],
        "earliest_problem_claim_index": label["earliest_problem_claim_index"],
        "hallucination_onset": label.get("hallucination_onset"),
        "confidence": label["confidence"],
        "summary": label["summary"],
        "claim_reviews": [
            {
                key: claim[key]
                for key in (
                    "claim_text",
                    "occurrence",
                    "status",
                    "reason",
                    "char_start",
                    "char_end",
                    "token_start",
                    "token_end_exclusive",
                )
            }
            for claim in label["claim_reviews"]
        ],
    }


def blocking_disagreement(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["path_status"] != right["path_status"]:
        return True
    return (
        left["path_status"] == "hallucinated"
        and int(left["hallucination_onset"]) != int(right["hallucination_onset"])
    )


def offsets_for(
    *,
    tokenizer: Any,
    response: str,
    output_token_ids: Sequence[int],
) -> list[tuple[int, int]]:
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    encoded_ids = [int(value) for value in encoded["input_ids"]]
    trailing = [int(value) for value in output_token_ids[len(encoded_ids) :]]
    return content_token_offsets(
        response=response,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir", default="/prodcpfs/user/panzhixin/hf_cache")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Adjudication materialization requires a clean committed worktree")

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-hallucination-adjudication-protocol-v1":
        raise ValueError("Unknown adjudication protocol schema")
    verify_input(resolve(protocol["base_protocol"]), protocol["base_protocol_sha256"])
    verify_input(
        resolve(protocol["comparison_protocol"]),
        protocol["comparison_protocol_sha256"],
    )
    inputs = protocol["inputs"]
    paths = {
        key: resolve(value)
        for key, value in inputs.items()
        if not key.endswith("_sha256")
    }
    for key, path in paths.items():
        verify_input(path, str(inputs[f"{key}_sha256"]))

    all_items = read_jsonl(paths["all_items"])
    all_lineage = read_jsonl(paths["all_lineage"])
    primary = read_jsonl(paths["primary_labels"])
    secondary = read_jsonl(paths["secondary_labels"])
    adjudication_items = read_jsonl(paths["adjudication_items"])
    adjudication_lineage = read_jsonl(paths["adjudication_lineage"])
    resolutions = read_jsonl(paths["adjudication_resolutions"])
    expected_rows = int(protocol["trainability_gate"]["expected_rows"])
    if not (
        len(all_items)
        == len(all_lineage)
        == len(primary)
        == len(secondary)
        == expected_rows
    ):
        raise ValueError("All-label inputs do not match the expected row count")
    if not (
        len(adjudication_items)
        == len(adjudication_lineage)
        == len(resolutions)
    ):
        raise ValueError("Adjudication package row counts differ")

    all_items_by_id = index_unique(all_items, name="all_items")
    all_lineage_by_id = index_unique(all_lineage, name="all_lineage")
    primary_by_id = index_unique(primary, name="primary_labels")
    secondary_by_id = index_unique(secondary, name="secondary_labels")
    adjudication_by_id = index_unique(
        adjudication_items,
        name="adjudication_items",
    )
    adjudication_lineage_by_id = index_unique(
        adjudication_lineage,
        name="adjudication_lineage",
    )
    resolutions_by_id = index_unique(resolutions, name="adjudication_resolutions")
    if not (
        set(all_items_by_id)
        == set(all_lineage_by_id)
        == set(primary_by_id)
        == set(secondary_by_id)
    ):
        raise ValueError("All-label item-id sets differ")
    if not (
        set(adjudication_by_id)
        == set(adjudication_lineage_by_id)
        == set(resolutions_by_id)
    ):
        raise ValueError("Adjudication item-id sets differ")

    expected_blocking = {
        item_id
        for item_id in all_items_by_id
        if blocking_disagreement(primary_by_id[item_id], secondary_by_id[item_id])
    }
    if expected_blocking != set(adjudication_by_id):
        raise ValueError("Adjudication package does not exactly cover blocking disagreements")

    final_raw_by_id: dict[str, dict[str, Any]] = {}
    decision_by_id: dict[str, Mapping[str, Any]] = {}
    adjudications_raw: list[dict[str, Any]] = []
    resolution_counts: Counter[str] = Counter()
    for package_item, role_lineage, decision in zip(
        adjudication_items,
        adjudication_lineage,
        resolutions,
    ):
        item_id = str(package_item["item_id"])
        if item_id != role_lineage["item_id"] or item_id != decision["item_id"]:
            raise ValueError("Adjudication input order or identity drifted")
        base_item = all_items_by_id[item_id]
        if (
            package_item["problem"] != base_item["problem"]
            or package_item["trajectory"] != base_item["trajectory"]
        ):
            raise ValueError("Adjudication package text differs from blind base item")
        left = primary_by_id[item_id]
        right = secondary_by_id[item_id]
        if canonical_sha256(left) != role_lineage["primary_label_sha256"]:
            raise ValueError("Primary adjudication lineage hash drifted")
        if canonical_sha256(right) != role_lineage["secondary_label_sha256"]:
            raise ValueError("Secondary adjudication lineage hash drifted")
        roles = {"primary": left, "secondary": right}
        for view_name in ("annotation_a", "annotation_b"):
            role = role_lineage[f"{view_name}_role"]
            if role not in roles:
                raise ValueError("Unknown adjudication view role")
            if package_item[view_name] != visible_annotation(roles[role]):
                raise ValueError("Adjudication A/B view differs from mapped source label")
        annotation = adjudication_decision_annotation(decision, package_item)
        final_raw_by_id[item_id] = annotation
        decision_by_id[item_id] = decision
        resolution_counts[str(decision["resolution"])] += 1
        adjudications_raw.append(
            {
                "schema_version": "clir-hallucination-adjudication-decision-v1",
                "item_id": item_id,
                "relation_to_views": decision["resolution"],
                "final_annotation": annotation,
                "rationale": decision["rationale"],
            }
        )

    base_protocol = json.loads(
        resolve(protocol["base_protocol"]).read_text(encoding="utf-8")
    )
    from transformers import AutoTokenizer

    token_cfg = base_protocol["token_mapping"]
    tokenizer = AutoTokenizer.from_pretrained(
        token_cfg["model_id"],
        revision=token_cfg["tokenizer_revision"],
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=bool(token_cfg["trust_remote_code"]),
    )
    if not tokenizer.is_fast:
        raise ValueError("Exact adjudication onset mapping requires a fast tokenizer")

    source_by_id = {
        str(row["id"]): row for row in read_jsonl(paths["source_manifest"])
    }
    protocol_sha256 = file_sha256(protocol_path)
    labels: list[dict[str, Any]] = []
    correctness_cross_tab: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    label_source_counts: Counter[str] = Counter()
    onsets: list[int] = []
    exact_token_identity = True
    for item, identity, left, right in zip(
        all_items,
        all_lineage,
        primary,
        secondary,
    ):
        item_id = str(item["item_id"])
        if not (
            item_id == identity["item_id"] == left["item_id"] == right["item_id"]
        ):
            raise ValueError("Final mapping input order or identity drifted")
        source = source_by_id[str(identity["source_id"])]
        if source["response"] != item["trajectory"]:
            raise ValueError("Frozen source response differs from adjudication item")
        output_token_ids = [int(value) for value in source["output_token_ids"]]
        token_hash = canonical_sha256(output_token_ids)
        if not (
            token_hash
            == identity["output_token_ids_sha256"]
            == left["output_token_ids_sha256"]
            == right["output_token_ids_sha256"]
        ):
            exact_token_identity = False
            raise ValueError("Frozen output-token identity drifted")
        if item_id in final_raw_by_id:
            annotation = final_raw_by_id[item_id]
            decision = decision_by_id[item_id]
            label_source = str(decision["resolution"])
            annotator = {
                "role": "internal_blinded_adjudication",
                "identity": decision["adjudicator"],
                "resolution": label_source,
                "resolution_sha256": canonical_sha256(decision),
                "adjudication_protocol_sha256": protocol_sha256,
                "claim_level_gold": False,
            }
        else:
            if blocking_disagreement(left, right):
                raise AssertionError("Blocking disagreement escaped adjudication")
            annotation = raw_annotation_from_visible(left, item_id=item_id)
            label_source = "path_onset_consensus_primary_claim_projection"
            annotator = {
                "role": "dual_annotation_path_onset_consensus",
                "identity": "primary_claim_projection_after_nonblocking_consensus",
                "primary_label_sha256": canonical_sha256(left),
                "secondary_label_sha256": canonical_sha256(right),
                "adjudication_protocol_sha256": protocol_sha256,
                "claim_level_gold": False,
            }
        offsets = offsets_for(
            tokenizer=tokenizer,
            response=str(source["response"]),
            output_token_ids=output_token_ids,
        )
        label = map_annotation(
            annotation,
            item,
            identity,
            offsets=offsets,
            annotator=annotator,
        )
        mapped_earliest_problem_claim(label)
        labels.append(label)
        path_counts[str(label["path_status"])] += 1
        label_source_counts[label_source] += 1
        correctness_cross_tab[
            f"correctness_{identity['correctness']}__{label['path_status']}"
        ] += 1
        if label.get("path_hallucinated") == 1:
            onsets.append(int(label["hallucination_onset"]))

    outputs = {key: resolve(value) for key, value in protocol["outputs"].items()}
    publish_or_verify(outputs["adjudications_raw"], jsonl_payload(adjudications_raw))
    publish_or_verify(outputs["adjudicated_labels"], jsonl_payload(labels))
    adjudicated_labels_sha256 = file_sha256(outputs["adjudicated_labels"])

    supervision: list[dict[str, Any]] = []
    for label in labels:
        if label["path_status"] == "uncertain":
            continue
        supervision.append(
            {
                "schema_version": "clir-supervision-annotation-v1",
                "id": label["id"],
                "query_id": label["query_id"],
                "output_token_ids_sha256": label["output_token_ids_sha256"],
                "path_hallucinated": label["path_hallucinated"],
                "hallucination_onset": label["hallucination_onset"],
                "annotation_provenance": {
                    "schema_version": "clir-supervision-provenance-v1",
                    "protocol_sha256": protocol_sha256,
                    "annotation_source": str(
                        outputs["adjudicated_labels"].relative_to(ROOT)
                    ),
                    "annotation_source_sha256": adjudicated_labels_sha256,
                    "evidence_tier": "pipeline_pilot",
                    "claim_level_gold": False,
                },
                "annotation_metadata": {
                    "item_id": label["item_id"],
                    "path_status": label["path_status"],
                    "confidence": label["confidence"],
                    "label_role": label["annotator"]["role"],
                },
            }
        )
    publish_or_verify(outputs["supervision_annotations"], jsonl_payload(supervision))

    gate = protocol["trainability_gate"]
    unresolved = resolution_counts["unresolved"]
    gate_checks = {
        "expected_rows": len(labels) == int(gate["expected_rows"]),
        "every_blocking_disagreement_resolved": (
            len(adjudications_raw) == len(expected_blocking) and unresolved == 0
        ),
        "minimum_clean_rows": path_counts["clean"] >= int(gate["minimum_clean_rows"]),
        "minimum_hallucinated_rows": (
            path_counts["hallucinated"] >= int(gate["minimum_hallucinated_rows"])
        ),
        "exact_frozen_token_identity": exact_token_identity,
        "no_pilot_test_access": not bool(protocol["pilot_test_access_allowed"]),
    }
    trainability_gate_passed = all(gate_checks.values())
    report = {
        "schema_version": "clir-hallucination-adjudication-report-v1",
        "evidence_tier": "pipeline_pilot",
        "claim_level_gold_status": "not_gold_internal_blinded_pipeline_pilot",
        "code": code,
        "adjudication_protocol": str(protocol_path.relative_to(ROOT)),
        "adjudication_protocol_sha256": protocol_sha256,
        "agreement_report_sha256": file_sha256(paths["agreement_report"]),
        "rows": len(labels),
        "consensus_rows": len(labels) - len(adjudications_raw),
        "adjudicated_rows": len(adjudications_raw),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "path_status_counts": dict(sorted(path_counts.items())),
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "hallucinated_onset": {
            "count": len(onsets),
            "min": min(onsets) if onsets else None,
            "max": max(onsets) if onsets else None,
            "mean": sum(onsets) / len(onsets) if onsets else None,
        },
        "correctness_path_cross_tab_private_diagnostic": dict(
            sorted(correctness_cross_tab.items())
        ),
        "adjudicator_disclosure": protocol["adjudicator_disclosure"],
        "gate_checks": gate_checks,
        "trainability_gate_passed": trainability_gate_passed,
        "training_labels_published": trainability_gate_passed,
        "supervision_rows_published": len(supervision),
        "artifacts": {
            "resolutions_sha256": file_sha256(paths["adjudication_resolutions"]),
            "adjudications_raw": str(outputs["adjudications_raw"].relative_to(ROOT)),
            "adjudications_raw_sha256": file_sha256(outputs["adjudications_raw"]),
            "adjudicated_labels": str(outputs["adjudicated_labels"].relative_to(ROOT)),
            "adjudicated_labels_sha256": adjudicated_labels_sha256,
            "supervision_annotations": str(
                outputs["supervision_annotations"].relative_to(ROOT)
            ),
            "supervision_annotations_sha256": file_sha256(
                outputs["supervision_annotations"]
            ),
        },
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    publish_or_verify(
        outputs["adjudication_report"],
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not trainability_gate_passed:
        raise RuntimeError("Adjudicated localization labels failed the trainability gate")


if __name__ == "__main__":
    main()
