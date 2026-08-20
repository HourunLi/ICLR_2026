#!/usr/bin/env python3
"""Materialize the frozen single-stream CLIR joint-training pilot manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_data import SemanticGroupBatchSampler, read_jsonl  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state  # noqa: E402
from src.clir_supervision import (  # noqa: E402
    audit_supervision_coverage,
    merge_supervision_annotations,
    output_token_ids_sha256,
)


DEFAULT_PROTOCOL = ROOT / "configs/joint_training_pilot_v1/data_protocol_v1.json"
SCHEMA = "clir-joint-training-data-protocol-v1"
PROTECTED_FIELDS = (
    "id",
    "query_id",
    "output_token_ids",
    "feature_sha256",
    "condition_sha256",
    "correctness",
)
FORBIDDEN_FIELDS = {
    "token_advantage",
    "progress_targets",
    "complete_reconstruction_target",
}
CLAIM_DIAGNOSTIC_FIELDS = (
    "hallucination_claim_spans",
    "hallucination_span_provenance",
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _indexed(
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, Mapping[str, Any]]:
    output = {str(row.get("id", "")): row for row in rows}
    if "" in output or len(output) != len(rows):
        raise ValueError(f"{name} requires unique non-empty row ids")
    return output


def _query_groups(
    rows: Sequence[Mapping[str, Any]], *, expected_width: int, name: str
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise ValueError(f"{name} contains a row without query_id")
        groups[query_id].append(row)
    for query_id, candidates in groups.items():
        indices = {int(row.get("candidate_index", -1)) for row in candidates}
        if len(candidates) != expected_width or indices != set(range(expected_width)):
            raise ValueError(
                f"{name}/{query_id}: expected candidates 0..{expected_width - 1}, "
                f"got {sorted(indices)}"
            )
    return groups


def _assert_identity(
    source: Mapping[str, Any], other: Mapping[str, Any], *, name: str
) -> None:
    for field in ("id", "query_id"):
        if str(source.get(field, "")) != str(other.get(field, "")):
            raise ValueError(f"{name}/{source.get('id')}: protected {field} drifted")
    for field in PROTECTED_FIELDS[2:]:
        if field in other and source.get(field) != other.get(field):
            raise ValueError(f"{name}/{source.get('id')}: protected {field} drifted")
    expected_hash = output_token_ids_sha256(source["output_token_ids"])
    other_hash = other.get("output_token_ids_sha256")
    if other_hash is not None and other_hash != expected_hash:
        raise ValueError(f"{name}/{source.get('id')}: output-token hash drifted")


def _checked_inputs(protocol: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(spec["path"])
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"Frozen input hash drifted: {name}")
        paths[name] = path
    integrity = json.loads(
        paths["feature_integrity_report"].read_text(encoding="utf-8")
    )
    required = int(
        protocol["inputs"]["feature_integrity_report"]["required_failure_count"]
    )
    if int(integrity.get("failure_count", -1)) != required:
        raise ValueError("Frozen feature-integrity report contains failures")
    return paths


def _artifact(paths: Mapping[str, Path], name: str) -> dict[str, Any]:
    return {
        "annotations_jsonl": str(paths[name].relative_to(ROOT)),
        "annotations_sha256": file_sha256(paths[name]),
    }


def _consistency_annotations(
    source_rows: Sequence[Mapping[str, Any]],
    consistency_rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    source_by_id = _indexed(source_rows, name="outcome train")
    selected = [
        row
        for row in consistency_rows
        if row.get("semantic_id") is not None or row.get("style_id") is not None
    ]
    expected = int(protocol["supervision_policy"]["consistency_rows"])
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} consistency rows, found {len(selected)}"
        )
    annotations: list[dict[str, Any]] = []
    selected_by_id: dict[str, Mapping[str, Any]] = {}
    protocol_sha = protocol["inputs"]["consistency_protocol"]["sha256"]
    manifest_sha = protocol["inputs"]["consistency_manifest"]["sha256"]
    for row in selected:
        row_id = str(row["id"])
        source = source_by_id.get(row_id)
        if source is None:
            raise ValueError(f"Consistency row is absent from outcome train: {row_id}")
        _assert_identity(source, row, name="consistency")
        provenance = row.get("consistency_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"Consistency row lacks route provenance: {row_id}")
        annotations.append(
            {
                "schema_version": "clir-supervision-annotation-v1",
                "id": row_id,
                "query_id": str(row["query_id"]),
                "output_token_ids_sha256": output_token_ids_sha256(
                    source["output_token_ids"]
                ),
                "annotation_provenance": {
                    "schema_version": "clir-supervision-provenance-v1",
                    "protocol_sha256": protocol_sha,
                    "annotation_source": str(
                        paths["consistency_manifest"].relative_to(ROOT)
                    ),
                    "annotation_source_sha256": manifest_sha,
                    "evidence_tier": "pipeline_pilot",
                    "route": "A_on_policy_equivalence_mining_repaired_v1a",
                },
                "annotation_metadata": {
                    "consistency_provenance": dict(provenance)
                },
                "semantic_id": row["semantic_id"],
                "style_id": row["style_id"],
            }
        )
        selected_by_id[row_id] = row
    return annotations, selected_by_id


def _merge(
    rows: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    row_ids = {str(row["id"]) for row in rows}
    selected = [row for row in annotations if str(row["id"]) in row_ids]
    if not selected:
        return [dict(row) for row in rows]
    return merge_supervision_annotations(
        rows,
        selected,
        annotation_artifact=artifact,
        require_all_rows_annotated=False,
    )


def _attach_claim_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    templates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    template_by_id = _indexed(templates, name="hallucination span templates")
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        template = template_by_id.get(str(row["id"]))
        if template is not None:
            _assert_identity(source, template, name="hallucination span template")
            for field in CLAIM_DIAGNOSTIC_FIELDS:
                if field not in template:
                    raise ValueError(f"Span template lacks {field}: {row['id']}")
                row[field] = template[field]
        output.append(row)
    return output


def _sampler_report(
    rows: Sequence[Mapping[str, Any]], *, batch_size: int, seed: int
) -> dict[str, Any]:
    dataset = SimpleNamespace(rows=list(rows))
    sampler = SemanticGroupBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        seed=seed,
    )
    batches = list(iter(sampler))
    flattened = [index for batch in batches for index in batch]
    if len(flattened) != len(rows) or set(flattened) != set(range(len(rows))):
        raise ValueError("Semantic-group sampler did not visit every row exactly once")
    positive = 0
    negative = 0
    for batch in batches:
        for left_position, left_index in enumerate(batch):
            left = rows[left_index]
            if left.get("semantic_id") is None or left.get("style_id") is None:
                continue
            for right_index in batch[left_position + 1 :]:
                right = rows[right_index]
                if right.get("semantic_id") is None or right.get("style_id") is None:
                    continue
                semantic_equal = left["semantic_id"] == right["semantic_id"]
                style_equal = left["style_id"] == right["style_id"]
                positive += int(semantic_equal and not style_equal)
                negative += int(not semantic_equal and style_equal)
    return {
        "batches": len(batches),
        "visited_rows": len(flattened),
        "unique_visited_rows": len(set(flattened)),
        "positive_pairs": positive,
        "negative_pairs": negative,
        "all_batch_sizes": dict(sorted(Counter(map(len, batches)).items())),
    }


def _feature_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    trajectory_paths: set[str] = set()
    condition_paths: set[str] = set()
    for row in rows:
        trajectory = Path(str(row["hidden_states_path"]))
        condition = Path(str(row["condition_states_path"]))
        if not trajectory.is_absolute() or not condition.is_absolute():
            raise ValueError("Joint manifests require absolute feature paths")
        if not trajectory.is_file() or not condition.is_file():
            raise FileNotFoundError(f"Missing feature payload for {row['id']}")
        trajectory_paths.add(str(trajectory))
        condition_paths.add(str(condition))
    return {
        "unique_trajectory_features": len(trajectory_paths),
        "unique_condition_features": len(condition_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("Unexpected joint-training data protocol schema")
    paths = _checked_inputs(protocol)
    outputs = {name: resolve(value) for name, value in protocol["outputs"].items()}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite joint data artifacts: {existing}")

    outcome = read_jsonl(paths["outcome_train"])
    ranking = read_jsonl(paths["ranking_validation"])
    consistency = read_jsonl(paths["consistency_manifest"])
    path_annotations = read_jsonl(paths["hallucination_path_annotations"])
    span_annotations = read_jsonl(paths["hallucination_span_annotations"])
    span_train_templates = read_jsonl(paths["hallucination_span_train"])
    span_dev_templates = read_jsonl(paths["hallucination_span_dev"])
    prior_train = read_jsonl(paths["dual_prior_train_annotations"])
    prior_dev = read_jsonl(paths["dual_prior_dev_annotations"])
    for name, rows in (
        ("outcome_train", outcome),
        ("ranking_validation", ranking),
        ("consistency_manifest", consistency),
        ("hallucination_path_annotations", path_annotations),
        ("hallucination_span_annotations", span_annotations),
        ("hallucination_span_train", span_train_templates),
        ("hallucination_span_dev", span_dev_templates),
        ("dual_prior_train_annotations", prior_train),
        ("dual_prior_dev_annotations", prior_dev),
    ):
        expected_rows = int(protocol["inputs"][name]["rows"])
        if len(rows) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} rows, found {len(rows)}")

    if any(FORBIDDEN_FIELDS.intersection(row) for row in outcome):
        raise ValueError("Outcome base already contains a forbidden deferred target")
    outcome_by_id = _indexed(outcome, name="outcome train")
    outcome_groups = _query_groups(
        outcome,
        expected_width=int(protocol["inputs"]["outcome_train"]["candidates_per_query"]),
        name="outcome train",
    )
    ranking_groups = _query_groups(
        ranking,
        expected_width=int(
            protocol["inputs"]["ranking_validation"]["candidates_per_query"]
        ),
        name="ranking validation",
    )

    train_mechanism_ids = {str(row["id"]) for row in span_train_templates}
    dev_mechanism_ids = {str(row["id"]) for row in span_dev_templates}
    if train_mechanism_ids != {str(row["id"]) for row in prior_train}:
        raise ValueError("Hallucination/prior train identities differ")
    if dev_mechanism_ids != {str(row["id"]) for row in prior_dev}:
        raise ValueError("Hallucination/prior dev identities differ")
    mechanism_ids = train_mechanism_ids | dev_mechanism_ids
    if len(mechanism_ids) != len(train_mechanism_ids) + len(dev_mechanism_ids):
        raise ValueError("Mechanism train/dev row overlap detected")
    for name, rows in (
        ("hallucination train", span_train_templates),
        ("hallucination dev", span_dev_templates),
        ("dual-prior train", prior_train),
        ("dual-prior dev", prior_dev),
    ):
        for other in rows:
            source = outcome_by_id.get(str(other["id"]))
            if source is None:
                raise ValueError(f"{name} row is absent from outcome train: {other['id']}")
            _assert_identity(source, other, name=name)

    train_mechanism_queries = {str(row["query_id"]) for row in span_train_templates}
    dev_mechanism_queries = {str(row["query_id"]) for row in span_dev_templates}
    ranking_queries = set(ranking_groups)
    if train_mechanism_queries & dev_mechanism_queries:
        raise ValueError("Mechanism train/dev query leakage detected")
    if (train_mechanism_queries | dev_mechanism_queries) & ranking_queries:
        raise ValueError("Mechanism queries overlap ranking validation")
    filtered = [
        dict(row)
        for row in outcome
        if str(row["query_id"]) not in dev_mechanism_queries
    ]
    filtered_groups = _query_groups(
        filtered,
        expected_width=int(protocol["inputs"]["outcome_train"]["candidates_per_query"]),
        name="joint train",
    )
    split = protocol["split_policy"]
    if len(filtered) != int(split["expected_train_rows"]):
        raise ValueError("Joint train row count differs from the frozen contract")
    if len(filtered_groups) != int(split["expected_train_queries"]):
        raise ValueError("Joint train query count differs from the frozen contract")
    if len(outcome) - len(filtered) != int(split["expected_excluded_rows"]):
        raise ValueError("Excluded mechanism-dev row count differs from the contract")
    if set(filtered_groups) & ranking_queries:
        raise ValueError("Joint train overlaps ranking validation")

    consistency_annotations, consistency_by_id = _consistency_annotations(
        outcome,
        consistency,
        protocol=protocol,
        paths=paths,
    )
    consistency_ids = set(consistency_by_id)
    if not consistency_ids <= {str(row["id"]) for row in filtered}:
        raise ValueError("A consistency row was excluded with mechanism dev")
    if consistency_ids & mechanism_ids:
        raise ValueError("Consistency and mechanism rows must be disjoint")

    joint = _merge(
        filtered,
        consistency_annotations,
        artifact=_artifact(paths, "consistency_manifest"),
    )
    joint = _merge(
        joint,
        path_annotations,
        artifact=_artifact(paths, "hallucination_path_annotations"),
    )
    joint = _merge(
        joint,
        span_annotations,
        artifact=_artifact(paths, "hallucination_span_annotations"),
    )
    joint = _merge(
        joint,
        prior_train,
        artifact=_artifact(paths, "dual_prior_train_annotations"),
    )
    joint = _attach_claim_diagnostics(joint, span_train_templates)
    for row in joint:
        source = consistency_by_id.get(str(row["id"]))
        if source is not None:
            row["consistency_provenance"] = source["consistency_provenance"]

    dev_sources = [outcome_by_id[str(row["id"])] for row in span_dev_templates]
    mechanism_dev = _merge(
        dev_sources,
        path_annotations,
        artifact=_artifact(paths, "hallucination_path_annotations"),
    )
    mechanism_dev = _merge(
        mechanism_dev,
        span_annotations,
        artifact=_artifact(paths, "hallucination_span_annotations"),
    )
    mechanism_dev = _merge(
        mechanism_dev,
        prior_dev,
        artifact=_artifact(paths, "dual_prior_dev_annotations"),
    )
    mechanism_dev = _attach_claim_diagnostics(mechanism_dev, span_dev_templates)

    joint_by_id = _indexed(joint, name="joint train")
    mechanism_train = [joint_by_id[str(row["id"])] for row in span_train_templates]
    train_coverage = audit_supervision_coverage(joint, require_provenance=True)
    mechanism_train_coverage = audit_supervision_coverage(
        mechanism_train, require_provenance=True
    )
    mechanism_dev_coverage = audit_supervision_coverage(
        mechanism_dev, require_provenance=True
    )
    supervision = protocol["supervision_policy"]
    expected_field_rows = {
        "semantic_id": supervision["consistency_rows"],
        "style_id": supervision["consistency_rows"],
        "path_hallucinated": supervision["mechanism_train_rows"],
        "hallucination_onset": supervision["mechanism_train_rows"],
        "token_hallucination_target": supervision["mechanism_train_rows"],
        "token_hallucination_mask": supervision["mechanism_train_rows"],
        "key_prior_target": supervision["mechanism_train_rows"],
        "complete_prior_target": supervision["mechanism_train_rows"],
    }
    for field, expected in expected_field_rows.items():
        if int(train_coverage["field_rows"][field]) != int(expected):
            raise ValueError(f"Joint train {field} coverage drifted")
    explicit_train = train_coverage["explicit_hallucination_tokens"]
    explicit_dev = mechanism_dev_coverage["explicit_hallucination_tokens"]
    for observed, prefix in ((explicit_train, "hallucination_train"), (explicit_dev, "hallucination_dev")):
        for key in ("supervised", "positive", "negative"):
            expected = int(supervision[f"{prefix}_{key}_tokens"])
            if int(observed[key]) != expected:
                raise ValueError(f"{prefix} {key} token count drifted")
    if train_coverage["field_rows"]["progress_targets"] != 0:
        raise ValueError("Joint train unexpectedly contains progress targets")
    if train_coverage["field_rows"]["complete_reconstruction_target"] != 0:
        raise ValueError("Joint train unexpectedly contains reconstruction targets")

    sampler_contract = protocol["sampler_contract"]
    sampler = _sampler_report(
        joint,
        batch_size=int(sampler_contract["batch_size"]),
        seed=int(sampler_contract["seed"]),
    )
    for key, expected_key in (
        ("batches", "expected_batches_per_epoch"),
        ("positive_pairs", "expected_positive_pairs_per_epoch"),
        ("negative_pairs", "expected_negative_pairs_per_epoch"),
    ):
        if int(sampler[key]) != int(sampler_contract[expected_key]):
            raise ValueError(f"Sampler {key} differs from the frozen contract")

    atomic_write_jsonl(outputs["train"], joint)
    atomic_write_jsonl(outputs["mechanism_train"], mechanism_train)
    atomic_write_jsonl(outputs["mechanism_dev"], mechanism_dev)
    classes = {
        "correctness_only_rows": len(joint) - len(consistency_ids) - len(train_mechanism_ids),
        "correctness_plus_consistency_rows": len(consistency_ids),
        "correctness_plus_hallucination_and_prior_rows": len(train_mechanism_ids),
    }
    report = {
        "schema_version": "clir-joint-training-data-report-v1",
        "status": "joint_manifests_ready",
        "evidence_tier": protocol["evidence_tier"],
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "rows": len(
                    joint
                    if name == "train"
                    else mechanism_train
                    if name == "mechanism_train"
                    else mechanism_dev
                ),
            }
            for name, path in outputs.items()
            if name != "report"
        },
        "train_queries": len(filtered_groups),
        "ranking_validation_queries": len(ranking_groups),
        "query_overlap": {
            "train_mechanism_dev": len(set(filtered_groups) & dev_mechanism_queries),
            "train_ranking_validation": len(set(filtered_groups) & ranking_queries),
            "mechanism_dev_ranking_validation": len(dev_mechanism_queries & ranking_queries),
        },
        "row_classes": classes,
        "consistency_mechanism_row_overlap": len(consistency_ids & train_mechanism_ids),
        "train_coverage": train_coverage,
        "mechanism_train_coverage": mechanism_train_coverage,
        "mechanism_dev_coverage": mechanism_dev_coverage,
        "sampler_epoch_1": sampler,
        "features": _feature_report(joint + mechanism_dev),
        "code": git_state(ROOT),
        "pilot_test_accessed": False,
        "final_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(outputs["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
