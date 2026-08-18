from copy import deepcopy

import pytest

from src.clir_supervision import (
    ANNOTATION_SCHEMA,
    PROVENANCE_SCHEMA,
    ROW_PROVENANCE_SCHEMA,
    audit_supervision_coverage,
    merge_supervision_annotations,
    output_token_ids_sha256,
    validate_supervision_annotation,
    validate_row_supervision_provenance,
)


def _row(row_id: str, query_id: str, token_ids=(10, 11, 12)):
    return {
        "id": row_id,
        "query_id": query_id,
        "output_token_ids": list(token_ids),
        "correctness": 1,
        "hidden_states_path": f"{row_id}.pt",
    }


def _annotation(row, **labels):
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "id": row["id"],
        "query_id": row["query_id"],
        "output_token_ids_sha256": output_token_ids_sha256(row["output_token_ids"]),
        "annotation_provenance": {
            "schema_version": PROVENANCE_SCHEMA,
            "protocol_sha256": "a" * 64,
            "annotation_source": "unit-test-explicit-annotation",
        },
        **labels,
    }


def test_merge_supervision_preserves_missing_targets_and_exact_token_identity():
    rows = [_row("r0", "q0"), _row("r1", "q0"), _row("r2", "q1")]
    annotations = [
        _annotation(
            rows[0],
            semantic_id="semantic-a",
            style_id="direct",
            path_hallucinated=0,
            hallucination_onset=-1,
            token_advantage=[0.1, 0.2, 0.3],
            progress_targets=[0.0, 0.5, 1.0],
            key_prior_target=[1.0, 0.0, 0.0],
            complete_prior_target=[1.0, 1.0, 0.0],
        ),
        _annotation(
            rows[1],
            semantic_id="semantic-a",
            style_id="verbose",
            path_hallucinated=1,
            hallucination_onset=1,
        ),
        _annotation(
            rows[2],
            semantic_id="semantic-b",
            style_id="direct",
        ),
    ]

    merged = merge_supervision_annotations(
        rows,
        annotations,
        annotation_artifact={"annotations_sha256": "b" * 64},
    )
    coverage = audit_supervision_coverage(merged)

    assert merged[0]["output_token_ids"] == rows[0]["output_token_ids"]
    assert "token_advantage" not in merged[1]
    assert "key_prior_target" not in merged[2]
    assert merged[1]["hallucination_onset"] == 1
    assert merged[0]["clir_supervision_provenance"]["schema_version"] == ROW_PROVENANCE_SCHEMA
    assert len(merged[0]["clir_supervision_provenance"]["sources"]) == 1
    assert coverage["field_rows"]["token_advantage"] == 1
    assert coverage["joint_prior_rows"] == 1
    assert coverage["eligible_components"]["consistency"] is True
    assert coverage["eligible_components"]["path_hallucination"] is True
    assert coverage["eligible_components"]["dual_prior"] is True


def test_annotation_rejects_token_hash_length_and_onset_drift():
    row = _row("r0", "q0")
    wrong_hash = _annotation(row, token_advantage=[0.0, 0.0, 0.0])
    wrong_hash["output_token_ids_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="output_token_ids_sha256"):
        validate_supervision_annotation(wrong_hash, row)

    wrong_length = _annotation(row, key_prior_target=[1.0, 0.0])
    with pytest.raises(ValueError, match="length mismatch"):
        validate_supervision_annotation(wrong_length, row)

    inconsistent_onset = _annotation(
        row,
        path_hallucinated=0,
        hallucination_onset=1,
    )
    with pytest.raises(ValueError, match="disagree"):
        validate_supervision_annotation(inconsistent_onset, row)

    typo = _annotation(row, path_hallucinated=0)
    typo["progress_target"] = [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="unknown top-level fields"):
        validate_supervision_annotation(typo, row)


def test_dual_prior_targets_require_nonempty_nested_evidence():
    row = _row("r0", "q0")

    empty_key = _annotation(
        row,
        key_prior_target=[0.0, 0.0, 0.0],
        complete_prior_target=[1.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="key_prior_target must contain"):
        validate_supervision_annotation(empty_key, row)

    non_nested = _annotation(
        row,
        key_prior_target=[1.0, 0.0, 0.0],
        complete_prior_target=[0.0, 1.0, 0.0],
    )
    with pytest.raises(ValueError, match="pointwise subset"):
        validate_supervision_annotation(non_nested, row)

    merged = deepcopy(row)
    merged.update(non_nested)
    merged["clir_supervision_provenance"] = {
        "schema_version": ROW_PROVENANCE_SCHEMA,
        "output_token_ids_sha256": output_token_ids_sha256(row["output_token_ids"]),
        "sources": [
            {
                "schema_version": PROVENANCE_SCHEMA,
                "protocol_sha256": "a" * 64,
                "annotation_source": "unit-test-explicit-annotation",
            }
        ],
    }
    with pytest.raises(ValueError, match="pointwise subset"):
        audit_supervision_coverage([merged])


def test_sparse_token_hallucination_labels_require_a_binary_explicit_mask():
    row = _row("r0", "q0")
    annotation = _annotation(
        row,
        path_hallucinated=1,
        hallucination_onset=1,
        token_hallucination_target=[0, 1, 0],
        token_hallucination_mask=[1, 1, 0],
    )
    merged = merge_supervision_annotations(
        [row],
        [annotation],
        annotation_artifact={"annotations_sha256": "b" * 64},
    )
    coverage = audit_supervision_coverage(merged, require_provenance=True)

    assert coverage["explicit_hallucination_tokens"] == {
        "supervised": 2,
        "positive": 1,
        "negative": 1,
    }

    missing_mask = _annotation(row, token_hallucination_target=[0, 1, 0])
    with pytest.raises(ValueError, match="must be provided together"):
        validate_supervision_annotation(missing_mask, row)
    outside_mask = _annotation(
        row,
        token_hallucination_target=[0, 1, 0],
        token_hallucination_mask=[1, 0, 0],
    )
    with pytest.raises(ValueError, match="zero outside"):
        validate_supervision_annotation(outside_mask, row)


def test_merge_refuses_to_replace_existing_supervision():
    row = _row("r0", "q0")
    row["progress"] = [0.0, 0.0, 1.0]
    annotation = _annotation(row, progress_targets=[0.0, 0.5, 1.0])

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        merge_supervision_annotations(
            [row],
            [annotation],
            annotation_artifact={"annotations_sha256": "b" * 64},
        )


def test_merge_requires_explicit_coverage_when_requested():
    rows = [_row("r0", "q0"), _row("r1", "q1")]
    annotation = _annotation(rows[0], path_hallucinated=0)

    with pytest.raises(ValueError, match="do not cover every source row"):
        merge_supervision_annotations(
            rows,
            [annotation],
            annotation_artifact={"annotations_sha256": "b" * 64},
            require_all_rows_annotated=True,
        )


def test_merge_rejects_empty_annotation_artifact():
    row = _row("r0", "q0")
    with pytest.raises(ValueError, match="At least one"):
        merge_supervision_annotations(
            [row],
            [],
            annotation_artifact={"annotations_sha256": "b" * 64},
        )
    with pytest.raises(ValueError, match="annotations_sha256"):
        merge_supervision_annotations(
            [row],
            [_annotation(row, path_hallucinated=0)],
            annotation_artifact={},
        )


def test_annotation_rejects_mutated_identity_even_with_reused_labels():
    row = _row("r0", "q0")
    annotation = _annotation(row, path_hallucinated=0)
    mutated = deepcopy(annotation)
    mutated["query_id"] = "different-query"

    with pytest.raises(ValueError, match="query_id"):
        validate_supervision_annotation(mutated, row)


def test_domain_id_can_be_the_consistency_attribute_fallback():
    row = _row("r0", "q0")
    annotation = _annotation(row, semantic_id="semantic-a", domain_id="math-shell")

    validated = validate_supervision_annotation(annotation, row)

    assert validated["labels"]["semantic_id"] == "semantic-a"
    assert validated["labels"]["domain_id"] == "math-shell"

    style_without_semantic = _annotation(row, style_id="verbose")
    with pytest.raises(ValueError, match="style_id requires semantic_id"):
        validate_supervision_annotation(style_without_semantic, row)


def test_coverage_rejects_inconsistent_existing_labels():
    missing_style = _row("r0", "q0")
    missing_style["semantic_id"] = "semantic-a"
    with pytest.raises(ValueError, match="semantic_id requires style_id or domain_id"):
        audit_supervision_coverage([missing_style])

    bad_onset = _row("r0", "q0")
    bad_onset.update(path_hallucinated=0, hallucination_onset=1)
    with pytest.raises(ValueError, match="disagree"):
        audit_supervision_coverage([bad_onset])


def test_coverage_requires_consistent_reconstruction_dimension():
    first = _row("r0", "q0")
    second = _row("r1", "q1")
    first["complete_reconstruction_target"] = [0.0, 1.0]
    second["complete_reconstruction_target"] = [0.0, 1.0, 2.0]

    with pytest.raises(ValueError, match="dimensions differ"):
        audit_supervision_coverage([first, second])
    with pytest.raises(ValueError, match="length mismatch"):
        audit_supervision_coverage([first], expected_reconstruction_dim=3)


def test_onset_eligibility_requires_a_positive_onset():
    negative = _row("r0", "q0")
    negative.update(path_hallucinated=0, hallucination_onset=-1)

    coverage = audit_supervision_coverage([negative])

    assert coverage["field_rows"]["hallucination_onset"] == 1
    assert coverage["positive_onset_rows"] == 0
    assert coverage["eligible_components"]["onset_localization"] is False


def test_coverage_can_require_field_level_provenance():
    row = _row("r0", "q0")
    row["path_hallucinated"] = 0

    with pytest.raises(ValueError, match="missing row provenance"):
        audit_supervision_coverage([row], require_provenance=True)


def test_incremental_merge_preserves_one_source_per_field():
    row = _row("r0", "q0")
    first = merge_supervision_annotations(
        [row],
        [_annotation(row, semantic_id="semantic-a", style_id="direct")],
        annotation_artifact={"annotations_sha256": "b" * 64},
    )
    second_annotation = _annotation(row, path_hallucinated=1, hallucination_onset=1)
    second = merge_supervision_annotations(
        first,
        [second_annotation],
        annotation_artifact={"annotations_sha256": "c" * 64},
    )

    provenance = validate_row_supervision_provenance(second[0])
    assert len(provenance["sources"]) == 2
    assert {field for source in provenance["sources"] for field in source["fields"]} == {
        "semantic_id",
        "style_id",
        "path_hallucinated",
        "hallucination_onset",
    }
    assert audit_supervision_coverage(second, require_provenance=True)[
        "unprovenanced_supervised_rows"
    ] == 0
