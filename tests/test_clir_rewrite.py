from copy import deepcopy

import pytest

from src.clir_rewrite import (
    REWRITE_SCHEMA,
    apply_rewrite_transform,
    build_rewrite_plan,
    invert_rewrite_transform,
    materialize_rewrite_views,
    select_pilot_sources,
    validate_rewrite_protocol,
    validate_rewrite_views,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(chr(token_id) for token_id in token_ids)


def _protocol():
    return {
        "schema_version": REWRITE_SCHEMA,
        "formal_mechanism_claim_allowed": False,
        "source": {
            "manifest": "artifacts/train.jsonl",
            "sha256": "a" * 64,
            "split": "train",
            "checker_version": "clir_gsm8k_numeric_v5",
        },
        "identity": {
            "query_id": "preserve_source_query_id",
            "semantic_id": "derive_from_source_trajectory_id",
            "style_id": "declared_view_style",
            "source_candidate_index": "lineage_only_not_top_level",
        },
        "selection": {
            "order": "source_manifest_order",
            "rows_per_correctness": 2,
            "correctness_values": [0, 1],
            "require_distinct_query_ids": True,
        },
        "rewrite_policy": {
            "trajectory_policy": "teacher_forced_exact_rewrite_token_ids",
            "condition_policy": "reuse_verified_prompt_only_source_condition",
            "semantic_certificate": "inverse_transform_byte_exact_source_response",
            "answer_gate": "same_normalized_candidate_answer_and_correctness",
        },
        "views": [
            {"style_id": "original", "transform": "identity"},
            {
                "style_id": "markdown_bullets",
                "transform": "prefix_every_physical_line_dash_space",
            },
            {
                "style_id": "blockquote",
                "transform": "prefix_every_physical_line_greater_space",
            },
        ],
        "model": {
            "repo_id": "model",
            "revision": "revision",
            "tokenizer_revision": "revision",
            "torch_dtype": "bfloat16",
        },
        "hidden_states": {
            "layer_policy": "all_outputs_hidden_states",
            "layer_count": 33,
            "per_layer_hidden_size": 3072,
            "feature_dim": 101376,
            "combine": "concatenate_last_dimension",
            "trajectory_slice": "exact_output_token_ids",
            "extractor_applied_additional_final_norm": False,
            "storage_dtype": "bfloat16",
        },
    }


def _row(index, *, correctness):
    answer = "3" if correctness else "4"
    response = f"First compute 1 + 2 = {answer}.\n\nFinal: \\boxed{{{answer}}}"
    return {
        "id": f"row-{index}",
        "query_id": f"query-{index}",
        "candidate_index": index,
        "split": "train",
        "response": response,
        "trajectory": response,
        "raw_reference": "Reference work.\n#### 3",
        "prompt": "Compute 1 + 2.",
        "prompt_token_ids": [1, 2, 3],
        "output_token_ids": CharacterTokenizer().encode(response),
        "correctness": correctness,
        "normalized_candidate_answer": answer,
        "condition_states_path": f"condition-{index}.pt",
        "condition_sha256": "b" * 64,
        "hidden_states_path": f"trajectory-{index}.pt",
        "feature_sha256": "c" * 64,
        "feature_metadata": {
            "layer_count": 33,
            "feature_dim": 101376,
            "storage_dtype": "bfloat16",
        },
        "provenance": {
            "model_id": "model",
            "model_revision": "revision",
            "tokenizer_revision": "revision",
            "protocol_version": "source-v1",
        },
    }


def test_reversible_surface_transforms_are_byte_exact():
    source = "Step one.\n\nStep two.\n"
    for transform in (
        "identity",
        "prefix_every_physical_line_dash_space",
        "prefix_every_physical_line_greater_space",
    ):
        rewritten = apply_rewrite_transform(source, transform)
        assert invert_rewrite_transform(rewritten, transform) == source

    with pytest.raises(ValueError, match="expected"):
        invert_rewrite_transform("not prefixed", "prefix_every_physical_line_dash_space")


def test_protocol_forbids_test_scope_and_formal_claims():
    protocol = _protocol()
    assert validate_rewrite_protocol(protocol)["source"]["split"] == "train"

    test_scope = deepcopy(protocol)
    test_scope["source"]["manifest"] = "pilot_test/secret.jsonl"
    with pytest.raises(ValueError, match="pilot_test"):
        validate_rewrite_protocol(test_scope)

    formal = deepcopy(protocol)
    formal["formal_mechanism_claim_allowed"] = True
    with pytest.raises(ValueError, match="forbid formal"):
        validate_rewrite_protocol(formal)


def test_selection_is_balanced_distinct_query_and_manifest_ordered():
    rows = [
        _row(0, correctness=1),
        {**_row(1, correctness=0), "query_id": "query-0"},
        _row(2, correctness=1),
        _row(3, correctness=0),
        _row(4, correctness=0),
    ]
    selected = select_pilot_sources(rows, _protocol())

    assert [row["id"] for row in selected] == ["row-0", "row-2", "row-3", "row-4"]
    assert [row["correctness"] for row in selected] == [1, 1, 0, 0]
    assert len({row["query_id"] for row in selected}) == 4


def test_plan_and_views_preserve_query_answer_and_activate_consistency():
    sources = [
        _row(0, correctness=1),
        _row(1, correctness=0),
        _row(2, correctness=1),
        _row(3, correctness=0),
    ]
    tokenizer = CharacterTokenizer()
    plan = build_rewrite_plan(
        sources,
        tokenizer,
        _protocol(),
        source_manifest_sha256="a" * 64,
        rewrite_protocol_sha256="d" * 64,
    )
    views = materialize_rewrite_views(
        sources,
        plan,
        plan_sha256="e" * 64,
        source_manifest_path="/artifacts/train.jsonl",
        source_manifest_sha256="a" * 64,
        rewrite_protocol_path="/configs/rewrite.json",
        rewrite_protocol_sha256="d" * 64,
        code={"commit": "f" * 40, "dirty": False},
    )
    summary = validate_rewrite_views(views, expected_plan_sha256="e" * 64)

    assert len(plan) == len(views) == 12
    assert summary["semantic_groups"] == 4
    assert summary["coverage"]["consistency_pairs"] == {
        "positive_same_semantic_different_style": 12,
        "negative_different_semantic_same_style": 18,
    }
    assert summary["coverage"]["eligible_components"]["consistency"] is True
    assert summary["coverage"]["eligible_component_count"] == 1
    for view in views:
        source = next(row for row in sources if row["id"] == view["rewrite_lineage"]["source_id"])
        assert view["query_id"] == source["query_id"]
        assert view["correctness"] == source["correctness"]
        assert view["semantic_id"] != view["query_id"]
        assert "candidate_index" not in view
        if view["style_id"] == "original":
            assert view["output_token_ids"] == source["output_token_ids"]
            assert view["hidden_states_path"] == source["hidden_states_path"]
        else:
            assert "hidden_states_path" not in view


def test_view_validation_rejects_query_and_response_drift():
    sources = [_row(index, correctness=index % 2) for index in range(4)]
    plan = build_rewrite_plan(
        sources,
        CharacterTokenizer(),
        _protocol(),
        source_manifest_sha256="a" * 64,
        rewrite_protocol_sha256="d" * 64,
    )
    views = materialize_rewrite_views(
        sources,
        plan,
        plan_sha256="e" * 64,
        source_manifest_path="/artifacts/train.jsonl",
        source_manifest_sha256="a" * 64,
        rewrite_protocol_path="/configs/rewrite.json",
        rewrite_protocol_sha256="d" * 64,
        code={"commit": "f" * 40, "dirty": False},
    )

    changed_query = deepcopy(views)
    changed_query[0]["query_id"] = "another-query"
    with pytest.raises(ValueError, match="changed query_id"):
        validate_rewrite_views(changed_query, expected_plan_sha256="e" * 64)

    changed_response = deepcopy(views)
    changed_response[1]["response"] += " extra claim"
    with pytest.raises(ValueError, match="semantic certificate|response hash"):
        validate_rewrite_views(changed_response, expected_plan_sha256="e" * 64)
