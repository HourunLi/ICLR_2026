import copy
import hashlib

import pytest

from src.clir_on_policy_pairs import canonical_json
from src.clir_on_policy_reaudit import (
    ANNOTATION_SCHEMA,
    TARGET_STYLE,
    build_reannotation_records,
    validate_annotation_set,
)
from src.clir_reasoning_rewrite import VERIFIER_SCHEMA


def _sha(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _source(row_id, query_id, response, output_tokens, candidate_index):
    return {
        "id": row_id,
        "query_id": query_id,
        "candidate_index": candidate_index,
        "response": response,
        "output_token_ids": output_tokens,
        "correctness": 1,
        "normalized_candidate_answer": "20",
        "checker_version": "clir_gsm8k_numeric_v5",
        "label_provenance": {"label_protocol_sha256": "a" * 64},
    }


def _fixture():
    anchor = _source("row-long", "q1", "Four groups of five give 20.", [1, 2, 3, 4], 0)
    alternate = _source("row-short", "q1", "4 times 5 is 20.", [1, 2], 1)
    candidate = {
        "pair_id": "OPA-test",
        "problem": "What is four times five?",
        "anchor": {
            "id": anchor["id"],
            "response": anchor["response"],
            "output_token_ids": anchor["output_token_ids"],
            "row_sha256": _sha(anchor),
        },
        "alternate": {
            "id": alternate["id"],
            "response": alternate["response"],
            "output_token_ids": alternate["output_token_ids"],
            "row_sha256": _sha(alternate),
        },
    }
    item, lineage = build_reannotation_records(
        candidate, source_rows={anchor["id"]: anchor, alternate["id"]: alternate}
    )
    return candidate, item, lineage


def _report(item):
    return {
        "schema_version": VERIFIER_SCHEMA,
        "source_to_rewrite": {
            "claims": [
                {
                    "claim_id": "S1",
                    "source_claim": "4 times 5 is 20.",
                    "essential": True,
                    "rewrite_evidence": "Four groups of five give 20.",
                    "relation": "equivalent",
                    "explanation": "The multiplication and result are unchanged.",
                }
            ],
            "all_essential_claims_preserved": True,
        },
        "rewrite_to_source": {
            "claims": [
                {
                    "claim_id": "R1",
                    "rewrite_claim": "Four groups of five give 20.",
                    "substantive": True,
                    "source_evidence": "4 times 5 is 20.",
                    "relation": "equivalent",
                    "explanation": "The grouped wording states the same operation.",
                }
            ],
            "all_substantive_claims_supported": True,
        },
        "global_relation": {
            "same_task_and_goal": True,
            "same_core_premises": True,
            "same_reasoning_method": True,
            "same_key_inferences": True,
            "same_intermediate_conclusions": True,
            "same_final_conclusion": True,
            "introduced_external_fact": False,
            "omitted_essential_claim": False,
            "contradicted_source_claim": False,
            "replaced_with_different_solution": False,
            "introduced_new_error": False,
        },
        "error_alignment": {
            "applicable": False,
            "source_error": None,
            "rewrite_error": None,
            "same_error_mechanism": None,
            "same_semantic_error_location": None,
            "same_downstream_effect": None,
        },
        "style_assessment": {
            "target_style": TARGET_STYLE,
            "satisfied": True,
            "evidence": "The longer trajectory spells out groups while preserving the calculation.",
        },
        "risk_review": [
            {
                "risk_id": risk_id,
                "resolved": True,
                "explanation": "The differing surface count is harmless here.",
            }
            for risk_id in item["risk_probe"]["expected_risk_ids"]
        ],
        "confidence": "high",
        "decision": "accept",
        "issues": [],
    }


def test_reannotation_orientation_uses_token_length_not_candidate_role():
    _, item, lineage = _fixture()
    assert item["source_trajectory"] == "4 times 5 is 20."
    assert item["rewrite_trajectory"] == "Four groups of five give 20."
    assert item["source_output_tokens"] == 2
    assert item["rewrite_output_tokens"] == 4
    assert lineage["source_original_role"] == "native_alternate"
    assert lineage["rewrite_original_role"] == "native_anchor"
    assert lineage["checker"]["source_correctness"] == 1


def test_complete_claim_annotation_derives_acceptance():
    _, item, lineage = _fixture()
    annotation = {
        "schema_version": ANNOTATION_SCHEMA,
        "item_id": item["item_id"],
        "report": _report(item),
    }
    result = validate_annotation_set(
        [annotation], items=[item], lineages=[lineage]
    )
    assert result[0]["acceptance"]["status"] == "accepted"


def test_missing_risk_review_routes_accept_request_to_review():
    _, item, lineage = _fixture()
    report = _report(item)
    report["risk_review"] = []
    annotation = {
        "schema_version": ANNOTATION_SCHEMA,
        "item_id": item["item_id"],
        "report": report,
    }
    result = validate_annotation_set(
        [annotation], items=[item], lineages=[lineage]
    )
    if item["risk_probe"]["expected_risk_ids"]:
        assert result[0]["acceptance"]["status"] == "review_required"


def test_target_style_and_strict_length_contrast_are_enforced():
    candidate, item, lineage = _fixture()
    report = _report(item)
    report["style_assessment"]["target_style"] = "terse_formal"
    annotation = {
        "schema_version": ANNOTATION_SCHEMA,
        "item_id": item["item_id"],
        "report": report,
    }
    with pytest.raises(ValueError, match="target style"):
        validate_annotation_set([annotation], items=[item], lineages=[lineage])

    equal = copy.deepcopy(candidate)
    equal["anchor"]["output_token_ids"] = [7, 8]
    # Recompute the referenced source and hash so the failure is specifically
    # the missing compact/expanded token contrast.
    anchor = _source("row-long", "q1", equal["anchor"]["response"], [7, 8], 0)
    alternate = _source("row-short", "q1", equal["alternate"]["response"], [1, 2], 1)
    equal["anchor"]["row_sha256"] = _sha(anchor)
    equal["alternate"]["row_sha256"] = _sha(alternate)
    with pytest.raises(ValueError, match="no strict output-token length contrast"):
        build_reannotation_records(
            equal, source_rows={anchor["id"]: anchor, alternate["id"]: alternate}
        )
