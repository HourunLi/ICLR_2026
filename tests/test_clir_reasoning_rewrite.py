import copy
import json

import pytest

from src.clir_reasoning_rewrite import (
    CHECKER_SCHEMA,
    VERIFIER_SCHEMA,
    build_generator_messages,
    build_soft_risk_probe,
    derive_acceptance_status,
    parse_strict_json_object,
    parse_tagged_rewritten_response,
    validate_verifier_report,
)


def _report(*, incorrect: bool = False) -> dict:
    return {
        "schema_version": VERIFIER_SCHEMA,
        "source_to_rewrite": {
            "claims": [
                {
                    "claim_id": "S1",
                    "source_claim": "Four groups each contain five items.",
                    "essential": True,
                    "rewrite_evidence": "Five items are in each of four groups.",
                    "relation": "equivalent",
                    "explanation": "The grouping claim is unchanged.",
                }
            ],
            "all_essential_claims_preserved": True,
        },
        "rewrite_to_source": {
            "claims": [
                {
                    "claim_id": "R1",
                    "rewrite_claim": "Five items are in each of four groups.",
                    "substantive": True,
                    "source_evidence": "Four groups each contain five items.",
                    "relation": "equivalent",
                    "explanation": "No new premise was introduced.",
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
            "applicable": incorrect,
            "source_error": "5 times 4 is evaluated as 25." if incorrect else None,
            "rewrite_error": "Four groups of 5 are evaluated as 25." if incorrect else None,
            "same_error_mechanism": True if incorrect else None,
            "same_semantic_error_location": True if incorrect else None,
            "same_downstream_effect": True if incorrect else None,
        },
        "style_assessment": {
            "target_style": "terse_formal",
            "satisfied": True,
            "evidence": "The rewrite is concise and formal.",
        },
        "risk_review": [],
        "confidence": "high",
        "decision": "accept",
        "issues": [],
    }


def _checker(*, correctness: int = 1) -> dict:
    return {
        "schema_version": CHECKER_SCHEMA,
        "available": True,
        "checker_id": "fixture-checker",
        "checker_revision": "v1",
        "source_outcome": "20" if correctness else "25",
        "rewrite_outcome": "20" if correctness else "25",
        "same_outcome": True,
        "source_correctness": correctness,
        "rewrite_correctness": correctness,
    }


def test_strict_json_parser_rejects_duplicate_keys_and_wrappers():
    assert parse_strict_json_object('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError, match="repeats key"):
        parse_strict_json_object('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="strict JSON"):
        parse_strict_json_object('```json\n{"a": 1}\n```')


def test_generator_wrapper_and_prompt_contract_are_strict():
    assert (
        parse_tagged_rewritten_response(
            "<rewritten_response>\nA reorganized trajectory.\n</rewritten_response>"
        )
        == "A reorganized trajectory."
    )
    with pytest.raises(ValueError, match="outside"):
        parse_tagged_rewritten_response(
            "Here it is:\n<rewritten_response>x</rewritten_response>"
        )
    with pytest.raises(ValueError, match="exactly one"):
        parse_tagged_rewritten_response(
            "<rewritten_response>x</rewritten_response>"
            "<rewritten_response>y</rewritten_response>"
        )

    messages = build_generator_messages(
        problem="What is the result?",
        source_trajectory="The source may be wrong.",
        style_id="explanatory_conversational",
    )
    serialized = json.dumps(messages)
    assert "reference answer" not in serialized.lower()
    assert "correctness label" not in serialized.lower()
    assert "examples, analogies" in serialized
    with pytest.raises(ValueError, match="style_id"):
        build_generator_messages(problem="p", source_trajectory="s", style_id="unknown")


def test_verifier_schema_rejects_unknown_fields_and_missing_evidence():
    report = _report()
    assert validate_verifier_report(report)["schema_version"] == VERIFIER_SCHEMA

    unknown = copy.deepcopy(report)
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="unknown"):
        validate_verifier_report(unknown)

    missing_evidence = copy.deepcopy(report)
    missing_evidence["source_to_rewrite"]["claims"][0]["rewrite_evidence"] = ""
    with pytest.raises(ValueError, match="rewrite_evidence is required"):
        validate_verifier_report(missing_evidence)


def test_program_accepts_only_complete_high_confidence_certificate():
    accepted = derive_acceptance_status(_report(), checker=_checker())
    assert accepted["status"] == "accepted"
    assert accepted["failed_checks"] == []

    uncertain = _report()
    uncertain["confidence"] = "medium"
    uncertain["decision"] = "review"
    reviewed = derive_acceptance_status(uncertain, checker=_checker())
    assert reviewed["status"] == "review_required"
    assert "high_confidence" in reviewed["failed_checks"]


def test_explicit_omission_and_checker_drift_are_rejected():
    omitted = _report()
    claim = omitted["source_to_rewrite"]["claims"][0]
    claim["relation"] = "omitted"
    claim["rewrite_evidence"] = ""
    omitted["source_to_rewrite"]["all_essential_claims_preserved"] = False
    omitted["global_relation"]["omitted_essential_claim"] = True
    omitted["decision"] = "reject"
    result = derive_acceptance_status(omitted, checker=_checker())
    assert result["status"] == "rejected"
    assert result["explicit_reject"] is True

    drifted_checker = _checker()
    drifted_checker["same_outcome"] = False
    drifted_checker["rewrite_outcome"] = "21"
    result = derive_acceptance_status(_report(), checker=drifted_checker)
    assert result["status"] == "rejected"


def test_entailed_elaboration_is_allowed_but_style_failure_rejects():
    report = _report()
    claim = report["rewrite_to_source"]["claims"][0]
    claim["relation"] = "entailed_elaboration"
    claim["explanation"] = "The multiplication bridge was implicit in the source."
    accepted = derive_acceptance_status(report, checker=_checker())
    assert accepted["status"] == "accepted"

    report["style_assessment"]["satisfied"] = False
    report["style_assessment"]["evidence"] = "The wording is effectively unchanged."
    report["decision"] = "reject"
    rejected = derive_acceptance_status(report, checker=_checker())
    assert rejected["status"] == "rejected"
    assert "target_style_satisfied" in rejected["failed_checks"]


def test_incorrect_source_requires_matching_error_alignment():
    accepted = derive_acceptance_status(_report(incorrect=True), checker=_checker(correctness=0))
    assert accepted["status"] == "accepted"

    changed_error = _report(incorrect=True)
    changed_error["error_alignment"]["same_error_mechanism"] = False
    changed_error["decision"] = "reject"
    rejected = derive_acceptance_status(changed_error, checker=_checker(correctness=0))
    assert rejected["status"] == "rejected"
    assert "incorrect_error_mechanism_preserved" in rejected["failed_checks"]


def test_checker_unavailable_routes_to_review_instead_of_auto_accept():
    checker = {"schema_version": CHECKER_SCHEMA, "available": False}
    result = derive_acceptance_status(_report(), checker=checker)
    assert result["status"] == "review_required"
    assert result["explicit_reject"] is False


def test_all_detected_soft_risks_must_be_reviewed_and_resolved():
    probe = build_soft_risk_probe(
        "It is not enough: 2/3 of the tank is full.",
        "Two thirds of the tank is already sufficient.",
    )
    assert "numeric-change-0" in probe["expected_risk_ids"]
    assert "negation-change-0" in probe["expected_risk_ids"]

    report = _report()
    missing = derive_acceptance_status(
        report,
        checker=_checker(),
        expected_risk_ids=probe["expected_risk_ids"],
    )
    assert missing["status"] == "review_required"

    report["risk_review"] = [
        {"risk_id": risk_id, "resolved": True, "explanation": "Manually aligned."}
        for risk_id in probe["expected_risk_ids"]
    ]
    accepted = derive_acceptance_status(
        report,
        checker=_checker(),
        expected_risk_ids=probe["expected_risk_ids"],
    )
    assert accepted["status"] == "accepted"


def test_risk_probe_is_diagnostic_and_stable():
    probe = build_soft_risk_probe("Each has 1,200 units.", "All have 1200 units.")
    assert probe["numeric_multiset_delta"] == {}
    assert probe["lexical_deltas"]["quantifier"] == {"all": 1, "each": -1}
    assert json.loads(json.dumps(probe))["schema_version"] == probe["schema_version"]
