"""Pure contracts for CLIR reasoning-equivalent rewrite verification.

This module deliberately contains no model runtime.  A generator or verifier
backend may propose text and annotations, but acceptance is derived here from a
strict, independently testable certificate.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Mapping, Sequence


VERIFIER_SCHEMA = "clir-reasoning-equivalence-verifier-v1"
ACCEPTANCE_SCHEMA = "clir-reasoning-equivalence-acceptance-v1"
RISK_PROBE_SCHEMA = "clir-reasoning-rewrite-risk-probe-v1"
CHECKER_SCHEMA = "clir-reasoning-rewrite-domain-checker-v1"

CLAIM_RELATIONS = {
    "equivalent",
    "entailed_elaboration",
    "preserved_implicitly",
    "omitted",
    "contradicted",
    "replaced_by_different_reasoning",
    "uncertain",
}
CONFIDENCE_VALUES = {"high", "medium", "low"}
MODEL_DECISIONS = {"accept", "reject", "review"}
STYLE_IDS = {"terse_formal", "explanatory_conversational"}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "source_to_rewrite",
    "rewrite_to_source",
    "global_relation",
    "error_alignment",
    "style_assessment",
    "risk_review",
    "confidence",
    "decision",
    "issues",
}
_SOURCE_DIRECTION_KEYS = {"claims", "all_essential_claims_preserved"}
_REWRITE_DIRECTION_KEYS = {"claims", "all_substantive_claims_supported"}
_SOURCE_CLAIM_KEYS = {
    "claim_id",
    "source_claim",
    "essential",
    "rewrite_evidence",
    "relation",
    "explanation",
}
_REWRITE_CLAIM_KEYS = {
    "claim_id",
    "rewrite_claim",
    "substantive",
    "source_evidence",
    "relation",
    "explanation",
}
_GLOBAL_KEYS = {
    "same_task_and_goal",
    "same_core_premises",
    "same_reasoning_method",
    "same_key_inferences",
    "same_intermediate_conclusions",
    "same_final_conclusion",
    "introduced_external_fact",
    "omitted_essential_claim",
    "contradicted_source_claim",
    "replaced_with_different_solution",
    "introduced_new_error",
}
_ERROR_KEYS = {
    "applicable",
    "source_error",
    "rewrite_error",
    "same_error_mechanism",
    "same_semantic_error_location",
    "same_downstream_effect",
}
_RISK_REVIEW_KEYS = {"risk_id", "resolved", "explanation"}
_STYLE_ASSESSMENT_KEYS = {"target_style", "satisfied", "evidence"}

_POSITIVE_GLOBAL_KEYS = {
    "same_task_and_goal",
    "same_core_premises",
    "same_reasoning_method",
    "same_key_inferences",
    "same_intermediate_conclusions",
    "same_final_conclusion",
}
_PROHIBITED_GLOBAL_KEYS = {
    "introduced_external_fact",
    "omitted_essential_claim",
    "contradicted_source_claim",
    "replaced_with_different_solution",
    "introduced_new_error",
}
_EXPLICIT_REJECTION_RELATIONS = {
    "omitted",
    "contradicted",
    "replaced_by_different_reasoning",
}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{path} keys differ: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _require_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _require_text(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{path} must be a non-empty string")
    return value


def parse_strict_json_object(text: str) -> dict[str, Any]:
    """Parse exactly one JSON object while rejecting duplicate keys."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Verifier output must be non-empty JSON text")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Verifier JSON repeats key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(text.strip(), object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("Verifier output is not one strict JSON value") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Verifier output must be one JSON object")
    return parsed


def parse_tagged_rewritten_response(text: str) -> str:
    """Extract one complete trajectory from the strict generator wrapper."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Generator output must be non-empty text")
    payload = text.strip()
    opening = "<rewritten_response>"
    closing = "</rewritten_response>"
    if payload.count(opening) != 1 or payload.count(closing) != 1:
        raise ValueError("Generator output must contain exactly one rewrite wrapper")
    if not payload.startswith(opening) or not payload.endswith(closing):
        raise ValueError("Generator output contains non-whitespace outside the wrapper")
    response = payload[len(opening) : -len(closing)]
    if response.startswith("\n"):
        response = response[1:]
    if response.endswith("\n"):
        response = response[:-1]
    if not response.strip():
        raise ValueError("Generator rewrite is empty")
    return response


_GENERATOR_SYSTEM_PROMPT = """You are rewriting an existing reasoning trajectory, not solving the problem again. Create a meaningfully different presentation and organization while preserving the same core premises, reasoning method, key inferences, intermediate conclusions, final conclusion, qualifications, and any original reasoning error. You may merge, split, or reorder explanatory steps only when logical dependencies remain unchanged. You may make an omitted reasoning bridge explicit only when it is directly entailed by the source's existing premises and method. Do not introduce a new premise, external fact, example, analogy, hypothetical scenario, different solution method, correction, or new error. If the source explicitly states a final conclusion, the rewrite must explicitly state an equivalent final conclusion; if it does not, do not add one. Return exactly one <rewritten_response>...</rewritten_response> wrapper and nothing else."""

_STYLE_INSTRUCTIONS = {
    "terse_formal": "Rewrite in a concise, formal style. You may merge closely related steps and remove purely redundant wording, but preserve every essential premise, inference, intermediate conclusion, qualification, final conclusion, and original error. Do not replace the reasoning with a shorter alternative solution.",
    "explanatory_conversational": "Rewrite in a clear, conversational, explanatory style. You may split dense steps, add transitions, and make an omitted reasoning bridge explicit when it is directly entailed by the source's existing premises and reasoning method. Do not add examples, analogies, hypothetical scenarios, new premises, external evidence, a different solution method, a correction, or a new error.",
}


def build_generator_messages(
    *, problem: str, source_trajectory: str, style_id: str
) -> list[dict[str, str]]:
    """Build the full-context v8 generator request without correctness leakage."""

    _require_text(problem, "problem")
    _require_text(source_trajectory, "source_trajectory")
    if style_id not in STYLE_IDS:
        raise ValueError(f"Unsupported generator style_id {style_id!r}")
    user = (
        f"TARGET STYLE\n{_STYLE_INSTRUCTIONS[style_id]}\n\n"
        f"PROBLEM\n<problem>\n{problem}\n</problem>\n\n"
        "SOURCE TRAJECTORY\n<source_trajectory>\n"
        f"{source_trajectory}\n</source_trajectory>"
    )
    return [
        {"role": "system", "content": _GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _validate_claims(
    value: Any,
    *,
    path: str,
    expected_keys: set[str],
    claim_text_key: str,
    evidence_key: str,
    applicability_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for index, raw in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{item_path} must be an object")
        item = dict(raw)
        _require_exact_keys(item, expected_keys, item_path)
        claim_id = _require_text(item["claim_id"], f"{item_path}.claim_id")
        if claim_id in observed_ids:
            raise ValueError(f"{path} repeats claim_id {claim_id!r}")
        observed_ids.add(claim_id)
        _require_text(item[claim_text_key], f"{item_path}.{claim_text_key}")
        _require_bool(item[applicability_key], f"{item_path}.{applicability_key}")
        _require_text(item[evidence_key], f"{item_path}.{evidence_key}", allow_empty=True)
        relation = item["relation"]
        if relation not in CLAIM_RELATIONS:
            raise ValueError(f"{item_path}.relation has an unsupported value")
        _require_text(item["explanation"], f"{item_path}.explanation")
        if relation in {"equivalent", "preserved_implicitly"} and not str(
            item[evidence_key]
        ).strip():
            raise ValueError(f"{item_path}.{evidence_key} is required for {relation}")
        normalized.append(item)
    return normalized


def validate_verifier_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact bidirectional reasoning-equivalence report schema."""

    if not isinstance(value, Mapping):
        raise ValueError("Verifier report must be an object")
    report = dict(value)
    _require_exact_keys(report, _TOP_LEVEL_KEYS, "verifier")
    if report["schema_version"] != VERIFIER_SCHEMA:
        raise ValueError("Verifier schema_version drifted")

    source_direction = report["source_to_rewrite"]
    if not isinstance(source_direction, Mapping):
        raise ValueError("verifier.source_to_rewrite must be an object")
    source_direction = dict(source_direction)
    _require_exact_keys(source_direction, _SOURCE_DIRECTION_KEYS, "source_to_rewrite")
    source_direction["claims"] = _validate_claims(
        source_direction["claims"],
        path="source_to_rewrite.claims",
        expected_keys=_SOURCE_CLAIM_KEYS,
        claim_text_key="source_claim",
        evidence_key="rewrite_evidence",
        applicability_key="essential",
    )
    _require_bool(
        source_direction["all_essential_claims_preserved"],
        "source_to_rewrite.all_essential_claims_preserved",
    )

    rewrite_direction = report["rewrite_to_source"]
    if not isinstance(rewrite_direction, Mapping):
        raise ValueError("verifier.rewrite_to_source must be an object")
    rewrite_direction = dict(rewrite_direction)
    _require_exact_keys(rewrite_direction, _REWRITE_DIRECTION_KEYS, "rewrite_to_source")
    rewrite_direction["claims"] = _validate_claims(
        rewrite_direction["claims"],
        path="rewrite_to_source.claims",
        expected_keys=_REWRITE_CLAIM_KEYS,
        claim_text_key="rewrite_claim",
        evidence_key="source_evidence",
        applicability_key="substantive",
    )
    _require_bool(
        rewrite_direction["all_substantive_claims_supported"],
        "rewrite_to_source.all_substantive_claims_supported",
    )

    global_relation = report["global_relation"]
    if not isinstance(global_relation, Mapping):
        raise ValueError("verifier.global_relation must be an object")
    global_relation = dict(global_relation)
    _require_exact_keys(global_relation, _GLOBAL_KEYS, "global_relation")
    for key, item in global_relation.items():
        _require_bool(item, f"global_relation.{key}")

    error_alignment = report["error_alignment"]
    if not isinstance(error_alignment, Mapping):
        raise ValueError("verifier.error_alignment must be an object")
    error_alignment = dict(error_alignment)
    _require_exact_keys(error_alignment, _ERROR_KEYS, "error_alignment")
    applicable = _require_bool(error_alignment["applicable"], "error_alignment.applicable")
    if applicable:
        _require_text(error_alignment["source_error"], "error_alignment.source_error")
        _require_text(error_alignment["rewrite_error"], "error_alignment.rewrite_error")
        for key in (
            "same_error_mechanism",
            "same_semantic_error_location",
            "same_downstream_effect",
        ):
            _require_bool(error_alignment[key], f"error_alignment.{key}")
    else:
        for key in _ERROR_KEYS - {"applicable"}:
            if error_alignment[key] is not None:
                raise ValueError(f"error_alignment.{key} must be null when not applicable")

    style_assessment = report["style_assessment"]
    if not isinstance(style_assessment, Mapping):
        raise ValueError("verifier.style_assessment must be an object")
    style_assessment = dict(style_assessment)
    _require_exact_keys(style_assessment, _STYLE_ASSESSMENT_KEYS, "style_assessment")
    if style_assessment["target_style"] not in STYLE_IDS:
        raise ValueError("style_assessment.target_style has an unsupported value")
    _require_bool(style_assessment["satisfied"], "style_assessment.satisfied")
    _require_text(style_assessment["evidence"], "style_assessment.evidence")

    risk_review = report["risk_review"]
    if not isinstance(risk_review, list):
        raise ValueError("verifier.risk_review must be a list")
    normalized_risks: list[dict[str, Any]] = []
    observed_risk_ids: set[str] = set()
    for index, raw in enumerate(risk_review):
        path = f"risk_review[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must be an object")
        item = dict(raw)
        _require_exact_keys(item, _RISK_REVIEW_KEYS, path)
        risk_id = _require_text(item["risk_id"], f"{path}.risk_id")
        if risk_id in observed_risk_ids:
            raise ValueError(f"risk_review repeats risk_id {risk_id!r}")
        observed_risk_ids.add(risk_id)
        _require_bool(item["resolved"], f"{path}.resolved")
        _require_text(item["explanation"], f"{path}.explanation")
        normalized_risks.append(item)

    if report["confidence"] not in CONFIDENCE_VALUES:
        raise ValueError("verifier.confidence has an unsupported value")
    if report["decision"] not in MODEL_DECISIONS:
        raise ValueError("verifier.decision has an unsupported value")
    if not isinstance(report["issues"], list) or not all(
        isinstance(item, str) and item.strip() for item in report["issues"]
    ):
        raise ValueError("verifier.issues must be a list of non-empty strings")

    return {
        **report,
        "source_to_rewrite": source_direction,
        "rewrite_to_source": rewrite_direction,
        "global_relation": global_relation,
        "error_alignment": error_alignment,
        "style_assessment": style_assessment,
        "risk_review": normalized_risks,
    }


def validate_domain_checker_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an optional task/domain checker result without assuming GSM8K."""

    if not isinstance(value, Mapping):
        raise ValueError("Domain checker result must be an object")
    result = dict(value)
    available = result.get("available")
    if type(available) is not bool:
        raise ValueError("Domain checker available must be a boolean")
    if not available:
        expected = {"schema_version", "available"}
        _require_exact_keys(result, expected, "checker")
        if result["schema_version"] != CHECKER_SCHEMA:
            raise ValueError("Domain checker schema_version drifted")
        return result

    expected = {
        "schema_version",
        "available",
        "checker_id",
        "checker_revision",
        "source_outcome",
        "rewrite_outcome",
        "same_outcome",
        "source_correctness",
        "rewrite_correctness",
    }
    _require_exact_keys(result, expected, "checker")
    if result["schema_version"] != CHECKER_SCHEMA:
        raise ValueError("Domain checker schema_version drifted")
    _require_text(result["checker_id"], "checker.checker_id")
    _require_text(result["checker_revision"], "checker.checker_revision")
    _require_text(result["source_outcome"], "checker.source_outcome")
    _require_text(result["rewrite_outcome"], "checker.rewrite_outcome")
    _require_bool(result["same_outcome"], "checker.same_outcome")
    for key in ("source_correctness", "rewrite_correctness"):
        if result[key] not in (0, 1) or type(result[key]) is not int:
            raise ValueError(f"checker.{key} must be integer 0 or 1")
    return result


def derive_acceptance_status(
    report: Mapping[str, Any],
    *,
    checker: Mapping[str, Any],
    expected_risk_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Derive accepted/rejected/review_required from validated evidence."""

    verifier = validate_verifier_report(report)
    checker_result = validate_domain_checker_result(checker)
    expected_risks = list(expected_risk_ids)
    if len(set(expected_risks)) != len(expected_risks):
        raise ValueError("expected_risk_ids must be unique")

    source_claims = verifier["source_to_rewrite"]["claims"]
    rewrite_claims = verifier["rewrite_to_source"]["claims"]
    global_relation = verifier["global_relation"]
    error = verifier["error_alignment"]
    reviewed_risk_ids = [item["risk_id"] for item in verifier["risk_review"]]

    checks = {
        "all_essential_source_claims_preserved": bool(
            verifier["source_to_rewrite"]["all_essential_claims_preserved"]
        )
        and all(
            not item["essential"]
            or item["relation"] in {"equivalent", "preserved_implicitly"}
            for item in source_claims
        ),
        "all_substantive_rewrite_claims_supported": bool(
            verifier["rewrite_to_source"]["all_substantive_claims_supported"]
        )
        and all(
            not item["substantive"]
            or item["relation"] in {"equivalent", "entailed_elaboration"}
            for item in rewrite_claims
        ),
        "positive_global_invariants": all(
            global_relation[key] for key in _POSITIVE_GLOBAL_KEYS
        ),
        "no_prohibited_global_changes": not any(
            global_relation[key] for key in _PROHIBITED_GLOBAL_KEYS
        ),
        "all_expected_risks_reviewed": set(reviewed_risk_ids) == set(expected_risks),
        "all_risks_resolved": all(item["resolved"] for item in verifier["risk_review"]),
        "high_confidence": verifier["confidence"] == "high",
        "model_accept": verifier["decision"] == "accept",
        "model_issues_empty": not verifier["issues"],
        "target_style_satisfied": verifier["style_assessment"]["satisfied"],
    }

    if checker_result["available"]:
        source_correctness = checker_result["source_correctness"]
        checks.update(
            {
                "checker_same_outcome": checker_result["same_outcome"],
                "checker_same_correctness": checker_result["rewrite_correctness"]
                == source_correctness,
                "incorrect_source_not_repaired": not (
                    source_correctness == 0
                    and checker_result["rewrite_correctness"] == 1
                ),
                "error_alignment_applicability": error["applicable"]
                == (source_correctness == 0),
                "incorrect_error_mechanism_preserved": source_correctness == 1
                or bool(
                    error["applicable"]
                    and error["same_error_mechanism"]
                    and error["same_semantic_error_location"]
                    and error["same_downstream_effect"]
                ),
            }
        )
    else:
        checks.update(
            {
                "checker_same_outcome": False,
                "checker_same_correctness": False,
                "incorrect_source_not_repaired": False,
                "error_alignment_applicability": not error["applicable"],
                "incorrect_error_mechanism_preserved": not error["applicable"],
            }
        )

    explicit_reject = (
        any(item["relation"] in _EXPLICIT_REJECTION_RELATIONS for item in source_claims)
        or any(item["relation"] in _EXPLICIT_REJECTION_RELATIONS for item in rewrite_claims)
        or any(global_relation[key] for key in _PROHIBITED_GLOBAL_KEYS)
        or verifier["decision"] == "reject"
        or not verifier["style_assessment"]["satisfied"]
        or (
            checker_result["available"]
            and (
                not checker_result["same_outcome"]
                or checker_result["source_correctness"]
                != checker_result["rewrite_correctness"]
            )
        )
        or (
            checker_result["available"]
            and checker_result["source_correctness"] == 0
            and error["applicable"]
            and any(
                error[key] is False
                for key in (
                    "same_error_mechanism",
                    "same_semantic_error_location",
                    "same_downstream_effect",
                )
            )
        )
    )

    if all(checks.values()):
        status = "accepted"
    elif explicit_reject:
        status = "rejected"
    else:
        status = "review_required"
    return {
        "schema_version": ACCEPTANCE_SCHEMA,
        "status": status,
        "checks": checks,
        "failed_checks": sorted(key for key, passed in checks.items() if not passed),
        "explicit_reject": explicit_reject,
    }


_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?\d+\s*/\s*\d+|(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*%|\s+percent)?",
    re.IGNORECASE,
)
_LEXICAL_RISK_PATTERNS = {
    "negation": re.compile(r"\b(?:not|no|never|without|neither|nor)\b", re.I),
    "quantifier": re.compile(r"\b(?:all|every|each|any|some|none|only|total)\b", re.I),
    "comparative": re.compile(
        r"\b(?:more|less|fewer|greater|smaller|at\s+least|at\s+most|before|after)\b",
        re.I,
    ),
    "modality": re.compile(r"\b(?:may|might|could|must|always|necessarily|possibly)\b", re.I),
    "causal": re.compile(r"\b(?:because|therefore|thus|hence|causes?|leads?\s+to)\b", re.I),
}


def _normalize_number(token: str) -> str:
    value = token.lower().replace(",", "").strip()
    percent = value.endswith("%") or value.endswith("percent")
    value = re.sub(r"(?:\s*%|\s+percent)$", "", value).strip()
    try:
        if "/" in value:
            numerator, denominator = (part.strip() for part in value.split("/", 1))
            normalized = f"{Decimal(numerator).normalize()}/{Decimal(denominator).normalize()}"
        else:
            normalized = str(Decimal(value).normalize())
    except InvalidOperation:
        normalized = value
    return normalized + ("%" if percent else "")


def _counter_delta(source: Counter[str], rewrite: Counter[str]) -> dict[str, int]:
    return {
        key: rewrite[key] - source[key]
        for key in sorted(set(source) | set(rewrite))
        if rewrite[key] != source[key]
    }


def build_soft_risk_probe(source: str, rewrite: str) -> dict[str, Any]:
    """Build domain-agnostic review signals; none are automatic rejections."""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Risk probe source must be non-empty text")
    if not isinstance(rewrite, str) or not rewrite.strip():
        raise ValueError("Risk probe rewrite must be non-empty text")

    risks: list[dict[str, Any]] = []
    source_numbers = Counter(_normalize_number(match.group(0)) for match in _NUMBER_PATTERN.finditer(source))
    rewrite_numbers = Counter(_normalize_number(match.group(0)) for match in _NUMBER_PATTERN.finditer(rewrite))
    number_delta = _counter_delta(source_numbers, rewrite_numbers)
    if number_delta:
        risks.append(
            {
                "risk_id": "numeric-change-0",
                "kind": "numeric_change",
                "details": {"multiset_delta": number_delta},
            }
        )

    lexical_deltas: dict[str, dict[str, int]] = {}
    for kind, pattern in _LEXICAL_RISK_PATTERNS.items():
        source_terms = Counter(match.group(0).lower() for match in pattern.finditer(source))
        rewrite_terms = Counter(match.group(0).lower() for match in pattern.finditer(rewrite))
        delta = _counter_delta(source_terms, rewrite_terms)
        if delta:
            lexical_deltas[kind] = delta
            risks.append(
                {
                    "risk_id": f"{kind}-change-0",
                    "kind": f"{kind}_change",
                    "details": {"term_delta": delta},
                }
            )

    length_ratio = len(rewrite) / len(source)
    source_lines = len(source.splitlines())
    rewrite_lines = len(rewrite.splitlines())
    return {
        "schema_version": RISK_PROBE_SCHEMA,
        "source_length": len(source),
        "rewrite_length": len(rewrite),
        "length_ratio": length_ratio,
        "source_line_count": source_lines,
        "rewrite_line_count": rewrite_lines,
        "numeric_multiset_delta": number_delta,
        "lexical_deltas": lexical_deltas,
        "risks": risks,
        "expected_risk_ids": [item["risk_id"] for item in risks],
    }


__all__ = [
    "ACCEPTANCE_SCHEMA",
    "CHECKER_SCHEMA",
    "RISK_PROBE_SCHEMA",
    "VERIFIER_SCHEMA",
    "build_soft_risk_probe",
    "build_generator_messages",
    "derive_acceptance_status",
    "parse_strict_json_object",
    "parse_tagged_rewritten_response",
    "validate_domain_checker_result",
    "validate_verifier_report",
]
