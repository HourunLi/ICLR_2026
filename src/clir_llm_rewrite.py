"""Fidelity-gated local-LLM semantic rewrites for CLIR.

This module contains the deterministic, testable part of the real rewrite
pilot.  Model loading and atomic shard orchestration live in the launcher.
Generator/verifier outputs are never trusted by themselves: exact-token,
answer, numeric-literal, lineage, and supervision gates remain mandatory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Callable, Dict, Mapping, Sequence

from .clir_real_data import (
    canonical_json_sha256,
    check_gsm8k_response,
    validate_rollout_row,
)
from .clir_rewrite import encode_exact_response, text_sha256
from .clir_supervision import (
    PROVENANCE_SCHEMA,
    ROW_PROVENANCE_SCHEMA,
    audit_supervision_coverage,
    output_token_ids_sha256,
)


LLM_REWRITE_SCHEMA = "clir-semantic-rewrite-llm-v1"
SUPPORTED_LLM_REWRITE_SCHEMAS = {
    LLM_REWRITE_SCHEMA,
    "clir-semantic-rewrite-llm-v2",
    "clir-semantic-rewrite-llm-v3",
    "clir-semantic-rewrite-llm-v4",
    "clir-semantic-rewrite-llm-v5",
    "clir-semantic-rewrite-llm-v6",
    "clir-semantic-rewrite-llm-v7",
}
LLM_VIEW_RESULT_SCHEMA = "clir-semantic-rewrite-llm-view-result-v1"
LLM_LINEAGE_SCHEMA = "clir-semantic-rewrite-llm-lineage-v1"

_PLACEHOLDERS = {
    "style": "<<STYLE_INSTRUCTION>>",
    "problem": "<<PROBLEM>>",
    "source": "<<SOURCE_RESPONSE>>",
    "rewrite": "<<REWRITTEN_RESPONSE>>",
    "annotated_source": "<<ANNOTATED_SOURCE_LINES>>",
    "expected_editable_ids": "<<EXPECTED_EDITABLE_IDS>>",
}
_NUMERIC_LITERAL_V1 = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)(?:\s*%|\s+percent)?",
    flags=re.IGNORECASE,
)
_NUMERIC_LITERAL_V2 = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)(?:%|\s+percent)?",
    flags=re.IGNORECASE,
)


def validate_llm_rewrite_protocol(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a frozen four-source local-LLM rewrite gate contract."""

    normalized = dict(protocol)
    if normalized.get("schema_version") not in SUPPORTED_LLM_REWRITE_SCHEMAS:
        raise ValueError(
            "LLM rewrite protocol must use one of "
            f"{sorted(SUPPORTED_LLM_REWRITE_SCHEMAS)}"
        )
    if normalized.get("formal_mechanism_claim_allowed") is not False:
        raise ValueError("The LLM rewrite pilot must forbid formal mechanism claims")

    source = normalized.get("source")
    if not isinstance(source, Mapping) or source.get("split") != "train":
        raise ValueError("LLM rewrite source must be an explicit train manifest")
    source_manifest = str(source.get("manifest", ""))
    if not source_manifest or "pilot_test" in source_manifest.split("/"):
        raise ValueError("LLM rewrite may not read pilot_test")
    source_sha = source.get("sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("LLM rewrite source requires a pinned SHA256")
    selected_ids = source.get("selected_source_ids")
    if (
        not isinstance(selected_ids, list)
        or len(selected_ids) != 4
        or len(set(selected_ids)) != 4
        or not all(isinstance(value, str) and value for value in selected_ids)
    ):
        raise ValueError("The v1 LLM gate requires four unique pinned source ids")
    if source.get("required_correctness_counts") != {"0": 2, "1": 2}:
        raise ValueError("The v1 LLM gate requires two correct and two incorrect sources")
    if source.get("require_distinct_query_ids") is not True:
        raise ValueError("The v1 LLM gate requires distinct source queries")

    identity = normalized.get("identity")
    expected_identity = {
        "query_id": "preserve_source_query_id",
        "semantic_id": "derive_from_source_trajectory_id",
        "style_id": "declared_rewrite_axis",
        "source_candidate_index": "lineage_only_not_top_level",
    }
    if not isinstance(identity, Mapping) or dict(identity) != expected_identity:
        raise ValueError("LLM rewrite identity policy drifted")

    for role in ("generator", "verifier"):
        model = normalized.get(role)
        if not isinstance(model, Mapping):
            raise ValueError(f"LLM rewrite protocol requires {role}")
        for key in (
            "repo_id",
            "revision",
            "tokenizer_revision",
            "torch_dtype",
            "deployment",
            "system_prompt",
            "user_prompt_template",
        ):
            if key not in model:
                raise ValueError(f"LLM rewrite {role} is missing {key}")
        if model["deployment"] != "local" or model["torch_dtype"] != "bfloat16":
            raise ValueError(f"LLM rewrite {role} must be local BF16")
        output_format = model.get("output_format", "json_object_v1")
        if role == "generator" and output_format not in {
            "json_object_v1",
            "single_rewritten_response_tag_v1",
            "indexed_editable_lines_v1",
        }:
            raise ValueError("Unsupported generator output format")
        if role == "verifier" and output_format != "json_object_v1":
            raise ValueError("Verifier output must remain strict JSON")
        if output_format == "json_object_v1" and (
            not isinstance(model.get("output_keys"), list) or not model["output_keys"]
        ):
            raise ValueError(f"LLM rewrite {role} output keys must be non-empty")
    generator = normalized["generator"]
    verifier = normalized["verifier"]
    for key in ("repo_id", "revision", "tokenizer_revision"):
        if verifier[key] != generator[key]:
            raise ValueError("The approved pilot requires the same generator/verifier checkpoint")
    if verifier.get("independent_from_generator") is not False:
        raise ValueError("The v1 verifier must disclose that it is not independent")
    if verifier.get("reference_answer_visible") is not False:
        raise ValueError("Verifier must not see the reference answer")
    if verifier.get("source_correctness_label_visible") is not False:
        raise ValueError("Verifier must not see the source correctness label")

    styles = normalized.get("styles")
    if not isinstance(styles, list) or len(styles) != 2:
        raise ValueError("The v1 LLM gate requires exactly two rewrite styles")
    style_ids = [style.get("style_id") for style in styles if isinstance(style, Mapping)]
    if len(style_ids) != 2 or len(set(style_ids)) != 2 or not all(style_ids):
        raise ValueError("LLM rewrite style ids must be unique and non-empty")
    if not all(isinstance(style.get("instruction"), str) and style["instruction"] for style in styles):
        raise ValueError("Every LLM rewrite style requires a frozen instruction")

    decoding = normalized.get("decoding")
    if not isinstance(decoding, Mapping):
        raise ValueError("LLM rewrite protocol requires decoding parameters")
    if decoding.get("do_sample") is not False or decoding.get("num_beams") != 1:
        raise ValueError("The v1 gate requires deterministic greedy decoding")
    if normalized["schema_version"].endswith(("v2", "v3", "v4", "v5", "v6", "v7")) and any(
        decoding.get(key) is not None for key in ("temperature", "top_p", "top_k")
    ):
        raise ValueError("The v2+ greedy protocol must explicitly disable sampling controls")
    for key in ("generator_max_new_tokens", "verifier_max_new_tokens", "max_input_tokens"):
        if isinstance(decoding.get(key), bool) or int(decoding.get(key, 0)) <= 0:
            raise ValueError(f"LLM rewrite decoding {key} must be positive")

    gates = normalized.get("deterministic_gates")
    required_true = {
        "response_must_change",
        "same_checker_correctness",
        "same_normalized_candidate_answer",
        "numeric_literal_set_equal",
        "incorrect_source_must_remain_incorrect",
        "verifier_accept_required",
    }
    if not isinstance(gates, Mapping) or any(gates.get(key) is not True for key in required_true):
        raise ValueError("LLM rewrite deterministic gates are incomplete")
    minimum_ratio = float(gates.get("minimum_length_ratio", 0.0))
    maximum_ratio = float(gates.get("maximum_length_ratio", 0.0))
    if not 0.0 < minimum_ratio <= 1.0 <= maximum_ratio:
        raise ValueError("LLM rewrite length-ratio gate is invalid")
    if normalized["schema_version"].endswith(("v3", "v4", "v5", "v6", "v7")):
        v3_required_true = {
            "numeric_literal_sequence_equal",
            "protected_math_lines_exact",
            "same_nonempty_line_count",
            "style_instruction_not_copied",
        }
        if any(gates.get(key) is not True for key in v3_required_true):
            raise ValueError("The v3 deterministic rewrite gates are incomplete")
        editable_ratio = float(gates.get("minimum_editable_line_change_ratio", 0.0))
        if not 0.0 < editable_ratio <= 1.0:
            raise ValueError("The v3 editable-line change ratio is invalid")

    quality = normalized.get("quality_gate")
    if not isinstance(quality, Mapping):
        raise ValueError("LLM rewrite protocol requires a quality gate")
    if quality.get("planned_views") != 8 or quality.get("required_accepted_views") != 8:
        raise ValueError("The v1 quality gate must require all 8 planned views")
    if quality.get("maximum_incorrect_source_repairs") != 0:
        raise ValueError("The v1 quality gate must forbid incorrect-source repair")
    if quality.get("human_blind_audit_required_before_scale") is not True:
        raise ValueError("Scaling requires a human blind audit")
    if normalized["schema_version"].endswith(("v3", "v4", "v5", "v6", "v7")) and quality.get(
        "require_distinct_rewrites_per_semantic_group"
    ) is not True:
        raise ValueError("The v3 gate requires distinct style rewrites per source")
    if normalized["schema_version"].endswith(("v4", "v5", "v6", "v7")):
        execution = normalized.get("execution")
        runtime = execution.get("required_runtime") if isinstance(execution, Mapping) else None
        runtime_keys = {
            "python_executable",
            "python_version",
            "torch_version",
            "transformers_version",
            "cuda_runtime_version",
        }
        if (
            not isinstance(runtime, Mapping)
            or set(runtime) != runtime_keys
            or not all(isinstance(runtime[key], str) and runtime[key] for key in runtime_keys)
        ):
            raise ValueError("The v4 rewrite protocol requires a complete pinned runtime")
    if normalized["schema_version"].endswith(("v5", "v6", "v7")):
        if normalized["generator"].get("output_format") != "indexed_editable_lines_v1":
            raise ValueError("The v5+ generator must use constrained editable-line output")
        template = str(normalized["generator"].get("user_prompt_template", ""))
        for placeholder in (
            _PLACEHOLDERS["annotated_source"],
            _PLACEHOLDERS["expected_editable_ids"],
        ):
            if template.count(placeholder) != 1:
                raise ValueError(f"The v5+ generator template requires exactly one {placeholder}")
    if normalized["schema_version"].endswith("v7"):
        diagnostic = normalized.get("diagnostic_scope")
        if not isinstance(diagnostic, Mapping):
            raise ValueError("The v7 model-scale run must declare its diagnostic scope")
        if diagnostic.get("source_gate_previously_exposed") is not True:
            raise ValueError("The v7 source gate must be disclosed as previously exposed")
        if diagnostic.get("blind_gate") is not False:
            raise ValueError("The v7 model-scale diagnostic may not be called a blind gate")
        if diagnostic.get("may_authorize_scale") is not False:
            raise ValueError("The v7 exposed diagnostic may not authorize scaling")
    return normalized


def select_configured_sources(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    """Resolve the four pinned rows and reproduce their source labels."""

    protocol = validate_llm_rewrite_protocol(protocol)
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    for row_index, value in enumerate(rows):
        row = dict(value)
        row_id = str(row.get("id", ""))
        if not row_id or row_id in rows_by_id:
            raise ValueError(f"Source manifest has an invalid id at row {row_index}")
        rows_by_id[row_id] = row

    selected: list[Dict[str, Any]] = []
    counts: Counter[int] = Counter()
    query_ids: set[str] = set()
    tokenization = protocol["tokenization"]
    for source_id in protocol["source"]["selected_source_ids"]:
        if source_id not in rows_by_id:
            raise ValueError(f"Pinned rewrite source is absent: {source_id}")
        row = rows_by_id[source_id]
        validate_rollout_row(row)
        if row.get("split") != protocol["source"]["split"]:
            raise ValueError(f"Rewrite source {source_id!r} is outside train")
        for field in ("question", "response", "raw_reference", "normalized_candidate_answer"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Rewrite source {source_id!r} lacks {field}")
        correctness = row.get("correctness")
        if isinstance(correctness, bool) or correctness not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Rewrite source {source_id!r} has invalid correctness")
        correctness = int(correctness)
        counts[correctness] += 1
        query_id = str(row["query_id"])
        if query_id in query_ids:
            raise ValueError("Pinned rewrite sources cross the distinct-query gate")
        query_ids.add(query_id)

        provenance = row["provenance"]
        expected_feature_identity = {
            "model_id": tokenization["feature_model_id"],
            "model_revision": tokenization["feature_model_revision"],
            "tokenizer_revision": tokenization["tokenizer_revision"],
        }
        for key, expected in expected_feature_identity.items():
            if provenance.get(key) != expected:
                raise ValueError(
                    f"Rewrite source {source_id!r} {key} differs from the feature model"
                )
        reproduced = check_gsm8k_response(
            row["response"],
            row["raw_reference"],
            checker_version=protocol["source"]["checker_version"],
        )
        if int(reproduced["correctness"]) != correctness:
            raise ValueError(f"Rewrite source label no longer reproduces for {source_id!r}")
        if reproduced.get("normalized_candidate_answer") != row["normalized_candidate_answer"]:
            raise ValueError(f"Rewrite source parsed answer drifted for {source_id!r}")
        selected.append(row)
    if {str(key): value for key, value in sorted(counts.items())} != protocol["source"][
        "required_correctness_counts"
    ]:
        raise ValueError("Pinned rewrite source correctness balance drifted")
    return selected


def render_prompt(template: str, **values: str) -> str:
    """Render only declared sentinels and reject unresolved prompt fields."""

    rendered = template
    for key, value in values.items():
        if key not in _PLACEHOLDERS:
            raise ValueError(f"Unknown rewrite prompt value: {key}")
        rendered = rendered.replace(_PLACEHOLDERS[key], value)
    unresolved = [placeholder for placeholder in _PLACEHOLDERS.values() if placeholder in rendered]
    if unresolved:
        raise ValueError(f"Rewrite prompt has unresolved placeholders: {unresolved}")
    return rendered


def parse_json_object(text: str, expected_keys: Sequence[str]) -> Dict[str, Any]:
    """Parse exactly one object, optionally enclosed in a single JSON fence."""

    payload = text.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) < 3 or lines[0] not in {"```", "```json", "```JSON"} or lines[-1] != "```":
            raise ValueError("Model output has a malformed JSON fence")
        payload = "\n".join(lines[1:-1]).strip()
        if "```" in payload:
            raise ValueError("Model output contains nested JSON fences")

    def unique_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Model JSON repeats key {key!r}")
            output[key] = value
        return output

    try:
        parsed = json.loads(payload, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not one valid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")
    expected = set(expected_keys)
    if set(parsed) != expected:
        raise ValueError(
            f"Model JSON keys differ: expected {sorted(expected)}, got {sorted(parsed)}"
        )
    return parsed


def parse_tagged_rewrite(text: str) -> str:
    """Parse one multiline rewrite without JSON string escaping ambiguity."""

    payload = text.strip()
    opening = "<rewritten_response>"
    closing = "</rewritten_response>"
    if payload.count(opening) != 1 or payload.count(closing) != 1:
        raise ValueError("Generator output must contain exactly one rewritten_response tag")
    if not payload.startswith(opening) or not payload.endswith(closing):
        raise ValueError("Generator output contains text outside rewritten_response tags")
    response = payload[len(opening) : -len(closing)]
    if response.startswith("\n"):
        response = response[1:]
    if response.endswith("\n"):
        response = response[:-1]
    if not response.strip():
        raise ValueError("Generator tagged rewrite is empty")
    return response


def parse_indexed_line_rewrites(
    text: str,
    expected_ids: Sequence[str],
) -> Dict[str, str]:
    """Parse one exact single-line replacement for every editable source id."""

    expected = list(expected_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Indexed rewrite requires unique expected editable ids")
    if not all(re.fullmatch(r"E\d{4}", line_id) for line_id in expected):
        raise ValueError("Indexed rewrite ids must use Edddd format")
    payload = text.strip()
    lines = payload.splitlines()
    if len(lines) != len(expected):
        raise ValueError(
            "Generator must return exactly one physical line per editable id: "
            f"expected {len(expected)}, got {len(lines)}"
        )
    pattern = re.compile(r"<(?P<id>E\d{4})>(?P<text>.*)</(?P=id)>")
    parsed: Dict[str, str] = {}
    observed_ids: list[str] = []
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError("Generator indexed line has invalid tag syntax")
        line_id = match.group("id")
        replacement = match.group("text")
        if not replacement.strip():
            raise ValueError(f"Generator returned an empty replacement for {line_id}")
        if line_id in parsed:
            raise ValueError(f"Generator repeated editable id {line_id}")
        parsed[line_id] = replacement
        observed_ids.append(line_id)
    if observed_ids != expected:
        raise ValueError(
            f"Generator editable ids drifted: expected {expected}, got {observed_ids}"
        )
    return parsed


def _normalize_decimal(value: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric literal {value!r}") from exc
    if number == number.to_integral():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def numeric_literal_sequence(
    text: str,
    parser_version: str = "clir_numeric_literal_set_v1",
) -> list[str]:
    """Extract canonical numeric/fraction/percent literals in occurrence order."""

    if parser_version == "clir_numeric_literal_set_v1":
        prepared = text
        pattern = _NUMERIC_LITERAL_V1
    elif parser_version == "clir_numeric_literal_set_v2":
        # Step/list ordinals are layout, not mathematical claims.  Removing
        # only line-leading labels keeps quantities such as "1 whole" intact.
        prepared = re.sub(
            r"(?im)^\s*step\s+\d+\s*[:.)-]?\s*",
            "",
            text,
        )
        prepared = re.sub(r"(?m)^\s*\d+\s*[.)]\s+", "", prepared)
        pattern = _NUMERIC_LITERAL_V2
    else:
        raise ValueError(f"Unsupported numeric literal parser: {parser_version}")
    literals: list[str] = []
    for match in pattern.finditer(prepared):
        token = match.group(0).lower().replace(",", "").strip()
        is_percent = token.endswith("%") or token.endswith("percent")
        token = re.sub(r"(?:\s*%|\s+percent)$", "", token).strip()
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            normalized = f"{_normalize_decimal(numerator)}/{_normalize_decimal(denominator)}"
        else:
            normalized = _normalize_decimal(token)
        literals.append(normalized + ("%" if is_percent else ""))
    return literals


def numeric_literal_set(
    text: str,
    parser_version: str = "clir_numeric_literal_set_v1",
) -> set[str]:
    """Extract the unique canonical numeric literals in ``text``."""

    return set(numeric_literal_sequence(text, parser_version))


def _is_protected_math_line(line: str) -> bool:
    normalized = line.strip()
    return bool(
        normalized
        and (
            "=" in normalized
            or "\\boxed" in normalized
            or re.search(r"(?i)\bfinal\s+answer\b", normalized)
        )
    )


def build_indexed_line_plan(response: str) -> Dict[str, Any]:
    """Mark protected/editable lines while preserving the exact source layout."""

    if not isinstance(response, str) or not response.strip():
        raise ValueError("Indexed rewrite source must be non-empty text")
    source_lines = response.split("\n")
    if len(source_lines) > 10_000:
        raise ValueError("Indexed rewrite supports at most 10,000 physical lines")
    entries: list[Dict[str, Any]] = []
    annotated_lines: list[str] = []
    editable_ids: list[str] = []
    protected_ids: list[str] = []
    for index, source_line in enumerate(source_lines):
        if not source_line.strip():
            entries.append(
                {
                    "physical_line_index": index,
                    "kind": "blank",
                    "line_id": None,
                    "source_text": source_line,
                }
            )
            annotated_lines.append("")
            continue
        protected = _is_protected_math_line(source_line)
        line_id = f"{'P' if protected else 'E'}{index:04d}"
        entries.append(
            {
                "physical_line_index": index,
                "kind": "protected" if protected else "editable",
                "line_id": line_id,
                "source_text": source_line,
            }
        )
        annotated_lines.append(f"[{line_id}] {source_line}")
        if protected:
            protected_ids.append(line_id)
        else:
            editable_ids.append(line_id)
    if not editable_ids:
        raise ValueError("Indexed rewrite source has no editable lines")
    return {
        "schema_version": "clir-indexed-editable-line-plan-v1",
        "source_response_sha256": text_sha256(response),
        "physical_line_count": len(source_lines),
        "entries": entries,
        "editable_ids": editable_ids,
        "protected_ids": protected_ids,
        "annotated_source": "\n".join(annotated_lines),
    }


def assemble_indexed_line_rewrite(
    plan: Mapping[str, Any],
    replacements: Mapping[str, str],
) -> str:
    """Rebuild a response by replacing editable lines and copying all others."""

    if plan.get("schema_version") != "clir-indexed-editable-line-plan-v1":
        raise ValueError("Indexed rewrite line-plan schema drifted")
    expected_ids = list(plan.get("editable_ids", []))
    if list(replacements) != expected_ids:
        raise ValueError("Indexed rewrite replacement ids/order differ from the line plan")
    output_lines: list[str] = []
    for entry in plan.get("entries", []):
        if not isinstance(entry, Mapping):
            raise ValueError("Indexed rewrite line plan contains an invalid entry")
        kind = entry.get("kind")
        source_text = entry.get("source_text")
        if not isinstance(source_text, str):
            raise ValueError("Indexed rewrite line plan contains non-text source content")
        if kind == "editable":
            line_id = entry.get("line_id")
            replacement = replacements.get(line_id)
            if not isinstance(replacement, str) or not replacement.strip():
                raise ValueError(f"Indexed rewrite lacks replacement for {line_id}")
            if "\n" in replacement or "\r" in replacement:
                raise ValueError(f"Indexed rewrite replacement {line_id} is not one line")
            output_lines.append(replacement)
        elif kind in {"protected", "blank"}:
            output_lines.append(source_text)
        else:
            raise ValueError("Indexed rewrite line plan contains an unknown line kind")
    assembled = "\n".join(output_lines)
    if len(output_lines) != int(plan.get("physical_line_count", -1)):
        raise ValueError("Indexed rewrite assembly changed the physical line count")
    return assembled


def _linewise_rewrite_metrics(source: str, rewrite: str) -> Dict[str, Any]:
    source_lines = [line.strip() for line in source.splitlines() if line.strip()]
    rewrite_lines = [line.strip() for line in rewrite.splitlines() if line.strip()]
    source_protected = [line for line in source_lines if _is_protected_math_line(line)]
    rewrite_protected = [line for line in rewrite_lines if _is_protected_math_line(line)]
    editable_indices = [
        index for index, line in enumerate(source_lines) if not _is_protected_math_line(line)
    ]
    changed_editable = 0
    if len(source_lines) == len(rewrite_lines):
        changed_editable = sum(
            source_lines[index] != rewrite_lines[index] for index in editable_indices
        )
    editable_ratio = changed_editable / len(editable_indices) if editable_indices else 0.0
    return {
        "source_nonempty_line_count": len(source_lines),
        "rewrite_nonempty_line_count": len(rewrite_lines),
        "source_protected_math_lines": source_protected,
        "rewrite_protected_math_lines": rewrite_protected,
        "editable_line_count": len(editable_indices),
        "changed_editable_line_count": changed_editable,
        "editable_line_change_ratio": editable_ratio,
    }


def deterministic_rewrite_gates(
    source: Mapping[str, Any],
    rewritten_response: str,
    protocol: Mapping[str, Any],
    *,
    style_instruction: str | None = None,
) -> Dict[str, Any]:
    """Apply deterministic gates without exposing the reference to either LLM call."""

    protocol = validate_llm_rewrite_protocol(protocol)
    source_response = str(source["response"])
    if not isinstance(rewritten_response, str) or not rewritten_response.strip():
        raise ValueError("Generator returned an empty rewritten_response")
    checker_version = protocol["source"]["checker_version"]
    source_check = check_gsm8k_response(
        source_response,
        str(source["raw_reference"]),
        checker_version=checker_version,
    )
    rewrite_check = check_gsm8k_response(
        rewritten_response,
        str(source["raw_reference"]),
        checker_version=checker_version,
    )
    parser_version = str(protocol["deterministic_gates"]["numeric_literal_parser"])
    source_number_sequence = numeric_literal_sequence(source_response, parser_version)
    rewrite_number_sequence = numeric_literal_sequence(rewritten_response, parser_version)
    source_numbers = numeric_literal_set(source_response, parser_version)
    rewrite_numbers = numeric_literal_set(rewritten_response, parser_version)
    length_ratio = len(rewritten_response) / max(len(source_response), 1)
    checks = {
        "response_changed": rewritten_response.strip() != source_response.strip(),
        "same_checker_correctness": int(rewrite_check["correctness"])
        == int(source_check["correctness"]),
        "same_normalized_candidate_answer": rewrite_check.get(
            "normalized_candidate_answer"
        )
        == source_check.get("normalized_candidate_answer"),
        "numeric_literal_set_equal": rewrite_numbers == source_numbers,
        "minimum_length_ratio": length_ratio
        >= float(protocol["deterministic_gates"]["minimum_length_ratio"]),
        "maximum_length_ratio": length_ratio
        <= float(protocol["deterministic_gates"]["maximum_length_ratio"]),
        "incorrect_source_not_repaired": not (
            int(source_check["correctness"]) == 0
            and int(rewrite_check["correctness"]) == 1
        ),
    }
    linewise: Dict[str, Any] | None = None
    if protocol["schema_version"].endswith(("v3", "v4", "v5", "v6", "v7")):
        if not isinstance(style_instruction, str) or not style_instruction:
            raise ValueError("The v3 rewrite gate requires the active style instruction")
        linewise = _linewise_rewrite_metrics(source_response, rewritten_response)
        checks.update(
            {
                "numeric_literal_sequence_equal": rewrite_number_sequence
                == source_number_sequence,
                "protected_math_lines_exact": linewise["rewrite_protected_math_lines"]
                == linewise["source_protected_math_lines"],
                "same_nonempty_line_count": linewise["rewrite_nonempty_line_count"]
                == linewise["source_nonempty_line_count"],
                "minimum_editable_line_change_ratio": linewise[
                    "editable_line_change_ratio"
                ]
                >= float(
                    protocol["deterministic_gates"][
                        "minimum_editable_line_change_ratio"
                    ]
                ),
                "style_instruction_not_copied": style_instruction.strip().lower()
                not in rewritten_response.lower(),
            }
        )
    return {
        "schema_version": "clir-semantic-rewrite-deterministic-gates-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "source_numeric_literals": sorted(source_numbers),
        "rewrite_numeric_literals": sorted(rewrite_numbers),
        "source_numeric_literal_sequence": source_number_sequence,
        "rewrite_numeric_literal_sequence": rewrite_number_sequence,
        "length_ratio": length_ratio,
        "linewise_metrics": linewise,
        "source_correctness_check": source_check,
        "rewrite_correctness_check": rewrite_check,
    }


def validate_verifier_decision(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the exact verifier JSON and derive its conservative acceptance."""

    parsed = dict(value)
    boolean_keys = (
        "reasoning_meaning_preserved",
        "final_answer_preserved",
        "numbers_operations_preserved",
        "style_changed",
        "introduced_correction",
        "added_or_removed_claims",
    )
    if any(type(parsed.get(key)) is not bool for key in boolean_keys):
        raise ValueError("Verifier decision fields must be JSON booleans")
    if parsed.get("decision") not in {"accept", "reject"}:
        raise ValueError("Verifier decision must be accept or reject")
    issues = parsed.get("issues")
    if not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
        raise ValueError("Verifier issues must be a string list")
    accepted = (
        parsed["decision"] == "accept"
        and parsed["reasoning_meaning_preserved"]
        and parsed["final_answer_preserved"]
        and parsed["numbers_operations_preserved"]
        and parsed["style_changed"]
        and not parsed["introduced_correction"]
        and not parsed["added_or_removed_claims"]
    )
    return {**parsed, "derived_accept": accepted}


def _completion_text(record: Mapping[str, Any], role: str) -> str:
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{role} backend returned no text")
    return text


def _build_view_row(
    source: Mapping[str, Any],
    *,
    style_id: str,
    rewritten_response: str,
    output_token_ids: Sequence[int],
    gates: Mapping[str, Any],
    verifier: Mapping[str, Any],
    generator_messages: Sequence[Mapping[str, str]],
    generator_constraint: Mapping[str, Any] | None,
    verifier_messages: Sequence[Mapping[str, str]],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    code: Mapping[str, Any],
) -> Dict[str, Any]:
    source_id = str(source["id"])
    protocol_suffix = str(protocol["schema_version"]).rsplit("-", 1)[-1]
    row = dict(source)
    for field in (
        "candidate_index",
        "generation",
        "backend_response_text",
        "decode_matches_backend_text",
        "hidden_states_path",
        "condition_states_path",
        "feature_sha256",
        "condition_sha256",
        "feature_metadata",
        "semantic_id",
        "style_id",
        "domain_id",
        "clir_supervision_provenance",
    ):
        row.pop(field, None)
    rewrite_check = dict(gates["rewrite_correctness_check"])
    row.update(
        {
            "id": f"{source_id}--semrw-llm-{protocol_suffix}--{style_id}",
            "query_id": str(source["query_id"]),
            "semantic_id": f"semrw-llm-{protocol_suffix}:{source_id}",
            "style_id": style_id,
            "response": rewritten_response,
            "trajectory": rewritten_response,
            "output_token_ids": list(output_token_ids),
            **rewrite_check,
        }
    )
    source_provenance = source["provenance"]
    tokenization = protocol["tokenization"]
    row["provenance"] = {
        "model_id": tokenization["feature_model_id"],
        "model_revision": tokenization["feature_model_revision"],
        "tokenizer_revision": tokenization["tokenizer_revision"],
        "protocol_version": protocol["schema_version"],
        "rewrite_protocol_sha256": protocol_sha256,
        "source_provenance_sha256": canonical_json_sha256(source_provenance),
        "trajectory_acquisition": "online_teacher_forced_exact_rewrite_token_ids",
        "code": dict(code),
    }
    certificate = {
        "source_id": source_id,
        "style_id": style_id,
        "source_response_sha256": text_sha256(str(source["response"])),
        "rewrite_response_sha256": text_sha256(rewritten_response),
        "generator_messages_sha256": canonical_json_sha256(generator_messages),
        "generator_constraint": (
            dict(generator_constraint) if generator_constraint is not None else None
        ),
        "verifier_messages_sha256": canonical_json_sha256(verifier_messages),
        "deterministic_gates": dict(gates),
        "verifier_decision": dict(verifier),
    }
    certificate_sha256 = canonical_json_sha256(certificate)
    row["rewrite_lineage"] = {
        "schema_version": LLM_LINEAGE_SCHEMA,
        **certificate,
        "certificate_sha256": certificate_sha256,
        "source_query_id": str(source["query_id"]),
        "source_candidate_index": source.get("candidate_index"),
        "source_output_token_ids_sha256": output_token_ids_sha256(
            source["output_token_ids"]
        ),
        "rewrite_protocol_sha256": protocol_sha256,
        "generator_model": {
            key: protocol["generator"][key]
            for key in ("repo_id", "revision", "tokenizer_revision")
        },
        "verifier_model": {
            key: protocol["verifier"][key]
            for key in ("repo_id", "revision", "tokenizer_revision")
        },
        "same_checkpoint_self_verification": True,
        "accepted": True,
    }
    generator_repo_id = str(protocol["generator"]["repo_id"])
    if generator_repo_id == "Qwen/Qwen2.5-3B-Instruct":
        annotation_family = "qwen2.5"
    else:
        annotation_family = re.sub(
            r"[^a-z0-9]+", "-", generator_repo_id.rsplit("/", 1)[-1].lower()
        ).strip("-")
    row["clir_supervision_provenance"] = {
        "schema_version": ROW_PROVENANCE_SCHEMA,
        "output_token_ids_sha256": output_token_ids_sha256(output_token_ids),
        "sources": [
            {
                "annotations_sha256": certificate_sha256,
                "fields": ["semantic_id", "style_id"],
                "annotation_protocol": {
                    "schema_version": PROVENANCE_SCHEMA,
                    "protocol_sha256": protocol_sha256,
                    "annotation_source": (
                        f"clir-{annotation_family}-semantic-rewrite-{protocol_suffix}"
                    ),
                },
                "merge_code": dict(code),
            }
        ],
    }
    return row


CompletionFunction = Callable[[Sequence[Mapping[str, str]], int], Mapping[str, Any]]


def rewrite_source_view(
    source: Mapping[str, Any],
    style: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    feature_tokenizer: Any,
    complete: CompletionFunction,
    protocol_sha256: str,
    code: Mapping[str, Any],
) -> Dict[str, Any]:
    """Generate, deterministically gate, verify, and tokenize one view."""

    protocol = validate_llm_rewrite_protocol(protocol)
    style_id = str(style["style_id"])
    protocol_suffix = str(protocol["schema_version"]).rsplit("-", 1)[-1]
    generator_output_format = protocol["generator"].get(
        "output_format", "json_object_v1"
    )
    line_plan: Dict[str, Any] | None = None
    generator_constraint: Dict[str, Any] | None = None
    if generator_output_format == "indexed_editable_lines_v1":
        line_plan = build_indexed_line_plan(str(source["response"]))
        generator_constraint = {
            "schema_version": "clir-indexed-editable-line-constraint-v1",
            "line_plan_sha256": canonical_json_sha256(line_plan),
            "editable_ids": list(line_plan["editable_ids"]),
            "protected_ids": list(line_plan["protected_ids"]),
            "assembly": "replace_editable_copy_protected_and_blank_source_lines",
        }
    generator_user = render_prompt(
        protocol["generator"]["user_prompt_template"],
        style=str(style["instruction"]),
        problem=str(source["question"]),
        source=str(source["response"]),
        annotated_source=(
            str(line_plan["annotated_source"]) if line_plan is not None else ""
        ),
        expected_editable_ids=(
            "\n".join(line_plan["editable_ids"]) if line_plan is not None else ""
        ),
    )
    generator_messages = [
        {"role": "system", "content": str(protocol["generator"]["system_prompt"])},
        {"role": "user", "content": generator_user},
    ]
    result: Dict[str, Any] = {
        "schema_version": LLM_VIEW_RESULT_SCHEMA,
        "id": f"{source['id']}--semrw-llm-{protocol_suffix}--{style_id}",
        "source_id": str(source["id"]),
        "query_id": str(source["query_id"]),
        "semantic_id": f"semrw-llm-{protocol_suffix}:{source['id']}",
        "style_id": style_id,
        "source_correctness": int(source["correctness"]),
        "source_response_sha256": text_sha256(str(source["response"])),
        "rewrite_protocol_sha256": protocol_sha256,
        "generator_messages": generator_messages,
        "generator_constraint": generator_constraint,
        "accepted": False,
        "rejection_reasons": [],
    }

    rewritten_response: str | None = None
    output_ids: list[int] | None = None
    gates: Dict[str, Any] | None = None
    verifier_value: Dict[str, Any] | None = None
    verifier_messages: list[Dict[str, str]] = []
    try:
        generator_completion = dict(
            complete(
                generator_messages,
                int(protocol["decoding"]["generator_max_new_tokens"]),
            )
        )
        result["generator_completion"] = generator_completion
        generator_text = _completion_text(generator_completion, "generator")
        if generator_output_format == "json_object_v1":
            generator_json = parse_json_object(
                generator_text,
                protocol["generator"]["output_keys"],
            )
            rewritten_response = generator_json["rewritten_response"]
        elif generator_output_format == "single_rewritten_response_tag_v1":
            rewritten_response = parse_tagged_rewrite(generator_text)
        elif generator_output_format == "indexed_editable_lines_v1":
            if line_plan is None or generator_constraint is None:
                raise AssertionError("Indexed generator lost its line plan")
            replacements = parse_indexed_line_rewrites(
                generator_text,
                line_plan["editable_ids"],
            )
            rewritten_response = assemble_indexed_line_rewrite(line_plan, replacements)
            generator_constraint.update(
                {
                    "generator_completion_text_sha256": text_sha256(generator_text),
                    "replacement_map_sha256": canonical_json_sha256(replacements),
                }
            )
            result["generator_constraint"] = generator_constraint
            result["generator_editable_line_rewrites"] = replacements
        else:
            raise AssertionError(f"Unsupported generator output format {generator_output_format}")
        if not isinstance(rewritten_response, str) or not rewritten_response.strip():
            raise ValueError("Generator rewritten_response must be a non-empty string")
        result["rewritten_response"] = rewritten_response
        result["rewritten_response_sha256"] = text_sha256(rewritten_response)
    except Exception as exc:
        result["generator_error"] = f"{type(exc).__name__}: {exc}"
        result["rejection_reasons"].append("generator_output_invalid")

    if rewritten_response is not None:
        try:
            gates = deterministic_rewrite_gates(
                source,
                rewritten_response,
                protocol,
                style_instruction=str(style["instruction"]),
            )
            result["deterministic_gates"] = gates
            if not gates["passed"]:
                result["rejection_reasons"].extend(
                    f"deterministic:{name}" for name in gates["failed_checks"]
                )
        except Exception as exc:
            result["deterministic_gate_error"] = f"{type(exc).__name__}: {exc}"
            result["rejection_reasons"].append("deterministic_gate_error")
        try:
            output_ids = encode_exact_response(feature_tokenizer, rewritten_response)
            result["output_token_ids"] = output_ids
            result["output_token_ids_sha256"] = output_token_ids_sha256(output_ids)
        except Exception as exc:
            result["tokenization_error"] = f"{type(exc).__name__}: {exc}"
            result["rejection_reasons"].append("feature_tokenization_roundtrip_failed")

        verifier_user = render_prompt(
            protocol["verifier"]["user_prompt_template"],
            style=str(style["instruction"]),
            problem=str(source["question"]),
            source=str(source["response"]),
            rewrite=rewritten_response,
        )
        verifier_messages = [
            {"role": "system", "content": str(protocol["verifier"]["system_prompt"])},
            {"role": "user", "content": verifier_user},
        ]
        result["verifier_messages"] = verifier_messages
        try:
            verifier_completion = dict(
                complete(
                    verifier_messages,
                    int(protocol["decoding"]["verifier_max_new_tokens"]),
                )
            )
            result["verifier_completion"] = verifier_completion
            verifier_json = parse_json_object(
                _completion_text(verifier_completion, "verifier"),
                protocol["verifier"]["output_keys"],
            )
            verifier_value = validate_verifier_decision(verifier_json)
            result["verifier_decision"] = verifier_value
            if not verifier_value["derived_accept"]:
                result["rejection_reasons"].append("verifier_rejected")
        except Exception as exc:
            result["verifier_error"] = f"{type(exc).__name__}: {exc}"
            result["rejection_reasons"].append("verifier_output_invalid")

    result["rejection_reasons"] = sorted(set(result["rejection_reasons"]))
    accepted = (
        rewritten_response is not None
        and output_ids is not None
        and gates is not None
        and gates["passed"]
        and verifier_value is not None
        and verifier_value["derived_accept"]
        and not result["rejection_reasons"]
    )
    result["accepted"] = accepted
    if accepted:
        result["accepted_view"] = _build_view_row(
            source,
            style_id=style_id,
            rewritten_response=rewritten_response,
            output_token_ids=output_ids,
            gates=gates,
            verifier=verifier_value,
            generator_messages=generator_messages,
            generator_constraint=generator_constraint,
            verifier_messages=verifier_messages,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            code=code,
        )
    return result


def validate_accepted_views(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    require_complete: bool,
) -> Dict[str, Any]:
    """Validate token-only accepted rows and their consistency-group identity."""

    protocol = validate_llm_rewrite_protocol(protocol)
    seen_ids: set[str] = set()
    semantic_styles: Dict[str, set[str]] = defaultdict(set)
    semantic_queries: Dict[str, set[str]] = defaultdict(set)
    semantic_responses: Dict[str, set[str]] = defaultdict(set)
    style_counts: Counter[str] = Counter()
    incorrect_repairs = 0
    for value in rows:
        row = dict(value)
        validate_rollout_row(row)
        row_id = str(row.get("id", ""))
        if not row_id or row_id in seen_ids:
            raise ValueError("Accepted rewrite views require unique ids")
        seen_ids.add(row_id)
        if row.get("split") != "train":
            raise ValueError("Accepted rewrite views must remain train-only")
        lineage = row.get("rewrite_lineage")
        if not isinstance(lineage, Mapping) or lineage.get("schema_version") != LLM_LINEAGE_SCHEMA:
            raise ValueError(f"Accepted rewrite {row_id!r} lacks LLM lineage")
        if lineage.get("rewrite_protocol_sha256") != protocol_sha256:
            raise ValueError(f"Accepted rewrite {row_id!r} has stale protocol lineage")
        if lineage.get("accepted") is not True:
            raise ValueError(f"Accepted rewrite {row_id!r} is not marked accepted")
        if row.get("query_id") != lineage.get("source_query_id"):
            raise ValueError(f"Accepted rewrite {row_id!r} changed query_id")
        semantic_id = str(row.get("semantic_id", ""))
        style_id = str(row.get("style_id", ""))
        if semantic_id == str(row.get("query_id")) or not semantic_id or not style_id:
            raise ValueError(f"Accepted rewrite {row_id!r} has invalid group identity")
        if style_id not in {style["style_id"] for style in protocol["styles"]}:
            raise ValueError(f"Accepted rewrite {row_id!r} has an undeclared style")
        semantic_styles[semantic_id].add(style_id)
        semantic_queries[semantic_id].add(str(row["query_id"]))
        semantic_responses[semantic_id].add(text_sha256(str(row["response"])))
        style_counts[style_id] += 1
        source_check = lineage["deterministic_gates"]["source_correctness_check"]
        rewrite_check = lineage["deterministic_gates"]["rewrite_correctness_check"]
        if int(source_check["correctness"]) == 0 and int(rewrite_check["correctness"]) == 1:
            incorrect_repairs += 1
    if any(len(queries) != 1 for queries in semantic_queries.values()):
        raise ValueError("An accepted semantic group crosses query identities")
    expected_styles = {style["style_id"] for style in protocol["styles"]}
    if require_complete:
        expected_rows = int(protocol["quality_gate"]["required_accepted_views"])
        if len(rows) != expected_rows:
            raise ValueError(
                f"Complete rewrite gate requires {expected_rows} accepted rows, got {len(rows)}"
            )
        if any(styles != expected_styles for styles in semantic_styles.values()):
            raise ValueError("Complete rewrite gate requires both styles for every source")
        if protocol["quality_gate"].get(
            "require_distinct_rewrites_per_semantic_group", False
        ) and any(
            len(responses) != len(expected_styles)
            for responses in semantic_responses.values()
        ):
            raise ValueError("Complete rewrite gate requires distinct rewrites per source")
    coverage = (
        audit_supervision_coverage(rows, require_provenance=True)
        if rows
        else None
    )
    return {
        "rows": len(rows),
        "semantic_groups": len(semantic_styles),
        "styles": dict(sorted(style_counts.items())),
        "incorrect_source_repairs": incorrect_repairs,
        "coverage": coverage,
    }


__all__ = [
    "LLM_LINEAGE_SCHEMA",
    "LLM_REWRITE_SCHEMA",
    "LLM_VIEW_RESULT_SCHEMA",
    "assemble_indexed_line_rewrite",
    "build_indexed_line_plan",
    "deterministic_rewrite_gates",
    "numeric_literal_sequence",
    "numeric_literal_set",
    "parse_json_object",
    "parse_indexed_line_rewrites",
    "parse_tagged_rewrite",
    "render_prompt",
    "rewrite_source_view",
    "select_configured_sources",
    "validate_accepted_views",
    "validate_llm_rewrite_protocol",
    "validate_verifier_decision",
]
