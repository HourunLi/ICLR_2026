"""Real-data contracts for CLIR generation and hidden-state extraction.

The invariant in this module is intentionally stricter than the legacy toy
loader: generated token ids are the source of truth.  Human-readable response
text and step segmentation are metadata and must never be re-tokenized to
decide the trajectory feature positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor


GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem step-by-step.\n"
    "Simplify your answer as much as possible. Present your final answer as "
    "\\boxed{Your Answer}.\n"
    "{question}"
)

TOKEN_LABEL_ALIASES = {
    "token_advantage": ("token_advantage", "token_advantages", "advantages"),
    "progress_targets": ("progress_targets", "progress", "progress_target"),
    "key_prior_target": ("key_prior_target", "key_prior"),
    "complete_prior_target": ("complete_prior_target", "complete_prior"),
}

TOKEN_LABEL_FIELDS = tuple(
    alias
    for aliases in TOKEN_LABEL_ALIASES.values()
    for alias in aliases
)

SUPPORTED_GSM8K_CHECKERS = {
    "clir_gsm8k_numeric_v2",
    "clir_gsm8k_numeric_v3",
    "clir_gsm8k_numeric_v4",
    "clir_gsm8k_numeric_v5",
}

PROTOCOL_HASH_SCHEMA = "clir-protocol-component-hashes-v1"

STORAGE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class AlignedHiddenStates:
    """All-layer features aligned to exact prompt and output token ids."""

    trajectory: Tensor
    condition: Tensor
    layer_count: int
    per_layer_hidden_size: int
    feature_dim: int


def load_protocol(path: str | Path) -> Dict[str, Any]:
    protocol_path = Path(path)
    try:
        value = json.loads(protocol_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid protocol JSON at {protocol_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Protocol must be a JSON object: {protocol_path}")
    return value


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_hashes(protocol: Mapping[str, Any]) -> Dict[str, str]:
    """Return separate immutable hashes for acquisition, labels, and evaluation.

    The legacy ``protocol_sha256`` covers the whole JSON document. Component
    hashes prevent a downstream evaluation-only edit from invalidating already
    acquired rollout or hidden-state payloads.
    """

    dataset = protocol.get("dataset", {})
    acquisition = {
        "hash_schema": PROTOCOL_HASH_SCHEMA,
        "component": "acquisition",
        "model": protocol.get("model", {}),
        "dataset": {
            key: dataset.get(key)
            for key in ("repo_id", "subset", "revision")
            if key in dataset
        },
        "prompt": protocol.get("prompt", {}),
        "generation": protocol.get("generation", {}),
        "hidden_states": protocol.get("hidden_states", {}),
    }
    labels = {
        "hash_schema": PROTOCOL_HASH_SCHEMA,
        "component": "labels",
        "correctness": protocol.get("correctness", {}),
    }
    evaluation = {
        "hash_schema": PROTOCOL_HASH_SCHEMA,
        "component": "evaluation",
        "evaluation": protocol.get("evaluation", {}),
    }
    return {
        "protocol_sha256": canonical_json_sha256(protocol),
        "acquisition_protocol_sha256": canonical_json_sha256(acquisition),
        "label_protocol_sha256": canonical_json_sha256(labels),
        "evaluation_protocol_sha256": canonical_json_sha256(evaluation),
    }


def validate_protocol_reference(reference: Mapping[str, Any], protocol: Mapping[str, Any]) -> Dict[str, str]:
    """Validate a split/marker protocol reference with legacy compatibility."""

    hashes = protocol_hashes(protocol)
    recorded_acquisition = reference.get("acquisition_protocol_sha256")
    if recorded_acquisition is not None:
        if recorded_acquisition != hashes["acquisition_protocol_sha256"]:
            raise ValueError("Acquisition protocol hash mismatch")
    elif reference.get("protocol_sha256") != hashes["protocol_sha256"]:
        raise ValueError("Legacy full protocol hash mismatch")
    return hashes


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_stem(identifier: Any) -> str:
    """Return a path-safe, deterministic artifact name for an arbitrary id."""

    text = str(identifier)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")[:48]
    readable = readable or "item"
    suffix = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{suffix}"


def build_gsm8k_prompt(question: str, template: str = GSM8K_PROMPT_TEMPLATE) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("GSM8K question must be a non-empty string")
    if template.count("{question}") != 1:
        raise ValueError("GSM8K prompt template must contain exactly one `{question}` placeholder")
    # Do not use str.format: the literal braces in ``\\boxed{Your Answer}``
    # are part of the research prompt, not formatting placeholders.
    return template.replace("{question}", question)


def extract_gsm8k_reference(raw_answer: str) -> str:
    if not isinstance(raw_answer, str):
        raise TypeError("GSM8K reference answer must be a string")
    match = re.search(r"####\s*(.+?)\s*$", raw_answer, flags=re.DOTALL)
    return (match.group(1) if match else raw_answer).strip()


def _boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            answers.append(text[start : index - 1].strip())
    return answers


def extract_gsm8k_candidate_answer(
    response: str,
    *,
    exclude_unit_exponents: bool = False,
    ignore_boxed_placeholders: bool = False,
) -> Optional[str]:
    if not isinstance(response, str) or not response.strip():
        return None
    boxed = _boxed_answers(response)
    if boxed:
        if not ignore_boxed_placeholders:
            return boxed[-1]
        for candidate in reversed(boxed):
            if not _is_boxed_placeholder(candidate):
                return candidate

        # Phi occasionally copies the prompt's literal ``\boxed{Your Answer}``
        # and then puts the actual numeric answer immediately after it.  Once
        # every boxed expression has been proven to be a placeholder, prefer a
        # governed answer in the suffix and then its last numeric expression.
        placeholder_matches = list(
            re.finditer(
                r"\\boxed\s*\{\s*(?:Your\s+Answer)?\s*\}",
                response,
                flags=re.IGNORECASE,
            )
        )
        if placeholder_matches:
            suffix = response[placeholder_matches[-1].end() :]
            governed = _answer_cue_numeric_expression(suffix)
            if governed is not None:
                return governed
            suffix_numeric = _last_numeric_expression(
                suffix,
                exclude_unit_exponents=exclude_unit_exponents,
            )
            if suffix_numeric is not None:
                return suffix_numeric

    final_patterns = (
        r"(?i)(?:final\s+answer|answer)\s*(?:is|=|:)\s*([^\n]+)",
        r"(?i)therefore[, ]+([^\n]+)",
    )
    for pattern in final_patterns:
        matches = re.findall(pattern, response)
        if matches:
            candidate = matches[-1].strip().rstrip(". ")
            numeric = _last_numeric_expression(
                candidate,
                exclude_unit_exponents=exclude_unit_exponents,
            )
            return numeric or candidate
    return _last_numeric_expression(
        response,
        exclude_unit_exponents=exclude_unit_exponents,
    )


def _is_boxed_placeholder(answer: str) -> bool:
    """Return whether boxed content is the prompt's literal placeholder.

    This is intentionally narrow.  Text such as ``x``, ``No solution``, or
    ``Impossible`` is a model answer and must remain authoritative even when
    it contains no number.
    """

    normalized = answer.strip()
    text_match = re.fullmatch(r"\\text\s*\{(.*)\}", normalized, flags=re.DOTALL)
    if text_match:
        normalized = text_match.group(1).strip()
    if not normalized:
        return True
    letters = re.sub(r"[^A-Za-z]+", "", normalized).lower()
    return letters in {"answer", "youranswer"} and not re.search(r"\d", normalized)


def _last_numeric_expression(
    text: str,
    *,
    exclude_unit_exponents: bool = False,
) -> Optional[str]:
    if exclude_unit_exponents:
        # Unit suffixes such as ``cm^2`` and ``m^{3}`` must not replace the
        # actual boxed numeric answer merely because their exponent ends later.
        text = re.sub(r"\^\s*(?:\{\s*[-+]?\d+\s*\}|[-+]?\d+)", "", text)
    patterns = (
        r"\\(?:d?frac|tfrac)\s*\{\s*-?\d+\s*\}\s*\{\s*-?\d+\s*\}",
        r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?",
    )
    candidates: list[tuple[int, int, str]] = []
    for priority, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text):
            # Prefer the expression ending latest. A LaTeX fraction ends after
            # its denominator digits, so it wins over the nested plain-number
            # regex; priority is only a deterministic tie breaker.
            candidates.append((match.end(), -priority, match.group(0)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2].strip().rstrip(". ")


def _first_numeric_expression(
    text: str,
    *,
    exclude_unit_exponents: bool = False,
) -> Optional[str]:
    """Return the first numeric expression in an answer span.

    Phi commonly emits boxed answer sentences such as ``$15 for 10 sprays``
    or ratios such as ``10:1``.  The answer is the first literal in those
    spans; selecting the last literal silently turns the qualifier into the
    answer.  LaTeX fractions retain priority over their nested plain numbers.
    """

    if exclude_unit_exponents:
        text = re.sub(r"\^\s*(?:\{\s*[-+]?\d+\s*\}|[-+]?\d+)", "", text)
    patterns = (
        r"\\(?:d?frac|tfrac)\s*\{\s*-?\d+\s*\}\s*\{\s*-?\d+\s*\}",
        r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?",
    )
    candidates: list[tuple[int, int, str]] = []
    for priority, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text):
            candidates.append((match.start(), priority, match.group(0)))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2].strip().rstrip(". ")


def _answer_span_numeric_expression(
    text: str,
    *,
    exclude_unit_exponents: bool = False,
) -> Optional[str]:
    """Extract the governed number from a non-numeric final-answer span.

    Equality spans retain the historical right-hand-side behavior needed for
    answers such as ``work = $29``.  Other prose/ratio spans use their first
    numeric literal so trailing durations, counts, and ratio denominators do
    not replace the answer.
    """

    if "=" in text:
        right_hand_side = text.rsplit("=", 1)[1]
        composite = _first_composite_numeric_expression(right_hand_side)
        if composite is not None:
            return composite
        return _last_numeric_expression(
            right_hand_side,
            exclude_unit_exponents=exclude_unit_exponents,
        )
    composite = _first_composite_numeric_expression(text)
    if composite is not None:
        return composite
    governed = _prose_cue_numeric_expression(text)
    if governed is not None:
        return governed
    return _first_numeric_expression(
        text,
        exclude_unit_exponents=exclude_unit_exponents,
    )


def _first_composite_numeric_expression(text: str) -> Optional[str]:
    """Preserve the value of a mixed number or compound duration.

    Selecting the first literal is correct for answer-plus-qualifier spans such
    as ``$15 for 10 sprays``, but not for values such as ``21\\frac{1}{2}``
    or ``3 hours 20 minutes``.  Only a composite beginning at the first numeric
    literal is eligible, so a later qualifier cannot take over the answer.
    The returned improper fraction is already understood by ``_numeric_value``.
    """

    mixed_number = re.search(
        r"([-+]?\d+)\s*\\(?:d?frac|tfrac)\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}",
        text,
    )
    duration = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:\\text\{\s*)?(?:hours?|hrs?)\b\s*\}?"
        r"\s*(?:and\s+)?(\d+(?:\.\d+)?)\s*"
        r"(?:\\text\{\s*)?(?:minutes?|mins?)\b\s*\}?",
        text,
        flags=re.IGNORECASE,
    )
    candidates: list[tuple[int, Fraction]] = []
    if mixed_number is not None:
        whole = int(mixed_number.group(1))
        denominator = int(mixed_number.group(3))
        if denominator != 0:
            fraction = Fraction(int(mixed_number.group(2)), denominator)
            value = Fraction(whole) + (-fraction if whole < 0 else fraction)
            candidates.append((mixed_number.start(), value))
    if duration is not None:
        try:
            hours = Fraction(Decimal(duration.group(1)))
            minutes = Fraction(Decimal(duration.group(2)))
        except InvalidOperation:
            pass
        else:
            candidates.append((duration.start(), hours + minutes / 60))
    if not candidates:
        return None

    start, value = min(candidates, key=lambda item: item[0])
    if _first_numeric_expression(text[:start]) is not None:
        return None
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def _prose_cue_numeric_expression(text: str) -> Optional[str]:
    """Extract a number governed by prose without mistaking ratio colons.

    A colon is an answer cue only when its left neighbor is alphabetic, so
    ``weight: 170`` and ``Ratio: 10:1`` select 170 and 10 respectively while
    the numeric ratio colon cannot select the trailing 1.
    """

    numeric = (
        r"(?:\\(?:d?frac|tfrac)\s*\{\s*-?\d+\s*\}\s*\{\s*-?\d+\s*\}"
        r"|[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?)"
    )
    pattern = rf"(?i)(?:answer\s*(?:is|:)|(?:is|be|equals)|(?<=[A-Za-z])\s*:)[ ]*({numeric})"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    return matches[-1].group(1).strip().rstrip(". ")


def _answer_cue_numeric_expression(text: str) -> Optional[str]:
    """Extract a number directly governed by an answer/is/be/equality cue."""

    numeric = (
        r"(?:\\(?:d?frac|tfrac)\s*\{\s*-?\d+\s*\}\s*\{\s*-?\d+\s*\}"
        r"|[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?%?)"
    )
    pattern = rf"(?i)(?:answer\s*(?:is|=|:)|(?:is|be|equals|=|:))\s*({numeric})"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    return matches[-1].group(1).strip().rstrip(". ")


def _numeric_value(
    answer: str,
    *,
    percent_as_fraction: bool = True,
) -> Optional[Fraction]:
    value = answer.strip()
    value = value.replace("\\%", "%")
    value = re.sub(r"^\$|\$$", "", value)
    value = value.replace(",", "").replace(" ", "")
    value = value.replace("\\$", "")
    text_match = re.fullmatch(r"\\text\{(.+)\}", value)
    if text_match:
        value = text_match.group(1)

    percent = value.endswith("%")
    if percent:
        value = value[:-1]

    latex_fraction = re.fullmatch(r"\\(?:d?frac|tfrac)\{([-+]?\d+)\}\{([-+]?\d+)\}", value)
    try:
        if latex_fraction:
            result = Fraction(int(latex_fraction.group(1)), int(latex_fraction.group(2)))
        elif re.fullmatch(r"[-+]?\d+/[-+]?\d+", value):
            numerator, denominator = value.split("/", 1)
            result = Fraction(int(numerator), int(denominator))
        else:
            result = Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return result / 100 if percent and percent_as_fraction else result


def _numeric_value_options_v3(answer: str) -> set[Fraction]:
    """Return context-safe numeric interpretations for GSM8K percentage answers.

    GSM8K references store percentage points as plain numbers, so ``60%`` must
    match reference ``60``. A conventional fractional reference such as ``0.6``
    is also accepted. Percent literals whose magnitude is at most one retain
    only their conventional fractional interpretation, avoiding the false
    equivalence ``0.6% == 0.6``.
    """

    literal = _numeric_value(answer, percent_as_fraction=False)
    if literal is None:
        return set()
    normalized = answer.strip().replace("\\%", "%")
    if not normalized.endswith("%"):
        return {literal}
    conventional = literal / 100
    if abs(literal) > 1:
        return {literal, conventional}
    return {conventional}


def _explicit_percentage_points(response: str) -> set[Fraction]:
    """Extract numeric percentage-point literals from an answer.

    A bare occurrence of ``percent`` elsewhere in a completion is not enough
    to reinterpret a final decimal.  Requiring the number and percent marker
    to be adjacent avoids false positives from words such as ``percentage``
    or unrelated intermediate calculations.
    """

    numeric = r"[-+]?\d[\d,]*(?:\.\d+)?"
    pattern = rf"(?i)(?<![A-Za-z0-9.])({numeric})\s*(?:\\?%|percent\b)"
    values: set[Fraction] = set()
    for match in re.finditer(pattern, response):
        value = _numeric_value(match.group(1), percent_as_fraction=False)
        if value is not None:
            values.add(value)
    return values


def _has_probability_percent_context(response: str) -> bool:
    return bool(
        _explicit_percentage_points(response)
        and re.search(r"(?i)\b(?:probability|odds|chance)\b", response)
    )


def check_gsm8k_response(
    response: str,
    raw_reference: str,
    *,
    checker_version: str = "clir_gsm8k_numeric_v5",
) -> Dict[str, Any]:
    if checker_version not in SUPPORTED_GSM8K_CHECKERS:
        raise ValueError(
            f"Unsupported GSM8K checker {checker_version!r}; "
            f"expected one of {sorted(SUPPORTED_GSM8K_CHECKERS)}"
        )
    is_v3_or_later = checker_version in {
        "clir_gsm8k_numeric_v3",
        "clir_gsm8k_numeric_v4",
        "clir_gsm8k_numeric_v5",
    }
    is_v4_or_later = checker_version in {
        "clir_gsm8k_numeric_v4",
        "clir_gsm8k_numeric_v5",
    }
    is_v5 = checker_version == "clir_gsm8k_numeric_v5"
    reference = extract_gsm8k_reference(raw_reference)
    parsed = extract_gsm8k_candidate_answer(
        response,
        exclude_unit_exponents=is_v3_or_later,
        ignore_boxed_placeholders=is_v4_or_later,
    )
    if parsed is None:
        return {
            "correctness": 0,
            "parsed_answer": None,
            "normalized_candidate_answer": None,
            "reference_answer": reference,
            "checker_status": "parse_failed",
            "checker_version": checker_version,
        }

    # SWIFT's official evaluator removes common unit suffixes before comparing
    # GSM8K answers. Preserve the full boxed string for auditability, but if it
    # is not itself numeric, compare its final numeric expression. This handles
    # answers such as ``\boxed{3 bolts}`` without accepting a wrong number.
    normalized_candidate = parsed
    parsed_values = (
        _numeric_value_options_v3(normalized_candidate)
        if is_v3_or_later
        else {_numeric_value(normalized_candidate)} - {None}
    )
    normalization = "direct"
    if not parsed_values:
        numeric_expression = (
            _answer_span_numeric_expression(
                parsed,
                exclude_unit_exponents=True,
            )
            if is_v5
            else _answer_cue_numeric_expression(parsed)
            if is_v3_or_later
            else None
        )
        if numeric_expression is None:
            numeric_expression = _last_numeric_expression(
                parsed,
                exclude_unit_exponents=is_v3_or_later,
            )
        numeric_values = (
            _numeric_value_options_v3(numeric_expression)
            if is_v3_or_later and numeric_expression is not None
            else {_numeric_value(numeric_expression)} - {None}
            if numeric_expression is not None
            else set()
        )
        if numeric_expression is not None and numeric_values:
            normalized_candidate = numeric_expression
            parsed_values = numeric_values
            normalization = "numeric_subexpression"
    percent_decimal_value: Optional[Fraction] = None
    if (
        is_v3_or_later
        and "%" not in normalized_candidate.replace("\\%", "%")
    ):
        literal = _numeric_value(normalized_candidate, percent_as_fraction=False)
        if literal is not None and abs(literal) <= 1:
            proposed = literal * 100
            if (
                not is_v4_or_later
                or proposed in _explicit_percentage_points(response)
                or _has_probability_percent_context(response)
            ):
                percent_decimal_value = proposed
                parsed_values.add(proposed)
    reference_values = (
        _numeric_value_options_v3(reference)
        if is_v3_or_later
        else {_numeric_value(reference)} - {None}
    )
    if parsed_values and reference_values:
        correct = bool(parsed_values & reference_values)
        if is_v3_or_later and "%" in parsed.replace("\\%", "%"):
            normalization = "percent_equivalence"
        elif correct and percent_decimal_value is not None and percent_decimal_value in reference_values:
            normalization = "percent_decimal_equivalence"
        status = "numeric_match" if correct else "numeric_mismatch"
    else:
        normalized_parsed = re.sub(r"\s+", "", parsed).strip("$.").lower()
        normalized_reference = re.sub(r"\s+", "", reference).strip("$.").lower()
        correct = normalized_parsed == normalized_reference
        status = "normalized_match" if correct else "non_numeric_mismatch"

    return {
        "correctness": int(correct),
        "parsed_answer": parsed,
        "normalized_candidate_answer": normalized_candidate,
        "reference_answer": reference,
        "checker_status": status,
        "checker_normalization": normalization,
        "checker_version": checker_version,
    }


def validate_rollout_row(row: Mapping[str, Any], *, require_provenance: bool = True) -> None:
    identifier = row.get("id", "<missing-id>")
    for key in ("id", "query_id", "prompt_token_ids", "output_token_ids", "response"):
        if key not in row:
            raise ValueError(f"Rollout {identifier!r} is missing required field `{key}`")

    prompt_ids = _validate_token_ids(row["prompt_token_ids"], "prompt_token_ids", identifier)
    output_ids = _validate_token_ids(row["output_token_ids"], "output_token_ids", identifier)
    if not prompt_ids:
        raise ValueError(f"Rollout {identifier!r} has an empty prompt_token_ids sequence")
    if not output_ids:
        raise ValueError(f"Rollout {identifier!r} has an empty output_token_ids sequence")

    if "correctness" in row:
        correctness = row["correctness"]
        if isinstance(correctness, bool) or correctness not in (0, 1, 0.0, 1.0):
            raise ValueError(
                f"Rollout {identifier!r} correctness must be numeric 0 or 1, got {correctness!r}"
            )

    if require_provenance:
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"Rollout {identifier!r} is missing provenance")
        for key in ("model_id", "model_revision", "tokenizer_revision", "protocol_version"):
            if not provenance.get(key):
                raise ValueError(f"Rollout {identifier!r} provenance is missing `{key}`")


def validate_extracted_row(
    row: Mapping[str, Any],
    trajectory: Tensor,
    condition: Optional[Tensor] = None,
    *,
    check_finite: bool = True,
    require_correctness: bool = True,
) -> None:
    validate_rollout_row(row)
    identifier = row.get("id", "<missing-id>")
    if trajectory.ndim != 2:
        raise ValueError(f"Trajectory feature for {identifier!r} must have shape [T,D]")
    output_length = len(row["output_token_ids"])
    if require_correctness and "correctness" not in row:
        raise ValueError(f"Extracted rollout {identifier!r} is missing required `correctness`")
    if "correctness" in row:
        correctness = row["correctness"]
        if isinstance(correctness, bool) or correctness not in (0, 1, 0.0, 1.0):
            raise ValueError(
                f"Extracted rollout {identifier!r} correctness must be numeric 0 or 1, "
                f"got {correctness!r}"
            )
    if trajectory.shape[0] != output_length:
        raise ValueError(
            f"Trajectory feature length mismatch for {identifier!r}: "
            f"expected {output_length}, got {trajectory.shape[0]}"
        )
    if check_finite and not torch.isfinite(trajectory).all():
        raise ValueError(f"Trajectory feature for {identifier!r} contains NaN or Inf")

    if condition is not None:
        if condition.ndim != 2:
            raise ValueError(f"Condition feature for {identifier!r} must have shape [C,D]")
        if condition.shape[0] != len(row["prompt_token_ids"]):
            raise ValueError(
                f"Condition feature length mismatch for {identifier!r}: "
                f"expected {len(row['prompt_token_ids'])}, got {condition.shape[0]}"
            )
        if condition.shape[1] != trajectory.shape[1]:
            raise ValueError(f"Condition/trajectory feature dimensions differ for {identifier!r}")
        if check_finite and not torch.isfinite(condition).all():
            raise ValueError(f"Condition feature for {identifier!r} contains NaN or Inf")

    for key in TOKEN_LABEL_FIELDS:
        if key in row and row[key] is not None and len(row[key]) != output_length:
            raise ValueError(
                f"Token label `{key}` length mismatch for {identifier!r}: "
                f"expected {output_length}, got {len(row[key])}"
            )
    onset = row.get("hallucination_onset")
    if onset is not None and int(onset) != -1 and not 0 <= int(onset) < output_length:
        raise ValueError(f"hallucination_onset out of range for {identifier!r}: {onset}")


def _validate_token_ids(values: Any, field: str, identifier: Any) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Rollout {identifier!r} field `{field}` must be a sequence")
    token_ids: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Rollout {identifier!r} field `{field}` contains invalid token id {value!r}")
        token_ids.append(value)
    return token_ids


def forward_all_layer_features(
    model: Any,
    token_ids: Sequence[int],
    *,
    device: Optional[str | torch.device] = None,
) -> tuple[Tensor, int, int]:
    """Run one exact unpadded sequence and concatenate returned layer states."""

    exact_ids = _validate_token_ids(token_ids, "token_ids", "forward-all-layers")
    if not exact_ids:
        raise ValueError("All-layer extraction cannot forward an empty token sequence")
    if device is None:
        device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")

    input_ids = torch.tensor([exact_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError("Model did not return hidden_states")

        layer_hidden_size: Optional[int] = None
        layers: list[Tensor] = []
        for layer_index, state in enumerate(hidden_states):
            if state.ndim != 3 or state.shape[0] != 1 or state.shape[1] != len(exact_ids):
                raise ValueError(
                    f"Hidden state {layer_index} has shape {tuple(state.shape)}, "
                    f"expected [1,{len(exact_ids)},H]"
                )
            if layer_hidden_size is None:
                layer_hidden_size = int(state.shape[-1])
            elif state.shape[-1] != layer_hidden_size:
                raise ValueError("All hidden-state layers must have the same hidden dimension")
            layers.append(state[0])
        features = torch.cat(layers, dim=-1)
    return features, len(layers), int(layer_hidden_size or 0)


def extract_aligned_hidden_states(
    model: Any,
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    *,
    storage_dtype: str | torch.dtype = "bfloat16",
    device: Optional[str | torch.device] = None,
    canonical_condition: Optional[Tensor] = None,
) -> AlignedHiddenStates:
    """Teacher-force exact ids and concatenate every returned hidden-state layer.

    Position ``t`` in ``trajectory`` always corresponds to
    ``output_token_ids[t]``. Condition states come from a prompt-only forward,
    or from the canonical per-query cache supplied by the caller. No response
    text or step metadata is consumed.
    """

    prompt_ids = _validate_token_ids(prompt_token_ids, "prompt_token_ids", "extract")
    output_ids = _validate_token_ids(output_token_ids, "output_token_ids", "extract")
    if not prompt_ids or not output_ids:
        raise ValueError("prompt_token_ids and output_token_ids must both be non-empty")
    dtype = STORAGE_DTYPES.get(storage_dtype, storage_dtype)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported storage dtype: {storage_dtype}")

    all_features, layer_count, per_layer_hidden_size = forward_all_layer_features(
        model,
        prompt_ids + output_ids,
        device=device,
    )
    all_features = all_features.to(dtype=dtype, device="cpu")
    # A contiguous slice can still retain the full prompt+output storage. Clone
    # both views so each artifact contains only its declared token positions;
    # this also prevents a condition file from carrying unused output bytes.
    trajectory = all_features[len(prompt_ids) :].clone()
    if canonical_condition is None:
        condition, condition_layer_count, condition_hidden_size = forward_all_layer_features(
            model,
            prompt_ids,
            device=device,
        )
        condition = condition.to(dtype=dtype, device="cpu")
        condition = condition.clone()
        if condition_layer_count != layer_count or condition_hidden_size != per_layer_hidden_size:
            raise ValueError("Prompt-only condition metadata differs from trajectory metadata")
    else:
        condition = canonical_condition
        if condition.ndim != 2 or tuple(condition.shape) != (len(prompt_ids), all_features.shape[-1]):
            raise ValueError(
                "Canonical condition shape mismatch: "
                f"expected {(len(prompt_ids), all_features.shape[-1])}, got {tuple(condition.shape)}"
            )
    if trajectory.shape[0] != len(output_ids):
        raise AssertionError("Internal output-token slicing error")

    return AlignedHiddenStates(
        trajectory=trajectory,
        condition=condition,
        layer_count=layer_count,
        per_layer_hidden_size=per_layer_hidden_size,
        feature_dim=int(all_features.shape[-1]),
    )


def validate_uniform_extraction(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize uniform feature metadata from extracted manifest rows."""

    layer_counts = {row.get("feature_metadata", {}).get("layer_count") for row in rows}
    feature_dims = {row.get("feature_metadata", {}).get("feature_dim") for row in rows}
    storage_dtypes = {row.get("feature_metadata", {}).get("storage_dtype") for row in rows}
    if len(layer_counts) != 1 or None in layer_counts:
        raise ValueError(f"Non-uniform or missing layer_count values: {layer_counts}")
    if len(feature_dims) != 1 or None in feature_dims:
        raise ValueError(f"Non-uniform or missing feature_dim values: {feature_dims}")
    if len(storage_dtypes) != 1 or None in storage_dtypes:
        raise ValueError(f"Non-uniform or missing storage_dtype values: {storage_dtypes}")
    return {
        "layer_count": next(iter(layer_counts)),
        "feature_dim": next(iter(feature_dims)),
        "storage_dtype": next(iter(storage_dtypes)),
    }


__all__ = [
    "AlignedHiddenStates",
    "GSM8K_PROMPT_TEMPLATE",
    "artifact_stem",
    "build_gsm8k_prompt",
    "canonical_json_sha256",
    "check_gsm8k_response",
    "extract_aligned_hidden_states",
    "forward_all_layer_features",
    "extract_gsm8k_candidate_answer",
    "extract_gsm8k_reference",
    "file_sha256",
    "load_protocol",
    "validate_extracted_row",
    "validate_rollout_row",
    "validate_uniform_extraction",
]
