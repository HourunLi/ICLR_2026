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

TOKEN_LABEL_FIELDS = (
    "token_advantage",
    "token_advantages",
    "advantages",
    "progress_targets",
    "key_prior_target",
    "complete_prior_target",
)

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


def extract_gsm8k_candidate_answer(response: str) -> Optional[str]:
    if not isinstance(response, str) or not response.strip():
        return None
    boxed = _boxed_answers(response)
    if boxed:
        return boxed[-1]

    final_patterns = (
        r"(?i)(?:final\s+answer|answer)\s*(?:is|=|:)\s*([^\n]+)",
        r"(?i)therefore[, ]+([^\n]+)",
    )
    for pattern in final_patterns:
        matches = re.findall(pattern, response)
        if matches:
            candidate = matches[-1].strip().rstrip(". ")
            numeric = _last_numeric_expression(candidate)
            return numeric or candidate
    return _last_numeric_expression(response)


def _last_numeric_expression(text: str) -> Optional[str]:
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


def _numeric_value(answer: str) -> Optional[Fraction]:
    value = answer.strip()
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
    return result / 100 if percent else result


def check_gsm8k_response(response: str, raw_reference: str) -> Dict[str, Any]:
    reference = extract_gsm8k_reference(raw_reference)
    parsed = extract_gsm8k_candidate_answer(response)
    if parsed is None:
        return {
            "correctness": 0,
            "parsed_answer": None,
            "normalized_candidate_answer": None,
            "reference_answer": reference,
            "checker_status": "parse_failed",
            "checker_version": "clir_gsm8k_numeric_v2",
        }

    # SWIFT's official evaluator removes common unit suffixes before comparing
    # GSM8K answers. Preserve the full boxed string for auditability, but if it
    # is not itself numeric, compare its final numeric expression. This handles
    # answers such as ``\boxed{3 bolts}`` without accepting a wrong number.
    normalized_candidate = parsed
    parsed_value = _numeric_value(normalized_candidate)
    normalization = "direct"
    if parsed_value is None:
        numeric_expression = _last_numeric_expression(parsed)
        if numeric_expression is not None and _numeric_value(numeric_expression) is not None:
            normalized_candidate = numeric_expression
            parsed_value = _numeric_value(numeric_expression)
            normalization = "numeric_subexpression"
    reference_value = _numeric_value(reference)
    if parsed_value is not None and reference_value is not None:
        correct = parsed_value == reference_value
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
        "checker_version": "clir_gsm8k_numeric_v2",
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
) -> None:
    validate_rollout_row(row)
    identifier = row.get("id", "<missing-id>")
    if trajectory.ndim != 2:
        raise ValueError(f"Trajectory feature for {identifier!r} must have shape [T,D]")
    output_length = len(row["output_token_ids"])
    if trajectory.shape[0] != output_length:
        raise ValueError(
            f"Trajectory feature length mismatch for {identifier!r}: "
            f"expected {output_length}, got {trajectory.shape[0]}"
        )
    if not torch.isfinite(trajectory).all():
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
        if not torch.isfinite(condition).all():
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

    if device is None:
        device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")

    def forward_features(token_ids: Sequence[int]) -> tuple[Tensor, int, int]:
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
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

        expected_length = len(token_ids)
        layer_hidden_size: Optional[int] = None
        layers: list[Tensor] = []
        for layer_index, state in enumerate(hidden_states):
            if state.ndim != 3 or state.shape[0] != 1 or state.shape[1] != expected_length:
                raise ValueError(
                    f"Hidden state {layer_index} has shape {tuple(state.shape)}, "
                    f"expected [1,{expected_length},H]"
                )
            if layer_hidden_size is None:
                layer_hidden_size = int(state.shape[-1])
            elif state.shape[-1] != layer_hidden_size:
                raise ValueError("All hidden-state layers must have the same hidden dimension")
            layers.append(state[0])
        features = torch.cat(layers, dim=-1).to(dtype=dtype, device="cpu")
        return features, len(layers), int(layer_hidden_size or 0)

    all_features, layer_count, per_layer_hidden_size = forward_features(prompt_ids + output_ids)
    # A contiguous slice can still retain the full prompt+output storage. Clone
    # both views so each artifact contains only its declared token positions;
    # this also prevents a condition file from carrying unused output bytes.
    trajectory = all_features[len(prompt_ids) :].clone()
    if canonical_condition is None:
        condition, condition_layer_count, condition_hidden_size = forward_features(prompt_ids)
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
    "extract_gsm8k_candidate_answer",
    "extract_gsm8k_reference",
    "file_sha256",
    "load_protocol",
    "validate_extracted_row",
    "validate_rollout_row",
    "validate_uniform_extraction",
]
