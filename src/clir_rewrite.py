"""Identity-safe deterministic semantic-view pilot for CLIR.

The reversible transforms in this module are an engineering gate only.  They
exercise rewrite grouping and token-aligned feature acquisition without
pretending that formatting-only views are research-grade semantic rewrites.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .clir_real_data import canonical_json_sha256, check_gsm8k_response
from .clir_supervision import (
    PROVENANCE_SCHEMA,
    ROW_PROVENANCE_SCHEMA,
    audit_supervision_coverage,
    output_token_ids_sha256,
)


REWRITE_SCHEMA = "clir-semantic-rewrite-pilot-v1"
PLAN_SCHEMA = "clir-semantic-rewrite-plan-row-v1"
LINEAGE_SCHEMA = "clir-semantic-rewrite-lineage-v1"
ANNOTATION_SOURCE = "clir-deterministic-reversible-rewrite-v1"
SUPPORTED_TRANSFORMS = {
    "identity": "",
    "prefix_every_physical_line_dash_space": "- ",
    "prefix_every_physical_line_greater_space": "> ",
}


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Text hash input must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_rewrite_protocol(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(protocol)
    if normalized.get("schema_version") != REWRITE_SCHEMA:
        raise ValueError(f"Rewrite protocol must use {REWRITE_SCHEMA}")
    if normalized.get("formal_mechanism_claim_allowed") is not False:
        raise ValueError("The deterministic pilot must forbid formal mechanism claims")

    source = normalized.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Rewrite protocol requires a source object")
    source_path = Path(str(source.get("manifest", "")))
    if not source_path.parts or "pilot_test" in source_path.parts:
        raise ValueError("Rewrite source must be explicit and must not reference pilot_test")
    if source.get("split") != "train":
        raise ValueError("The v1 engineering pilot is train-only")
    source_sha = source.get("sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("Rewrite source requires a pinned SHA256")
    if source.get("checker_version") != "clir_gsm8k_numeric_v5":
        raise ValueError("The v1 pilot requires the frozen v5 GSM8K checker")

    identity = normalized.get("identity")
    expected_identity = {
        "query_id": "preserve_source_query_id",
        "semantic_id": "derive_from_source_trajectory_id",
        "style_id": "declared_view_style",
        "source_candidate_index": "lineage_only_not_top_level",
    }
    if not isinstance(identity, Mapping) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("Rewrite identity policy does not match the v1 data contract")

    model = normalized.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Rewrite protocol requires a pinned feature model")
    for field in ("repo_id", "revision", "tokenizer_revision", "torch_dtype"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise ValueError(f"Rewrite model requires {field}")
    if model["torch_dtype"] != "bfloat16":
        raise ValueError("The v1 rewrite feature model must use bfloat16")

    hidden = normalized.get("hidden_states")
    expected_hidden = {
        "layer_policy": "all_outputs_hidden_states",
        "layer_count": 33,
        "per_layer_hidden_size": 3072,
        "feature_dim": 101376,
        "combine": "concatenate_last_dimension",
        "trajectory_slice": "exact_output_token_ids",
        "extractor_applied_additional_final_norm": False,
        "storage_dtype": "bfloat16",
    }
    if not isinstance(hidden, Mapping) or any(
        hidden.get(key) != value for key, value in expected_hidden.items()
    ):
        raise ValueError("Rewrite hidden-state policy does not match the frozen 33-layer layout")

    selection = normalized.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Rewrite protocol requires selection rules")
    if selection.get("order") != "source_manifest_order":
        raise ValueError("Pilot selection order must be source_manifest_order")
    rows_per_correctness = selection.get("rows_per_correctness")
    if isinstance(rows_per_correctness, bool) or not isinstance(rows_per_correctness, int):
        raise ValueError("rows_per_correctness must be an integer")
    if rows_per_correctness <= 0:
        raise ValueError("rows_per_correctness must be positive")
    if selection.get("correctness_values") != [0, 1]:
        raise ValueError("Pilot selection must cover numeric correctness classes [0, 1]")
    if selection.get("require_distinct_query_ids") is not True:
        raise ValueError("Pilot source trajectories must come from distinct queries")

    views = normalized.get("views")
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)) or len(views) < 2:
        raise ValueError("Rewrite protocol requires at least two declared views")
    style_ids: list[str] = []
    transforms: list[str] = []
    for view in views:
        if not isinstance(view, Mapping):
            raise ValueError("Each rewrite view must be an object")
        style_id = view.get("style_id")
        transform = view.get("transform")
        if not isinstance(style_id, str) or not style_id.strip():
            raise ValueError("Each rewrite view requires a non-empty style_id")
        if transform not in SUPPORTED_TRANSFORMS:
            raise ValueError(f"Unsupported rewrite transform: {transform!r}")
        style_ids.append(style_id)
        transforms.append(str(transform))
    if len(set(style_ids)) != len(style_ids):
        raise ValueError("Rewrite style_id values must be unique")
    if style_ids[0] != "original" or transforms[0] != "identity":
        raise ValueError("The first rewrite view must be original/identity")

    policy = normalized.get("rewrite_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("Rewrite protocol requires rewrite_policy")
    expected_policy = {
        "trajectory_policy": "teacher_forced_exact_rewrite_token_ids",
        "condition_policy": "reuse_verified_prompt_only_source_condition",
        "semantic_certificate": "inverse_transform_byte_exact_source_response",
        "answer_gate": "same_normalized_candidate_answer_and_correctness",
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"Rewrite policy {key} must be {expected!r}")
    return normalized


def apply_rewrite_transform(text: str, transform: str) -> str:
    if not isinstance(text, str) or not text:
        raise ValueError("Rewrite source response must be a non-empty string")
    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(f"Unsupported rewrite transform: {transform!r}")
    prefix = SUPPORTED_TRANSFORMS[transform]
    if not prefix:
        return text
    return "".join(prefix + line for line in text.splitlines(keepends=True))


def invert_rewrite_transform(text: str, transform: str) -> str:
    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(f"Unsupported rewrite transform: {transform!r}")
    prefix = SUPPORTED_TRANSFORMS[transform]
    if not prefix:
        return text
    restored: list[str] = []
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        if not line.startswith(prefix):
            raise ValueError(
                f"Rewrite line {line_number} does not carry the expected {prefix!r} prefix"
            )
        restored.append(line[len(prefix) :])
    return "".join(restored)


def select_pilot_sources(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    protocol = validate_rewrite_protocol(protocol)
    split = str(protocol["source"]["split"])
    per_class = int(protocol["selection"]["rows_per_correctness"])
    needed = {0: per_class, 1: per_class}
    selected: list[Dict[str, Any]] = []
    selected_queries: set[str] = set()
    seen_ids: set[str] = set()
    for row_index, source in enumerate(rows):
        row = dict(source)
        row_id = str(row.get("id", ""))
        query_id = str(row.get("query_id", ""))
        if not row_id or row_id in seen_ids:
            raise ValueError(f"Source manifest has an invalid id at row {row_index}")
        seen_ids.add(row_id)
        if row.get("split") != split:
            continue
        correctness = row.get("correctness")
        if isinstance(correctness, bool) or correctness not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Source row {row_id!r} has invalid correctness")
        correctness = int(correctness)
        if needed[correctness] <= 0 or query_id in selected_queries:
            continue
        required = (
            "response",
            "raw_reference",
            "prompt_token_ids",
            "output_token_ids",
            "condition_states_path",
            "condition_sha256",
        )
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"Source row {row_id!r} is missing fields: {missing}")
        selected.append(row)
        selected_queries.add(query_id)
        needed[correctness] -= 1
        if all(value == 0 for value in needed.values()):
            break
    if any(value != 0 for value in needed.values()):
        raise ValueError(f"Source manifest cannot satisfy balanced pilot selection: {needed}")
    return selected


def encode_exact_response(tokenizer: Any, response: str) -> list[int]:
    """Encode a response on the frozen feature-token axis with exact roundtrip."""

    encoded = tokenizer.encode(response, add_special_tokens=False)
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)) or not encoded:
        raise ValueError("Tokenizer returned no rewrite token IDs")
    token_ids = [int(value) for value in encoded]
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if decoded != response:
        raise ValueError("Rewrite response does not round-trip through the pinned tokenizer")
    return token_ids


def build_rewrite_plan(
    source_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    protocol: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    rewrite_protocol_sha256: str,
) -> list[Dict[str, Any]]:
    protocol = validate_rewrite_protocol(protocol)
    checker_version = str(protocol["source"]["checker_version"])
    plan: list[Dict[str, Any]] = []
    seen_view_ids: set[str] = set()
    for source in source_rows:
        source_id = str(source["id"])
        query_id = str(source["query_id"])
        source_response = str(source["response"])
        source_tokens = [int(value) for value in source["output_token_ids"]]
        source_provenance = source.get("provenance")
        if not isinstance(source_provenance, Mapping):
            raise ValueError(f"Source provenance is missing for {source_id!r}")
        model = protocol["model"]
        expected_source_model = {
            "model_id": model["repo_id"],
            "model_revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"],
        }
        for field, expected in expected_source_model.items():
            if source_provenance.get(field) != expected:
                raise ValueError(
                    f"Source {field} mismatch for {source_id!r}: "
                    f"expected {expected!r}, got {source_provenance.get(field)!r}"
                )
        decoded_source = tokenizer.decode(
            source_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded_source != source_response:
            raise ValueError(f"Source response/token mismatch for {source_id!r}")
        source_check = check_gsm8k_response(
            source_response,
            str(source["raw_reference"]),
            checker_version=checker_version,
        )
        if int(source_check["correctness"]) != int(source["correctness"]):
            raise ValueError(f"Frozen correctness no longer reproduces for {source_id!r}")
        if source_check.get("normalized_candidate_answer") != source.get(
            "normalized_candidate_answer"
        ):
            raise ValueError(f"Frozen parsed answer no longer reproduces for {source_id!r}")
        semantic_id = f"semrw-v1:{source_id}"
        for view in protocol["views"]:
            style_id = str(view["style_id"])
            transform = str(view["transform"])
            rewritten_response = apply_rewrite_transform(source_response, transform)
            if invert_rewrite_transform(rewritten_response, transform) != source_response:
                raise AssertionError("Internal reversible rewrite failure")
            output_ids = source_tokens if transform == "identity" else encode_exact_response(
                tokenizer, rewritten_response
            )
            rewrite_check = check_gsm8k_response(
                rewritten_response,
                str(source["raw_reference"]),
                checker_version=checker_version,
            )
            if int(rewrite_check["correctness"]) != int(source_check["correctness"]):
                raise ValueError(f"Rewrite changed correctness for {source_id!r}/{style_id}")
            if (
                rewrite_check.get("normalized_candidate_answer")
                != source_check.get("normalized_candidate_answer")
            ):
                raise ValueError(f"Rewrite changed the parsed answer for {source_id!r}/{style_id}")
            view_id = (
                source_id
                if transform == "identity"
                else f"{source_id}--semrw-v1--{style_id}"
            )
            if view_id in seen_view_ids:
                raise ValueError(f"Duplicate rewrite view id: {view_id}")
            seen_view_ids.add(view_id)
            plan.append(
                {
                    "schema_version": PLAN_SCHEMA,
                    "id": view_id,
                    "query_id": query_id,
                    "semantic_id": semantic_id,
                    "style_id": style_id,
                    "transform": transform,
                    "split": source["split"],
                    "source_id": source_id,
                    "source_candidate_index": source.get("candidate_index"),
                    "source_output_token_ids_sha256": output_token_ids_sha256(source_tokens),
                    "source_response_sha256": text_sha256(source_response),
                    "response": rewritten_response,
                    "response_sha256": text_sha256(rewritten_response),
                    "output_token_ids": output_ids,
                    "output_token_ids_sha256": output_token_ids_sha256(output_ids),
                    "correctness_check": rewrite_check,
                    "source_manifest_sha256": source_manifest_sha256,
                    "rewrite_protocol_sha256": rewrite_protocol_sha256,
                }
            )
    return plan


def materialize_rewrite_views(
    source_rows: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
    *,
    plan_sha256: str,
    source_manifest_path: str,
    source_manifest_sha256: str,
    rewrite_protocol_path: str,
    rewrite_protocol_sha256: str,
    code: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    source_by_id = {str(row["id"]): dict(row) for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Selected rewrite sources must have unique ids")
    output: list[Dict[str, Any]] = []
    for entry_value in plan:
        entry = dict(entry_value)
        source_id = str(entry["source_id"])
        if source_id not in source_by_id:
            raise ValueError(f"Rewrite plan references unknown source {source_id!r}")
        source = source_by_id[source_id]
        row = dict(source)
        for field in (
            "candidate_index",
            "generation",
            "backend_response_text",
            "decode_matches_backend_text",
            "correctness",
            "parsed_answer",
            "normalized_candidate_answer",
            "reference_answer",
            "checker_status",
            "checker_normalization",
            "checker_version",
            "semantic_id",
            "style_id",
            "domain_id",
            "clir_supervision_provenance",
        ):
            row.pop(field, None)
        is_original = entry["transform"] == "identity"
        if not is_original:
            for field in ("hidden_states_path", "feature_sha256", "feature_metadata"):
                row.pop(field, None)
        row.update(
            {
                "id": entry["id"],
                "query_id": entry["query_id"],
                "response": entry["response"],
                "trajectory": entry["response"],
                "output_token_ids": list(entry["output_token_ids"]),
                "semantic_id": entry["semantic_id"],
                "style_id": entry["style_id"],
                **dict(entry["correctness_check"]),
            }
        )
        source_provenance = source.get("provenance", {})
        if not isinstance(source_provenance, Mapping):
            raise ValueError(f"Source provenance is missing for {source_id!r}")
        row["provenance"] = {
            "model_id": source_provenance.get("model_id"),
            "model_revision": source_provenance.get("model_revision"),
            "tokenizer_revision": source_provenance.get("tokenizer_revision"),
            "protocol_version": REWRITE_SCHEMA,
            "rewrite_protocol_sha256": rewrite_protocol_sha256,
            "source_provenance_sha256": canonical_json_sha256(source_provenance),
            "trajectory_acquisition": (
                "reused_immutable_source_feature"
                if is_original
                else "teacher_forced_exact_rewrite_token_ids"
            ),
            "code": dict(code),
        }
        row["rewrite_lineage"] = {
            "schema_version": LINEAGE_SCHEMA,
            "source_id": source_id,
            "source_query_id": source["query_id"],
            "source_candidate_index": entry.get("source_candidate_index"),
            "source_output_token_ids_sha256": entry["source_output_token_ids_sha256"],
            "source_response_sha256": entry["source_response_sha256"],
            "response_sha256": entry["response_sha256"],
            "transform": entry["transform"],
            "inverse_transform_verified": True,
            "answer_gate_verified": True,
            "source_manifest": source_manifest_path,
            "source_manifest_sha256": source_manifest_sha256,
            "rewrite_protocol": rewrite_protocol_path,
            "rewrite_protocol_sha256": rewrite_protocol_sha256,
            "rewrite_plan_sha256": plan_sha256,
        }
        row["clir_supervision_provenance"] = {
            "schema_version": ROW_PROVENANCE_SCHEMA,
            "output_token_ids_sha256": entry["output_token_ids_sha256"],
            "sources": [
                {
                    "annotations_sha256": plan_sha256,
                    "fields": ["semantic_id", "style_id"],
                    "annotation_protocol": {
                        "schema_version": PROVENANCE_SCHEMA,
                        "protocol_sha256": rewrite_protocol_sha256,
                        "annotation_source": ANNOTATION_SOURCE,
                    },
                    "rewrite_plan": "rewrite_plan.jsonl",
                    "merge_code": dict(code),
                }
            ],
        }
        output.append(row)
    validate_rewrite_views(output, expected_plan_sha256=plan_sha256)
    return output


def validate_rewrite_views(
    rows: Sequence[Mapping[str, Any]], *, expected_plan_sha256: str | None = None
) -> Dict[str, Any]:
    seen_ids: set[str] = set()
    semantic_styles: Dict[str, set[str]] = defaultdict(set)
    semantic_queries: Dict[str, set[str]] = defaultdict(set)
    style_counts = Counter()
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in seen_ids:
            raise ValueError("Rewrite views require unique non-empty ids")
        seen_ids.add(row_id)
        lineage = row.get("rewrite_lineage")
        if not isinstance(lineage, Mapping) or lineage.get("schema_version") != LINEAGE_SCHEMA:
            raise ValueError(f"Rewrite view {row_id!r} lacks valid lineage")
        if expected_plan_sha256 and lineage.get("rewrite_plan_sha256") != expected_plan_sha256:
            raise ValueError(f"Rewrite view {row_id!r} has a stale plan hash")
        if row.get("split") != "train":
            raise ValueError("The v1 rewrite pilot must remain train-only")
        if row.get("query_id") != lineage.get("source_query_id"):
            raise ValueError(f"Rewrite view {row_id!r} changed query_id")
        transform = str(lineage.get("transform", ""))
        restored = invert_rewrite_transform(str(row.get("response", "")), transform)
        if text_sha256(restored) != lineage.get("source_response_sha256"):
            raise ValueError(f"Rewrite view {row_id!r} failed inverse semantic certificate")
        if text_sha256(str(row["response"])) != lineage.get("response_sha256"):
            raise ValueError(f"Rewrite view {row_id!r} response hash drifted")
        if output_token_ids_sha256(row.get("output_token_ids", ())) != row.get(
            "clir_supervision_provenance", {}
        ).get("output_token_ids_sha256"):
            raise ValueError(f"Rewrite view {row_id!r} token identity drifted")
        semantic_id = str(row.get("semantic_id", ""))
        style_id = str(row.get("style_id", ""))
        if not semantic_id or not style_id:
            raise ValueError(f"Rewrite view {row_id!r} lacks semantic/style identity")
        if semantic_id == str(row.get("query_id")):
            raise ValueError("semantic_id must not alias query_id")
        semantic_styles[semantic_id].add(style_id)
        semantic_queries[semantic_id].add(str(row["query_id"]))
        style_counts[style_id] += 1
    if any(len(queries) != 1 for queries in semantic_queries.values()):
        raise ValueError("A semantic rewrite group crosses query identities")
    style_sets = {tuple(sorted(styles)) for styles in semantic_styles.values()}
    if len(style_sets) != 1:
        raise ValueError("Every semantic group must contain the same declared styles")
    coverage = audit_supervision_coverage(rows, require_provenance=True)
    if not coverage["eligible_components"]["consistency"]:
        raise ValueError("Rewrite views do not activate both positive and negative consistency pairs")
    return {
        "rows": len(rows),
        "semantic_groups": len(semantic_styles),
        "styles": dict(sorted(style_counts.items())),
        "coverage": coverage,
    }


__all__ = [
    "ANNOTATION_SOURCE",
    "LINEAGE_SCHEMA",
    "PLAN_SCHEMA",
    "REWRITE_SCHEMA",
    "apply_rewrite_transform",
    "build_rewrite_plan",
    "encode_exact_response",
    "invert_rewrite_transform",
    "materialize_rewrite_views",
    "select_pilot_sources",
    "text_sha256",
    "validate_rewrite_protocol",
    "validate_rewrite_views",
]
