"""Versioned import and coverage audit for externally produced CLIR supervision."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Dict, Mapping, Sequence

from .clir_real_data import TOKEN_LABEL_ALIASES, canonical_json_sha256


ANNOTATION_SCHEMA = "clir-supervision-annotation-v1"
PROVENANCE_SCHEMA = "clir-supervision-provenance-v1"
ROW_PROVENANCE_SCHEMA = "clir-supervision-row-provenance-v1"
TOKEN_FIELDS = (
    "token_hallucination_target",
    "token_hallucination_mask",
    "token_advantage",
    "progress_targets",
    "key_prior_target",
    "complete_prior_target",
)
SCALAR_FIELDS = ("path_hallucinated", "hallucination_onset")
GROUP_FIELDS = ("semantic_id", "style_id", "domain_id")
VECTOR_FIELDS = ("complete_reconstruction_target",)
MERGED_FIELDS = (*SCALAR_FIELDS, *TOKEN_FIELDS, *GROUP_FIELDS, *VECTOR_FIELDS)
ANNOTATION_METADATA_FIELDS = {
    "schema_version",
    "id",
    "query_id",
    "output_token_ids_sha256",
    "annotation_provenance",
    "annotation_metadata",
}

SCALAR_ALIASES = {
    "path_hallucinated": ("path_hallucinated", "hallucinated", "hallucination"),
    "hallucination_onset": ("hallucination_onset", "hallucination_start", "onset"),
    "semantic_id": (
        "semantic_id",
        "semantic_ids",
        "augmentation_group",
        "augmentation_group_id",
        "group_id",
    ),
    "style_id": (
        "style_id",
        "style_ids",
        "augmentation_style",
        "rewrite_style",
        "style",
    ),
    "domain_id": ("domain_id", "domain"),
    "complete_reconstruction_target": ("complete_reconstruction_target", "csr_target"),
}


def output_token_ids_sha256(token_ids: Sequence[int]) -> str:
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ValueError("output_token_ids must be a sequence")
    normalized: list[int] = []
    for value in token_ids:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("output_token_ids must contain non-negative integers")
        normalized.append(int(value))
    if not normalized:
        raise ValueError("output_token_ids must not be empty")
    return canonical_json_sha256(normalized)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _numeric_sequence(value: Any, *, field: str, length: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a numeric sequence")
    if len(value) != length:
        raise ValueError(f"{field} length mismatch: expected {length}, got {len(value)}")
    normalized: list[float] = []
    for element in value:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            raise ValueError(f"{field} must contain only numeric values")
        numeric = float(element)
        if not math.isfinite(numeric):
            raise ValueError(f"{field} contains a non-finite value")
        normalized.append(numeric)
    if field in {
        "token_hallucination_target",
        "token_hallucination_mask",
        "key_prior_target",
        "complete_prior_target",
    } and any(
        not 0.0 <= element <= 1.0 for element in normalized
    ):
        raise ValueError(f"{field} values must be in [0, 1]")
    if field in {"token_hallucination_target", "token_hallucination_mask"} and any(
        element not in {0.0, 1.0} for element in normalized
    ):
        raise ValueError(f"{field} values must be binary")
    return normalized


def _validate_provenance(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("annotation_provenance is required")
    provenance = dict(value)
    if provenance.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError(f"annotation_provenance must use {PROVENANCE_SCHEMA}")
    if not _valid_sha256(provenance.get("protocol_sha256")):
        raise ValueError("annotation_provenance requires a valid protocol_sha256")
    source = provenance.get("annotation_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("annotation_provenance requires annotation_source")
    return provenance


def validate_supervision_annotation(
    annotation: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    expected_reconstruction_dim: int | None = None,
) -> Dict[str, Any]:
    """Validate one annotation against the immutable trajectory token identity."""

    if annotation.get("schema_version") != ANNOTATION_SCHEMA:
        raise ValueError(f"Annotation must use {ANNOTATION_SCHEMA}")
    unknown_fields = sorted(set(annotation) - ANNOTATION_METADATA_FIELDS - set(MERGED_FIELDS))
    if unknown_fields:
        raise ValueError(f"Annotation contains unknown top-level fields: {unknown_fields}")
    if "annotation_metadata" in annotation and not isinstance(
        annotation["annotation_metadata"], Mapping
    ):
        raise ValueError("annotation_metadata must be an object")
    for identity in ("id", "query_id"):
        annotation_identity = str(annotation.get(identity, ""))
        row_identity = str(row.get(identity, ""))
        if not annotation_identity or not row_identity:
            raise ValueError(f"Annotation and source trajectory require non-empty {identity}")
        if annotation_identity != row_identity:
            raise ValueError(f"Annotation {identity} does not match source trajectory")
    token_ids = row.get("output_token_ids")
    expected_token_hash = output_token_ids_sha256(token_ids)
    if annotation.get("output_token_ids_sha256") != expected_token_hash:
        raise ValueError("Annotation output_token_ids_sha256 does not match source trajectory")
    token_count = len(token_ids)
    provenance = _validate_provenance(annotation.get("annotation_provenance"))

    labels: Dict[str, Any] = {}
    for field in TOKEN_FIELDS:
        if field in annotation:
            labels[field] = _numeric_sequence(
                annotation[field], field=field, length=token_count
            )

    for field in ("key_prior_target", "complete_prior_target"):
        if field in labels and not any(value > 0.0 for value in labels[field]):
            raise ValueError(f"{field} must contain at least one positive token")
    if "key_prior_target" in labels and "complete_prior_target" in labels:
        if any(
            key_value > complete_value
            for key_value, complete_value in zip(
                labels["key_prior_target"], labels["complete_prior_target"]
            )
        ):
            raise ValueError("key_prior_target must be a pointwise subset of complete_prior_target")

    explicit_target = labels.get("token_hallucination_target")
    explicit_mask = labels.get("token_hallucination_mask")
    if (explicit_target is None) != (explicit_mask is None):
        raise ValueError(
            "token_hallucination_target and token_hallucination_mask must be provided together"
        )
    if explicit_target is not None:
        if not any(explicit_mask):
            raise ValueError("token_hallucination_mask must supervise at least one token")
        if any(target and not mask for target, mask in zip(explicit_target, explicit_mask)):
            raise ValueError(
                "token_hallucination_target must be zero outside token_hallucination_mask"
            )

    if "path_hallucinated" in annotation:
        path_label = annotation["path_hallucinated"]
        if isinstance(path_label, bool) or path_label not in (0, 1):
            raise ValueError("path_hallucinated must be numeric 0 or 1")
        labels["path_hallucinated"] = int(path_label)
    if "hallucination_onset" in annotation:
        onset = annotation["hallucination_onset"]
        if isinstance(onset, bool) or not isinstance(onset, int):
            raise ValueError("hallucination_onset must be an integer")
        if onset < -1 or onset >= token_count:
            raise ValueError("hallucination_onset is outside [-1, token_count)")
        if "path_hallucinated" not in labels:
            raise ValueError("hallucination_onset requires an explicit path_hallucinated label")
        if (onset == -1) != (labels["path_hallucinated"] == 0):
            raise ValueError("hallucination_onset and path_hallucinated disagree")
        labels["hallucination_onset"] = onset

    for field in GROUP_FIELDS:
        if field in annotation:
            value = annotation[field]
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"{field} must be non-empty when provided")
            labels[field] = value
    if "semantic_id" in labels and not {
        "style_id",
        "domain_id",
    }.intersection(labels):
        raise ValueError("semantic_id requires style_id or domain_id")
    if "style_id" in labels and "semantic_id" not in labels:
        raise ValueError("style_id requires semantic_id")

    if "complete_reconstruction_target" in annotation:
        value = annotation["complete_reconstruction_target"]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise ValueError("complete_reconstruction_target must be a non-empty vector")
        if expected_reconstruction_dim is not None and len(value) != expected_reconstruction_dim:
            raise ValueError(
                "complete_reconstruction_target length mismatch: expected "
                f"{expected_reconstruction_dim}, got {len(value)}"
            )
        labels["complete_reconstruction_target"] = _numeric_sequence(
            value,
            field="complete_reconstruction_target",
            length=len(value),
        )

    if not labels:
        raise ValueError("Annotation does not contain any CLIR supervision field")
    return {
        "labels": labels,
        "annotation_provenance": provenance,
        "output_token_ids_sha256": expected_token_hash,
    }


def _existing_aliases(row: Mapping[str, Any], field: str) -> list[str]:
    aliases = TOKEN_LABEL_ALIASES.get(field, SCALAR_ALIASES.get(field, (field,)))
    return [alias for alias in aliases if alias in row]


def _present_canonical_fields(row: Mapping[str, Any]) -> set[str]:
    return {field for field in MERGED_FIELDS if _existing_aliases(row, field)}


def validate_row_supervision_provenance(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify that every present auxiliary field has exactly one recorded source."""

    present_fields = _present_canonical_fields(row)
    if not present_fields:
        raise ValueError("Row does not contain CLIR supervision")
    provenance_value = row.get("clir_supervision_provenance")
    if not isinstance(provenance_value, Mapping):
        raise ValueError("CLIR supervision is missing row provenance")
    provenance = dict(provenance_value)
    if provenance.get("schema_version") != ROW_PROVENANCE_SCHEMA:
        raise ValueError(f"Row supervision provenance must use {ROW_PROVENANCE_SCHEMA}")
    expected_token_hash = output_token_ids_sha256(row.get("output_token_ids"))
    if provenance.get("output_token_ids_sha256") != expected_token_hash:
        raise ValueError("Row supervision provenance has a stale output_token_ids_sha256")
    sources = provenance.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("Row supervision provenance requires non-empty sources")

    sourced_fields: set[str] = set()
    normalized_sources: list[Dict[str, Any]] = []
    for source_index, source_value in enumerate(sources):
        if not isinstance(source_value, Mapping):
            raise ValueError(f"Row supervision source {source_index} must be an object")
        source = dict(source_value)
        fields = source.get("fields")
        if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)) or not fields:
            raise ValueError(f"Row supervision source {source_index} requires fields")
        normalized_fields = [str(field) for field in fields]
        if len(set(normalized_fields)) != len(normalized_fields):
            raise ValueError(f"Row supervision source {source_index} repeats fields")
        unknown = sorted(set(normalized_fields) - set(MERGED_FIELDS))
        if unknown:
            raise ValueError(f"Row supervision source {source_index} has unknown fields: {unknown}")
        overlap = sourced_fields.intersection(normalized_fields)
        if overlap:
            raise ValueError(f"Row supervision fields have multiple sources: {sorted(overlap)}")
        if not _valid_sha256(source.get("annotations_sha256")):
            raise ValueError(f"Row supervision source {source_index} lacks annotations_sha256")
        _validate_provenance(source.get("annotation_protocol"))
        sourced_fields.update(normalized_fields)
        source["fields"] = sorted(normalized_fields)
        normalized_sources.append(source)
    if sourced_fields != present_fields:
        raise ValueError(
            "Row supervision provenance fields differ from manifest fields: "
            f"present={sorted(present_fields)}, sourced={sorted(sourced_fields)}"
        )
    return {
        **provenance,
        "output_token_ids_sha256": expected_token_hash,
        "sources": normalized_sources,
    }


def merge_supervision_annotations(
    rows: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    annotation_artifact: Mapping[str, Any],
    expected_reconstruction_dim: int | None = None,
    require_all_rows_annotated: bool = False,
) -> list[Dict[str, Any]]:
    """Merge explicit labels without synthesizing or zero-filling missing targets."""

    by_id: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in by_id:
            raise ValueError("Source rows require unique non-empty id values")
        if not str(row.get("query_id", "")):
            raise ValueError(f"Source row {row_id!r} requires a non-empty query_id")
        by_id[row_id] = row

    if not annotations:
        raise ValueError("At least one supervision annotation is required")
    annotations_by_id: Dict[str, Mapping[str, Any]] = {}
    for annotation in annotations:
        row_id = str(annotation.get("id", ""))
        if not row_id or row_id in annotations_by_id:
            raise ValueError("Annotations require unique non-empty id values")
        if row_id not in by_id:
            raise ValueError(f"Annotation references unknown trajectory id {row_id!r}")
        annotations_by_id[row_id] = annotation
    if require_all_rows_annotated and set(annotations_by_id) != set(by_id):
        missing = sorted(set(by_id) - set(annotations_by_id))
        raise ValueError(f"Annotations do not cover every source row; first missing ids: {missing[:5]}")

    output: list[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row_id = str(row["id"])
        existing_fields = _present_canonical_fields(source)
        annotation = annotations_by_id.get(row_id)
        if annotation is None:
            if existing_fields:
                validate_row_supervision_provenance(source)
            output.append(row)
            continue
        validated = validate_supervision_annotation(
            annotation,
            source,
            expected_reconstruction_dim=expected_reconstruction_dim,
        )
        for field, value in validated["labels"].items():
            existing = _existing_aliases(source, field)
            if existing:
                raise ValueError(
                    f"Refusing to overwrite existing supervision for {row_id}/{field}: {existing}"
                )
            row[field] = value
        existing_provenance = (
            validate_row_supervision_provenance(source)
            if existing_fields
            else None
        )
        artifact = dict(annotation_artifact)
        if not _valid_sha256(artifact.get("annotations_sha256")):
            raise ValueError("annotation_artifact requires a valid annotations_sha256")
        for reserved in ("schema_version", "output_token_ids_sha256", "sources", "fields", "annotation_protocol"):
            if reserved in artifact:
                raise ValueError(f"annotation_artifact may not set reserved key {reserved!r}")
        prior_sources = (
            list(existing_provenance["sources"])
            if existing_provenance is not None
            else []
        )
        prior_sources.append(
            {
                **artifact,
                "fields": sorted(validated["labels"]),
                "annotation_protocol": validated["annotation_provenance"],
            }
        )
        row["clir_supervision_provenance"] = {
            "schema_version": ROW_PROVENANCE_SCHEMA,
            "output_token_ids_sha256": validated["output_token_ids_sha256"],
            "sources": prior_sources,
        }
        validate_row_supervision_provenance(row)
        output.append(row)
    return output


def audit_supervision_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_reconstruction_dim: int | None = None,
    require_provenance: bool = False,
) -> Dict[str, Any]:
    """Count actual target presence; absent fields remain absent rather than zero labels."""

    if expected_reconstruction_dim is not None and expected_reconstruction_dim <= 0:
        raise ValueError("expected_reconstruction_dim must be positive")

    field_rows = Counter()
    field_tokens = Counter()
    path_values = Counter()
    positive_onset_rows = 0
    reconstruction_dims = Counter()
    supervised_rows = 0
    provenanced_supervised_rows = 0
    unprovenanced_supervised_rows = 0
    semantic_styles: Dict[str, Counter[str]] = defaultdict(Counter)
    style_semantics: Dict[str, Counter[str]] = defaultdict(Counter)
    joint_prior_rows = 0
    joint_prior_tokens = 0
    explicit_hallucination_tokens = Counter()
    seen_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        row_id = str(row.get("id", ""))
        if not row_id or row_id in seen_ids:
            raise ValueError(f"Coverage audit requires unique row ids; invalid row {row_index}")
        seen_ids.add(row_id)
        token_ids = row.get("output_token_ids", ())
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise ValueError(f"{row_id}/output_token_ids must be a sequence")
        token_count = len(token_ids)
        has_supervision = any(_existing_aliases(row, field) for field in MERGED_FIELDS)
        if token_count:
            output_token_ids_sha256(token_ids)
        elif has_supervision:
            raise ValueError(f"{row_id} has CLIR supervision without output_token_ids")
        if has_supervision:
            supervised_rows += 1
            try:
                validate_row_supervision_provenance(row)
            except ValueError:
                unprovenanced_supervised_rows += 1
                if require_provenance:
                    raise
            else:
                provenanced_supervised_rows += 1
        for field in TOKEN_FIELDS:
            aliases = _existing_aliases(row, field)
            if aliases:
                values = [row[alias] for alias in aliases]
                if any(value != values[0] for value in values[1:]):
                    raise ValueError(f"Conflicting aliases for {row_id}/{field}: {aliases}")
                value = values[0]
                if token_count <= 0:
                    raise ValueError(f"{row_id}/{field} requires output_token_ids")
                _numeric_sequence(value, field=field, length=token_count)
                field_rows[field] += 1
                field_tokens[field] += len(value)
        key_aliases = _existing_aliases(row, "key_prior_target")
        complete_aliases = _existing_aliases(row, "complete_prior_target")
        key_present = bool(key_aliases)
        complete_present = bool(complete_aliases)
        if key_present:
            key_target = _numeric_sequence(
                row[key_aliases[0]], field="key_prior_target", length=token_count
            )
            if not any(value > 0.0 for value in key_target):
                raise ValueError(f"{row_id}/key_prior_target must contain a positive token")
        if complete_present:
            complete_target = _numeric_sequence(
                row[complete_aliases[0]], field="complete_prior_target", length=token_count
            )
            if not any(value > 0.0 for value in complete_target):
                raise ValueError(f"{row_id}/complete_prior_target must contain a positive token")
        if key_present and complete_present:
            if any(
                key_value > complete_value
                for key_value, complete_value in zip(key_target, complete_target)
            ):
                raise ValueError(
                    f"{row_id}/key_prior_target must be a pointwise subset of complete_prior_target"
                )
            joint_prior_rows += 1
            joint_prior_tokens += token_count
        for field in (*SCALAR_FIELDS, *GROUP_FIELDS, *VECTOR_FIELDS):
            aliases = _existing_aliases(row, field)
            if aliases:
                values = [row[alias] for alias in aliases]
                if any(value != values[0] for value in values[1:]):
                    raise ValueError(f"Conflicting aliases for {row_id}/{field}: {aliases}")
                field_rows[field] += 1
                if field == "path_hallucinated":
                    value = values[0]
                    if isinstance(value, bool) or value not in (0, 1):
                        raise ValueError(f"{row_id}/path_hallucinated must be numeric 0 or 1")
                    path_values[str(int(value))] += 1
                elif field == "hallucination_onset":
                    value = values[0]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < -1
                        or value >= token_count
                    ):
                        raise ValueError(f"{row_id}/hallucination_onset is outside token range")
                    if value >= 0:
                        positive_onset_rows += 1
                elif field == "complete_reconstruction_target":
                    value = values[0]
                    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
                        raise ValueError(
                            f"{row_id}/complete_reconstruction_target must be a non-empty vector"
                        )
                    _numeric_sequence(
                        value,
                        field="complete_reconstruction_target",
                        length=len(value),
                    )
                    if (
                        expected_reconstruction_dim is not None
                        and len(value) != expected_reconstruction_dim
                    ):
                        raise ValueError(
                            f"{row_id}/complete_reconstruction_target length mismatch: "
                            f"expected {expected_reconstruction_dim}, got {len(value)}"
                        )
                    reconstruction_dims[len(value)] += 1
                elif field in GROUP_FIELDS:
                    value = values[0]
                    if value is None or (isinstance(value, str) and not value.strip()):
                        raise ValueError(f"{row_id}/{field} must be non-empty")
        semantic_aliases = _existing_aliases(row, "semantic_id")
        style_aliases = _existing_aliases(row, "style_id")
        domain_aliases = _existing_aliases(row, "domain_id")
        if semantic_aliases and not (style_aliases or domain_aliases):
            raise ValueError(f"{row_id}/semantic_id requires style_id or domain_id")
        if style_aliases and not semantic_aliases:
            raise ValueError(f"{row_id}/style_id requires semantic_id")
        effective_style_aliases = style_aliases or domain_aliases
        if semantic_aliases and effective_style_aliases:
            semantic_value = row[semantic_aliases[0]]
            style_value = row[effective_style_aliases[0]]
            if semantic_value is None or (
                isinstance(semantic_value, str) and not semantic_value.strip()
            ):
                raise ValueError(f"{row_id}/semantic_id must be non-empty")
            if style_value is None or (
                isinstance(style_value, str) and not style_value.strip()
            ):
                raise ValueError(f"{row_id}/style_id must be non-empty")
            semantic = repr(semantic_value)
            style = repr(style_value)
            semantic_styles[semantic][style] += 1
            style_semantics[style][semantic] += 1

        path_aliases = _existing_aliases(row, "path_hallucinated")
        onset_aliases = _existing_aliases(row, "hallucination_onset")
        explicit_target_aliases = _existing_aliases(row, "token_hallucination_target")
        explicit_mask_aliases = _existing_aliases(row, "token_hallucination_mask")
        if bool(explicit_target_aliases) != bool(explicit_mask_aliases):
            raise ValueError(
                f"{row_id}/token_hallucination_target and mask must be provided together"
            )
        if explicit_target_aliases:
            targets = _numeric_sequence(
                row[explicit_target_aliases[0]],
                field="token_hallucination_target",
                length=token_count,
            )
            explicit_mask = _numeric_sequence(
                row[explicit_mask_aliases[0]],
                field="token_hallucination_mask",
                length=token_count,
            )
            supervised = sum(int(value) for value in explicit_mask)
            positives = sum(
                int(target)
                for target, known in zip(targets, explicit_mask)
                if known
            )
            if supervised == 0:
                raise ValueError(
                    f"{row_id}/token_hallucination_mask must supervise at least one token"
                )
            if any(target and not known for target, known in zip(targets, explicit_mask)):
                raise ValueError(
                    f"{row_id}/token_hallucination_target must be zero outside its mask"
                )
            if path_aliases and bool(positives) != bool(int(row[path_aliases[0]])):
                raise ValueError(
                    f"{row_id}/explicit token positives disagree with path_hallucinated"
                )
            explicit_hallucination_tokens["supervised"] += supervised
            explicit_hallucination_tokens["positive"] += positives
            explicit_hallucination_tokens["negative"] += supervised - positives
        if onset_aliases and not path_aliases:
            raise ValueError(f"{row_id}/hallucination_onset requires path_hallucinated")
        if onset_aliases:
            path_value = int(row[path_aliases[0]])
            onset_value = int(row[onset_aliases[0]])
            if (onset_value == -1) != (path_value == 0):
                raise ValueError(
                    f"{row_id}/hallucination_onset and path_hallucinated disagree"
                )

    if len(reconstruction_dims) > 1:
        raise ValueError(
            "complete_reconstruction_target dimensions differ across rows: "
            f"{sorted(reconstruction_dims)}"
        )

    positive_consistency_pairs = sum(
        sum(counts.values()) * (sum(counts.values()) - 1) // 2
        - sum(count * (count - 1) // 2 for count in counts.values())
        for counts in semantic_styles.values()
    )
    negative_consistency_pairs = sum(
        sum(counts.values()) * (sum(counts.values()) - 1) // 2
        - sum(count * (count - 1) // 2 for count in counts.values())
        for counts in style_semantics.values()
    )
    eligibility = {
        "consistency": positive_consistency_pairs > 0 and negative_consistency_pairs > 0,
        "path_hallucination": path_values["0"] > 0 and path_values["1"] > 0,
        "onset_localization": positive_onset_rows > 0,
        "token_advantage": field_rows["token_advantage"] > 0,
        "progress": field_rows["progress_targets"] > 0,
        "dual_prior": joint_prior_rows > 0,
        "reconstruction": field_rows["complete_reconstruction_target"] > 0,
    }
    return {
        "schema_version": "clir-supervision-coverage-v1",
        "rows": len(rows),
        "supervised_rows": supervised_rows,
        "provenanced_supervised_rows": provenanced_supervised_rows,
        "unprovenanced_supervised_rows": unprovenanced_supervised_rows,
        "field_rows": {field: int(field_rows[field]) for field in MERGED_FIELDS},
        "field_tokens": {field: int(field_tokens[field]) for field in TOKEN_FIELDS},
        "path_label_values": dict(sorted(path_values.items())),
        "joint_prior_rows": joint_prior_rows,
        "joint_prior_tokens": joint_prior_tokens,
        "positive_onset_rows": positive_onset_rows,
        "explicit_hallucination_tokens": {
            key: int(explicit_hallucination_tokens[key])
            for key in ("supervised", "positive", "negative")
        },
        "reconstruction_dimensions": {
            str(dimension): int(count)
            for dimension, count in sorted(reconstruction_dims.items())
        },
        "consistency_pairs": {
            "positive_same_semantic_different_style": positive_consistency_pairs,
            "negative_different_semantic_same_style": negative_consistency_pairs,
        },
        "eligible_components": eligibility,
        "eligible_component_count": sum(eligibility.values()),
    }


__all__ = [
    "ANNOTATION_SCHEMA",
    "PROVENANCE_SCHEMA",
    "ROW_PROVENANCE_SCHEMA",
    "audit_supervision_coverage",
    "merge_supervision_annotations",
    "output_token_ids_sha256",
    "validate_supervision_annotation",
    "validate_row_supervision_provenance",
]
