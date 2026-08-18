#!/usr/bin/env python3
"""Map blind secondary labels, report agreement, and build A/B adjudication items."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    canonical_json,
    canonical_sha256,
    cohen_kappa,
    compare_mapped_path_labels,
    content_token_offsets,
    file_sha256,
    map_annotation,
    mapped_earliest_problem_claim,
    read_jsonl,
    validate_annotation,
)
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_COMPARISON = (
    ROOT / "configs/hallucination_localization_v1/comparison_protocol_v1.json"
)
VIEW_SALT = "clir-hallucination-adjudication-view-v1"
REASON_POLICY_KEYS = {
    "path_status": "adjudicate_path_status_disagreement",
    "onset_token": "adjudicate_common_positive_onset_token_disagreement",
    "earliest_problem_claim_span": (
        "adjudicate_common_positive_earliest_claim_span_disagreement"
    ),
    "earliest_problem_claim_status": (
        "adjudicate_common_positive_problem_status_disagreement"
    ),
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def publish_or_verify(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite drifted artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def verify_input(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen input hash drifted for {path}: expected {expected_sha256}, got {observed}"
        )


def visible_annotation(label: Mapping[str, Any]) -> dict[str, Any]:
    claims = []
    for claim in label["claim_reviews"]:
        claims.append(
            {
                "claim_text": claim["claim_text"],
                "occurrence": claim["occurrence"],
                "status": claim["status"],
                "reason": claim["reason"],
                "char_start": claim["char_start"],
                "char_end": claim["char_end"],
                "token_start": claim["token_start"],
                "token_end_exclusive": claim["token_end_exclusive"],
            }
        )
    return {
        "path_status": label["path_status"],
        "earliest_problem_claim_index": label["earliest_problem_claim_index"],
        "hallucination_onset": label.get("hallucination_onset"),
        "confidence": label["confidence"],
        "summary": label["summary"],
        "claim_reviews": claims,
    }


def assignment(item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{VIEW_SALT}:{item_id}".encode("utf-8")).digest()
    return ("primary", "secondary") if digest[0] % 2 == 0 else ("secondary", "primary")


def onset_metrics(distances: Sequence[int]) -> dict[str, Any]:
    if not distances:
        return {
            "count": 0,
            "mean_absolute_token_distance": None,
            "median_absolute_token_distance": None,
            "max_absolute_token_distance": None,
            "exact": {"count": 0, "rate": None},
            "within_1": {"count": 0, "rate": None},
            "within_3": {"count": 0, "rate": None},
            "within_5": {"count": 0, "rate": None},
        }
    ordered = sorted(int(value) for value in distances)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    result: dict[str, Any] = {
        "count": len(ordered),
        "mean_absolute_token_distance": sum(ordered) / len(ordered),
        "median_absolute_token_distance": median,
        "max_absolute_token_distance": max(ordered),
    }
    for name, tolerance in (
        ("exact", 0),
        ("within_1", 1),
        ("within_3", 3),
        ("within_5", 5),
    ):
        count = sum(value <= tolerance for value in ordered)
        result[name] = {"count": count, "rate": count / len(ordered)}
    return result


def adjudication_prompt(*, guide: Path, items: Path, output: Path) -> str:
    return f"""# Blind A/B adjudication prompt — hallucination localization v1

You are adjudicating only trajectory items where two independent annotations disagree about path status, the first
material problem claim, or its exact Phi-token onset.

Read only:

- `{guide}`
- `{items}`

Do not read adjudication lineage, primary/secondary source files, correctness, reference answers, source/query IDs,
README/handoff, or model identities. `annotation_a` and `annotation_b` are deterministically shuffled per item.

For each row, independently re-evaluate the supplied problem and trajectory. The two annotations are evidence, not
votes. A different valid derivation and a derivable explanatory step are allowed. Select the earliest material
`contradicted` or `unsupported` claim; use `uncertain` if it cannot be resolved from the item.

Write one JSONL row per item, in input order, to:

`{output}`

Each row must have exactly this form:

```json
{{
  "schema_version": "clir-hallucination-adjudication-decision-v1",
  "item_id": "copy exactly",
  "relation_to_views": "annotation_a",
  "final_annotation": {{
    "item_id": "copy exactly",
    "claim_reviews": [
      {{
        "claim_text": "exact trajectory substring",
        "occurrence": 0,
        "status": "supported",
        "reason": "item-specific evidence"
      }}
    ],
    "path_status": "clean",
    "earliest_problem_claim_index": null,
    "confidence": "high",
    "summary": "item-specific final judgment"
  }},
  "rationale": "why this resolves the disagreement"
}}
```

`relation_to_views` must be one of `annotation_a`, `annotation_b`, `revised`, or `unresolved`. Even when selecting an
existing view, copy a complete valid `final_annotation`. Every claim quote must be an exact trajectory substring and
the earliest index must follow the annotation guide. Use `unresolved` with a valid `path_status=uncertain` annotation
instead of forcing a class.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--cache-dir", default="/prodcpfs/user/panzhixin/hf_cache")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Annotation comparison requires a clean committed worktree")
    comparison_path = args.comparison.resolve()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    inputs = comparison["inputs"]
    paths = {
        key: resolve(value)
        for key, value in inputs.items()
        if not key.endswith("_sha256")
    }
    for key, path in paths.items():
        verify_input(path, str(inputs[f"{key}_sha256"]))
    base_protocol_path = resolve(comparison["base_protocol"])
    verify_input(base_protocol_path, comparison["base_protocol_sha256"])
    base_protocol = json.loads(base_protocol_path.read_text(encoding="utf-8"))

    items = read_jsonl(paths["items"])
    lineage = read_jsonl(paths["lineage"])
    primary = read_jsonl(paths["primary_labels"])
    secondary_raw = read_jsonl(paths["secondary_raw_labels"])
    if not (len(items) == len(lineage) == len(primary) == len(secondary_raw) == 64):
        raise ValueError("Comparison input row counts must all equal 64")
    source_by_id = {
        str(row["id"]): row for row in read_jsonl(paths["source_manifest"])
    }

    from transformers import AutoTokenizer

    token_cfg = base_protocol["token_mapping"]
    tokenizer = AutoTokenizer.from_pretrained(
        token_cfg["model_id"],
        revision=token_cfg["tokenizer_revision"],
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=bool(token_cfg["trust_remote_code"]),
    )
    if not tokenizer.is_fast:
        raise ValueError("Exact onset comparison requires a fast tokenizer")

    secondary: list[dict[str, Any]] = []
    confidence_counts: Counter[str] = Counter()
    secondary_path_counts: Counter[str] = Counter()
    correctness_cross_tab: Counter[str] = Counter()
    secondary_onsets: list[int] = []
    for row_number, (item, identity, primary_label, raw_label) in enumerate(
        zip(items, lineage, primary, secondary_raw), start=1
    ):
        item_id = item["item_id"]
        if not (
            item_id
            == identity["item_id"]
            == primary_label["item_id"]
            == raw_label["item_id"]
        ):
            raise ValueError(f"Comparison identity/order drifted at row {row_number}")
        validate_annotation(raw_label, item)
        source = source_by_id[identity["source_id"]]
        response = str(source["response"])
        if response != item["trajectory"]:
            raise ValueError("Blind trajectory differs from frozen source")
        output_token_ids = [int(value) for value in source["output_token_ids"]]
        if canonical_sha256(output_token_ids) != identity["output_token_ids_sha256"]:
            raise ValueError("Output-token identity hash drifted")
        if primary_label["output_token_ids_sha256"] != identity["output_token_ids_sha256"]:
            raise ValueError("Primary mapped token identity drifted")
        mapped_earliest_problem_claim(primary_label)
        encoded = tokenizer(
            response,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = [int(value) for value in encoded["input_ids"]]
        trailing = output_token_ids[len(encoded_ids) :]
        offsets = content_token_offsets(
            response=response,
            output_token_ids=output_token_ids,
            encoded_token_ids=encoded_ids,
            offsets=encoded["offset_mapping"],
            trailing_token_decodes_to_empty=[
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                == ""
                for token_id in trailing
            ],
        )
        mapped = map_annotation(
            raw_label,
            item,
            identity,
            offsets=offsets,
            annotator={
                "role": "independent_secondary",
                "identity": "externally_supplied_blind_annotator_not_recorded",
                "raw_labels_sha256": inputs["secondary_raw_labels_sha256"],
                "blind_items_sha256": inputs["items_sha256"],
            },
        )
        secondary.append(mapped)
        confidence_counts[mapped["confidence"]] += 1
        secondary_path_counts[mapped["path_status"]] += 1
        correctness_cross_tab[
            f"correctness_{identity['correctness']}__{mapped['path_status']}"
        ] += 1
        if mapped.get("path_hallucinated") == 1:
            secondary_onsets.append(int(mapped["hallucination_onset"]))

    outputs = {
        key: resolve(value) for key, value in comparison["outputs"].items()
    }
    publish_or_verify(outputs["secondary_labels"], jsonl_payload(secondary))

    primary_statuses = [str(row["path_status"]) for row in primary]
    secondary_statuses = [str(row["path_status"]) for row in secondary]
    confusion = Counter(
        f"{left}->{right}" for left, right in zip(primary_statuses, secondary_statuses)
    )
    comparisons = [
        compare_mapped_path_labels(left, right)
        for left, right in zip(primary, secondary)
    ]
    comparison_policy = comparison["comparison"]
    for compared in comparisons:
        compared["requires_adjudication"] = any(
            bool(comparison_policy[REASON_POLICY_KEYS[reason]])
            for reason in compared["disagreement_reasons"]
        )
    reason_counts = Counter(
        reason
        for row in comparisons
        for reason in row["disagreement_reasons"]
    )
    path_agreements = sum(row["path_agreement"] for row in comparisons)
    common_positive_distances = [
        abs(int(left["hallucination_onset"]) - int(right["hallucination_onset"]))
        for left, right in zip(primary, secondary)
        if left["path_status"] == right["path_status"] == "hallucinated"
    ]

    adjudication_items: list[dict[str, Any]] = []
    adjudication_lineage: list[dict[str, Any]] = []
    consensus_counts: Counter[str] = Counter()
    for item, left, right, compared in zip(items, primary, secondary, comparisons):
        if not compared["requires_adjudication"]:
            consensus_counts[str(left["path_status"])] += 1
            continue
        first_role, second_role = assignment(item["item_id"])
        role_rows = {"primary": left, "secondary": right}
        adjudication_items.append(
            {
                "schema_version": "clir-hallucination-adjudication-item-v1",
                "item_id": item["item_id"],
                "problem": item["problem"],
                "trajectory": item["trajectory"],
                "disagreement_reasons": compared["disagreement_reasons"],
                "annotation_a": visible_annotation(role_rows[first_role]),
                "annotation_b": visible_annotation(role_rows[second_role]),
            }
        )
        adjudication_lineage.append(
            {
                "schema_version": "clir-hallucination-adjudication-lineage-v1",
                "item_id": item["item_id"],
                "annotation_a_role": first_role,
                "annotation_b_role": second_role,
                "primary_label_sha256": canonical_sha256(left),
                "secondary_label_sha256": canonical_sha256(right),
                "comparison_sha256": canonical_sha256(compared),
            }
        )
    publish_or_verify(outputs["adjudication_items"], jsonl_payload(adjudication_items))
    publish_or_verify(
        outputs["adjudication_lineage"], jsonl_payload(adjudication_lineage)
    )
    prompt = adjudication_prompt(
        guide=resolve(base_protocol["annotation"]["guide"]),
        items=outputs["adjudication_items"],
        output=outputs["adjudications_raw"],
    )
    publish_or_verify(outputs["adjudication_prompt"], prompt)

    secondary_report = {
        "schema_version": "clir-hallucination-secondary-report-v1",
        "evidence_tier": "pipeline_pilot",
        "code": code,
        "comparison_protocol_sha256": file_sha256(comparison_path),
        "base_protocol_sha256": file_sha256(base_protocol_path),
        "rows": len(secondary),
        "parse_and_token_map_valid": len(secondary),
        "path_status_counts": dict(sorted(secondary_path_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "hallucinated_onset": {
            "count": len(secondary_onsets),
            "min": min(secondary_onsets) if secondary_onsets else None,
            "max": max(secondary_onsets) if secondary_onsets else None,
            "mean": (
                sum(secondary_onsets) / len(secondary_onsets)
                if secondary_onsets
                else None
            ),
        },
        "correctness_path_cross_tab_private_diagnostic": dict(
            sorted(correctness_cross_tab.items())
        ),
        "secondary_raw_labels_sha256": file_sha256(paths["secondary_raw_labels"]),
        "secondary_labels": str(outputs["secondary_labels"].relative_to(ROOT)),
        "secondary_labels_sha256": file_sha256(outputs["secondary_labels"]),
        "secondary_was_blind_to_primary_and_correctness": True,
        "secondary_annotator_identity_recorded": False,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    publish_or_verify(
        outputs["secondary_report"],
        json.dumps(secondary_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    agreement_report = {
        "schema_version": "clir-hallucination-agreement-report-v1",
        "evidence_tier": "pipeline_pilot",
        "code": code,
        "comparison_protocol": str(comparison_path.relative_to(ROOT)),
        "comparison_protocol_sha256": file_sha256(comparison_path),
        "rows": len(primary),
        "primary_path_status_counts": dict(sorted(Counter(primary_statuses).items())),
        "secondary_path_status_counts": dict(
            sorted(Counter(secondary_statuses).items())
        ),
        "path_confusion_primary_to_secondary": dict(sorted(confusion.items())),
        "path_agreement_count": path_agreements,
        "path_agreement_rate": path_agreements / len(primary),
        "cohen_kappa": cohen_kappa(primary_statuses, secondary_statuses),
        "common_hallucinated_onset_metrics": onset_metrics(
            common_positive_distances
        ),
        "disagreement_reason_counts": dict(sorted(reason_counts.items())),
        "rows_requiring_adjudication": len(adjudication_items),
        "consensus_usable_path_counts_before_adjudication": dict(
            sorted(consensus_counts.items())
        ),
        "primary_labels_sha256": file_sha256(paths["primary_labels"]),
        "secondary_raw_labels_sha256": file_sha256(paths["secondary_raw_labels"]),
        "secondary_mapped_labels_sha256": file_sha256(outputs["secondary_labels"]),
        "adjudication_items": str(outputs["adjudication_items"].relative_to(ROOT)),
        "adjudication_items_sha256": file_sha256(outputs["adjudication_items"]),
        "adjudication_lineage_sha256": file_sha256(
            outputs["adjudication_lineage"]
        ),
        "adjudication_prompt_sha256": file_sha256(outputs["adjudication_prompt"]),
        "adjudication_package_contains_correctness": False,
        "calibration_status": "awaiting_adjudication",
        "training_labels_published": False,
        "trainability_gate_passed": False,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    publish_or_verify(
        outputs["agreement_report"],
        json.dumps(agreement_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(agreement_report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
