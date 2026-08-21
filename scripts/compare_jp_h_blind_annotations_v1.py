#!/usr/bin/env python3
"""Map and compare both score-blind JP-H annotations on exact Phi tokens."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    canonical_json,
    canonical_sha256,
    cohen_kappa,
    content_token_offsets,
    file_sha256,
    map_annotation,
    read_jsonl,
    validate_annotation,
)
from src.clir_stage_a import git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/jp_h_blind_validation_v1/comparison_protocol_v1.json"
)
VIEW_SALT = "clir-jp-h-blind-adjudication-view-v1"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen input hash drifted for {path}: expected {expected_sha256}, got {observed}"
        )


def jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def publish_or_verify(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite drifted artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def positive_token_positions(
    label: Mapping[str, Any],
    *,
    positive_statuses: set[str] | frozenset[str] = frozenset(
        {"contradicted", "unsupported"}
    ),
) -> frozenset[int]:
    positions: set[int] = set()
    for claim in label["claim_reviews"]:
        if str(claim["status"]) not in positive_statuses:
            continue
        start = int(claim["token_start"])
        end = int(claim["token_end_exclusive"])
        if start < 0 or end <= start:
            raise ValueError("Mapped claim has an invalid token span")
        positions.update(range(start, end))
    return frozenset(positions)


def reviewed_negative_token_positions(
    label: Mapping[str, Any],
    *,
    negative_statuses: set[str] | frozenset[str] = frozenset(
        {"supported", "non_claim"}
    ),
) -> frozenset[int]:
    positions: set[int] = set()
    for claim in label["claim_reviews"]:
        if str(claim["status"]) not in negative_statuses:
            continue
        positions.update(
            range(int(claim["token_start"]), int(claim["token_end_exclusive"]))
        )
    return frozenset(positions)


def token_set_jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def compare_labels(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *,
    positive_statuses: set[str] | frozenset[str] = frozenset(
        {"contradicted", "unsupported"}
    ),
    negative_statuses: set[str] | frozenset[str] = frozenset(
        {"supported", "non_claim"}
    ),
) -> dict[str, Any]:
    if primary["item_id"] != secondary["item_id"]:
        raise ValueError("Cannot compare labels with different item IDs")
    if primary["output_token_ids_sha256"] != secondary["output_token_ids_sha256"]:
        raise ValueError("Cannot compare labels with different token identities")
    primary_positive = positive_token_positions(
        primary, positive_statuses=positive_statuses
    )
    secondary_positive = positive_token_positions(
        secondary, positive_statuses=positive_statuses
    )
    primary_negative = reviewed_negative_token_positions(
        primary, negative_statuses=negative_statuses
    )
    secondary_negative = reviewed_negative_token_positions(
        secondary, negative_statuses=negative_statuses
    )
    reasons: list[str] = []
    if primary["path_status"] != secondary["path_status"]:
        reasons.append("path_status_mismatch")
    if primary_positive != secondary_positive:
        reasons.append("exact_positive_token_set_mismatch")
    if (
        primary["path_status"] == secondary["path_status"] == "hallucinated"
        and primary.get("hallucination_onset") != secondary.get("hallucination_onset")
    ):
        reasons.append("onset_token_mismatch")
    if primary_negative != secondary_negative:
        reasons.append("reviewed_negative_token_set_mismatch")
    blocking_reasons = {
        "path_status_mismatch",
        "exact_positive_token_set_mismatch",
    }
    return {
        "schema_version": "clir-jp-h-blind-annotation-comparison-v1",
        "item_id": primary["item_id"],
        "output_token_ids_sha256": primary["output_token_ids_sha256"],
        "primary_path_status": primary["path_status"],
        "secondary_path_status": secondary["path_status"],
        "primary_hallucination_onset": primary.get("hallucination_onset"),
        "secondary_hallucination_onset": secondary.get("hallucination_onset"),
        "primary_positive_tokens": len(primary_positive),
        "secondary_positive_tokens": len(secondary_positive),
        "positive_token_intersection": len(primary_positive & secondary_positive),
        "positive_token_union": len(primary_positive | secondary_positive),
        "positive_token_jaccard": token_set_jaccard(
            primary_positive, secondary_positive
        ),
        "primary_reviewed_negative_tokens": len(primary_negative),
        "secondary_reviewed_negative_tokens": len(secondary_negative),
        "reviewed_negative_token_jaccard": token_set_jaccard(
            primary_negative, secondary_negative
        ),
        "disagreement_reasons": reasons,
        "blocking_disagreement_reasons": [
            reason for reason in reasons if reason in blocking_reasons
        ],
        "requires_adjudication": any(reason in blocking_reasons for reason in reasons),
    }


def assignment(item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{VIEW_SALT}:{item_id}".encode("utf-8")).digest()
    return ("primary", "secondary") if digest[0] % 2 == 0 else ("secondary", "primary")


def visible_annotation(label: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path_status": label["path_status"],
        "earliest_problem_claim_index": label["earliest_problem_claim_index"],
        "hallucination_onset": label.get("hallucination_onset"),
        "confidence": label["confidence"],
        "summary": label["summary"],
        "claim_reviews": [
            {
                key: claim[key]
                for key in (
                    "claim_text",
                    "occurrence",
                    "status",
                    "reason",
                    "char_start",
                    "char_end",
                    "token_start",
                    "token_end_exclusive",
                )
            }
            for claim in label["claim_reviews"]
        ],
    }


def offsets_for(
    *, tokenizer: Any, response: str, output_token_ids: Sequence[int]
) -> list[tuple[int, int]]:
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    encoded_ids = [int(value) for value in encoded["input_ids"]]
    trailing = [int(value) for value in output_token_ids[len(encoded_ids) :]]
    return content_token_offsets(
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


def adjudication_prompt(*, guide: Path, items: Path, output: Path) -> str:
    return f"""# CLIR JP-H blind A/B adjudication prompt v1

You are the third, score-blind adjudicator for a mixed-domain hallucination-localization validation set.

Read only:

- `{guide}`
- `{items}`

Do not read any lineage, primary/secondary source-label files, answer keys, correctness/domain fields, README or
handoff documents, model identities, or prediction/score files. `annotation_a` and `annotation_b` are shuffled.

Each input row has a disagreement that changes path status or at least one positive contradicted/unsupported token.
Independently re-evaluate the complete problem and trajectory. Treat A and B as evidence, not votes. Return a
complete final annotation covering all material claims under the guide; do not merely choose a span mechanically.

After judging each individual row, append and fsync that one decision immediately to `{output}` before moving to
the next row. Never hold multiple finished decisions only in memory. Resume by `item_id`; never rewrite a completed
valid row. Finish in input order and produce exactly one output row per input row.

Each JSONL row must contain exactly:

```json
{{
  "schema_version": "clir-jp-h-blind-adjudication-decision-v1",
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
  "rationale": "why this resolves the target-changing disagreement"
}}
```

`relation_to_views` must be `annotation_a`, `annotation_b`, `revised`, or `unresolved`. Even when choosing A or B,
copy a complete valid `final_annotation`. Every claim quote must be an exact non-empty trajectory substring and use
the correct zero-based occurrence. If the evidence cannot settle the first possible problem, use `unresolved` with
a valid `path_status="uncertain"` annotation instead of forcing a decision.
"""


def median_or_none(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir", default="/prodcpfs/user/panzhixin/hf_cache")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Blind annotation comparison requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-jp-h-blind-comparison-protocol-v1":
        raise ValueError("Unknown JP-H blind comparison protocol schema")
    if protocol.get("prediction_values_may_be_opened_during_comparison") is not False:
        raise ValueError("Comparison must keep prediction values sealed")

    paths: dict[str, Path] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(spec["path"])
        verify(path, str(spec["sha256"]))
        paths[name] = path
    items = read_jsonl(paths["items"])
    lineage = read_jsonl(paths["private_lineage"])
    primary_raw = read_jsonl(paths["primary_raw_labels"])
    secondary_raw = read_jsonl(paths["secondary_raw_labels"])
    expected = int(protocol["token_mapping"]["expected_rows"])
    if not (
        len(items)
        == len(lineage)
        == len(primary_raw)
        == len(secondary_raw)
        == expected
    ):
        raise ValueError("Blind comparison inputs are not exactly 96 aligned rows")

    from transformers import AutoTokenizer

    token_cfg = protocol["token_mapping"]
    tokenizer = AutoTokenizer.from_pretrained(
        token_cfg["model_id"],
        revision=token_cfg["tokenizer_revision"],
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=bool(token_cfg["trust_remote_code"]),
        use_fast=True,
    )
    if bool(token_cfg["require_fast_tokenizer"]) and not tokenizer.is_fast:
        raise ValueError("Exact claim mapping requires a fast Phi tokenizer")

    protocol_sha256 = file_sha256(protocol_path)
    positive_statuses = set(protocol["target_semantics"]["positive_claim_statuses"])
    negative_statuses = set(protocol["target_semantics"]["negative_claim_statuses"])
    primary: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for row_number, (item, identity, left_raw, right_raw) in enumerate(
        zip(items, lineage, primary_raw, secondary_raw), start=1
    ):
        item_id = str(item["item_id"])
        if not (
            item_id
            == str(identity["item_id"])
            == str(left_raw["item_id"])
            == str(right_raw["item_id"])
        ):
            raise ValueError(f"Blind item identity/order drifted at row {row_number}")
        validate_annotation(left_raw, item)
        validate_annotation(right_raw, item)
        response = str(item["trajectory"])
        if response != identity["trajectory"] or item["problem"] != identity["problem"]:
            raise ValueError(f"Private lineage text drifted at row {row_number}")
        output_ids = [int(value) for value in identity["output_token_ids"]]
        if canonical_sha256(output_ids) != identity["output_token_ids_sha256"]:
            raise ValueError(f"Saved output-token identity drifted at row {row_number}")
        offsets = offsets_for(
            tokenizer=tokenizer,
            response=response,
            output_token_ids=output_ids,
        )
        left = map_annotation(
            left_raw,
            item,
            identity,
            offsets=offsets,
            annotator={
                "role": "independent_primary",
                "raw_labels_sha256": protocol["inputs"]["primary_raw_labels"]["sha256"],
                "comparison_protocol_sha256": protocol_sha256,
                "claim_level_gold": False,
            },
        )
        right = map_annotation(
            right_raw,
            item,
            identity,
            offsets=offsets,
            annotator={
                "role": "independent_secondary",
                "identity": "externally_supplied_blind_annotator_not_recorded",
                "raw_labels_sha256": protocol["inputs"]["secondary_raw_labels"]["sha256"],
                "comparison_protocol_sha256": protocol_sha256,
                "claim_level_gold": False,
            },
        )
        primary.append(left)
        secondary.append(right)
        comparisons.append(
            compare_labels(
                left,
                right,
                positive_statuses=positive_statuses,
                negative_statuses=negative_statuses,
            )
        )

    outputs = {name: resolve(path) for name, path in protocol["outputs"].items()}
    publish_or_verify(outputs["primary_mapped_labels"], jsonl_payload(primary))
    publish_or_verify(outputs["secondary_mapped_labels"], jsonl_payload(secondary))
    publish_or_verify(outputs["row_comparisons"], jsonl_payload(comparisons))

    adjudication_items: list[dict[str, Any]] = []
    adjudication_lineage: list[dict[str, Any]] = []
    for item, left, right, compared in zip(items, primary, secondary, comparisons):
        if not compared["requires_adjudication"]:
            continue
        role_a, role_b = assignment(str(item["item_id"]))
        by_role = {"primary": left, "secondary": right}
        adjudication_items.append(
            {
                "schema_version": "clir-jp-h-blind-adjudication-item-v1",
                "item_id": item["item_id"],
                "problem": item["problem"],
                "trajectory": item["trajectory"],
                "disagreement_reasons": compared["blocking_disagreement_reasons"],
                "annotation_a": visible_annotation(by_role[role_a]),
                "annotation_b": visible_annotation(by_role[role_b]),
            }
        )
        adjudication_lineage.append(
            {
                "schema_version": "clir-jp-h-blind-adjudication-lineage-v1",
                "item_id": item["item_id"],
                "annotation_a_role": role_a,
                "annotation_b_role": role_b,
                "primary_label_sha256": canonical_sha256(left),
                "secondary_label_sha256": canonical_sha256(right),
                "comparison_sha256": canonical_sha256(compared),
            }
        )
    publish_or_verify(outputs["adjudication_items"], jsonl_payload(adjudication_items))
    publish_or_verify(
        outputs["adjudication_lineage"], jsonl_payload(adjudication_lineage)
    )
    publish_or_verify(
        outputs["adjudication_prompt"],
        adjudication_prompt(
            guide=paths["annotation_guide"],
            items=outputs["adjudication_items"],
            output=outputs["adjudication_resolutions_raw"],
        ),
    )

    primary_statuses = [str(row["path_status"]) for row in primary]
    secondary_statuses = [str(row["path_status"]) for row in secondary]
    path_agreement = sum(a == b for a, b in zip(primary_statuses, secondary_statuses))
    reason_counts = Counter(
        reason for row in comparisons for reason in row["disagreement_reasons"]
    )
    positive_jaccards = [float(row["positive_token_jaccard"]) for row in comparisons]
    negative_jaccards = [
        float(row["reviewed_negative_token_jaccard"]) for row in comparisons
    ]
    confusion = Counter(
        f"{left}->{right}" for left, right in zip(primary_statuses, secondary_statuses)
    )
    report = {
        "schema_version": "clir-jp-h-blind-agreement-report-v1",
        "status": "awaiting_score_blind_adjudication",
        "evidence_tier": "blind_validation_annotation_audit",
        "code": code,
        "comparison_protocol": str(protocol_path.relative_to(ROOT)),
        "comparison_protocol_sha256": protocol_sha256,
        "rows": expected,
        "path_status_counts": {
            "primary": dict(sorted(Counter(primary_statuses).items())),
            "secondary": dict(sorted(Counter(secondary_statuses).items())),
        },
        "path_confusion_primary_to_secondary": dict(sorted(confusion.items())),
        "path_agreement_count": path_agreement,
        "path_agreement_rate": path_agreement / expected,
        "path_cohen_kappa": cohen_kappa(primary_statuses, secondary_statuses),
        "exact_positive_token_set_agreement_count": sum(
            row["positive_token_jaccard"] == 1.0 for row in comparisons
        ),
        "positive_token_jaccard": {
            "mean": sum(positive_jaccards) / expected,
            "median": median_or_none(positive_jaccards),
            "minimum": min(positive_jaccards),
        },
        "reviewed_negative_token_jaccard": {
            "mean": sum(negative_jaccards) / expected,
            "median": median_or_none(negative_jaccards),
            "minimum": min(negative_jaccards),
        },
        "disagreement_reason_counts": dict(sorted(reason_counts.items())),
        "rows_requiring_adjudication": len(adjudication_items),
        "blocking_policy": protocol["comparison"]["blocking_disagreements"],
        "post_adjudication_evaluation_views": protocol[
            "post_adjudication_evaluation_views"
        ],
        "artifacts": {
            "primary_mapped_labels": str(
                outputs["primary_mapped_labels"].relative_to(ROOT)
            ),
            "primary_mapped_labels_sha256": file_sha256(
                outputs["primary_mapped_labels"]
            ),
            "secondary_mapped_labels": str(
                outputs["secondary_mapped_labels"].relative_to(ROOT)
            ),
            "secondary_mapped_labels_sha256": file_sha256(
                outputs["secondary_mapped_labels"]
            ),
            "row_comparisons": str(outputs["row_comparisons"].relative_to(ROOT)),
            "row_comparisons_sha256": file_sha256(outputs["row_comparisons"]),
            "adjudication_items": str(
                outputs["adjudication_items"].relative_to(ROOT)
            ),
            "adjudication_items_sha256": file_sha256(
                outputs["adjudication_items"]
            ),
            "adjudication_lineage": str(
                outputs["adjudication_lineage"].relative_to(ROOT)
            ),
            "adjudication_lineage_sha256": file_sha256(
                outputs["adjudication_lineage"]
            ),
            "adjudication_prompt": str(
                outputs["adjudication_prompt"].relative_to(ROOT)
            ),
            "adjudication_prompt_sha256": file_sha256(
                outputs["adjudication_prompt"]
            ),
        },
        "prediction_values_accessed": False,
        "correctness_or_answer_key_visible_to_adjudicator": False,
        "adoption_gate_run": False,
        "automatic_score_coupling": False,
    }
    publish_or_verify(
        outputs["agreement_report"],
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
