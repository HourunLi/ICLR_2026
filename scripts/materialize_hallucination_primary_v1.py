#!/usr/bin/env python3
"""Map primary exact quotes to Phi tokens and publish a blind secondary package."""

from __future__ import annotations

import argparse
from collections import Counter
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
    content_token_offsets,
    file_sha256,
    map_annotation,
    read_jsonl,
)


DEFAULT_PROTOCOL = ROOT / "configs/hallucination_localization_v1/protocol_v1.json"


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


def secondary_prompt(
    *,
    guide_path: Path,
    items_path: Path,
    output_path: Path,
) -> str:
    return f"""# Independent secondary annotation prompt — hallucination localization v1

You are the independent second annotator for 64 blind trajectory items.

Read only these task files:

- `{guide_path}`
- `{items_path}`

Do not read the selection manifest, annotation lineage, primary labels, primary model run, package reports,
README/handoff discussion, source correctness, reference answers, or any previous annotation. Do not browse for a
hidden answer key. The problem and trajectory in each blind item are the complete annotation evidence.

Follow the guide exactly. For each input row, output one strict JSON object with exactly these top-level keys:

`item_id`, `claim_reviews`, `path_status`, `earliest_problem_claim_index`, `confidence`, `summary`.

Important requirements:

1. Preserve the 64-item input order and copy every `item_id` exactly.
2. Write JSONL: exactly 64 one-line JSON objects, with no Markdown fences and no extra prose.
3. Every `claim_text` must be an exact non-empty substring of that item's `trajectory`; use zero-based
   `occurrence` when a quote repeats.
4. Order `claim_reviews` by response position. For `hallucinated`, the earliest index must point to the first
   `contradicted` or `unsupported` reviewed claim.
5. Do not infer path status from whether the final answer looks correct. A valid different derivation is allowed;
   an unsupported or false intermediate claim is not.
6. Use `uncertain` instead of forcing `clean` whenever the earliest possible bad claim cannot be resolved.

Write the completed file to:

`{output_path}`

Then run this structure-only blind validator:

```bash
/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  {ROOT / 'scripts/validate_hallucination_annotations_v1.py'} \
  --items {items_path} \
  --labels {output_path}
```

Fix only schema, ordering, or exact-quote errors reported by the validator; do not consult prohibited files. Stop only
after it prints `\"status\": \"passed\"`, and report the output path and SHA256.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir", default="/prodcpfs/user/panzhixin/hf_cache")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    outputs = protocol["outputs"]
    items_path = resolve(outputs["annotation_items"])
    lineage_path = resolve(outputs["annotation_lineage"])
    source_path = resolve(protocol["source"]["manifest"])
    run_root = resolve(outputs["primary_run_root"])
    raw_results_path = run_root / "all_results.jsonl"
    run_path = run_root / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    if run_record.get("status") != "complete":
        raise ValueError("Primary annotation run is incomplete")
    if run_record["frozen"]["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError("Primary run protocol hash drifted")
    if run_record["results_sha256"] != file_sha256(raw_results_path):
        raise ValueError("Primary result hash drifted")

    resolved_results_path = run_root / "all_results_resolved_v1.jsonl"
    repair_report_path = run_root / "contract_repair_report_v1.json"
    if resolved_results_path.exists() != repair_report_path.exists():
        raise ValueError("Resolved results and contract-repair report must coexist")
    repair_report = None
    results_path = raw_results_path
    if resolved_results_path.exists():
        repair_report = json.loads(repair_report_path.read_text(encoding="utf-8"))
        if repair_report["raw_results_sha256"] != run_record["results_sha256"]:
            raise ValueError("Contract repair is not bound to the frozen raw results")
        if repair_report["resolved_results_sha256"] != file_sha256(
            resolved_results_path
        ):
            raise ValueError("Resolved primary-result hash drifted")
        if repair_report.get("semantic_judgments_changed") is not False:
            raise ValueError("Contract repair did not preserve semantic judgments")
        results_path = resolved_results_path

    items = read_jsonl(items_path)
    lineage = read_jsonl(lineage_path)
    results = read_jsonl(results_path)
    source_by_id = {str(row["id"]): row for row in read_jsonl(source_path)}
    if not (len(items) == len(lineage) == len(results) == 64):
        raise ValueError("Primary materialization row count drifted")

    from transformers import AutoTokenizer

    token_cfg = protocol["token_mapping"]
    tokenizer = AutoTokenizer.from_pretrained(
        token_cfg["model_id"],
        revision=token_cfg["tokenizer_revision"],
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=bool(token_cfg["trust_remote_code"]),
    )
    if not tokenizer.is_fast:
        raise ValueError("Exact onset mapping requires a fast tokenizer")

    labels: list[dict[str, Any]] = []
    path_counts: Counter[str] = Counter()
    cross_tab: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    onsets: list[int] = []
    primary_cfg = protocol["primary_annotator"]
    for row_number, (item, identity, result) in enumerate(
        zip(items, lineage, results), start=1
    ):
        if not (
            item["item_id"] == identity["item_id"] == result["item_id"]
        ):
            raise ValueError(f"Primary row {row_number} identity/order drifted")
        if not result["parse_valid"] or result["annotation"] is None:
            raise ValueError(
                f"Primary row {row_number}/{item['item_id']} is invalid: "
                f"{result['parse_error']}"
            )
        source = source_by_id[identity["source_id"]]
        response = str(source["response"])
        if response != item["trajectory"]:
            raise ValueError("Blind trajectory differs from frozen source")
        output_token_ids = [int(value) for value in source["output_token_ids"]]
        if canonical_sha256(output_token_ids) != identity["output_token_ids_sha256"]:
            raise ValueError("Output-token identity hash drifted")
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
        contract_repair = result.get("contract_repair")
        label = map_annotation(
            result["annotation"],
            item,
            identity,
            offsets=offsets,
            annotator={
                "role": "candidate_primary",
                "model_id": primary_cfg["model_id"],
                "model_revision": primary_cfg["model_revision"],
                "run_results_sha256": run_record["results_sha256"],
                "resolved_results_sha256": file_sha256(results_path),
                "completion_sha256": result["completion_sha256"],
                "contract_repair_applied": contract_repair is not None,
                "contract_repair_semantic_decision_signature_sha256": (
                    contract_repair["semantic_decision_signature_sha256"]
                    if contract_repair is not None
                    else None
                ),
            },
        )
        labels.append(label)
        path_counts[label["path_status"]] += 1
        cross_tab[
            f"correctness_{identity['correctness']}__{label['path_status']}"
        ] += 1
        confidence_counts[label["confidence"]] += 1
        if label.get("path_hallucinated") == 1:
            onsets.append(int(label["hallucination_onset"]))

    primary_labels_path = resolve(outputs["primary_labels"])
    publish_or_verify(primary_labels_path, jsonl_payload(labels))
    secondary_items_path = resolve(outputs["secondary_items"])
    publish_or_verify(secondary_items_path, items_path.read_text(encoding="utf-8"))
    if file_sha256(secondary_items_path) != file_sha256(items_path):
        raise ValueError("Secondary blind items differ from primary blind items")
    secondary_output_path = (
        ROOT
        / "configs/hallucination_localization_v1/labels_secondary_raw_v1.jsonl"
    )
    secondary_prompt_path = resolve(outputs["secondary_prompt"])
    prompt = secondary_prompt(
        guide_path=resolve(protocol["annotation"]["guide"]),
        items_path=secondary_items_path,
        output_path=secondary_output_path,
    )
    publish_or_verify(secondary_prompt_path, prompt)

    positive = path_counts["hallucinated"]
    clean = path_counts["clean"]
    report = {
        "schema_version": "clir-hallucination-primary-report-v1",
        "evidence_tier": "pipeline_pilot",
        "protocol_sha256": file_sha256(protocol_path),
        "primary_run": str(run_path.relative_to(ROOT)),
        "primary_run_sha256": file_sha256(run_path),
        "primary_raw_results_sha256": file_sha256(raw_results_path),
        "primary_resolved_results": str(results_path.relative_to(ROOT)),
        "primary_resolved_results_sha256": file_sha256(results_path),
        "primary_contract_repair": (
            str(repair_report_path.relative_to(ROOT))
            if repair_report is not None
            else None
        ),
        "primary_contract_repair_sha256": (
            file_sha256(repair_report_path) if repair_report is not None else None
        ),
        "primary_contract_repaired_rows": (
            int(repair_report["repaired_rows"]) if repair_report is not None else 0
        ),
        "rows": len(labels),
        "parse_and_token_map_valid": len(labels),
        "path_status_counts": dict(sorted(path_counts.items())),
        "correctness_path_cross_tab_private_diagnostic": dict(sorted(cross_tab.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "hallucinated_onset": {
            "count": len(onsets),
            "min": min(onsets) if onsets else None,
            "max": max(onsets) if onsets else None,
            "mean": sum(onsets) / len(onsets) if onsets else None,
        },
        "primary_preliminary_class_gate": {
            "minimum_hallucinated": 12,
            "minimum_clean": 12,
            "passed": positive >= 12 and clean >= 12,
        },
        "primary_labels": str(primary_labels_path.relative_to(ROOT)),
        "primary_labels_sha256": file_sha256(primary_labels_path),
        "secondary_items": str(secondary_items_path.relative_to(ROOT)),
        "secondary_items_sha256": file_sha256(secondary_items_path),
        "secondary_prompt": str(secondary_prompt_path.relative_to(ROOT)),
        "secondary_prompt_sha256": file_sha256(secondary_prompt_path),
        "primary_is_candidate_not_gold": True,
        "secondary_is_blind_to_primary_and_correctness": True,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    if not report["primary_preliminary_class_gate"]["passed"]:
        raise RuntimeError(
            "Primary labels lack the preregistered preliminary path-class support; "
            "secondary package was retained for inspection but should not be dispatched"
        )
    report_path = resolve(outputs["primary_report"])
    publish_or_verify(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
