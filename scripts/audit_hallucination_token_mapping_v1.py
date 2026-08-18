#!/usr/bin/env python3
"""Preflight exact character-to-frozen-token mapping for all 64 blind items."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    canonical_sha256,
    content_token_offsets,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = ROOT / "configs/hallucination_localization_v1/protocol_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "configs/hallucination_localization_v1/token_mapping_preflight_v1.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default="/prodcpfs/user/panzhixin/hf_cache")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_path = resolve(protocol["source"]["manifest"])
    items_path = resolve(protocol["outputs"]["annotation_items"])
    lineage_path = resolve(protocol["outputs"]["annotation_lineage"])
    if file_sha256(source_path) != protocol["source"]["sha256"]:
        raise ValueError("Source manifest hash drifted")
    items = read_jsonl(items_path)
    lineage = read_jsonl(lineage_path)
    source_by_id = {str(row["id"]): row for row in read_jsonl(source_path)}
    if len(items) != 64 or len(lineage) != 64:
        raise ValueError("Annotation item/lineage row count drifted")

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
        raise ValueError("Exact offset mapping requires a fast tokenizer")

    trailing_sequences: Counter[str] = Counter()
    visible_counts: list[int] = []
    for item, identity in zip(items, lineage):
        if item["item_id"] != identity["item_id"]:
            raise ValueError("Item/lineage order or identity drifted")
        source = source_by_id[identity["source_id"]]
        response = str(source["response"])
        if item["trajectory"] != response or item["problem"] != source["question"]:
            raise ValueError("Blind item text differs from its frozen source row")
        output_token_ids = [int(value) for value in source["output_token_ids"]]
        if canonical_sha256(output_token_ids) != identity["output_token_ids_sha256"]:
            raise ValueError("Frozen output-token hash differs from lineage")
        encoded = tokenizer(
            response,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = [int(value) for value in encoded["input_ids"]]
        trailing = output_token_ids[len(encoded_ids) :]
        trailing_empty = [
            tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            == ""
            for token_id in trailing
        ]
        offsets = content_token_offsets(
            response=response,
            output_token_ids=output_token_ids,
            encoded_token_ids=encoded_ids,
            offsets=encoded["offset_mapping"],
            trailing_token_decodes_to_empty=trailing_empty,
        )
        decoded = tokenizer.decode(
            output_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded != response:
            raise ValueError("Frozen output IDs do not decode exactly to response")
        trailing_sequences[json.dumps(trailing)] += 1
        visible_counts.append(len(offsets))

    report = {
        "schema_version": "clir-hallucination-token-mapping-preflight-v1",
        "evidence_tier": "pipeline_pilot",
        "status": "passed",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "annotation_items_sha256": file_sha256(items_path),
        "annotation_lineage_sha256": file_sha256(lineage_path),
        "source_manifest_sha256": file_sha256(source_path),
        "rows": len(items),
        "tokenizer": {
            "model_id": token_cfg["model_id"],
            "revision": token_cfg["tokenizer_revision"],
            "class": type(tokenizer).__name__,
            "is_fast": tokenizer.is_fast,
        },
        "visible_token_count": {
            "min": min(visible_counts),
            "max": max(visible_counts),
            "mean": sum(visible_counts) / len(visible_counts),
        },
        "trailing_invisible_token_sequences": dict(sorted(trailing_sequences.items())),
        "exact_encoded_prefix_rows": len(items),
        "exact_decode_rows": len(items),
        "offset_coverage_rows": len(items),
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    output = args.output.resolve()
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"Refusing to overwrite drifted mapping report: {output}")
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    print(payload, end="")


if __name__ == "__main__":
    main()
