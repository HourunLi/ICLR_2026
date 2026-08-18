#!/usr/bin/env python3
"""Validate blind raw hallucination annotations without reading private lineage."""

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
    file_sha256,
    read_jsonl,
    validate_annotation,
)


DEFAULT_ITEMS = (
    ROOT / "configs/hallucination_localization_v1/secondary_items_v1.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    items = read_jsonl(args.items)
    labels = read_jsonl(args.labels)
    if len(labels) != len(items):
        raise ValueError(f"Expected {len(items)} labels, found {len(labels)}")
    counts: Counter[str] = Counter()
    for row_number, (item, label) in enumerate(zip(items, labels), start=1):
        if item["item_id"] != label.get("item_id"):
            raise ValueError(
                f"Row {row_number} item_id/order drifted: "
                f"expected {item['item_id']!r}, got {label.get('item_id')!r}"
            )
        validate_annotation(label, item)
        counts[str(label["path_status"])] += 1
    report = {
        "schema_version": "clir-hallucination-blind-annotation-validation-v1",
        "status": "passed",
        "items": len(items),
        "items_sha256": file_sha256(args.items),
        "labels_sha256": file_sha256(args.labels),
        "path_status_counts": dict(sorted(counts.items())),
        "private_lineage_accessed": False,
        "primary_labels_accessed": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
