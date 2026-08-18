#!/usr/bin/env python3
"""Durably append one validated dual-prior secondary annotation at a time."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_dual_prior_annotation import validate_annotation  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    canonical_json,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_context(
    protocol_path: Path,
    output_override: Path | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-dual-prior-evidence-protocol-v1":
        raise ValueError("Unexpected dual-prior protocol schema")
    items = read_jsonl(resolve(protocol["outputs"]["annotation_items"]))
    expected = int(protocol["source"]["rows"])
    if len(items) != expected:
        raise ValueError(f"Frozen item count drifted: expected {expected}, got {len(items)}")
    output = (
        output_override.resolve()
        if output_override is not None
        else resolve(protocol["annotation"]["secondary_output"])
    )
    return items, output


def read_valid_prefix(
    output: Path,
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Read an absent/partial output only when it is an exact valid item prefix."""

    if not output.exists():
        return []
    text = output.read_text(encoding="utf-8")
    if not text:
        return []
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("Checkpoint JSONL contains a blank line")
    if len(lines) > len(items):
        raise ValueError("Checkpoint JSONL contains more rows than the frozen item set")
    rows: list[dict[str, Any]] = []
    for row_index, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Checkpoint row {row_index + 1} is not complete valid JSON; "
                "do not discard the preceding valid prefix"
            ) from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"Checkpoint row {row_index + 1} must be one JSON object")
        item = items[row_index]
        if raw.get("item_id") != item["item_id"]:
            raise ValueError(
                f"Checkpoint row {row_index + 1} is not the matching frozen item"
            )
        rows.append(validate_annotation(raw, item))
    return rows


def durable_atomic_jsonl(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Publish the whole tiny prefix atomically after flushing file and directory."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    payload = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def progress_record(
    rows: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    completed = len(rows)
    return {
        "schema_version": "clir-checkpointed-annotation-progress-v1",
        "output": str(output),
        "completed_rows": completed,
        "total_rows": len(items),
        "remaining_rows": len(items) - completed,
        "next_row_number_1_based": completed + 1 if completed < len(items) else None,
        "next_item_id": items[completed]["item_id"] if completed < len(items) else None,
        "complete": completed == len(items),
        "output_sha256": file_sha256(output) if output.exists() else None,
    }


def append_one(
    *,
    annotation_file: Path,
    output: Path,
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if annotation_file.resolve() == output.resolve():
        raise ValueError("The one-row scratch file must differ from the checkpoint output")
    raw = json.loads(annotation_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("The one-row scratch file must contain exactly one JSON object")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rows = read_valid_prefix(output, items)
        if len(rows) == len(items):
            raise ValueError("All frozen items are already checkpointed")
        expected_item = items[len(rows)]
        normalized = validate_annotation(raw, expected_item)
        durable_atomic_jsonl(output, [*rows, normalized])
        verified = read_valid_prefix(output, items)
        if len(verified) != len(rows) + 1:
            raise RuntimeError("Durable checkpoint verification failed")
        return {
            **progress_record(verified, items, output),
            "appended_item_id": normalized["item_id"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("status", "append", "finalize"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--annotation-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items, output = load_context(args.protocol.resolve(), args.output)
    if args.stage == "append":
        if args.annotation_file is None:
            raise ValueError("append requires --annotation-file with one JSON object")
        record = append_one(
            annotation_file=args.annotation_file.resolve(),
            output=output,
            items=items,
        )
    else:
        rows = read_valid_prefix(output, items)
        record = progress_record(rows, items, output)
        if args.stage == "finalize" and not record["complete"]:
            raise ValueError(
                f"Cannot finalize: {record['completed_rows']}/{record['total_rows']} rows checkpointed"
            )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
