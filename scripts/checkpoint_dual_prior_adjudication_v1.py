#!/usr/bin/env python3
"""Durably checkpoint one validated role-blind dual-prior adjudication at a time."""

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

from src.clir_dual_prior_adjudication import validate_adjudication  # noqa: E402
from src.clir_hallucination_annotation import (  # noqa: E402
    canonical_json,
    file_sha256,
    read_jsonl,
)


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/dual_prior_evidence_v1/agreement_adjudication_protocol_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/dual_prior_evidence_v1/adjudications_raw_v1.jsonl"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_context(
    protocol_path: Path,
    output_override: Path | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-dual-prior-agreement-adjudication-protocol-v1"
    ):
        raise ValueError("Unexpected dual-prior agreement protocol schema")
    packets = read_jsonl(resolve(protocol["outputs"]["adjudication_items"]))
    report = json.loads(
        resolve(protocol["outputs"]["agreement_report"]).read_text(encoding="utf-8")
    )
    if not report.get("agreement_gate_passed") or not report.get(
        "adjudication_authorized"
    ):
        raise ValueError("Frozen agreement report does not authorize adjudication")
    if file_sha256(resolve(protocol["outputs"]["adjudication_items"])) != report[
        "outputs"
    ]["adjudication_items_sha256"]:
        raise ValueError("Adjudication packet hash differs from agreement report")
    if len(packets) != int(report["disagreement_rows"]):
        raise ValueError("Adjudication packet count differs from agreement report")
    output = output_override.resolve() if output_override is not None else DEFAULT_OUTPUT
    return packets, output


def read_valid_prefix(
    output: Path,
    packets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not output.exists():
        return []
    text = output.read_text(encoding="utf-8")
    if not text:
        return []
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("Adjudication checkpoint contains a blank line")
    if len(lines) > len(packets):
        raise ValueError("Adjudication checkpoint has more rows than the frozen packet")
    rows: list[dict[str, Any]] = []
    for row_index, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Adjudication checkpoint row {row_index + 1} is incomplete JSON"
            ) from exc
        if not isinstance(raw, Mapping):
            raise ValueError("Every adjudication checkpoint row must be one object")
        rows.append(validate_adjudication(raw, packets[row_index]))
    return rows


def durable_atomic_jsonl(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
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
    packets: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    completed = len(rows)
    return {
        "schema_version": "clir-checkpointed-adjudication-progress-v1",
        "output": str(output),
        "completed_rows": completed,
        "total_rows": len(packets),
        "remaining_rows": len(packets) - completed,
        "next_row_number_1_based": completed + 1 if completed < len(packets) else None,
        "next_item_id": packets[completed]["item_id"] if completed < len(packets) else None,
        "complete": completed == len(packets),
        "output_sha256": file_sha256(output) if output.exists() else None,
    }


def append_one(
    *,
    adjudication_file: Path,
    output: Path,
    packets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if adjudication_file.resolve() == output.resolve():
        raise ValueError("The one-row scratch file must differ from checkpoint output")
    raw = json.loads(adjudication_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("The one-row scratch file must contain exactly one object")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rows = read_valid_prefix(output, packets)
        if len(rows) == len(packets):
            raise ValueError("All frozen disagreements are already adjudicated")
        normalized = validate_adjudication(raw, packets[len(rows)])
        durable_atomic_jsonl(output, [*rows, normalized])
        verified = read_valid_prefix(output, packets)
        if len(verified) != len(rows) + 1:
            raise RuntimeError("Durable adjudication checkpoint verification failed")
        return {
            **progress_record(verified, packets, output),
            "appended_item_id": normalized["item_id"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("status", "append", "finalize"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adjudication-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packets, output = load_context(args.protocol.resolve(), args.output)
    if args.stage == "append":
        if args.adjudication_file is None:
            raise ValueError("append requires --adjudication-file with one JSON object")
        record = append_one(
            adjudication_file=args.adjudication_file.resolve(),
            output=output,
            packets=packets,
        )
    else:
        rows = read_valid_prefix(output, packets)
        record = progress_record(rows, packets, output)
        if args.stage == "finalize" and not record["complete"]:
            raise ValueError(
                f"Cannot finalize: {record['completed_rows']}/{record['total_rows']} "
                "disagreements checkpointed"
            )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
