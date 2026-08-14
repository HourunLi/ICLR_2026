#!/usr/bin/env python
"""Verify every feature payload referenced by one or more extracted manifests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl
from src.clir_real_data import file_sha256
from src.clir_stage_a import atomic_write_json, git_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_payload(manifest: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else manifest.parent / path


def collect_payloads(manifests: list[Path]) -> tuple[Dict[Path, str], int]:
    payloads: Dict[Path, str] = {}
    rows = 0
    fields = (
        ("hidden_states_path", "feature_sha256"),
        ("condition_states_path", "condition_sha256"),
    )
    for manifest in manifests:
        for row in read_jsonl(manifest):
            rows += 1
            for path_field, hash_field in fields:
                raw_path = row.get(path_field)
                expected = row.get(hash_field)
                if raw_path is None:
                    continue
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError(
                        f"Row {row.get('id')!r} has {path_field} without a valid {hash_field}"
                    )
                path = _resolve_payload(manifest, str(raw_path)).resolve()
                previous = payloads.setdefault(path, expected)
                if previous != expected:
                    raise ValueError(f"Conflicting expected hashes for {path}")
    return payloads, rows


def verify_payloads(payloads: Dict[Path, str], workers: int) -> Dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")

    def verify(item: tuple[Path, str]) -> tuple[Path, str, str | None, int]:
        path, expected = item
        if not path.is_file():
            return path, expected, None, 0
        return path, expected, file_sha256(path), path.stat().st_size

    failures = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for path, expected, actual, size in executor.map(verify, payloads.items()):
            total_bytes += size
            if actual != expected:
                failures.append(
                    {
                        "path": str(path),
                        "expected_sha256": expected,
                        "actual_sha256": actual,
                        "status": "missing" if actual is None else "hash_mismatch",
                    }
                )
    return {
        "unique_payloads": len(payloads),
        "verified_payloads": len(payloads) - len(failures),
        "total_bytes": total_bytes,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    manifests = [Path(path).resolve() for path in args.manifest]
    output = Path(args.output_json).resolve() if args.output_json else None
    if output is not None and output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")

    payloads, rows = collect_payloads(manifests)
    verification = verify_payloads(payloads, args.workers)
    report = {
        "schema_version": "clir-feature-mirror-verification-v1",
        "manifests": [
            {"path": str(path), "sha256": file_sha256(path)} for path in manifests
        ],
        "rows": rows,
        **verification,
        "code": git_state(PROJECT_ROOT),
    }
    if output is not None:
        atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
