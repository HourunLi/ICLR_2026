#!/usr/bin/env python3
"""Materialize the frozen four-fold hallucination tail comparison split."""

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

from src.clir_data import read_jsonl  # noqa: E402
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/hallucination_localization_v2/tail_cv_data_protocol_v2c.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def partition_key(row: Mapping[str, Any]) -> str:
    material = (
        "clir-tail-cv-v2c|"
        f"{int(row['path_hallucinated'])}|{row['query_id']}|{row['id']}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def partition_original_train(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[Mapping[str, Any]]]:
    """Split 31 clean / 17 hallucinated rows into fixed 16-row folds."""

    strata = {
        label: sorted(
            (row for row in rows if int(row["path_hallucinated"]) == label),
            key=partition_key,
        )
        for label in (0, 1)
    }
    if [len(strata[label]) for label in (0, 1)] != [31, 17]:
        raise ValueError("Tail CV expects exactly 31 clean and 17 hallucinated train rows")
    allocations = {0: [10, 10, 11], 1: [6, 6, 5]}
    folds: dict[int, list[Mapping[str, Any]]] = {1: [], 2: [], 3: []}
    for label in (0, 1):
        offset = 0
        for fold, count in zip((1, 2, 3), allocations[label]):
            folds[fold].extend(strata[label][offset : offset + count])
            offset += count
        if offset != len(strata[label]):
            raise AssertionError("Tail CV stratum allocation did not consume every row")
    for fold in folds:
        folds[fold].sort(key=partition_key)
    return folds


def token_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        target = row.get("token_hallucination_target")
        mask = row.get("token_hallucination_mask")
        token_ids = row.get("output_token_ids")
        if not isinstance(target, list) or not isinstance(mask, list):
            raise ValueError("Tail CV requires sparse token target and mask")
        if not isinstance(token_ids, list) or len(target) != len(mask) or len(mask) != len(token_ids):
            raise ValueError("Tail CV token fields are not aligned")
        if any(value not in (0, 1) for value in target + mask):
            raise ValueError("Tail CV sparse token fields must be binary")
        if any(label and not known for label, known in zip(target, mask)):
            raise ValueError("Tail CV contains a positive token outside its mask")
        counts["trajectory_tokens"] += len(token_ids)
        counts["supervised_tokens"] += sum(mask)
        counts["positive_tokens"] += sum(target)
        counts["negative_tokens"] += sum(mask) - sum(target)
    return dict(counts)


def split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(row["id"]) for row in rows]
    queries = [str(row["query_id"]) for row in rows]
    if len(set(ids)) != len(ids) or len(set(queries)) != len(queries):
        raise ValueError("Tail CV split contains duplicate ids or queries")
    path = Counter(int(row["path_hallucinated"]) for row in rows)
    correctness = Counter(int(row["correctness"]) for row in rows)
    if set(path) != {0, 1} or set(correctness) != {0, 1}:
        raise ValueError("Tail CV split must contain both path and correctness classes")
    counts = token_counts(rows)
    if counts["positive_tokens"] <= 0 or counts["negative_tokens"] <= 0:
        raise ValueError("Tail CV split must contain positive and negative explicit token labels")
    return {
        "rows": len(rows),
        "queries": len(queries),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "queries_sha256": hashlib.sha256("\n".join(queries).encode("utf-8")).hexdigest(),
        "path_clean": path[0],
        "path_hallucinated": path[1],
        "correct": correctness[1],
        "incorrect": correctness[0],
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Tail CV materialization requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-hallucination-tail-cv-data-protocol-v2c":
        raise ValueError("Unknown tail CV data protocol")
    sources: dict[str, list[dict[str, Any]]] = {}
    for name, spec in protocol["sources"].items():
        path = resolve(spec["path"])
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"Tail CV source hash drifted for {name}")
        rows = read_jsonl(path)
        if len(rows) != int(spec["rows"]):
            raise ValueError(f"Tail CV source row count drifted for {name}")
        sources[name] = rows

    original_train = sources["original_train"]
    original_dev = sources["original_dev"]
    all_rows = [*original_train, *original_dev]
    all_ids = [str(row["id"]) for row in all_rows]
    all_queries = [str(row["query_id"]) for row in all_rows]
    if len(set(all_ids)) != 64 or len(set(all_queries)) != 64:
        raise ValueError("Tail CV sources must contain 64 unique ids and queries")
    if set(str(row["query_id"]) for row in original_train) & set(
        str(row["query_id"]) for row in original_dev
    ):
        raise ValueError("Tail CV source train/dev query overlap")

    dev_folds: dict[int, list[Mapping[str, Any]]] = {0: original_dev}
    dev_folds.update(partition_original_train(original_train))
    dev_id_sets = {fold: {str(row["id"]) for row in rows} for fold, rows in dev_folds.items()}
    if set().union(*dev_id_sets.values()) != set(all_ids):
        raise ValueError("Tail CV dev folds do not cover all source rows")
    for left in dev_id_sets:
        for right in dev_id_sets:
            if left < right and dev_id_sets[left] & dev_id_sets[right]:
                raise ValueError("Tail CV dev folds overlap")

    output_root = resolve(protocol["outputs"]["root"])
    audit_path = resolve(protocol["outputs"]["audit"])
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite tail CV data: {output_root}")
    ordered = {str(row["id"]): row for row in all_rows}
    fold_reports: dict[str, Any] = {}
    for fold in range(4):
        dev_ids = dev_id_sets[fold]
        train_rows = [row for identifier, row in ordered.items() if identifier not in dev_ids]
        dev_rows = list(dev_folds[fold])
        if len(train_rows) != 48 or len(dev_rows) != 16:
            raise ValueError("Tail CV fold size drifted")
        if {str(row["query_id"]) for row in train_rows} & {
            str(row["query_id"]) for row in dev_rows
        }:
            raise ValueError("Tail CV fold train/dev query overlap")
        fold_root = output_root / f"fold_{fold}"
        train_path = fold_root / "train48.jsonl"
        dev_path = fold_root / "dev16.jsonl"
        atomic_write_jsonl(train_path, train_rows)
        atomic_write_jsonl(dev_path, dev_rows)
        fold_reports[str(fold)] = {
            "selection_exposed": fold == 0,
            "train": {
                "path": str(train_path.relative_to(ROOT)),
                "sha256": file_sha256(train_path),
                **split_summary(train_rows),
            },
            "dev": {
                "path": str(dev_path.relative_to(ROOT)),
                "sha256": file_sha256(dev_path),
                **split_summary(dev_rows),
            },
            "query_overlap": 0,
        }

    report = {
        "schema_version": "clir-hallucination-tail-cv-fold-audit-v2c",
        "status": "passed",
        "evidence_tier": "pipeline_pilot",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": file_sha256(protocol_path),
        },
        "source_union": {
            "rows": 64,
            "queries": 64,
            "ids_sha256": hashlib.sha256("\n".join(all_ids).encode("utf-8")).hexdigest(),
            "queries_sha256": hashlib.sha256("\n".join(all_queries).encode("utf-8")).hexdigest(),
        },
        "folds": fold_reports,
        "confirmatory_folds": [1, 2, 3],
        "confirmatory_rows": 48,
        "all_dev_folds_pairwise_disjoint": True,
        "all_dev_folds_cover_source_union": True,
        "all_train_dev_query_overlaps_zero": True,
        "code": state,
        "pilot_test_accessed": False,
        "formal_mechanism_claim_allowed": False,
    }
    atomic_write_json(audit_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
