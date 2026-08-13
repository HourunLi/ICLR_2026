"""Stage A split, atomic artifact, and query-shard utilities."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from .clir_real_data import artifact_stem, canonical_json_sha256, file_sha256


def atomic_write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, payload)


def read_json(path: str | Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_state(project_root: str | Path) -> Dict[str, Any]:
    """Return code provenance while ignoring generated/untracked artifacts."""
    root = Path(project_root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tracked_changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"commit": commit, "dirty": bool(tracked_changes)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def stable_query_id(split: str, source_index: int) -> str:
    return f"gsm8k-{split}-{source_index:05d}"


def _query_entries(records: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for source_index, record in enumerate(records):
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing GSM8K question at {split}[{source_index}]")
        query_id = stable_query_id(split, source_index)
        entries[query_id] = {
            "query_id": query_id,
            "source_split": split,
            "source_index": source_index,
            "question_sha256": sha256_text(question),
        }
    return entries


def build_gsm8k_split_manifest(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    *,
    dataset: Mapping[str, Any],
    protocol_version: str,
    protocol_sha256: str,
    seed: int = 42,
    train_primary_size: int = 6000,
    validation_size: int = 500,
    development_size: int = 32,
    pilot_test_size: int = 500,
) -> Dict[str, Any]:
    if train_primary_size <= 0 or validation_size <= 0:
        raise ValueError("train_primary_size and validation_size must be positive")
    if train_primary_size + validation_size > len(train_records):
        raise ValueError("Requested train_primary + validation exceeds the train split")
    if not 0 < development_size <= train_primary_size:
        raise ValueError("development_size must be in [1, train_primary_size]")
    if not 0 < pilot_test_size <= len(test_records):
        raise ValueError("pilot_test_size must be in [1, len(test_records)]")

    train_entries = _query_entries(train_records, "train")
    test_entries = _query_entries(test_records, "test")
    train_ids = list(train_entries)
    random.Random(seed).shuffle(train_ids)
    train_primary = train_ids[:train_primary_size]
    validation = train_ids[train_primary_size : train_primary_size + validation_size]
    reserve = train_ids[train_primary_size + validation_size :]
    final_test = list(test_entries)

    memberships = {
        "train_primary": train_primary,
        "validation": validation,
        "reserve": reserve,
        "development_32": train_primary[:development_size],
        "pilot_test": final_test[:pilot_test_size],
        "final_test": final_test,
    }
    manifest: Dict[str, Any] = {
        "schema_version": "clir-gsm8k-query-splits-v1",
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "dataset": dict(dataset),
        "split_unit": "original_query",
        "permutation": {
            "algorithm": "python_random_mt19937_shuffle",
            "seed": seed,
        },
        "queries": {**train_entries, **test_entries},
        "memberships": memberships,
        "membership_sha256": {
            name: canonical_json_sha256(query_ids)
            for name, query_ids in memberships.items()
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != "clir-gsm8k-query-splits-v1":
        raise ValueError("Unsupported split manifest schema_version")
    queries = manifest.get("queries")
    memberships = manifest.get("memberships")
    if not isinstance(queries, dict) or not isinstance(memberships, dict):
        raise ValueError("Split manifest requires queries and memberships objects")

    required = {
        "train_primary",
        "validation",
        "reserve",
        "development_32",
        "pilot_test",
        "final_test",
    }
    if set(memberships) != required:
        raise ValueError(f"Split memberships must be exactly {sorted(required)}")
    for name, query_ids in memberships.items():
        if not isinstance(query_ids, list) or len(query_ids) != len(set(query_ids)):
            raise ValueError(f"Membership {name} must be a duplicate-free list")
        missing = [query_id for query_id in query_ids if query_id not in queries]
        if missing:
            raise ValueError(f"Membership {name} contains unknown queries: {missing[:3]}")

    train_primary = set(memberships["train_primary"])
    validation = set(memberships["validation"])
    reserve = set(memberships["reserve"])
    if train_primary & validation or train_primary & reserve or validation & reserve:
        raise ValueError("train_primary, validation, and reserve must be disjoint")
    if not set(memberships["development_32"]).issubset(train_primary):
        raise ValueError("development_32 must be a subset of train_primary")
    if not set(memberships["pilot_test"]).issubset(set(memberships["final_test"])):
        raise ValueError("pilot_test must be a subset of final_test")

    train_all = train_primary | validation | reserve
    query_train = {
        query_id for query_id, entry in queries.items() if entry.get("source_split") == "train"
    }
    query_test = {
        query_id for query_id, entry in queries.items() if entry.get("source_split") == "test"
    }
    if train_all != query_train:
        raise ValueError("Train memberships do not cover exactly the source train split")
    if set(memberships["final_test"]) != query_test:
        raise ValueError("final_test does not cover exactly the source test split")

    expected_membership_hashes = {
        name: canonical_json_sha256(query_ids)
        for name, query_ids in memberships.items()
    }
    if manifest.get("membership_sha256") != expected_membership_hashes:
        raise ValueError("Split membership hash mismatch")
    without_hash = deepcopy(dict(manifest))
    recorded_hash = without_hash.pop("manifest_sha256", None)
    if recorded_hash != canonical_json_sha256(without_hash):
        raise ValueError("Split manifest hash mismatch")


def load_split_manifest(path: str | Path) -> Dict[str, Any]:
    manifest = read_json(path)
    validate_split_manifest(manifest)
    return manifest


def membership_entries(
    manifest: Mapping[str, Any],
    membership: str,
    *,
    max_queries: int | None = None,
    num_shards: int = 1,
    shard_id: int = 0,
) -> list[Dict[str, Any]]:
    validate_split_manifest(manifest)
    if membership not in manifest["memberships"]:
        raise KeyError(f"Unknown split membership: {membership}")
    if num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_id < num_shards")
    query_ids = list(manifest["memberships"][membership])
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        query_ids = query_ids[:max_queries]
    selected = [query_id for position, query_id in enumerate(query_ids) if position % num_shards == shard_id]
    return [dict(manifest["queries"][query_id]) for query_id in selected]


def query_shard_dir(shard_root: str | Path, query_id: str) -> Path:
    return Path(shard_root).resolve() / artifact_stem(query_id)


def build_payload_record(path: str | Path, *, role: str, root: str | Path) -> Dict[str, Any]:
    payload = Path(path).resolve()
    base = Path(root).resolve()
    return {
        "role": role,
        "path": str(payload.relative_to(base)),
        "bytes": payload.stat().st_size,
        "sha256": file_sha256(payload),
    }


def publish_completion_marker(
    query_dir: str | Path,
    marker_name: str,
    marker: Mapping[str, Any],
) -> Path:
    output = Path(query_dir) / marker_name
    atomic_write_json(output, dict(marker))
    return output


def validate_completion_marker(
    query_dir: str | Path,
    marker_name: str,
    *,
    stage: str,
    query_id: str,
    protocol_sha256: str,
    split_manifest_sha256: str,
    expected_candidate_count: int,
    rows_loader: Callable[[Path], Sequence[Mapping[str, Any]]],
    row_validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    root = Path(query_dir).resolve()
    marker_path = root / marker_name
    if not marker_path.exists():
        raise FileNotFoundError(marker_path)
    marker = read_json(marker_path)
    expected = {
        "schema_version": "clir-query-shard-v1",
        "stage": stage,
        "query_id": query_id,
        "protocol_sha256": protocol_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "candidate_count": expected_candidate_count,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"Completion marker {key} mismatch for {query_id}")
    payloads = marker.get("payloads")
    if not isinstance(payloads, list) or not payloads:
        raise ValueError(f"Completion marker has no payloads for {query_id}")
    rows_paths: list[Path] = []
    for payload in payloads:
        path = (root / payload["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Payload escapes query shard: {path}")
        if not path.exists() or path.stat().st_size != payload.get("bytes"):
            raise ValueError(f"Payload missing/size mismatch: {path}")
        if file_sha256(path) != payload.get("sha256"):
            raise ValueError(f"Payload checksum mismatch: {path}")
        if payload.get("role") == "rows":
            rows_paths.append(path)
    if len(rows_paths) != 1:
        raise ValueError(f"Completion marker requires exactly one rows payload for {query_id}")
    rows_path = rows_paths[0]
    rows = list(rows_loader(rows_path))
    if len(rows) != expected_candidate_count:
        raise ValueError(f"Candidate count mismatch for {query_id}")
    if {str(row.get("query_id")) for row in rows} != {query_id}:
        raise ValueError(f"Rows contain the wrong query_id for {query_id}")
    candidate_indices = [int(row.get("candidate_index", -1)) for row in rows]
    if candidate_indices != list(range(expected_candidate_count)):
        raise ValueError(f"Candidate indices are not contiguous for {query_id}")
    if row_validator is not None:
        for row in rows:
            row_validator(row)
    marker["_rows_path"] = str(rows_path)
    return marker


__all__ = [
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "build_gsm8k_split_manifest",
    "build_payload_record",
    "git_state",
    "load_split_manifest",
    "membership_entries",
    "publish_completion_marker",
    "query_shard_dir",
    "read_json",
    "sha256_text",
    "stable_query_id",
    "validate_completion_marker",
    "validate_split_manifest",
]
