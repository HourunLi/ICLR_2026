import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.clir_data import read_jsonl
from src.clir_real_data import canonical_json_sha256, validate_rollout_row
from src.clir_stage_a import (
    atomic_write_jsonl,
    build_gsm8k_split_manifest,
    build_payload_record,
    candidate_count_for_membership,
    git_state,
    membership_entries,
    publish_completion_marker,
    query_shard_dir,
    validate_completion_marker,
    validate_split_manifest,
)
from scripts.merge_query_shards import merge_query_shards, validate_candidate_index_policy
from scripts.generate_gsm8k_rollouts import ordered_vllm_candidates


def fake_records(count: int, prefix: str):
    return [
        {"question": f"{prefix} question {index}", "answer": f"work #### {index}"}
        for index in range(count)
    ]


def test_candidate_count_uses_membership_instead_of_dataset_source_split():
    generation = {
        "train_candidates": 8,
        "pilot_eval_candidates": 16,
        "formal_eval_candidates": 64,
    }

    assert candidate_count_for_membership(generation, "train_primary") == 8
    assert candidate_count_for_membership(generation, "validation") == 16
    assert candidate_count_for_membership(generation, "pilot_test") == 16
    assert candidate_count_for_membership(generation, "final_test") == 64


def test_vllm_candidates_are_restored_to_original_sample_order():
    request_output = SimpleNamespace(
        outputs=[
            SimpleNamespace(index=2, cumulative_logprob=-1.0),
            SimpleNamespace(index=0, cumulative_logprob=-2.0),
            SimpleNamespace(index=1, cumulative_logprob=-3.0),
        ]
    )
    ordered = ordered_vllm_candidates(request_output, 3)
    assert [candidate.index for candidate in ordered] == [0, 1, 2]

    request_output.outputs[2].index = 2
    with pytest.raises(ValueError, match="unique and contiguous"):
        ordered_vllm_candidates(request_output, 3)


def test_candidate_index_policy_is_protocol_gated():
    rows = [
        {"generation": {"candidate_index_policy": "vllm_completion_output_index"}},
        {"generation": {"candidate_index_policy": "vllm_completion_output_index"}},
    ]
    validate_candidate_index_policy(rows, "vllm_completion_output_index", "q")
    rows[1]["generation"]["candidate_index_policy"] = "likelihood_rank"
    with pytest.raises(ValueError, match="Candidate index policy mismatch"):
        validate_candidate_index_policy(rows, "vllm_completion_output_index", "q")


def make_manifest():
    return build_gsm8k_split_manifest(
        fake_records(12, "train"),
        fake_records(7, "test"),
        dataset={"repo_id": "fake/gsm8k", "revision": "fixed"},
        protocol_version="test-v1",
        protocol_sha256="protocol-hash",
        seed=42,
        train_primary_size=7,
        validation_size=3,
        development_size=2,
        pilot_test_size=4,
    )


def test_split_manifest_is_deterministic_group_safe_and_hashed():
    first = make_manifest()
    second = make_manifest()

    assert first == second
    assert len(first["memberships"]["train_primary"]) == 7
    assert len(first["memberships"]["validation"]) == 3
    assert len(first["memberships"]["reserve"]) == 2
    assert set(first["memberships"]["development_32"]).issubset(
        first["memberships"]["train_primary"]
    )
    assert set(first["memberships"]["pilot_test"]).issubset(
        first["memberships"]["final_test"]
    )
    validate_split_manifest(first)

    tampered = json.loads(json.dumps(first))
    tampered["memberships"]["validation"][0] = tampered["memberships"]["train_primary"][0]
    with pytest.raises(ValueError):
        validate_split_manifest(tampered)


def test_membership_partition_is_complete_and_disjoint():
    manifest = make_manifest()
    partitions = [
        membership_entries(
            manifest,
            "train_primary",
            max_queries=6,
            num_shards=3,
            shard_id=shard_id,
        )
        for shard_id in range(3)
    ]
    ids = [[entry["query_id"] for entry in partition] for partition in partitions]

    assert all(len(partition) == 2 for partition in ids)
    assert len(set().union(*(set(partition) for partition in ids))) == 6
    assert not (set(ids[0]) & set(ids[1]) or set(ids[0]) & set(ids[2]) or set(ids[1]) & set(ids[2]))


def test_query_completion_marker_checks_payload_and_candidate_contract(tmp_path: Path):
    query_id = "gsm8k-train-00003"
    query_dir = query_shard_dir(tmp_path, query_id)
    rows_path = query_dir / "rollouts.jsonl"
    rows = []
    for candidate_index in range(2):
        rows.append(
            {
                "id": f"{query_id}-cand-{candidate_index:03d}",
                "query_id": query_id,
                "candidate_index": candidate_index,
                "prompt_token_ids": [1, 2],
                "output_token_ids": [3 + candidate_index],
                "response": "x",
                "correctness": candidate_index,
                "provenance": {
                    "model_id": "fake/model",
                    "model_revision": "rev",
                    "tokenizer_revision": "rev",
                    "protocol_version": "test-v1",
                },
            }
        )
    atomic_write_jsonl(rows_path, rows)
    marker = {
        "schema_version": "clir-query-shard-v1",
        "stage": "rollout",
        "query_id": query_id,
        "protocol_sha256": "protocol-hash",
        "split_manifest_sha256": "split-hash",
        "candidate_count": 2,
        "payloads": [build_payload_record(rows_path, role="rows", root=query_dir)],
    }
    publish_completion_marker(query_dir, "_ROLLOUT_SUCCESS.json", marker)

    validated = validate_completion_marker(
        query_dir,
        "_ROLLOUT_SUCCESS.json",
        stage="rollout",
        query_id=query_id,
        protocol_sha256="protocol-hash",
        split_manifest_sha256="split-hash",
        expected_candidate_count=2,
        rows_loader=read_jsonl,
        row_validator=validate_rollout_row,
    )
    assert Path(validated["_rows_path"]) == rows_path

    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_completion_marker(
            query_dir,
            "_ROLLOUT_SUCCESS.json",
            stage="rollout",
            query_id=query_id,
            protocol_sha256="protocol-hash",
            split_manifest_sha256="split-hash",
            expected_candidate_count=2,
            rows_loader=read_jsonl,
        )


def test_component_marker_rejects_label_hash_drift(tmp_path: Path):
    query_id = "gsm8k-train-00000"
    query_dir = query_shard_dir(tmp_path, query_id)
    rows_path = query_dir / "rollouts.jsonl"
    rows = [{
        "id": f"{query_id}-cand-000",
        "query_id": query_id,
        "candidate_index": 0,
        "prompt_token_ids": [1],
        "output_token_ids": [2],
        "response": "x",
        "correctness": 1,
        "provenance": {},
    }]
    atomic_write_jsonl(rows_path, rows)
    publish_completion_marker(query_dir, "_ROLLOUT_SUCCESS.json", {
        "schema_version": "clir-query-shard-v1",
        "stage": "rollout",
        "query_id": query_id,
        "protocol_sha256": "full",
        "acquisition_protocol_sha256": "acquisition",
        "label_protocol_sha256": "labels-v2",
        "split_manifest_sha256": "split",
        "candidate_count": 1,
        "payloads": [build_payload_record(rows_path, role="rows", root=query_dir)],
    })

    with pytest.raises(ValueError, match="label_protocol_sha256 mismatch"):
        validate_completion_marker(
            query_dir,
            "_ROLLOUT_SUCCESS.json",
            stage="rollout",
            query_id=query_id,
            protocol_sha256="full",
            acquisition_protocol_sha256="acquisition",
            label_protocol_sha256="labels-v3",
            split_manifest_sha256="split",
            expected_candidate_count=1,
            rows_loader=read_jsonl,
        )


def test_merge_query_shards_preserves_frozen_query_and_candidate_order(tmp_path: Path):
    protocol = {"protocol_version": "test-v1"}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_hash = canonical_json_sha256(protocol)
    manifest = build_gsm8k_split_manifest(
        fake_records(4, "train"),
        fake_records(2, "test"),
        dataset={"repo_id": "fake/gsm8k", "revision": "fixed"},
        protocol_version="test-v1",
        protocol_sha256=protocol_hash,
        seed=42,
        train_primary_size=2,
        validation_size=1,
        development_size=2,
        pilot_test_size=1,
    )
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    shard_root = tmp_path / "shards"
    frozen_ids = manifest["memberships"]["development_32"]
    for query_position, query_id in enumerate(frozen_ids):
        query_dir = query_shard_dir(shard_root, query_id)
        rows_path = query_dir / "rollouts.jsonl"
        rows = []
        for candidate_index in range(2):
            rows.append({
                "id": f"{query_id}-cand-{candidate_index:03d}",
                "query_id": query_id,
                "candidate_index": candidate_index,
                "prompt_token_ids": [1, 2],
                "output_token_ids": [3 + candidate_index],
                "response": "x",
                "correctness": int(query_position == candidate_index),
                "provenance": {
                    "model_id": "fake/model",
                    "model_revision": "rev",
                    "tokenizer_revision": "rev",
                    "protocol_version": "test-v1",
                },
            })
        atomic_write_jsonl(rows_path, rows)
        publish_completion_marker(query_dir, "_ROLLOUT_SUCCESS.json", {
            "schema_version": "clir-query-shard-v1",
            "stage": "rollout",
            "query_id": query_id,
            "protocol_sha256": protocol_hash,
            "split_manifest_sha256": manifest["manifest_sha256"],
            "candidate_count": 2,
            "payloads": [build_payload_record(rows_path, role="rows", root=query_dir)],
        })

    rows, report = merge_query_shards(
        protocol_path=protocol_path,
        split_manifest_path=manifest_path,
        membership="development_32",
        shard_root=shard_root,
        stage="rollout",
        expected_candidates=2,
        max_queries=None,
    )

    assert [(row["query_id"], row["candidate_index"]) for row in rows] == [
        (query_id, candidate_index)
        for query_id in frozen_ids
        for candidate_index in range(2)
    ]
    assert report["queries"] == 2
    assert report["rows"] == 4
    assert report["correct"] == 2


def test_generation_resume_exits_before_heavy_dependencies_and_does_not_rewrite(tmp_path: Path):
    protocol = {
        "protocol_version": "test-v1",
        "generation": {"train_candidates": 2, "pilot_eval_candidates": 3},
        "model": {},
        "dataset": {},
        "prompt": {},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_hash = canonical_json_sha256(protocol)
    manifest = build_gsm8k_split_manifest(
        fake_records(3, "train"),
        fake_records(1, "test"),
        dataset={"repo_id": "fake/gsm8k", "revision": "fixed"},
        protocol_version="test-v1",
        protocol_sha256=protocol_hash,
        seed=42,
        train_primary_size=1,
        validation_size=1,
        development_size=1,
        pilot_test_size=1,
    )
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    query_id = manifest["memberships"]["development_32"][0]
    query_dir = query_shard_dir(tmp_path / "shards", query_id)
    rows_path = query_dir / "rollouts.jsonl"
    rows = [
        {
            "id": f"{query_id}-cand-{candidate_index:03d}",
            "query_id": query_id,
            "candidate_index": candidate_index,
            "prompt_token_ids": [1],
            "output_token_ids": [2 + candidate_index],
            "response": "x",
            "correctness": candidate_index,
            "provenance": {
                "model_id": "fake/model",
                "model_revision": "rev",
                "tokenizer_revision": "rev",
                "protocol_version": "test-v1",
            },
        }
        for candidate_index in range(2)
    ]
    atomic_write_jsonl(rows_path, rows)
    marker_path = publish_completion_marker(query_dir, "_ROLLOUT_SUCCESS.json", {
        "schema_version": "clir-query-shard-v1",
        "stage": "rollout",
        "query_id": query_id,
        "protocol_sha256": protocol_hash,
        "split_manifest_sha256": manifest["manifest_sha256"],
        "candidate_count": 2,
        "payloads": [build_payload_record(rows_path, role="rows", root=query_dir)],
    })
    before = marker_path.stat().st_mtime_ns

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_gsm8k_rollouts.py",
            "--protocol-config", str(protocol_path),
            "--split-manifest", str(manifest_path),
            "--membership", "development_32",
            "--shard-root", str(tmp_path / "shards"),
            "--split", "train",
            "--max-queries", "1",
            "--n-rollouts", "2",
            "--resume",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    assert '"status": "all_complete"' in result.stdout
    assert '"skipped_queries": 1' in result.stdout
    assert marker_path.stat().st_mtime_ns == before


def test_git_state_ignores_untracked_artifacts_but_detects_tracked_changes(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "CLIR Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "clir-test@example.com"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    (tmp_path / "artifact.pt").write_bytes(b"generated")
    assert git_state(tmp_path)["dirty"] is False

    tracked.write_text("v2\n", encoding="utf-8")
    assert git_state(tmp_path)["dirty"] is True
