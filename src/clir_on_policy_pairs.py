"""Deterministic candidate mining for CLIR on-policy equivalence pilots."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_SCHEMA = "clir-on-policy-equivalence-candidate-v1"
REPORT_SCHEMA = "clir-on-policy-equivalence-candidate-report-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_source_row(row: Mapping[str, Any]) -> None:
    required = {
        "id", "query_id", "candidate_index", "question", "prompt", "response",
        "prompt_token_ids", "output_token_ids", "correctness",
        "normalized_candidate_answer", "source_index", "split",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Source row is missing fields: {sorted(missing)}")
    if row["correctness"] not in (0, 1) or isinstance(row["correctness"], bool):
        raise ValueError(f"Source row {row['id']!r} has invalid correctness")
    if not row["response"] or not row["output_token_ids"]:
        raise ValueError(f"Source row {row['id']!r} has an empty response")


def mine_candidate_pairs(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_query_ids: Sequence[str],
    correctness: int = 1,
    max_pairs_per_query: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mine high-recall pairs; this function never claims semantic equivalence."""

    if correctness not in (0, 1) or isinstance(correctness, bool):
        raise ValueError("correctness must be integer 0 or 1")
    if max_pairs_per_query <= 0:
        raise ValueError("max_pairs_per_query must be positive")
    membership = list(allowed_query_ids)
    if len(membership) != len(set(membership)):
        raise ValueError("allowed_query_ids contains duplicates")
    order = {query_id: index for index, query_id in enumerate(membership)}

    selected_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in rows:
        row = dict(raw)
        _validate_source_row(row)
        if row["query_id"] not in order:
            continue
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate source row id {row['id']!r}")
        seen_ids.add(row["id"])
        selected_rows.append(row)

    observed_queries = {row["query_id"] for row in selected_rows}
    missing_queries = set(membership) - observed_queries
    if missing_queries:
        raise ValueError(f"Membership queries missing from source: {sorted(missing_queries)}")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        if row["correctness"] != correctness:
            continue
        outcome = row["normalized_candidate_answer"]
        if outcome is None or not str(outcome).strip():
            continue
        groups[(row["query_id"], str(outcome))].append(row)

    by_query: dict[str, list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    eligible_pair_count = 0
    for (query_id, outcome), group in groups.items():
        group.sort(key=lambda row: (int(row["candidate_index"]), str(row["id"])))
        for left, right in itertools.combinations(group, 2):
            if left["response"] == right["response"]:
                continue
            eligible_pair_count += 1
            token_gap = abs(len(left["output_token_ids"]) - len(right["output_token_ids"]))
            rank_key = (-token_gap, int(left["candidate_index"]), int(right["candidate_index"]), left["id"], right["id"])
            by_query[query_id].append((rank_key, left, right))

    candidates: list[dict[str, Any]] = []
    for query_id in membership:
        ranked = sorted(by_query.get(query_id, []), key=lambda item: item[0])
        for query_rank, (_, left, right) in enumerate(ranked[:max_pairs_per_query], start=1):
            pair_identity = {
                "query_id": query_id,
                "anchor_id": left["id"],
                "alternate_id": right["id"],
                "normalized_outcome": str(left["normalized_candidate_answer"]),
            }
            pair_id = "OPA-" + object_sha256(pair_identity)[:16]
            candidates.append({
                "schema_version": CANDIDATE_SCHEMA,
                "pair_id": pair_id,
                "candidate_status": "unverified_high_recall_candidate",
                "query_rank": query_rank,
                "query_id": query_id,
                "source_index": left["source_index"],
                "source_split": left["split"],
                "problem": left["question"],
                "prompt_sha256": hashlib.sha256(canonical_json(left["prompt_token_ids"]).encode()).hexdigest(),
                "correctness": correctness,
                "normalized_outcome": str(left["normalized_candidate_answer"]),
                "selection_features": {
                    "anchor_output_tokens": len(left["output_token_ids"]),
                    "alternate_output_tokens": len(right["output_token_ids"]),
                    "absolute_output_token_gap": abs(len(left["output_token_ids"]) - len(right["output_token_ids"])),
                },
                "anchor": {
                    "id": left["id"], "candidate_index": left["candidate_index"],
                    "response": left["response"], "output_token_ids": left["output_token_ids"],
                    "row_sha256": object_sha256(left), "view_role": "native_anchor",
                },
                "alternate": {
                    "id": right["id"], "candidate_index": right["candidate_index"],
                    "response": right["response"], "output_token_ids": right["output_token_ids"],
                    "row_sha256": object_sha256(right), "view_role": "native_alternate",
                },
                "relation_label": None,
                "relation_verifier": None,
            })

    report = {
        "schema_version": REPORT_SCHEMA,
        "evidence_tier": "pipeline_pilot",
        "membership_queries": len(membership),
        "source_rows_in_membership": len(selected_rows),
        "requested_correctness": correctness,
        "eligible_pair_count_before_cap": eligible_pair_count,
        "max_pairs_per_query": max_pairs_per_query,
        "candidate_pair_count": len(candidates),
        "queries_with_candidates": len({row["query_id"] for row in candidates}),
        "candidate_status": "unverified_high_recall_candidate",
        "reasoning_equivalence_claim_allowed": False,
    }
    return candidates, report


def audit_candidates(candidates: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> None:
    pair_ids: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for row in candidates:
        if row.get("schema_version") != CANDIDATE_SCHEMA:
            raise ValueError("Candidate schema drifted")
        if row.get("candidate_status") != "unverified_high_recall_candidate":
            raise ValueError("Candidate was prematurely labeled")
        if row.get("relation_label") is not None or row.get("relation_verifier") is not None:
            raise ValueError("Candidate contains a relation verdict")
        pair_id = str(row.get("pair_id"))
        if pair_id in pair_ids:
            raise ValueError(f"Duplicate pair_id {pair_id!r}")
        pair_ids.add(pair_id)
        counts[str(row["query_id"])] += 1
        if row["correctness"] != report["requested_correctness"]:
            raise ValueError("Candidate correctness drifted")
        if row["anchor"]["id"] == row["alternate"]["id"]:
            raise ValueError("Candidate pairs a row with itself")
        if row["anchor"]["response"] == row["alternate"]["response"]:
            raise ValueError("Candidate responses are identical")
        if row["anchor"]["view_role"] != "native_anchor" or row["alternate"]["view_role"] != "native_alternate":
            raise ValueError("View roles drifted")
    if any(value > report["max_pairs_per_query"] for value in counts.values()):
        raise ValueError("Per-query candidate cap was exceeded")
    if len(candidates) != report["candidate_pair_count"]:
        raise ValueError("Candidate count differs from report")
    if report.get("reasoning_equivalence_claim_allowed") is not False:
        raise ValueError("Candidate report must prohibit equivalence claims")
