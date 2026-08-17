#!/usr/bin/env python3
"""Publish one Silver native pair per query as an exact-token online manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.clir_data import read_jsonl
from src.clir_on_policy_pairs import canonical_json


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "run_artifacts/on_policy_pilot0_v1/candidate_build/candidates.jsonl"
VERIFIER = ROOT / "run_artifacts/on_policy_pilot0_v1/verifier_mistral24_v1/all_results.jsonl"
SOURCE = ROOT / "run_artifacts/stage1b_v3/labels/train_extracted.v5.jsonl"
OUTPUT_ROOT = ROOT / "run_artifacts/on_policy_pilot0_v1/published_silver_v1"
DROP_FIELDS = {
    "hidden_states_path", "hidden_states", "feature_sha256", "feature_metadata",
    "condition_states_path", "condition_states", "condition_sha256",
    "hallucination_onset", "path_hallucinated", "token_advantage", "token_advantages",
    "progress_targets", "key_prior_target", "complete_prior_target",
    "complete_reconstruction_target",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    candidates = read_jsonl(CANDIDATES)
    results = {row["pair_id"]: row for row in read_jsonl(VERIFIER)}
    sources = {row["id"]: row for row in read_jsonl(SOURCE)}
    accepted = [row for row in candidates if results[row["pair_id"]]["acceptance_status"] == "silver_accepted"]
    by_query: dict[str, list[dict]] = {}
    for row in accepted:
        by_query.setdefault(row["query_id"], []).append(row)
    selected = []
    for query_id in sorted(by_query):
        ranked = sorted(
            by_query[query_id],
            key=lambda row: (-row["selection_features"]["absolute_output_token_gap"], row["query_rank"], row["pair_id"]),
        )
        selected.append(ranked[0])

    published = []
    for pair in selected:
        for side, role in (("anchor", "native_anchor"), ("alternate", "native_alternate")):
            source = sources[pair[side]["id"]]
            if hashlib.sha256(canonical_json(source).encode()).hexdigest() != pair[side]["row_sha256"]:
                raise ValueError(f"Source identity drifted for {source['id']}")
            row = {key: value for key, value in source.items() if key not in DROP_FIELDS}
            row.update({
                "id": f"{source['id']}--{pair['pair_id']}--{role}",
                "semantic_id": pair["pair_id"], "style_id": role,
                "augmentation_provenance": {
                    "schema_version": "clir-on-policy-published-view-v1",
                    "route": "A_on_policy_equivalence_mining",
                    "source_row_id": source["id"], "source_row_sha256": pair[side]["row_sha256"],
                    "pair_id": pair["pair_id"], "view_role": role,
                    "verifier_tier": "silver", "verifier_result_sha256": hashlib.sha256(
                        canonical_json(results[pair["pair_id"]]).encode()
                    ).hexdigest(),
                },
            })
            published.append(row)

    if len(published) != 2 * len(selected):
        raise ValueError("Published view count drifted")
    if len({row["id"] for row in published}) != len(published):
        raise ValueError("Published row IDs are not unique")
    if any(set(row) & DROP_FIELDS for row in published):
        raise ValueError("A forbidden inherited supervision/feature field survived")
    if any(row["correctness"] != 1 for row in published):
        raise ValueError("Pilot-0 must be correct-only")
    groups = {}
    for row in published:
        groups.setdefault(row["semantic_id"], []).append(row)
    if any({row["style_id"] for row in group} != {"native_anchor", "native_alternate"} for group in groups.values()):
        raise ValueError("Every semantic group must contain the two native view roles")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT_ROOT / "train_online.jsonl"
    manifest.write_text("".join(canonical_json(row) + "\n" for row in published), encoding="utf-8")
    report = {
        "schema_version": "clir-on-policy-published-report-v1", "evidence_tier": "pipeline_pilot",
        "candidate_manifest_sha256": sha256(CANDIDATES), "verifier_results_sha256": sha256(VERIFIER),
        "source_manifest_sha256": sha256(SOURCE), "selection": "one_max_token_gap_silver_pair_per_query",
        "silver_candidate_pairs": len(accepted), "published_pairs": len(selected),
        "published_views": len(published), "semantic_groups": len(groups),
        "query_count": len({row["query_id"] for row in published}),
        "correct_views": sum(row["correctness"] for row in published),
        "inherited_auxiliary_supervision_fields": [],
        "style_ids_are_view_roles_not_taxonomy": True,
        "formal_mechanism_claim_allowed": False,
        "manifest": str(manifest.relative_to(ROOT)), "manifest_sha256": sha256(manifest),
    }
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
