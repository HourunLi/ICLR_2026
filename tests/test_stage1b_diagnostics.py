from pathlib import Path

from scripts.diagnose_stage1b import build_diagnostic_report
from src.clir_data import write_jsonl


def _scored_rows(score_shift: float = 0.0):
    rows = []
    labels = {
        "mixed": [0, 1, 0, 1],
        "correct": [1, 1, 1, 1],
        "wrong": [0, 0, 0, 0],
    }
    scores = {
        "mixed": [0.1, 0.8 + score_shift, 0.4, 0.7],
        "correct": [0.1, 0.2, 0.3, 0.4],
        "wrong": [0.4, 0.3, 0.2, 0.1],
    }
    for query_id in labels:
        for candidate_index in range(4):
            rows.append({
                "id": f"{query_id}-{candidate_index}",
                "query_id": query_id,
                "candidate_index": candidate_index,
                "correctness": labels[query_id][candidate_index],
                "reward_score": scores[query_id][candidate_index],
                "output_token_ids": list(range(candidate_index + 2)),
                "generation": {"candidate_index_policy": "vllm_completion_output_index"},
                "mean_layer_pool_attention": [[0.25, 0.75], [0.5, 0.5]],
            })
    return rows


def test_stage1b_diagnostic_uses_only_mixed_full_pools(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    labels = tmp_path / "labels.jsonl"
    write_jsonl(first, _scored_rows())
    write_jsonl(second, _scored_rows(0.1))
    label_rows = [dict(row, checker_version="clir_gsm8k_numeric_v3") for row in _scored_rows()]
    write_jsonl(labels, label_rows)

    report = build_diagnostic_report(
        [("clir", 42, first), ("clir", 43, second)],
        label_path=labels,
        max_k=4,
    )

    assert report["candidate_order"]["ordered_prefix_valid"] is True
    assert report["runs"][0]["pool_counts"] == {
        "mixed": 1,
        "all_correct": 1,
        "all_wrong": 1,
    }
    assert report["runs"][0]["mixed_pool_metrics"]["selection_accuracy"] == 1.0
    assert report["runs"][0]["layer_attention"]["layer_count"] == 2
    assert report["method_aggregates"]["clir"]["unanimous_selected_candidate_fraction"] == 1.0


def test_stage1b_diagnostic_marks_legacy_candidate_order(tmp_path: Path):
    manifest = tmp_path / "legacy.jsonl"
    rows = _scored_rows()
    for row in rows:
        row["generation"].pop("candidate_index_policy")
    write_jsonl(manifest, rows)

    report = build_diagnostic_report(
        [("strict_swift", 42, manifest)],
        label_path=None,
        max_k=4,
    )

    assert report["candidate_order"]["ordered_prefix_valid"] is False
    assert "full max-k pool only" in report["candidate_order"]["diagnostic_scope"]
