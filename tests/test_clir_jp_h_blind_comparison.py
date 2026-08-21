import json
from pathlib import Path

from scripts.compare_jp_h_blind_annotations_v1 import (
    VIEW_SALT,
    adjudication_prompt,
    assignment,
    compare_labels,
    positive_token_positions,
)
from src.clir_hallucination_annotation import file_sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/jp_h_blind_validation_v1/comparison_protocol_v1.json"
RUNNER = ROOT / "scripts/compare_jp_h_blind_annotations_v1.py"


def mapped_label(
    item_id: str,
    *,
    path_status: str,
    claims: list[tuple[int, int, str]],
) -> dict:
    positive = [claim for claim in claims if claim[2] in {"contradicted", "unsupported"}]
    onset = min((claim[0] for claim in positive), default=-1)
    return {
        "item_id": item_id,
        "output_token_ids_sha256": "token-hash",
        "path_status": path_status,
        "hallucination_onset": onset,
        "earliest_problem_claim_index": 0 if positive else None,
        "confidence": "high",
        "summary": "test",
        "claim_reviews": [
            {
                "claim_text": f"claim-{index}",
                "occurrence": 0,
                "status": status,
                "reason": "test",
                "char_start": start,
                "char_end": end,
                "token_start": start,
                "token_end_exclusive": end,
            }
            for index, (start, end, status) in enumerate(claims)
        ],
    }


def test_protocol_freezes_hashes_and_dual_view_gate() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_opening_primary_or_secondary_label_values"
    for spec in protocol["inputs"].values():
        assert file_sha256(ROOT / spec["path"]) == spec["sha256"]
    assert protocol["prediction_values_may_be_opened_during_comparison"] is False
    views = protocol["post_adjudication_evaluation_views"]
    assert views["views"] == ["resolved_primary", "resolved_secondary"]
    assert views["fusion"] == "none"
    assert "both resolved views" in views["adoption_gate"]


def test_positive_positions_are_exact_union() -> None:
    label = mapped_label(
        "x",
        path_status="hallucinated",
        claims=[(1, 4, "unsupported"), (3, 6, "contradicted"), (7, 9, "supported")],
    )
    assert positive_token_positions(label) == frozenset({1, 2, 3, 4, 5})


def test_path_or_positive_target_mismatch_is_blocking() -> None:
    clean = mapped_label("x", path_status="clean", claims=[(0, 3, "supported")])
    bad = mapped_label(
        "x", path_status="hallucinated", claims=[(0, 3, "unsupported")]
    )
    compared = compare_labels(clean, bad)
    assert compared["requires_adjudication"] is True
    assert set(compared["blocking_disagreement_reasons"]) == {
        "path_status_mismatch",
        "exact_positive_token_set_mismatch",
    }


def test_negative_coverage_only_is_dual_view_not_blocking() -> None:
    left = mapped_label("x", path_status="clean", claims=[(0, 3, "supported")])
    right = mapped_label("x", path_status="clean", claims=[(0, 5, "supported")])
    compared = compare_labels(left, right)
    assert compared["requires_adjudication"] is False
    assert compared["disagreement_reasons"] == [
        "reviewed_negative_token_set_mismatch"
    ]


def test_positive_boundary_difference_is_blocking_even_with_same_onset() -> None:
    left = mapped_label(
        "x", path_status="hallucinated", claims=[(2, 4, "unsupported")]
    )
    right = mapped_label(
        "x", path_status="hallucinated", claims=[(2, 6, "unsupported")]
    )
    compared = compare_labels(left, right)
    assert compared["primary_hallucination_onset"] == 2
    assert compared["secondary_hallucination_onset"] == 2
    assert compared["requires_adjudication"] is True
    assert compared["blocking_disagreement_reasons"] == [
        "exact_positive_token_set_mismatch"
    ]


def test_view_assignment_is_deterministic_and_balanced_role_only() -> None:
    assert VIEW_SALT == "clir-jp-h-blind-adjudication-view-v1"
    assert assignment("item-a") == assignment("item-a")
    assert set(assignment("item-a")) == {"primary", "secondary"}


def test_prompt_requires_per_row_checkpoint_and_hides_scores(tmp_path: Path) -> None:
    prompt = adjudication_prompt(
        guide=tmp_path / "guide.md",
        items=tmp_path / "items.jsonl",
        output=tmp_path / "out.jsonl",
    )
    assert "append and fsync that one decision immediately" in prompt
    assert "prediction/score files" in prompt
    assert "annotation_a" in prompt and "annotation_b" in prompt


def test_comparison_runner_does_not_name_prediction_artifact() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "predictions_v1.jsonl" not in source
    assert "raw_scores" not in source
    assert "smoothed_scores" not in source
