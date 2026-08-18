from __future__ import annotations

import pytest

from scripts.materialize_hallucination_span_targets_v2 import ROOT, derive_annotation
from src.clir_supervision import output_token_ids_sha256


MAPPING = {
    "positive_statuses": ["unsupported", "contradicted"],
    "negative_statuses": ["supported"],
    "masked_statuses": ["non_claim"],
}


def _row():
    return {
        "id": "candidate",
        "query_id": "query",
        "output_token_ids": [10, 11, 12, 13, 14, 15],
    }


def _label():
    row = _row()
    return {
        "id": row["id"],
        "query_id": row["query_id"],
        "output_token_ids_sha256": output_token_ids_sha256(row["output_token_ids"]),
        "path_hallucinated": 1,
        "hallucination_onset": 3,
        "claim_reviews": [
            {
                "claim_index": 0,
                "status": "supported",
                "token_start": 0,
                "token_end_exclusive": 2,
            },
            {
                "claim_index": 1,
                "status": "non_claim",
                "token_start": 2,
                "token_end_exclusive": 3,
            },
            {
                "claim_index": 2,
                "status": "contradicted",
                "token_start": 3,
                "token_end_exclusive": 5,
            },
        ],
    }


def _derive(label):
    return derive_annotation(
        _row(),
        label,
        protocol_path=ROOT / "configs/hallucination_localization_v2/span_target_protocol_v2.json",
        protocol_sha256="a" * 64,
        labels_path=ROOT / "configs/hallucination_localization_v1/labels_adjudicated_v1.jsonl",
        labels_sha256="b" * 64,
        mapping=MAPPING,
    )


def test_claim_spans_become_sparse_exact_token_targets_without_tail_fill():
    annotation, spans, counts = _derive(_label())

    assert annotation["token_hallucination_target"] == [0, 0, 0, 1, 1, 0]
    assert annotation["token_hallucination_mask"] == [1, 1, 0, 1, 1, 0]
    assert [span["target"] for span in spans] == [0, 1]
    assert counts["supervised_tokens"] == 4
    assert counts["positive_tokens"] == 2


def test_claim_span_derivation_rejects_onset_or_conflicting_overlap_drift():
    wrong_onset = _label()
    wrong_onset["hallucination_onset"] = 4
    with pytest.raises(ValueError, match="first explicit positive token"):
        _derive(wrong_onset)

    overlap = _label()
    overlap["claim_reviews"].append(
        {
            "claim_index": 3,
            "status": "supported",
            "token_start": 4,
            "token_end_exclusive": 6,
        }
    )
    with pytest.raises(ValueError, match="conflicting token targets"):
        _derive(overlap)
