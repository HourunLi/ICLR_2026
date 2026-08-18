from __future__ import annotations

import pytest

from scripts.calibrate_hallucination_span_thresholds_v2 import explicit_token_targets
from scripts.materialize_hallucination_span_targets_v2 import ROOT, derive_annotation
from scripts.materialize_hallucination_tail_cv_v2c import partition_original_train
from scripts.run_hallucination_localization_pilot_v2 import (
    resolve_fold_inputs,
    resolve_training_seed,
    weight_args,
)
from scripts.run_hallucination_tail_cv_matrix_v2c import matrix_jobs
from scripts.summarize_hallucination_tail_cv_v2c import adoption_decision
from scripts.summarize_hallucination_tail_comparison_v2b import (
    paired_bootstrap,
    tail_gate,
    tail_label_composition,
    value_diagnostics,
)
from scripts.summarize_hallucination_span_pilot_v2 import select_span_cell
from src.clir_real_data import load_protocol
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


def test_span_threshold_calibration_uses_only_explicitly_reviewed_tokens():
    labels, scores = explicit_token_targets(
        [
            {
                "clir_token_hallucination_probs": [0.1, 0.8, 0.99],
                "token_hallucination_target": [0, 1, 0],
                "token_hallucination_mask": [1, 1, 0],
            }
        ]
    )

    assert labels == [0, 1]
    assert scores == [0.1, 0.8]


def test_pilot_v2_cells_keep_downstream_shaping_off_and_change_only_declared_factors():
    protocol = load_protocol(
        ROOT / "configs/hallucination_localization_v2/training_protocol_v2.json"
    )
    s0 = weight_args(protocol, "s0_tail_bce")
    s3 = weight_args(protocol, "s3_span_balanced_path")

    assert s0[s0.index("--mil_weight") + 1] == "0.0"
    assert s3[s3.index("--mil_weight") + 1] == "0.25"
    for args in (s0, s3):
        assert args[args.index("--tail_weight") + 1] == "0.0"
        assert args[args.index("--pseudo_tail_weight") + 1] == "0.0"
        assert args[args.index("--consistency_weight") + 1] == "0.0"
        assert args[args.index("--prior_weight") + 1] == "0.0"


def test_pilot_v2_launcher_allows_explicit_per_cell_tail_ablation():
    protocol = load_protocol(
        ROOT / "configs/hallucination_localization_v2/training_protocol_v2.json"
    )
    protocol["cells"]["s1_span_bce"]["tail_weight"] = 0.1

    args = weight_args(protocol, "s1_span_bce")

    assert args[args.index("--tail_weight") + 1] == "0.1"
    assert args[args.index("--hallucination_weight") + 1] == "1.0"
    assert args[args.index("--mil_weight") + 1] == "0.0"


def test_tail_cv_partition_is_deterministic_stratified_and_exhaustive():
    rows = [
        {
            "id": f"row-{label}-{index}",
            "query_id": f"query-{label}-{index}",
            "path_hallucinated": label,
        }
        for label, count in ((0, 31), (1, 17))
        for index in range(count)
    ]

    first = partition_original_train(rows)
    second = partition_original_train(list(reversed(rows)))

    assert {
        fold: [row["id"] for row in fold_rows] for fold, fold_rows in first.items()
    } == {
        fold: [row["id"] for row in fold_rows] for fold, fold_rows in second.items()
    }
    assert [len(first[fold]) for fold in (1, 2, 3)] == [16, 16, 16]
    assert [
        sum(int(row["path_hallucinated"]) for row in first[fold])
        for fold in (1, 2, 3)
    ] == [6, 6, 5]
    assert len({row["id"] for fold in first.values() for row in fold}) == 48


def test_tail_cv_launcher_requires_frozen_seed_and_fold():
    protocol = {
        "inputs": {},
        "matched_training": {"seeds": [42, 43, 44]},
        "cross_validation": {
            "folds": {
                "1": {"train": {"path": "train"}, "dev": {"path": "dev"}}
            }
        },
    }

    assert resolve_training_seed(protocol, 43) == 43
    fold, train, dev = resolve_fold_inputs(protocol, 1)
    assert fold == 1
    assert train["path"] == "train"
    assert dev["path"] == "dev"
    with pytest.raises(ValueError, match="explicit --seed"):
        resolve_training_seed(protocol, None)
    with pytest.raises(ValueError, match="not frozen"):
        resolve_training_seed(protocol, 45)
    with pytest.raises(ValueError, match="explicit --fold"):
        resolve_fold_inputs(protocol, None)


def test_tail_cv_matrix_freezes_22_new_cells_and_reuses_fold0_seed42():
    protocol = load_protocol(
        ROOT / "configs/hallucination_localization_v2/tail_cv_protocol_v2c.json"
    )

    jobs = matrix_jobs(protocol)

    assert len(jobs) == 22
    assert {job["cell"] for job in jobs} == {
        "t0_span_only",
        "t2_span_tail_historical",
    }
    assert not any(job["fold"] == 0 and job["seed"] == 42 for job in jobs)


def test_tail_cv_adoption_requires_two_seed_guards_mean_guards_and_no_catastrophe():
    protocol = load_protocol(
        ROOT / "configs/hallucination_localization_v2/tail_cv_protocol_v2c.json"
    )
    gates = {
        "42": {"all_pilot_guards_passed": True},
        "43": {"all_pilot_guards_passed": True},
        "44": {"all_pilot_guards_passed": False},
    }
    base_delta = {
        "tail_minus_pre_gap": -0.2,
        "tail_minus_clean_gap": -0.1,
        "explicit_token_value_risk_average_precision": 0.03,
        "span_hallucination_probability_average_precision": 0.01,
        "reward_score_correctness_roc_auc": 0.01,
    }
    deltas = {seed: dict(base_delta) for seed in gates}

    passed = adoption_decision(gates, deltas, protocol)
    deltas["44"]["span_hallucination_probability_average_precision"] = -0.06
    failed = adoption_decision(gates, deltas, protocol)

    assert passed["passed"] is True
    assert passed["selected_cell"] == "t2_span_tail_historical"
    assert failed["passed"] is False
    assert failed["no_catastrophic_seed"] is False


def test_tail_value_diagnostics_keep_supported_post_onset_tokens_visible():
    diagnostics = value_diagnostics(
        [
            {
                "id": "positive",
                "correctness": 0,
                "reward_score": -1.0,
                "hallucination_onset": 2,
                "clir_token_values": [1.0, 0.5, -2.0, -1.0, -0.25],
                "token_hallucination_target": [0, 0, 1, 0, 0],
                "token_hallucination_mask": [1, 1, 1, 1, 0],
            },
            {
                "id": "clean",
                "correctness": 1,
                "reward_score": 1.0,
                "hallucination_onset": -1,
                "clir_token_values": [2.0, 1.5],
                "token_hallucination_target": [0, 0],
                "token_hallucination_mask": [1, 1],
            },
        ]
    )

    populations = diagnostics["token_value_populations"]
    semantic = diagnostics["post_onset_semantic_audit"]
    assert populations["mean_pre_onset"] == 0.75
    assert populations["mean_tail"] == pytest.approx(-3.25 / 3)
    assert populations["mean_clean"] == 1.75
    assert semantic["explicit_hallucinated_tokens"] == 1
    assert semantic["explicit_supported_tokens"] == 1
    assert semantic["unreviewed_tokens"] == 1
    assert semantic["hallucinated_mean_minus_supported_mean"] == -1.0


def test_tail_label_composition_does_not_hide_supported_or_unreviewed_tail():
    composition = tail_label_composition(
        [
            {
                "hallucination_onset": 1,
                "token_hallucination_target": [0, 1, 0, 0],
                "token_hallucination_mask": [1, 1, 1, 0],
            },
            {
                "hallucination_onset": -1,
                "token_hallucination_target": [0, 0],
                "token_hallucination_mask": [1, 1],
            },
        ]
    )

    assert composition["hallucinated_rows"] == 1
    assert composition["tail_tokens"] == 3
    assert composition["explicit_hallucinated_tokens"] == 1
    assert composition["explicit_supported_tokens"] == 1
    assert composition["unreviewed_tokens"] == 1


def test_tail_gate_requires_locality_semantic_span_and_correctness_guards():
    def cell(tail_pre, tail_clean, value_ap, span_ap, correctness_auc):
        return {
            "span_token_average_precision": span_ap,
            "value_diagnostics": {
                "token_value_populations": {
                    "tail_mean_minus_pre_onset_mean": tail_pre,
                    "tail_mean_minus_clean_mean": tail_clean,
                },
                "explicit_token_value_localization": {
                    "average_precision": value_ap,
                },
                "reward_score_correctness": {"roc_auc": correctness_auc},
            },
        }

    control = cell(0.2, -1.0, 0.4, 0.42, 0.70)
    passed = tail_gate(control, cell(-0.2, -1.5, 0.41, 0.41, 0.66))
    failed = tail_gate(control, cell(-0.2, -1.5, 0.39, 0.41, 0.66))

    assert passed["all_pilot_guards_passed"] is True
    assert failed["semantic_value_guard_passed"] is False
    assert failed["all_pilot_guards_passed"] is False


def test_tail_bootstrap_skips_single_class_resamples_without_failing():
    control = [
        {
            "id": "negative",
            "correctness": 1,
            "reward_score": 1.0,
            "hallucination_onset": -1,
            "clir_token_values": [1.0],
            "clir_token_hallucination_probs": [0.1],
            "token_hallucination_target": [0],
            "token_hallucination_mask": [1],
        },
        {
            "id": "positive",
            "correctness": 0,
            "reward_score": -1.0,
            "hallucination_onset": 1,
            "clir_token_values": [0.0, -0.5],
            "clir_token_hallucination_probs": [0.1, 0.7],
            "token_hallucination_target": [0, 1],
            "token_hallucination_mask": [0, 1],
        },
    ]
    candidate = [
        dict(control[0]),
        {
            **control[1],
            "clir_token_values": [0.0, -1.0],
            "clir_token_hallucination_probs": [0.1, 0.8],
        },
    ]

    report = paired_bootstrap(control, candidate, samples=100, seed=42)

    metrics = report["candidate_minus_control"]
    assert metrics["span_hallucination_probability_average_precision"][
        "valid_resamples"
    ] < 100
    assert metrics["span_hallucination_probability_average_precision"][
        "valid_resamples"
    ] > 0


def test_span_cell_selection_prefers_simpler_cell_on_ties_and_requires_both_shortcuts():
    cells = {
        "s0_tail_bce": {
            "span_token_average_precision": 0.30,
            "claim_mean_average_precision": 0.30,
        },
        "s1_span_bce": {
            "span_token_average_precision": 0.50,
            "claim_mean_average_precision": 0.50,
        },
        "s2_span_balanced": {
            "span_token_average_precision": 0.50,
            "claim_mean_average_precision": 0.60,
        },
        "s3_span_balanced_path": {
            "span_token_average_precision": 0.40,
            "claim_mean_average_precision": 0.70,
        },
    }

    selected = select_span_cell(
        cells,
        token_position_ap=0.40,
        claim_position_ap=0.40,
    )

    assert selected["selected_cell"] == "s1_span_bce"
    assert selected["token_gate_passed"] is True
