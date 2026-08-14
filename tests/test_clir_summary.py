from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from evaluate_clir import evaluate_candidate_rows
from src.clir_real_data import file_sha256
from summarize_clir import (
    _evaluation_failure_evidence,
    _health_failure_evidence,
    summarize_evaluation_reports,
)


def _frozen_contract(protocol_sha256="9" * 64):
    return {
        "sha256": protocol_sha256,
        "minimum_prior_relative_improvement": 0.01,
        "minimum_score_population_std": 0.1,
        "minimum_within_query_pairwise_accuracy": 0.6,
        "train_input_sha256": "e" * 64,
        "validation_input_sha256": "f" * 64,
        "scoring_batch_size": 2,
        "scoring_amp_dtype": "none",
        "scoring_compute_dtype": "float32",
        "evaluation_query_count": 2,
        "evaluation_k": [1, 2],
        "bootstrap_replicates": 10,
        "confidence_level": 0.95,
    }


def _report(
    scores,
    variant="clir",
    checkpoint_character="a",
    *,
    minimum_pairwise_accuracy=0.0,
    protocol_sha256=None,
    minimum_score_std=0.0,
):
    flattened_scores = np.asarray(
        [score for query_scores in scores.values() for score in query_scores],
        dtype=np.float64,
    )
    provenance = {
        "schema_version": "clir-reward-scoring-v2",
        "model_variant": variant,
        "checkpoint_sha256": checkpoint_character * 64,
        "input_sha256": "f" * 64,
        "batch_size": 2,
        "amp_dtype": "none",
        "compute_dtype": "float32",
        "min_score_std": minimum_score_std,
        "score_distribution": {
            "count": int(flattened_scores.size),
            "mean": float(flattened_scores.mean()),
            "population_std": float(flattened_scores.std(ddof=0)),
            "min": float(flattened_scores.min()),
            "max": float(flattened_scores.max()),
        },
    }
    if protocol_sha256 is not None:
        provenance["experiment_protocol"] = {"sha256": protocol_sha256}
    rows = []
    labels = {"q0": [0, 1], "q1": [1, 0]}
    for query_id in ("q0", "q1"):
        for candidate_index in range(2):
            rows.append({
                "query_id": query_id,
                "candidate_index": candidate_index,
                "correctness": labels[query_id][candidate_index],
                "reward_score": scores[query_id][candidate_index],
                "generation": {"candidate_index_policy": "vllm_completion_output_index"},
                "reward_model_variant": variant,
                "reward_scoring_provenance": dict(provenance),
            })
    return evaluate_candidate_rows(
        rows,
        score_field="reward_score",
        k_values=[1, 2],
        bootstrap_replicates=10,
        seed=1,
        minimum_within_query_pairwise_accuracy=minimum_pairwise_accuracy,
    )


def test_multiseed_summary_reports_sample_std_and_paired_comparisons():
    strict = _report({"q0": [0.9, 0.1], "q1": [0.9, 0.1]}, "strict_swift", "a")
    encoded_good = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b")
    encoded_bad = _report({"q0": [0.9, 0.1], "q1": [0.1, 0.9]}, "encoded_swift", "c")
    clir = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "d")
    strict_second_seed = _report(
        {"q0": [0.9, 0.1], "q1": [0.9, 0.1]}, "strict_swift", "e"
    )
    clir_second_seed = _report(
        {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "f"
    )
    reports = {
        1: {"strict_swift": strict, "encoded_swift": encoded_good, "clir": clir},
        2: {
            "strict_swift": strict_second_seed,
            "encoded_swift": encoded_bad,
            "clir": clir_second_seed,
        },
    }

    summary = summarize_evaluation_reports(
        reports,
        primary_k=2,
        bootstrap_replicates=20,
        bootstrap_seed=7,
    )

    encoded = summary["per_variant"]["encoded_swift"]["metrics"]["2"][
        "reward_bon_accuracy"
    ]
    assert encoded["by_seed"] == {"1": 1.0, "2": 0.0}
    assert encoded["mean"] == 0.5
    assert encoded["sample_std"] == pytest.approx(2 ** -0.5)
    comparison = summary["primary_comparisons"]["encoded_swift_to_clir"]
    assert comparison["by_seed"] == {"1": 0.0, "2": 1.0}
    assert comparison["mean"] == 0.5
    assert set(comparison["paired_query_bootstrap_ci_by_seed"]) == {"1", "2"}
    aggregate = comparison["aggregate_query_paired"]
    assert aggregate["unit"] == "query"
    assert aggregate["seed_aggregation_within_query"] == "arithmetic_mean"
    assert aggregate["query_count"] == 2
    assert aggregate["training_seed_count"] == 2
    assert aggregate["mean"] == 0.5
    assert aggregate["bootstrap_ci"] == [0.5, 0.5]
    assert summary["paired_bootstrap"]["unit"] == "query"
    assert summary["paired_bootstrap"]["aggregate_definition"] == (
        "mean_across_training_seeds_within_query_then_bootstrap_queries"
    )


def test_multiseed_summary_rejects_candidate_baseline_mismatch():
    strict = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a")
    encoded = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b")
    clir = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "c")
    corrupted = deepcopy(encoded)
    corrupted["metrics"]["2"]["oracle_accuracy"]["value"] = 0.0
    reports = {
        1: {
            "strict_swift": strict,
            "encoded_swift": corrupted,
            "clir": clir,
        }
    }

    with pytest.raises(ValueError, match="baseline mismatch"):
        summarize_evaluation_reports(reports, primary_k=2)


def test_multiseed_summary_rejects_per_query_baseline_mismatch_with_same_mean():
    strict = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a")
    encoded = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b")
    clir = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "c")
    corrupted = deepcopy(encoded)
    corrupted["per_query"][0]["k"]["2"]["random_expected"] = 1.0
    corrupted["per_query"][1]["k"]["2"]["random_expected"] = 0.0
    reports = {
        1: {
            "strict_swift": strict,
            "encoded_swift": corrupted,
            "clir": clir,
        }
    }

    with pytest.raises(ValueError, match="Per-query baseline mismatch"):
        summarize_evaluation_reports(reports, primary_k=2)


def test_multiseed_summary_rejects_historical_candidate_subset():
    strict = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a")
    encoded = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b")
    clir = _report({"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "c")
    strict["candidate_subset"] = "first_k_in_generation_order"

    with pytest.raises(ValueError, match="Candidate subset mismatch"):
        summarize_evaluation_reports(
            {1: {"strict_swift": strict, "encoded_swift": encoded, "clir": clir}},
            primary_k=2,
        )


def test_multiseed_summary_rejects_checkpoint_reuse_across_seeds():
    reports = {
        1: {
            "strict_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a"
            ),
            "encoded_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b"
            ),
            "clir": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "c"
            ),
        },
        2: {
            "strict_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a"
            ),
            "encoded_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "d"
            ),
            "clir": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "e"
            ),
        },
    }

    with pytest.raises(ValueError, match="reuses one checkpoint across cells"):
        summarize_evaluation_reports(reports, primary_k=2)


def test_incomplete_summary_uses_all_healthy_cells_but_disables_formal_claim():
    reports = {
        1: {
            "strict_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "a"
            ),
            "encoded_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "b"
            ),
            "clir": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "clir", "c"
            ),
        },
        2: {
            "strict_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "strict_swift", "d"
            ),
            "encoded_swift": _report(
                {"q0": [0.1, 0.9], "q1": [0.9, 0.1]}, "encoded_swift", "e"
            ),
        },
    }

    summary = summarize_evaluation_reports(
        reports,
        primary_k=2,
        bootstrap_replicates=10,
        allow_incomplete=True,
    )

    assert summary["matrix_complete"] is False
    assert summary["formal_primary_claim_allowed"] is False
    assert summary["result_status"] == "incomplete_diagnostic_only"
    assert summary["available_seeds_by_variant"] == {
        "strict_swift": [1, 2],
        "encoded_swift": [1, 2],
        "clir": [1],
    }
    assert summary["primary_comparisons"]["encoded_swift_to_clir"]["by_seed"] == {
        "1": 0.0
    }


def test_all_health_failed_cells_still_produce_diagnostic_summary_shell():
    summary = summarize_evaluation_reports(
        {42: {}, 43: {}, 44: {}},
        primary_k=16,
        allow_incomplete=True,
    )

    assert summary["result_status"] == "all_cells_failed_diagnostic_only"
    assert summary["formal_primary_claim_allowed"] is False
    assert summary["available_seeds_by_variant"] == {
        "strict_swift": [],
        "encoded_swift": [],
        "clir": [],
    }
    assert summary["primary_comparisons"] == {}


def test_frozen_summary_contract_rejects_disabled_ranking_gate():
    protocol_sha256 = "9" * 64
    report = _report(
        {"q0": [0.1, 0.9], "q1": [0.9, 0.1]},
        "clir",
        "a",
        protocol_sha256=protocol_sha256,
        minimum_score_std=0.1,
    )
    contract = _frozen_contract(protocol_sha256)

    with pytest.raises(ValueError, match="did not enable"):
        summarize_evaluation_reports(
            {1: {"clir": report}},
            variants=("clir",),
            primary_k=2,
            protocol_contract=contract,
        )

    gated_report = _report(
        {"q0": [0.1, 0.9], "q1": [0.9, 0.1]},
        "clir",
        "b",
        minimum_pairwise_accuracy=0.6,
        protocol_sha256=protocol_sha256,
        minimum_score_std=0.1,
    )
    summary = summarize_evaluation_reports(
        {1: {"clir": gated_report}},
        variants=("clir",),
        primary_k=2,
        protocol_contract=contract,
    )
    assert summary["matrix_complete"] is True


def test_failed_cell_requires_explicit_health_evidence(tmp_path: Path):
    run_dir = tmp_path / "models"
    scored_dir = tmp_path / "scored"
    run_path = run_dir / "seed_43" / "clir.run.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps({
            "status": "health_gate_failed",
            "model_variant": "clir",
            "experiment_protocol": {"sha256": "9" * 64},
            "data_state": {
                "train_sha256": "e" * 64,
                "val_sha256": "f" * 64,
            },
            "health_gate": {
                "schema_version": "clir-training-health-v2",
                "enabled": True,
                "passed": False,
                "gate": "constant_class_prior_bce",
                "constant_prior_bce": 0.5,
                "observed_train_correctness_bce": 0.5,
                "relative_improvement_over_prior_bce": 0.0,
                "minimum_relative_improvement": 0.01,
            },
        }),
        encoding="utf-8",
    )

    evidence = _health_failure_evidence(
        seed=43,
        variant="clir",
        run_dir=run_dir,
        scored_dir=scored_dir,
    )

    assert evidence["stage"] == "train"
    assert evidence["reason"] == "constant_prior_health_gate_failed"
    frozen_evidence = _health_failure_evidence(
        seed=43,
        variant="clir",
        run_dir=run_dir,
        scored_dir=scored_dir,
        protocol_contract=_frozen_contract(),
    )
    assert frozen_evidence["stage"] == "train"
    mismatched_contract = _frozen_contract()
    mismatched_contract["minimum_prior_relative_improvement"] = 0.02
    with pytest.raises(ValueError, match="does not match"):
        _health_failure_evidence(
            seed=43,
            variant="clir",
            run_dir=run_dir,
            scored_dir=scored_dir,
            protocol_contract=mismatched_contract,
        )
    with pytest.raises(FileNotFoundError, match="without explicit"):
        _health_failure_evidence(
            seed=44,
            variant="clir",
            run_dir=run_dir,
            scored_dir=scored_dir,
        )


def test_scoring_failure_evidence_is_bound_to_checkpoint_and_protocol(tmp_path: Path):
    run_dir = tmp_path / "models"
    scored_dir = tmp_path / "scored"
    model_path = run_dir / "seed_44" / "clir.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"checkpoint")
    score_health_path = scored_dir / "seed_44" / "clir.jsonl.health.json"
    score_health_path.parent.mkdir(parents=True)
    distribution = {
        "count": 2,
        "mean": 0.0,
        "population_std": 0.05,
        "min": -0.05,
        "max": 0.05,
    }
    provenance = {
        "model_variant": "clir",
        "checkpoint_sha256": file_sha256(model_path),
        "input_sha256": "f" * 64,
        "batch_size": 2,
        "amp_dtype": "none",
        "compute_dtype": "float32",
        "min_score_std": 0.1,
        "score_distribution": distribution,
        "experiment_protocol": {"sha256": "9" * 64},
    }
    score_health_path.write_text(
        json.dumps({
            "schema_version": "clir-scoring-health-v1",
            "status": "health_gate_failed",
            "model_variant": "clir",
            "checkpoint_sha256": file_sha256(model_path),
            "input_sha256": "f" * 64,
            "experiment_protocol": {"sha256": "9" * 64},
            "scoring_provenance": provenance,
            "health_gate": {
                "gate": "minimum_validation_score_population_std",
                "enabled": True,
                "passed": False,
                "minimum_population_std": 0.1,
                "observed_distribution": distribution,
            },
        }),
        encoding="utf-8",
    )

    evidence = _health_failure_evidence(
        seed=44,
        variant="clir",
        run_dir=run_dir,
        scored_dir=scored_dir,
        protocol_contract=_frozen_contract(),
    )

    assert evidence["stage"] == "score"
    provenance["checkpoint_sha256"] = "0" * 64
    corrupted = json.loads(score_health_path.read_text(encoding="utf-8"))
    corrupted["scoring_provenance"] = provenance
    score_health_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _health_failure_evidence(
            seed=44,
            variant="clir",
            run_dir=run_dir,
            scored_dir=scored_dir,
            protocol_contract=_frozen_contract(),
        )


def test_evaluation_failure_evidence_requires_frozen_full_pool_contract(tmp_path: Path):
    report = _report(
        {"q0": [0.9, 0.1], "q1": [0.1, 0.9]},
        "clir",
        "a",
        minimum_pairwise_accuracy=0.6,
        protocol_sha256="9" * 64,
        minimum_score_std=0.1,
    )
    path = tmp_path / "clir.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    evidence = _evaluation_failure_evidence(
        report,
        path=path,
        seed=1,
        variant="clir",
        protocol_contract=_frozen_contract(),
    )

    assert evidence is not None
    assert evidence["stage"] == "evaluate"
    report["k"] = [1]
    with pytest.raises(ValueError, match="does not identify"):
        _evaluation_failure_evidence(
            report,
            path=path,
            seed=1,
            variant="clir",
            protocol_contract=_frozen_contract(),
        )
