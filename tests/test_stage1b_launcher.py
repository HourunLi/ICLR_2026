from pathlib import Path
import json

import pytest

from scripts.run_stage1b_validation import (
    DEFAULT_PROTOCOL,
    _require_completed_model,
    evaluation_command,
    load_stage1b_protocol,
    scoring_command,
    summary_command,
    training_command,
    validate_supervision_contract,
)


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_stage1b_v4_launcher_pins_training_scoring_and_ranking_contracts():
    protocol_path, protocol = load_stage1b_protocol(DEFAULT_PROTOCOL)
    train = training_command(
        "python",
        protocol_path,
        protocol,
        42,
        "clir",
        "cuda",
        force=False,
    )
    score = scoring_command(
        "python",
        protocol_path,
        protocol,
        42,
        "clir",
        "cuda",
        None,
        require_inputs=False,
    )
    evaluate = evaluation_command(
        "python",
        protocol_path,
        protocol,
        42,
        "clir",
        None,
        require_inputs=False,
    )

    assert _value_after(train, "--batch_size") == "2"
    assert _value_after(train, "--lr") == "0.0001"
    assert _value_after(train, "--amp_dtype") == "bfloat16"
    assert _value_after(train, "--max_grad_norm") == "1.0"
    assert _value_after(train, "--prior_collapse_tolerance") == "0.01"
    assert _value_after(train, "--num_workers") == "4"
    assert "--no-group_by_semantic_id" in train
    assert "--skip_feature_finite_check" in train
    assert "--epoch_checkpoint_dir" in train
    assert _value_after(score, "--amp_dtype") == "none"
    assert _value_after(score, "--min_score_std") == "0.1"
    assert _value_after(evaluate, "--bootstrap-replicates") == "10000"
    assert _value_after(evaluate, "--expected-scoring-amp-dtype") == "none"
    assert _value_after(
        evaluate, "--minimum-within-query-pairwise-accuracy"
    ) == "0.6"
    assert protocol["schema_version"] == "clir-stage1b-validation-v4"


def test_stage1b_strict_swift_command_does_not_build_the_shared_encoder():
    protocol_path, protocol = load_stage1b_protocol(DEFAULT_PROTOCOL)
    command = training_command(
        "python",
        protocol_path,
        protocol,
        44,
        "strict_swift",
        "cuda",
        force=False,
    )

    assert _value_after(command, "--encoder_type") == "identity"
    assert "--model_dim" not in command
    assert Path(_value_after(command, "--output_model")).name == "strict_swift.pt"


def test_stage1b_v4_freezes_zero_clir_supervision_coverage():
    _, protocol = load_stage1b_protocol(DEFAULT_PROTOCOL)
    rows = [
        {"id": "r0", "query_id": "q0", "output_token_ids": [1, 2]},
        {"id": "r1", "query_id": "q1", "output_token_ids": [3]},
    ]

    coverage = validate_supervision_contract(
        rows,
        protocol["supervision_audit"],
        split_name="train",
    )

    assert coverage["eligible_component_count"] == 0


def test_stage1b_v4_rejects_accidental_auxiliary_labels():
    _, protocol = load_stage1b_protocol(DEFAULT_PROTOCOL)
    row = {
        "id": "r0",
        "query_id": "q0",
        "output_token_ids": [1, 2],
        "progress_targets": [0.0, 1.0],
    }

    try:
        validate_supervision_contract(
            [row],
            protocol["supervision_audit"],
            split_name="train",
        )
    except ValueError as error:
        assert "supervision" in str(error).lower()
    else:
        raise AssertionError("Stage 1B v4 accepted non-frozen auxiliary supervision")


def test_epoch_snapshot_uses_its_own_training_health(tmp_path: Path):
    model = tmp_path / "epoch_001.pt"
    metrics = tmp_path / "clir.metrics.jsonl"
    run = tmp_path / "clir.run.json"
    model.write_bytes(b"snapshot")
    metrics.write_text(
        json.dumps({
            "epoch": 1,
            "training_health": {"enabled": True, "passed": True},
        }) + "\n",
        encoding="utf-8",
    )
    run.write_text(
        json.dumps({
            "status": "health_gate_failed",
            "health_gate": {"enabled": True, "passed": False},
        }),
        encoding="utf-8",
    )
    paths = {"model": model, "metrics": metrics, "run": run}

    evidence = _require_completed_model(paths, checkpoint_epoch=1)

    assert evidence["status"] == "epoch_snapshot_health_passed"
    with pytest.raises(RuntimeError, match="did not pass health gates"):
        _require_completed_model(paths)


def test_incomplete_summary_command_requires_explicit_failure_evidence_roots():
    _, protocol = load_stage1b_protocol(DEFAULT_PROTOCOL)

    command = summary_command(
        "python",
        DEFAULT_PROTOCOL,
        protocol,
        None,
        allow_failed_cells=True,
    )

    assert "--allow-failed-cells" in command
    assert _value_after(command, "--experiment-protocol-config") == str(DEFAULT_PROTOCOL)
    assert "--run-dir" in command
    assert "--scored-dir" in command
