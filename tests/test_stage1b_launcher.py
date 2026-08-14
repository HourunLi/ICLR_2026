from pathlib import Path

from scripts.run_stage1b_validation import (
    DEFAULT_PROTOCOL,
    evaluation_command,
    load_stage1b_protocol,
    scoring_command,
    training_command,
    validate_supervision_contract,
)


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_stage1b_v3_launcher_pins_training_and_scoring_contracts():
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
    assert _value_after(train, "--num_workers") == "4"
    assert "--no-group_by_semantic_id" in train
    assert "--skip_feature_finite_check" in train
    assert "--epoch_checkpoint_dir" in train
    assert _value_after(score, "--amp_dtype") == "none"
    assert _value_after(score, "--min_score_std") == "0.1"
    assert _value_after(evaluate, "--bootstrap-replicates") == "10000"
    assert _value_after(evaluate, "--expected-scoring-amp-dtype") == "none"


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


def test_stage1b_v3_freezes_zero_clir_supervision_coverage():
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


def test_stage1b_v3_rejects_accidental_auxiliary_labels():
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
        raise AssertionError("Stage 1B v3 accepted non-frozen auxiliary supervision")
