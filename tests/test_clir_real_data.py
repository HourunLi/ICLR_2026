import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.clir_data import CLIRTrajectoryDataset, write_jsonl
from src.clir_real_data import (
    build_gsm8k_prompt,
    check_gsm8k_response,
    extract_aligned_hidden_states,
    load_protocol,
    validate_extracted_row,
)


class FakeCausalLM(torch.nn.Module):
    """Deterministic causal-model-shaped fixture with embedding + two blocks."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input_ids,
        attention_mask,
        output_hidden_states,
        use_cache,
        return_dict,
    ):
        assert output_hidden_states is True
        assert use_cache is False
        assert return_dict is True
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
        base = torch.stack((input_ids.float(), positions.float()), dim=-1)
        return SimpleNamespace(hidden_states=(base, base + 100.0, base + 200.0))


def _provenance():
    return {
        "model_id": "fake/model",
        "model_revision": "revision",
        "tokenizer_revision": "revision",
        "protocol_version": "test-v1",
    }


def test_exact_token_ids_drive_all_layer_extraction():
    result = extract_aligned_hidden_states(
        FakeCausalLM(),
        prompt_token_ids=[10, 11, 12],
        output_token_ids=[20, 21],
        storage_dtype="float32",
    )

    assert result.layer_count == 3
    assert result.per_layer_hidden_size == 2
    assert result.feature_dim == 6
    assert result.condition.shape == (3, 6)
    assert result.trajectory.shape == (2, 6)
    assert result.condition.untyped_storage().nbytes() == (
        result.condition.numel() * result.condition.element_size()
    )
    assert result.trajectory.untyped_storage().nbytes() == (
        result.trajectory.numel() * result.trajectory.element_size()
    )
    assert result.trajectory[0].tolist() == [20.0, 3.0, 120.0, 103.0, 220.0, 203.0]
    assert result.trajectory[1].tolist() == [21.0, 4.0, 121.0, 104.0, 221.0, 204.0]

    reused = extract_aligned_hidden_states(
        FakeCausalLM(),
        prompt_token_ids=[10, 11, 12],
        output_token_ids=[30],
        storage_dtype="float32",
        canonical_condition=result.condition,
    )
    assert reused.condition.data_ptr() == result.condition.data_ptr()
    assert reused.trajectory[0].tolist() == [30.0, 3.0, 130.0, 103.0, 230.0, 203.0]


def test_strict_alignment_rejects_token_label_mismatch():
    row = {
        "id": "candidate",
        "query_id": "query",
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4],
        "response": "answer",
        "token_advantage": [0.0],
        "provenance": _provenance(),
    }
    with pytest.raises(ValueError, match="token_advantage.*length mismatch"):
        validate_extracted_row(row, torch.zeros(2, 4), torch.zeros(2, 4))


def test_real_manifest_loader_rejects_hidden_state_mismatch(tmp_path: Path):
    hidden_path = tmp_path / "bad.pt"
    torch.save(torch.zeros(3, 4), hidden_path)
    row = {
        "id": "candidate",
        "query_id": "query",
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4],
        "response": "answer",
        "hidden_states_path": str(hidden_path),
        "provenance": _provenance(),
    }
    manifest = tmp_path / "bad.jsonl"
    write_jsonl(manifest, [row])
    dataset = CLIRTrajectoryDataset(manifest)

    with pytest.raises(ValueError, match="Trajectory feature length mismatch"):
        dataset[0]


def test_gsm8k_prompt_and_checker():
    prompt = build_gsm8k_prompt("What is 2 + 2?")
    assert prompt.endswith("What is 2 + 2?")
    assert "\\boxed{Your Answer}" in prompt

    correct = check_gsm8k_response("Therefore, \\boxed{1,234}.", "work #### 1234")
    assert correct["correctness"] == 1
    assert correct["checker_status"] == "numeric_match"

    fraction = check_gsm8k_response("The result is \\boxed{\\frac{3}{2}}.", "#### 1.5")
    assert fraction["correctness"] == 1

    with_units = check_gsm8k_response("The result is \\boxed{160 minutes}.", "#### 160")
    assert with_units["correctness"] == 1
    assert with_units["normalized_candidate_answer"] == "160"
    assert with_units["checker_normalization"] == "numeric_subexpression"

    wrong_with_units = check_gsm8k_response("The result is \\boxed{140 minutes}.", "#### 160")
    assert wrong_with_units["correctness"] == 0
    assert wrong_with_units["checker_status"] == "numeric_mismatch"

    fraction_with_units = check_gsm8k_response(
        "The result is \\boxed{\\frac{3}{2} meters}.", "#### 1.5"
    )
    assert fraction_with_units["correctness"] == 1
    assert fraction_with_units["normalized_candidate_answer"] == "\\frac{3}{2}"

    failed = check_gsm8k_response("I cannot solve this.", "#### 4")
    assert failed["correctness"] == 0
    assert failed["checker_status"] == "parse_failed"


def test_frozen_protocol_uses_full_layers_and_exact_ids():
    protocol_path = Path(__file__).resolve().parents[1] / "configs" / "phi35_gsm8k_pilot_v1.json"
    protocol = load_protocol(protocol_path)

    assert protocol["model"]["repo_id"] == "microsoft/Phi-3.5-mini-instruct"
    assert protocol["generation"]["pilot_eval_candidates"] == 16
    assert protocol["hidden_states"]["include_embedding_output"] is True
    assert protocol["hidden_states"]["include_all_transformer_blocks"] is True
    assert protocol["hidden_states"]["trajectory_slice"] == "exact_output_token_ids"
    assert protocol["hidden_states"]["condition_forward"] == "prompt_only_once_per_query"
    assert protocol["reward_architecture"]["input_dim"] == 33 * 3072
    assert protocol["reward_architecture"]["primary_encoder"]["type"] == "layer_transformer"
    assert protocol["reward_architecture"]["primary_encoder"]["model_dim"] == 768
    assert set(protocol["reward_architecture"]["required_variants"]) == {
        "strict_swift",
        "encoded_swift",
        "clir",
    }
    reconstruction = protocol["reward_architecture"]["complete_reconstruction_target"]
    assert reconstruction["status"] == "absent_in_stage1"
    assert reconstruction["candidate_self_pooling_forbidden"] is True
    assert reconstruction["loss_when_absent"] == 0.0
    assert protocol["correctness"]["checker"] == "clir_gsm8k_numeric_v2"
    assert protocol["evaluation"]["pilot_primary"] == "bon_at_16_accuracy"
