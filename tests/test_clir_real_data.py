import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.clir_data import CLIRTrajectoryDataset, write_jsonl
from src.clir_real_data import (
    TOKEN_LABEL_ALIASES,
    build_gsm8k_prompt,
    check_gsm8k_response,
    extract_aligned_hidden_states,
    load_protocol,
    protocol_hashes,
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
        "correctness": 1,
        "token_advantage": [0.0],
        "provenance": _provenance(),
    }
    with pytest.raises(ValueError, match="token_advantage.*length mismatch"):
        validate_extracted_row(row, torch.zeros(2, 4), torch.zeros(2, 4))


def test_finite_scan_can_only_be_skipped_explicitly_after_artifact_validation():
    row = {
        "id": "candidate",
        "query_id": "query",
        "prompt_token_ids": [1],
        "output_token_ids": [2],
        "response": "answer",
        "correctness": 1,
        "provenance": _provenance(),
    }
    feature = torch.tensor([[float("nan"), 0.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_extracted_row(row, feature)
    validate_extracted_row(row, feature, check_finite=False)


def test_real_extraction_requires_binary_correctness():
    base = {
        "id": "candidate",
        "query_id": "query",
        "prompt_token_ids": [1],
        "output_token_ids": [2],
        "response": "answer",
        "provenance": _provenance(),
    }
    with pytest.raises(ValueError, match="missing required `correctness`"):
        validate_extracted_row(base, torch.zeros(1, 2))

    invalid = {**base, "correctness": -1}
    with pytest.raises(ValueError, match="numeric 0 or 1"):
        validate_extracted_row(invalid, torch.zeros(1, 2))


@pytest.mark.parametrize(
    "alias",
    [alias for aliases in TOKEN_LABEL_ALIASES.values() for alias in aliases],
)
def test_every_token_label_alias_uses_strict_length_gate(alias: str):
    row = {
        "id": "candidate",
        "query_id": "query",
        "prompt_token_ids": [1],
        "output_token_ids": [2, 3],
        "response": "answer",
        "correctness": 1,
        alias: [1.0],
        "provenance": _provenance(),
    }
    with pytest.raises(ValueError, match=f"{alias}.*length mismatch"):
        validate_extracted_row(row, torch.zeros(2, 2))


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


def test_gsm8k_checker_v3_handles_percent_and_unit_exponents_without_rewriting_v2():
    percent = check_gsm8k_response(
        r"Therefore, \boxed{60%}.",
        "#### 60",
        checker_version="clir_gsm8k_numeric_v3",
    )
    escaped_percent = check_gsm8k_response(
        r"Therefore, \boxed{60\%}.",
        "#### 60",
        checker_version="clir_gsm8k_numeric_v3",
    )
    squared_unit = check_gsm8k_response(
        r"Therefore, \boxed{36\text{ cm}^2}.",
        "#### 36",
        checker_version="clir_gsm8k_numeric_v3",
    )

    assert percent["correctness"] == 1
    assert escaped_percent["correctness"] == 1
    assert squared_unit["correctness"] == 1
    assert squared_unit["normalized_candidate_answer"] == "36"
    assert percent["checker_version"] == "clir_gsm8k_numeric_v3"

    historical = check_gsm8k_response(
        r"Therefore, \boxed{60%}.",
        "#### 60",
        checker_version="clir_gsm8k_numeric_v2",
    )
    assert historical["correctness"] == 0
    assert historical["checker_version"] == "clir_gsm8k_numeric_v2"


def test_gsm8k_checker_v3_handles_decimal_percent_and_textual_boxed_answer():
    decimal_percent = check_gsm8k_response(
        "The probability is 0.24, or 24%. Final answer: \\boxed{0.24}.",
        "#### 24",
        checker_version="clir_gsm8k_numeric_v3",
    )
    textual_box = check_gsm8k_response(
        "\\boxed{James will be 44 years old after 5 years.}",
        "#### 44",
        checker_version="clir_gsm8k_numeric_v3",
    )

    assert decimal_percent["correctness"] == 1
    assert decimal_percent["checker_normalization"] == "percent_decimal_equivalence"
    assert textual_box["correctness"] == 1
    assert textual_box["normalized_candidate_answer"] == "44"


def test_gsm8k_checker_v4_ignores_only_literal_boxed_placeholders():
    after_placeholder = check_gsm8k_response(
        r"The final answer, presented as \boxed{Your Answer}, is: $2000 in 5 days.",
        "#### 2000",
    )
    earlier_numeric_box = check_gsm8k_response(
        r"Final answer: \boxed{45}. Formatting may also be written as \boxed{}.",
        "#### 45",
    )
    substantive_text_answer = check_gsm8k_response(
        r"The calculation gives 13, but the final answer is \boxed{x}.",
        "#### 13",
    )
    numeric_box_remains_authoritative = check_gsm8k_response(
        r"Final answer: \boxed{26}. This was computed in 14 steps.",
        "#### 26",
    )

    assert after_placeholder["correctness"] == 1
    assert after_placeholder["parsed_answer"] == "$2000"
    assert earlier_numeric_box["correctness"] == 1
    assert earlier_numeric_box["parsed_answer"] == "45"
    assert substantive_text_answer["correctness"] == 0
    assert substantive_text_answer["parsed_answer"] == "x"
    assert numeric_box_remains_authoritative["correctness"] == 1
    assert numeric_box_remains_authoritative["parsed_answer"] == "26"
    assert after_placeholder["checker_version"] == "clir_gsm8k_numeric_v4"


def test_gsm8k_checker_v4_requires_numeric_evidence_for_decimal_percent_equivalence():
    explicit_equivalence = check_gsm8k_response(
        r"The probability is 0.24, or 24%. Final answer: \boxed{0.24}.",
        "#### 24",
    )
    unrelated_percent_word = check_gsm8k_response(
        r"The percentage calculation is discussed above. Final answer: \boxed{0.12 years}.",
        "#### 12",
    )
    historical_v3 = check_gsm8k_response(
        r"The percentage calculation is discussed above. Final answer: \boxed{0.12 years}.",
        "#### 12",
        checker_version="clir_gsm8k_numeric_v3",
    )
    probability_fraction = check_gsm8k_response(
        r"The probability is 80% * 40% * 75% = 24/100, so the odds are \boxed{\frac{6}{25}}.",
        "#### 24",
    )

    assert explicit_equivalence["correctness"] == 1
    assert explicit_equivalence["checker_normalization"] == "percent_decimal_equivalence"
    assert unrelated_percent_word["correctness"] == 0
    assert historical_v3["correctness"] == 1
    assert probability_fraction["correctness"] == 1


def test_component_protocol_hashes_isolate_evaluation_edits():
    protocol = {
        "model": {"repo_id": "model", "revision": "r"},
        "dataset": {"repo_id": "dataset", "subset": "main", "revision": "r"},
        "prompt": {"template": "{question}"},
        "generation": {"temperature": 1.0},
        "hidden_states": {"layer_policy": "all"},
        "correctness": {"checker": "v3"},
        "evaluation": {"k": [1, 2]},
    }
    changed = json.loads(json.dumps(protocol))
    changed["evaluation"]["k"] = [1, 2, 4]

    first = protocol_hashes(protocol)
    second = protocol_hashes(changed)
    assert first["protocol_sha256"] != second["protocol_sha256"]
    assert first["evaluation_protocol_sha256"] != second["evaluation_protocol_sha256"]
    assert first["acquisition_protocol_sha256"] == second["acquisition_protocol_sha256"]
    assert first["label_protocol_sha256"] == second["label_protocol_sha256"]


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
