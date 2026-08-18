from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.clir_llm_rewrite import (
    assemble_indexed_line_rewrite,
    build_indexed_line_plan,
    deterministic_rewrite_gates,
    numeric_literal_sequence,
    numeric_literal_set,
    parse_json_object,
    parse_indexed_line_rewrites,
    parse_tagged_rewrite,
    rewrite_source_view,
    select_configured_sources,
    validate_accepted_views,
    validate_llm_rewrite_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(chr(token_id) for token_id in token_ids)


def _protocol():
    return json.loads(
        (PROJECT_ROOT / "configs/semantic_rewrite_llm_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _protocol_v2():
    return json.loads(
        (PROJECT_ROOT / "configs/semantic_rewrite_llm_v2.json").read_text(
            encoding="utf-8"
        )
    )


def _protocol_v3():
    return json.loads(
        (PROJECT_ROOT / "configs/semantic_rewrite_llm_v3.json").read_text(
            encoding="utf-8"
        )
    )


def _protocol_v4():
    return json.loads(
        (PROJECT_ROOT / "configs/semantic_rewrite_llm_v4.json").read_text(
            encoding="utf-8"
        )
    )


def _protocol_v5():
    return json.loads(
        (PROJECT_ROOT / "configs/semantic_rewrite_llm_v5.json").read_text(
            encoding="utf-8"
        )
    )


def _protocol_v6():
    return json.loads(
        (PROJECT_ROOT / "configs/semantic_rewrite_llm_v6.json").read_text(
            encoding="utf-8"
        )
    )


def _protocol_v7():
    return json.loads(
        (PROJECT_ROOT / "configs/semantic_rewrite_llm_v7.json").read_text(
            encoding="utf-8"
        )
    )


def _source(source_id: str, index: int, *, correctness: int):
    answer = "3" if correctness else "4"
    response = f"Compute 1 + 2 = {answer}.\nFinal answer: \\boxed{{{answer}}}."
    return {
        "id": source_id,
        "query_id": f"query-{index}",
        "candidate_index": index,
        "split": "train",
        "question": "What is one plus two?",
        "prompt": "Solve: one plus two.",
        "response": response,
        "trajectory": response,
        "raw_reference": "Reference.\n#### 3",
        "prompt_token_ids": [10, 11],
        "output_token_ids": CharacterTokenizer().encode(response),
        "correctness": correctness,
        "normalized_candidate_answer": answer,
        "hidden_states_path": f"trajectory-{index}.pt",
        "condition_states_path": f"condition-{index}.pt",
        "feature_sha256": "a" * 64,
        "condition_sha256": "b" * 64,
        "provenance": {
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "model_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
            "tokenizer_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
            "protocol_version": "source-v1",
        },
    }


def _sources():
    ids = _protocol()["source"]["selected_source_ids"]
    return [
        _source(ids[0], 0, correctness=1),
        _source(ids[1], 1, correctness=1),
        _source(ids[2], 2, correctness=0),
        _source(ids[3], 3, correctness=0),
    ]


def test_llm_protocol_and_pinned_source_selection_are_strict():
    protocol = _protocol()
    assert validate_llm_rewrite_protocol(protocol)["quality_gate"][
        "required_accepted_views"
    ] == 8
    assert [row["id"] for row in select_configured_sources(_sources(), protocol)] == protocol[
        "source"
    ]["selected_source_ids"]

    formal = deepcopy(protocol)
    formal["formal_mechanism_claim_allowed"] = True
    with pytest.raises(ValueError, match="forbid formal"):
        validate_llm_rewrite_protocol(formal)

    test_data = deepcopy(protocol)
    test_data["source"]["manifest"] = "pilot_test/secret.jsonl"
    with pytest.raises(ValueError, match="pilot_test"):
        validate_llm_rewrite_protocol(test_data)

    independent = deepcopy(protocol)
    independent["verifier"]["independent_from_generator"] = True
    with pytest.raises(ValueError, match="not independent"):
        validate_llm_rewrite_protocol(independent)


def test_json_parser_rejects_chatter_extra_and_duplicate_keys():
    assert parse_json_object('{"value": 1}', ["value"]) == {"value": 1}
    assert parse_json_object('```json\n{"value": 1}\n```', ["value"]) == {
        "value": 1
    }
    with pytest.raises(ValueError, match="not one valid"):
        parse_json_object('comment {"value": 1}', ["value"])
    with pytest.raises(ValueError, match="keys differ"):
        parse_json_object('{"value": 1, "extra": 2}', ["value"])
    with pytest.raises(ValueError, match="repeats key"):
        parse_json_object('{"value": 1, "value": 2}', ["value"])


def test_v2_tag_parser_and_numeric_parser_remove_only_layout_ordinals():
    assert validate_llm_rewrite_protocol(_protocol_v2())["schema_version"].endswith("v2")
    assert parse_tagged_rewrite(
        "<rewritten_response>\nLine with \\boxed{3}.\n</rewritten_response>"
    ) == "Line with \\boxed{3}."
    with pytest.raises(ValueError, match="outside"):
        parse_tagged_rewrite(
            "comment <rewritten_response>value</rewritten_response>"
        )

    source = "Step 1: begin\n2. continue\nKeep 2/3 and 40 % 60 and 20%."
    rewritten = "Begin\nContinue\nKeep 2/3 and 40 % 60 and 20%."
    expected = {"2/3", "20%", "40", "60"}
    assert numeric_literal_set(source, "clir_numeric_literal_set_v2") == expected
    assert numeric_literal_set(rewritten, "clir_numeric_literal_set_v2") == expected
    assert numeric_literal_sequence(
        "Keep 2, then 2, then 3.", "clir_numeric_literal_set_v2"
    ) == ["2", "2", "3"]


def test_v3_linewise_gates_require_real_edits_and_exact_math_lines():
    protocol = _protocol_v3()
    assert validate_llm_rewrite_protocol(protocol)["schema_version"].endswith("v3")
    source = _source("linewise", 0, correctness=1)
    source["response"] = (
        "Step 1: Start with the values.\n"
        "Compute 1 + 2 = 3.\n"
        "This gives the result.\n"
        "Final answer: \\boxed{3}."
    )
    style = protocol["styles"][0]["instruction"]
    rewritten = (
        "First, use the given values.\n"
        "Compute 1 + 2 = 3.\n"
        "That yields the result.\n"
        "Final answer: \\boxed{3}."
    )
    gates = deterministic_rewrite_gates(
        source,
        rewritten,
        protocol,
        style_instruction=style,
    )
    assert gates["passed"] is True
    assert gates["linewise_metrics"]["editable_line_change_ratio"] == 1.0

    altered_equation = rewritten.replace("Compute 1 + 2 = 3.", "Compute 2 + 1 = 3.")
    altered = deterministic_rewrite_gates(
        source,
        altered_equation,
        protocol,
        style_instruction=style,
    )
    assert altered["passed"] is False
    assert altered["checks"]["protected_math_lines_exact"] is False
    assert altered["checks"]["numeric_literal_sequence_equal"] is False


def test_v4_only_pins_runtime_and_keeps_v3_rewrite_contract():
    v3 = _protocol_v3()
    v4 = _protocol_v4()
    assert validate_llm_rewrite_protocol(v4)["schema_version"].endswith("v4")
    for key in (
        "source",
        "identity",
        "model_selection",
        "generator",
        "verifier",
        "styles",
        "decoding",
        "deterministic_gates",
        "tokenization",
        "quality_gate",
    ):
        assert v4[key] == v3[key]
    assert v4["execution"]["required_runtime"]["transformers_version"] == "4.51.2"


def test_v5_indexed_parser_and_programmatic_assembly_are_strict():
    source = (
        "Start with 1 item.\n"
        "Total = 1 + 2\n"
        "\n"
        "Explain the result plainly.\n"
        "Final answer: \\boxed{3}."
    )
    plan = build_indexed_line_plan(source)
    assert plan["editable_ids"] == ["E0000", "E0003"]
    assert plan["protected_ids"] == ["P0001", "P0004"]
    generated = (
        "<E0000>Begin with 1 item.</E0000>\n"
        "<E0003>State the result simply.</E0003>"
    )
    replacements = parse_indexed_line_rewrites(generated, plan["editable_ids"])
    assembled = assemble_indexed_line_rewrite(plan, replacements)
    assert assembled == (
        "Begin with 1 item.\n"
        "Total = 1 + 2\n"
        "\n"
        "State the result simply.\n"
        "Final answer: \\boxed{3}."
    )

    with pytest.raises(ValueError, match="exactly one physical line"):
        parse_indexed_line_rewrites("comment\n" + generated, plan["editable_ids"])
    with pytest.raises(ValueError, match="ids drifted"):
        parse_indexed_line_rewrites(
            "<E0003>State it.</E0003>\n<E0000>Begin with 1 item.</E0000>",
            plan["editable_ids"],
        )


def test_v5_constrained_view_reassembles_before_all_existing_gates():
    protocol = _protocol_v5()
    assert validate_llm_rewrite_protocol(protocol)["schema_version"].endswith("v5")
    source = _source("constrained", 0, correctness=1)
    source["response"] = (
        "Step 1: Start with 1 and 2.\n"
        "Compute 1 + 2 = 3.\n"
        "This gives 3.\n"
        "Final answer: \\boxed{3}."
    )
    source["trajectory"] = source["response"]
    source["output_token_ids"] = CharacterTokenizer().encode(source["response"])

    def complete(messages, max_new_tokens):
        del max_new_tokens
        if "fidelity auditor" in messages[0]["content"]:
            return {
                "text": json.dumps(
                    {
                        "reasoning_meaning_preserved": True,
                        "final_answer_preserved": True,
                        "numbers_operations_preserved": True,
                        "style_changed": True,
                        "introduced_correction": False,
                        "added_or_removed_claims": False,
                        "decision": "accept",
                        "issues": [],
                    }
                )
            }
        return {
            "text": (
                "<E0000>First, use 1 and 2.</E0000>\n"
                "<E0002>The result is 3.</E0002>"
            )
        }

    result = rewrite_source_view(
        source,
        protocol["styles"][0],
        protocol,
        feature_tokenizer=CharacterTokenizer(),
        complete=complete,
        protocol_sha256="e" * 64,
        code={"commit": "f" * 40, "dirty": False},
    )
    assert result["accepted"] is True
    assert result["rewritten_response"].splitlines()[1] == "Compute 1 + 2 = 3."
    assert result["rewritten_response"].splitlines()[3] == (
        "Final answer: \\boxed{3}."
    )
    assert result["deterministic_gates"]["linewise_metrics"][
        "editable_line_change_ratio"
    ] == 1.0
    lineage_constraint = result["accepted_view"]["rewrite_lineage"][
        "generator_constraint"
    ]
    assert lineage_constraint["editable_ids"] == ["E0000", "E0002"]
    assert len(lineage_constraint["replacement_map_sha256"]) == 64


def test_v6_switches_only_the_frozen_model_and_preserves_the_v5_gate():
    v5 = _protocol_v5()
    v6 = _protocol_v6()
    assert validate_llm_rewrite_protocol(v6)["schema_version"].endswith("v6")
    assert v6["generator"]["repo_id"] == "tiiuae/Falcon3-3B-Instruct"
    assert v6["generator"]["revision"] == v6["verifier"]["revision"]
    assert v6["model_selection"]["config"].endswith(
        "semantic_rewrite_models_v2.json"
    )
    for key in (
        "source",
        "identity",
        "styles",
        "decoding",
        "deterministic_gates",
        "tokenization",
        "quality_gate",
    ):
        assert v6[key] == v5[key]
    for key in (
        "output_format",
        "system_prompt",
        "user_prompt_template",
        "protected_content_policy",
    ):
        assert v6["generator"][key] == v5["generator"][key]
    for key in (
        "output_format",
        "independent_from_generator",
        "system_prompt",
        "user_prompt_template",
        "output_keys",
        "reference_answer_visible",
        "source_correctness_label_visible",
    ):
        assert v6["verifier"][key] == v5["verifier"][key]


def test_v7_changes_only_qwen_scale_and_preserves_the_v5_diagnostic_contract():
    v5 = _protocol_v5()
    v7 = _protocol_v7()
    assert validate_llm_rewrite_protocol(v7)["schema_version"].endswith("v7")
    assert v7["generator"]["repo_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert v7["generator"]["revision"] == (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    assert v7["generator"]["revision"] == v7["verifier"]["revision"]
    assert v7["model_selection"] == {
        "config": "configs/semantic_rewrite_models_v3.json",
        "sha256": "071f93fcec13bef2118652960bab99bea8630e3078dd725fde79e246369c7a76",
    }
    assert v7["diagnostic_scope"] == {
        "comparison_baseline": "configs/semantic_rewrite_llm_v5.json",
        "only_intended_factor": "qwen2.5_checkpoint_scale_3.09B_to_7.61B",
        "source_gate_previously_exposed": True,
        "blind_gate": False,
        "may_authorize_scale": False,
    }
    for key in (
        "source",
        "identity",
        "styles",
        "decoding",
        "deterministic_gates",
        "tokenization",
        "quality_gate",
    ):
        assert v7[key] == v5[key]
    for key in (
        "output_format",
        "system_prompt",
        "user_prompt_template",
        "protected_content_policy",
    ):
        assert v7["generator"][key] == v5["generator"][key]
    for key in (
        "output_format",
        "independent_from_generator",
        "system_prompt",
        "user_prompt_template",
        "output_keys",
        "reference_answer_visible",
        "source_correctness_label_visible",
    ):
        assert v7["verifier"][key] == v5["verifier"][key]

    scale_claim = deepcopy(v7)
    scale_claim["diagnostic_scope"]["may_authorize_scale"] = True
    with pytest.raises(ValueError, match="may not authorize scaling"):
        validate_llm_rewrite_protocol(scale_claim)


def test_deterministic_gate_preserves_incorrectness_and_numeric_literals():
    source = _source("incorrect", 0, correctness=0)
    preserved = "Restated reasoning: Compute 1 + 2 = 4.\nFinal answer: \\boxed{4}."
    gates = deterministic_rewrite_gates(source, preserved, _protocol())
    assert gates["passed"] is True
    assert numeric_literal_set(source["response"]) == {"1", "2", "4"}

    repaired = "Restated reasoning: Compute 1 + 2 = 3.\nFinal answer: \\boxed{3}."
    repaired_gates = deterministic_rewrite_gates(source, repaired, _protocol())
    assert repaired_gates["passed"] is False
    assert repaired_gates["checks"]["incorrect_source_not_repaired"] is False
    assert repaired_gates["checks"]["same_checker_correctness"] is False


def test_fake_local_rewrite_gate_builds_eight_token_only_provenanced_views():
    protocol = _protocol()
    accepted = []
    tokenizer = CharacterTokenizer()

    for source in _sources():
        for style in protocol["styles"]:
            rewritten = "Restated: " + source["response"]

            def complete(messages, max_new_tokens, *, rewritten=rewritten):
                del max_new_tokens
                if "fidelity auditor" in messages[0]["content"]:
                    return {
                        "text": json.dumps(
                            {
                                "reasoning_meaning_preserved": True,
                                "final_answer_preserved": True,
                                "numbers_operations_preserved": True,
                                "style_changed": True,
                                "introduced_correction": False,
                                "added_or_removed_claims": False,
                                "decision": "accept",
                                "issues": [],
                            }
                        )
                    }
                return {"text": json.dumps({"rewritten_response": rewritten})}

            result = rewrite_source_view(
                source,
                style,
                protocol,
                feature_tokenizer=tokenizer,
                complete=complete,
                protocol_sha256="c" * 64,
                code={"commit": "d" * 40, "dirty": False},
            )
            assert result["accepted"] is True
            accepted.append(result["accepted_view"])

    summary = validate_accepted_views(
        accepted,
        protocol,
        protocol_sha256="c" * 64,
        require_complete=True,
    )
    assert summary["rows"] == 8
    assert summary["semantic_groups"] == 4
    assert summary["incorrect_source_repairs"] == 0
    assert summary["coverage"]["eligible_components"]["consistency"] is True
    assert all("hidden_states_path" not in row for row in accepted)
    assert all("condition_states_path" not in row for row in accepted)
    assert all(row["provenance"]["trajectory_acquisition"].startswith("online") for row in accepted)
