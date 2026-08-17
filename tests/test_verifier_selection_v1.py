import hashlib
from pathlib import Path

from scripts.audit_verifier_selection_v1 import load_jsonl, validate_labels
from scripts.run_verifier_selection_v1 import parse_candidate_annotation, prompt_for


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs" / "verifier_selection_v1"


def test_selection_v1_is_anonymous_balanced_and_schema_valid():
    items = load_jsonl(DATA / "verifier_selection_items_v1.jsonl")
    labels = load_jsonl(DATA / "verifier_selection_labels_primary_v1.jsonl")
    assert len(items) == len(labels) == 64
    assert len({row["base_source_id"] for row in items}) == 16
    assert {row["domain"] for row in items} == {
        "quantitative",
        "code_rules",
        "evidence",
        "logic_procedure",
    }
    assert all("source_correctness" not in row for row in items)
    assert all("decision" not in row and "reason" not in row for row in items)
    validate_labels(items, labels, primary=True)


def test_selection_v1_persistent_hashes_match():
    recorded = {}
    for line in (DATA / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    assert recorded
    for name, expected in recorded.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == expected


def test_secondary_prompt_enforces_annotation_isolation():
    prompt = (DATA / "verifier_selection_secondary_prompt_v1.md").read_text(
        encoding="utf-8"
    )
    assert "verifier_selection_labels_primary_v1.jsonl" in prompt
    assert "Do not read or search" in prompt
    assert "Do not report label counts" in prompt


def test_secondary_and_adjudicated_gold_are_complete():
    items = load_jsonl(DATA / "verifier_selection_items_v1.jsonl")
    secondary = load_jsonl(DATA / "verifier_selection_labels_secondary_v1.jsonl")
    gold = load_jsonl(DATA / "verifier_selection_gold_v1.jsonl")
    adjudications = load_jsonl(DATA / "verifier_selection_adjudications_v1.jsonl")
    validate_labels(items, secondary, primary=False)
    validate_labels(items, gold, primary=False)
    assert len(adjudications) == 12
    assert {row["decision"] for row in gold} == {"accept", "reject"}
    assert sum(row["decision"] == "accept" for row in gold) == 32
    assert sum(row["decision"] == "reject" for row in gold) == 32
    assert all(row["gold_decision"] == row["secondary_decision"] for row in adjudications)


def test_candidate_prompt_is_model_blind_and_item_bound():
    item = load_jsonl(DATA / "verifier_selection_items_v1.jsonl")[0]
    guide = (DATA / "verifier_selection_annotation_guide_v1.md").read_text(
        encoding="utf-8"
    )
    messages = prompt_for(item, guide)
    rendered = "\n".join(message["content"] for message in messages)
    assert item["item_id"] in rendered
    assert item["source_trajectory"] in rendered
    assert "verifier_selection_gold" not in rendered
    assert "labels_primary" not in rendered


def test_candidate_parser_allows_only_one_exact_json_fence():
    assert parse_candidate_annotation('{"a":1}') == ({"a": 1}, "strict_json")
    assert parse_candidate_annotation('```json\n{"a":1}\n```') == (
        {"a": 1},
        "single_json_fence",
    )
    with __import__("pytest").raises(ValueError):
        parse_candidate_annotation('Here:\n```json\n{"a":1}\n```')
