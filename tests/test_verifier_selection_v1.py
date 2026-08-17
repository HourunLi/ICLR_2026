import hashlib
from pathlib import Path

from scripts.audit_verifier_selection_v1 import load_jsonl, validate_labels


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
