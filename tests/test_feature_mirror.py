import hashlib
from pathlib import Path

from src.clir_data import write_jsonl
from scripts.verify_feature_mirror import collect_payloads, verify_payloads


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_feature_mirror_verifies_unique_shared_payloads(tmp_path: Path):
    trajectory = tmp_path / "trajectory.pt"
    condition = tmp_path / "condition.pt"
    trajectory.write_bytes(b"trajectory")
    condition.write_bytes(b"condition")
    manifest = tmp_path / "rows.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": f"row-{index}",
                "hidden_states_path": trajectory.name,
                "feature_sha256": _sha256(trajectory),
                "condition_states_path": condition.name,
                "condition_sha256": _sha256(condition),
            }
            for index in range(2)
        ],
    )

    payloads, rows = collect_payloads([manifest])
    report = verify_payloads(payloads, workers=2)

    assert rows == 2
    assert report["unique_payloads"] == 2
    assert report["verified_payloads"] == 2
    assert report["failure_count"] == 0


def test_feature_mirror_reports_hash_mismatch(tmp_path: Path):
    payload = tmp_path / "trajectory.pt"
    payload.write_bytes(b"corrupted")
    report = verify_payloads({payload: "0" * 64}, workers=1)

    assert report["failure_count"] == 1
    assert report["failures"][0]["status"] == "hash_mismatch"
