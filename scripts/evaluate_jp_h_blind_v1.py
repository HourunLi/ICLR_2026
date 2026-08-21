#!/usr/bin/env python3
"""Open frozen JP-H predictions and run the precommitted two-view adoption gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import (  # noqa: E402
    file_sha256,
    read_jsonl,
)
from src.clir_jp_h_blind_evaluation import evaluate_rows  # noqa: E402
from src.clir_jp_h_blind_scoring import validate_prediction_row  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/jp_h_blind_validation_v1/evaluation_protocol_v1.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify(path: Path, expected_sha256: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen input hash drifted for {path}: expected {expected_sha256}, got {observed}"
        )


def index_unique(
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in indexed:
            raise ValueError(f"{name} requires unique non-empty item_id values")
        indexed[item_id] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Blind evaluation requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-jp-h-blind-evaluation-protocol-v1":
        raise ValueError("Unknown JP-H blind evaluation protocol schema")
    if protocol.get("prediction_values_accessed_when_protocol_frozen") is not False:
        raise ValueError("Evaluation protocol was not frozen before score access")

    paths: dict[str, Path] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(spec["path"])
        verify(path, str(spec["sha256"]))
        paths[name] = path
    scoring_protocol = json.loads(paths["scoring_protocol"].read_text(encoding="utf-8"))
    adjudication = json.loads(paths["adjudication_report"].read_text(encoding="utf-8"))
    receipt = json.loads(
        paths["scoring_completion_receipt"].read_text(encoding="utf-8")
    )
    if receipt.get("status") != "predictions_frozen_annotations_still_sealed":
        raise ValueError("Prediction-freeze receipt status drifted")
    if adjudication.get("status") != "adjudication_complete_predictions_still_sealed":
        raise ValueError("Adjudication was not complete before score access")
    if adjudication.get("prediction_values_accessed") is not False:
        raise ValueError("Adjudication report records premature score access")

    # This read is the intentional, protocol-authorized score unseal point.
    predictions = read_jsonl(paths["frozen_predictions"])
    views = {
        "resolved_primary": read_jsonl(paths["resolved_primary_labels"]),
        "resolved_secondary": read_jsonl(paths["resolved_secondary_labels"]),
    }
    lineage = read_jsonl(paths["private_lineage_for_domain_names_only"])
    population = protocol["population"]
    expected_rows = int(population["rows"])
    if not (
        len(predictions)
        == len(lineage)
        == len(views["resolved_primary"])
        == len(views["resolved_secondary"])
        == expected_rows
    ):
        raise ValueError("Blind evaluation inputs differ from the frozen row count")

    scoring_protocol_sha256 = file_sha256(paths["scoring_protocol"])
    domains: list[str] = []
    allowed_domains = list(population["domains"])
    for row_number, (prediction, identity) in enumerate(
        zip(predictions, lineage), start=1
    ):
        if prediction["item_id"] != identity["item_id"]:
            raise ValueError(f"Prediction/lineage order drifted at row {row_number}")
        validate_prediction_row(
            prediction,
            protocol_sha256=scoring_protocol_sha256,
        )
        domain = str(identity["domain"])
        if domain not in allowed_domains:
            raise ValueError("Private lineage contains an unexpected domain")
        domains.append(domain)
    domain_counts = Counter(domains)
    if domain_counts != Counter(
        {domain: int(population["rows_per_domain"]) for domain in allowed_domains}
    ):
        raise ValueError("Frozen domain row counts drifted")

    seeds = [int(seed) for seed in population["seeds"]]
    minimum = int(protocol["adoption_gate"]["minimum_passing_seeds_per_view"])
    pooled: dict[str, Any] = {}
    descriptive_domains: dict[str, Any] = {}
    view_passes: dict[str, bool] = {}
    for name, labels in views.items():
        result = evaluate_rows(
            labels,
            predictions,
            seeds=seeds,
            domains=domains,
        )
        result["minimum_passing_seeds"] = minimum
        result["view_passed"] = int(result["passing_seeds"]) >= minimum
        pooled[name] = result
        view_passes[name] = bool(result["view_passed"])
        descriptive_domains[name] = {
            domain: {
                **evaluate_rows(
                    labels,
                    predictions,
                    seeds=seeds,
                    domains=domains,
                    include_domain=domain,
                ),
                "descriptive_only": True,
                "affects_adoption_gate": False,
            }
            for domain in allowed_domains
        }

    overall_passed = all(view_passes.values())
    result = {
        "schema_version": "clir-jp-h-blind-evaluation-result-v1",
        "status": (
            "blind_gate_passed_offline_smoother_retained"
            if overall_passed
            else "blind_gate_failed_smoother_rejected"
        ),
        "evidence_tier": "blind_validation",
        "code": code,
        "evaluation_protocol": str(protocol_path.relative_to(ROOT)),
        "evaluation_protocol_sha256": file_sha256(protocol_path),
        "scoring_protocol_sha256": scoring_protocol_sha256,
        "prediction_sha256": file_sha256(paths["frozen_predictions"]),
        "resolved_label_sha256": {
            "primary": file_sha256(paths["resolved_primary_labels"]),
            "secondary": file_sha256(paths["resolved_secondary_labels"]),
        },
        "rows": expected_rows,
        "seeds": seeds,
        "domain_counts": dict(sorted(domain_counts.items())),
        "pooled_primary_results": pooled,
        "view_passes": view_passes,
        "minimum_passing_seeds_per_view": minimum,
        "overall_adoption_gate_passed": overall_passed,
        "per_domain_descriptive_results": descriptive_domains,
        "interpretation": (
            protocol["interpretation"]["pass_authorizes"]
            if overall_passed
            else protocol["interpretation"]["failure_action"]
        ),
        "limitations": {
            "claim_level_gold": False,
            "adjudicator": adjudication["adjudicator_disclosure"],
            "bootstrap_or_significance_test_predeclared": False,
            "per_domain_results_affect_gate": False,
        },
        "prediction_values_opened_after_protocol_commit": True,
        "threshold_tuning_performed": False,
        "additional_model_or_window_selection_performed": False,
        "tail_targets_used": False,
        "automatic_score_coupling": False,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
    }
    output = resolve(protocol["outputs"]["evaluation_result"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite blind evaluation result: {output}")
    atomic_write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
