#!/usr/bin/env python
"""Prepare, extract, and audit the deterministic CLIR semantic-view pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl
from src.clir_real_data import (
    artifact_stem,
    extract_aligned_hidden_states,
    file_sha256,
    validate_extracted_row,
    validate_uniform_extraction,
)
from src.clir_rewrite import (
    build_rewrite_plan,
    materialize_rewrite_views,
    select_pilot_sources,
    validate_rewrite_protocol,
    validate_rewrite_views,
)
from src.clir_stage_a import atomic_write_json, atomic_write_jsonl, git_state


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "semantic_rewrite_pilot_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--stage", choices=("prepare", "extract", "audit"), required=True)
    parser.add_argument("--device-map", default="auto", help="Transformers device_map for extract")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _load_protocol(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Rewrite protocol must be a JSON object")
    return validate_rewrite_protocol(value)


def _paths(protocol: Mapping[str, Any]) -> Dict[str, Path]:
    outputs = protocol.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Rewrite protocol requires outputs")
    root = _resolve(str(outputs["root"]))
    return {
        "root": root,
        "plan": root / str(outputs["plan"]),
        "views": root / str(outputs["views"]),
        "prepare_report": root / str(outputs["prepare_report"]),
        "extracted": root / str(outputs["extracted"]),
        "extraction_report": root / str(outputs["extraction_report"]),
        "audit_report": root / str(outputs["audit_report"]),
        "feature_dir": root / str(outputs["feature_dir"]),
    }


def _assert_targets(paths: Sequence[Path], overwrite: bool) -> None:
    collisions = [str(path) for path in paths if path.exists()]
    if collisions and not overwrite:
        raise FileExistsError(f"Refusing to overwrite rewrite pilot outputs: {collisions}")


def _jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)


def _text_sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_feature(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "hidden_states" in payload:
        payload = payload["hidden_states"]
    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"Feature payload is not a tensor: {path}")
    return payload


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _source_context(
    protocol_path: Path, protocol: Mapping[str, Any]
) -> tuple[Path, str, str]:
    source_path = _resolve(str(protocol["source"]["manifest"]))
    if "pilot_test" in source_path.parts:
        raise ValueError("Rewrite pilot must not access pilot_test")
    source_sha256 = file_sha256(source_path)
    expected = str(protocol["source"]["sha256"])
    if source_sha256 != expected:
        raise ValueError(
            f"Rewrite source SHA256 mismatch: expected {expected}, got {source_sha256}"
        )
    return source_path, source_sha256, file_sha256(protocol_path)


def _load_tokenizer(protocol: Mapping[str, Any], cache_dir: str | None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Activate the SWIFT environment before running the rewrite pilot") from exc
    model = protocol["model"]
    return AutoTokenizer.from_pretrained(
        model["repo_id"],
        revision=model["tokenizer_revision"],
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        cache_dir=cache_dir,
    )


def prepare(
    args: argparse.Namespace,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    source_path, source_sha256, protocol_sha256 = _source_context(protocol_path, protocol)
    selected = select_pilot_sources(read_jsonl(source_path), protocol)
    preview = {
        "stage": "prepare",
        "execute": bool(args.execute),
        "source_manifest": str(source_path),
        "source_sha256": source_sha256,
        "rewrite_protocol": str(protocol_path),
        "rewrite_protocol_sha256": protocol_sha256,
        "selected_sources": [
            {
                "id": row["id"],
                "query_id": row["query_id"],
                "correctness": int(row["correctness"]),
            }
            for row in selected
        ],
        "view_count": len(protocol["views"]),
        "planned_rows": len(selected) * len(protocol["views"]),
        "outputs": {key: str(value) for key, value in paths.items() if key != "root"},
    }
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    _assert_targets(
        (paths["plan"], paths["views"], paths["prepare_report"]), args.overwrite
    )
    tokenizer = _load_tokenizer(protocol, args.cache_dir)
    plan = build_rewrite_plan(
        selected,
        tokenizer,
        protocol,
        source_manifest_sha256=source_sha256,
        rewrite_protocol_sha256=protocol_sha256,
    )
    plan_payload = _jsonl_payload(plan)
    plan_sha256 = _text_sha256(plan_payload)
    code = git_state(PROJECT_ROOT)
    views = materialize_rewrite_views(
        selected,
        plan,
        plan_sha256=plan_sha256,
        source_manifest_path=str(source_path),
        source_manifest_sha256=source_sha256,
        rewrite_protocol_path=str(protocol_path),
        rewrite_protocol_sha256=protocol_sha256,
        code=code,
    )
    view_summary = validate_rewrite_views(views, expected_plan_sha256=plan_sha256)
    atomic_write_jsonl(paths["plan"], plan)
    if file_sha256(paths["plan"]) != plan_sha256:
        raise AssertionError("Published rewrite plan hash differs from the in-memory plan")
    atomic_write_jsonl(paths["views"], views)
    report = {
        "schema_version": "clir-semantic-rewrite-prepare-report-v1",
        **preview,
        "execute": True,
        "rewrite_plan_sha256": plan_sha256,
        "views_sha256": file_sha256(paths["views"]),
        "view_summary": view_summary,
        "code": code,
    }
    atomic_write_json(paths["prepare_report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_prepared(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    source_path, source_sha256, protocol_sha256 = _source_context(protocol_path, protocol)
    del source_path
    report = json.loads(paths["prepare_report"].read_text(encoding="utf-8"))
    if report.get("schema_version") != "clir-semantic-rewrite-prepare-report-v1":
        raise ValueError("Prepared rewrite report has the wrong schema")
    expected = {
        "source_sha256": source_sha256,
        "rewrite_protocol_sha256": protocol_sha256,
        "rewrite_plan_sha256": file_sha256(paths["plan"]),
        "views_sha256": file_sha256(paths["views"]),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Prepared rewrite {key} drifted")
    views = read_jsonl(paths["views"])
    validate_rewrite_views(views, expected_plan_sha256=report["rewrite_plan_sha256"])
    return views, report


def extract(
    args: argparse.Namespace,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    views, prepare_report = _validate_prepared(protocol_path, protocol, paths)
    rewritten = [row for row in views if row["style_id"] != "original"]
    preview = {
        "stage": "extract",
        "execute": bool(args.execute),
        "views": len(views),
        "reused_original_features": len(views) - len(rewritten),
        "teacher_forced_rewrite_features": len(rewritten),
        "device_map": args.device_map,
        "output": str(paths["extracted"]),
    }
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return
    _assert_targets((paths["extracted"], paths["extraction_report"]), args.overwrite)

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("Activate the SWIFT environment before extracting rewrite features") from exc
    tokenizer = _load_tokenizer(protocol, args.cache_dir)
    model_cfg = protocol["model"]
    torch_dtype = getattr(torch, str(model_cfg["torch_dtype"]))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["repo_id"],
        revision=model_cfg["revision"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )
    model.eval()

    expected_hidden = protocol["hidden_states"]
    condition_cache: Dict[str, torch.Tensor] = {}
    extracted: list[Dict[str, Any]] = []
    new_feature_bytes = 0
    for row_value in views:
        row = dict(row_value)
        decoded = tokenizer.decode(
            row["output_token_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded != row["response"]:
            raise ValueError(f"Rewrite token alignment drifted for {row['id']!r}")
        condition_path = Path(str(row["condition_states_path"]))
        if file_sha256(condition_path) != row["condition_sha256"]:
            raise ValueError(f"Condition checksum mismatch for {row['id']!r}")
        query_id = str(row["query_id"])
        condition = condition_cache.get(query_id)
        if condition is None:
            condition = _load_feature(condition_path)
            condition_cache[query_id] = condition

        if row["style_id"] == "original":
            trajectory_path = Path(str(row["hidden_states_path"]))
            if file_sha256(trajectory_path) != row["feature_sha256"]:
                raise ValueError(f"Original trajectory checksum mismatch for {row['id']!r}")
            trajectory = _load_feature(trajectory_path)
            validate_extracted_row(row, trajectory, condition)
            extracted.append(row)
            continue

        aligned = extract_aligned_hidden_states(
            model,
            row["prompt_token_ids"],
            row["output_token_ids"],
            storage_dtype=expected_hidden["storage_dtype"],
            canonical_condition=condition,
        )
        expected = (
            int(expected_hidden["layer_count"]),
            int(expected_hidden["per_layer_hidden_size"]),
            int(expected_hidden["feature_dim"]),
        )
        observed = (aligned.layer_count, aligned.per_layer_hidden_size, aligned.feature_dim)
        if observed != expected:
            raise ValueError(f"Rewrite feature layout mismatch: expected {expected}, got {observed}")
        validate_extracted_row(row, aligned.trajectory, condition)
        trajectory_path = paths["feature_dir"] / f"{artifact_stem(row['id'])}.pt"
        if trajectory_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite rewrite feature: {trajectory_path}")
        metadata = {
            "layer_policy": expected_hidden["layer_policy"],
            "layer_count": aligned.layer_count,
            "per_layer_hidden_size": aligned.per_layer_hidden_size,
            "feature_dim": aligned.feature_dim,
            "hidden_states_source": "model_outputs.hidden_states_as_returned",
            "extractor_applied_additional_final_norm": False,
            "storage_dtype": expected_hidden["storage_dtype"],
            "feature_role": "trajectory",
            "trajectory_tokens": len(row["output_token_ids"]),
            "condition_tokens": len(row["prompt_token_ids"]),
            "trajectory_acquisition": "teacher_forced_exact_rewrite_token_ids",
        }
        _atomic_torch_save(
            {"hidden_states": aligned.trajectory, "metadata": metadata}, trajectory_path
        )
        row["hidden_states_path"] = str(trajectory_path)
        row["feature_sha256"] = file_sha256(trajectory_path)
        row["feature_metadata"] = metadata
        extracted.append(row)
        new_feature_bytes += trajectory_path.stat().st_size

    view_summary = validate_rewrite_views(
        extracted, expected_plan_sha256=prepare_report["rewrite_plan_sha256"]
    )
    uniform = validate_uniform_extraction(extracted)
    atomic_write_jsonl(paths["extracted"], extracted)
    report = {
        "schema_version": "clir-semantic-rewrite-extraction-report-v1",
        **preview,
        "execute": True,
        "input_views_sha256": prepare_report["views_sha256"],
        "rewrite_plan_sha256": prepare_report["rewrite_plan_sha256"],
        "rewrite_protocol_sha256": prepare_report["rewrite_protocol_sha256"],
        "output_sha256": file_sha256(paths["extracted"]),
        "new_feature_bytes": new_feature_bytes,
        "uniform_extraction": uniform,
        "view_summary": view_summary,
        "code": git_state(PROJECT_ROOT),
    }
    atomic_write_json(paths["extraction_report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_extracted_artifact(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[list[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    views, prepare_report = _validate_prepared(protocol_path, protocol, paths)
    del views
    if not paths["extracted"].is_file() or not paths["extraction_report"].is_file():
        raise FileNotFoundError(
            "Rewrite extraction is incomplete; run --stage extract --execute before audit"
        )
    report = json.loads(paths["extraction_report"].read_text(encoding="utf-8"))
    if report.get("schema_version") != "clir-semantic-rewrite-extraction-report-v1":
        raise ValueError("Rewrite extraction report has the wrong schema")
    expected = {
        "input_views_sha256": prepare_report["views_sha256"],
        "rewrite_plan_sha256": prepare_report["rewrite_plan_sha256"],
        "rewrite_protocol_sha256": prepare_report["rewrite_protocol_sha256"],
        "output_sha256": file_sha256(paths["extracted"]),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Rewrite extraction {key} drifted")
    rows = read_jsonl(paths["extracted"])
    validate_rewrite_views(rows, expected_plan_sha256=prepare_report["rewrite_plan_sha256"])
    return rows, prepare_report, report


def audit(
    args: argparse.Namespace,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    rows, prepare_report, extraction_report = _validate_extracted_artifact(
        protocol_path, protocol, paths
    )
    preview = {
        "stage": "audit",
        "execute": bool(args.execute),
        "rows": len(rows),
        "output": str(paths["audit_report"]),
    }
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return
    _assert_targets((paths["audit_report"],), args.overwrite)

    checked_trajectory_paths: set[Path] = set()
    checked_condition_paths: set[Path] = set()
    total_feature_bytes = 0
    for row in rows:
        trajectory_path = Path(str(row["hidden_states_path"]))
        condition_path = Path(str(row["condition_states_path"]))
        if file_sha256(trajectory_path) != row["feature_sha256"]:
            raise ValueError(f"Trajectory checksum mismatch for {row['id']!r}")
        if file_sha256(condition_path) != row["condition_sha256"]:
            raise ValueError(f"Condition checksum mismatch for {row['id']!r}")
        trajectory = _load_feature(trajectory_path)
        condition = _load_feature(condition_path)
        validate_extracted_row(row, trajectory, condition, check_finite=True)
        if trajectory_path not in checked_trajectory_paths:
            total_feature_bytes += trajectory_path.stat().st_size
            checked_trajectory_paths.add(trajectory_path)
        if condition_path not in checked_condition_paths:
            total_feature_bytes += condition_path.stat().st_size
            checked_condition_paths.add(condition_path)

    summary = validate_rewrite_views(
        rows, expected_plan_sha256=prepare_report["rewrite_plan_sha256"]
    )
    coverage = summary["coverage"]
    if coverage["eligible_component_count"] != 1:
        raise ValueError("The deterministic rewrite pilot must activate consistency only")
    report = {
        "schema_version": "clir-semantic-rewrite-audit-report-v1",
        **preview,
        "execute": True,
        "status": "passed_pipeline_only_no_mechanism_claim",
        "formal_mechanism_claim_allowed": False,
        "rewrite_protocol": str(protocol_path),
        "rewrite_protocol_sha256": prepare_report["rewrite_protocol_sha256"],
        "rewrite_plan_sha256": prepare_report["rewrite_plan_sha256"],
        "extracted_sha256": extraction_report["output_sha256"],
        "trajectory_payloads": len(checked_trajectory_paths),
        "condition_payloads": len(checked_condition_paths),
        "total_feature_bytes": total_feature_bytes,
        "summary": summary,
        "code": git_state(PROJECT_ROOT),
    }
    atomic_write_json(paths["audit_report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol_config).resolve()
    protocol = _load_protocol(protocol_path)
    paths = _paths(protocol)
    if args.stage == "prepare":
        prepare(args, protocol_path, protocol, paths)
    elif args.stage == "extract":
        extract(args, protocol_path, protocol, paths)
    else:
        audit(args, protocol_path, protocol, paths)


if __name__ == "__main__":
    main()
