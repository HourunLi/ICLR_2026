#!/usr/bin/env python3
"""Run a frozen local-LLM four-source semantic rewrite quality gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_data import read_jsonl
from src.clir_llm_rewrite import (
    rewrite_source_view,
    select_configured_sources,
    validate_accepted_views,
    validate_llm_rewrite_protocol,
)
from src.clir_real_data import canonical_json_sha256, file_sha256
from src.clir_rewrite import text_sha256
from src.clir_stage_a import artifact_stem, atomic_write_json, atomic_write_jsonl, git_state


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/semantic_rewrite_llm_v7.json"
SHARD_SCHEMA = "clir-semantic-rewrite-llm-source-shard-v1"
MARKER_SCHEMA = "clir-semantic-rewrite-llm-source-complete-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight, run, or audit a frozen local-LLM semantic rewrite gate."
    )
    parser.add_argument(
        "--protocol-config",
        default=str(DEFAULT_PROTOCOL),
        help="Frozen LLM rewrite protocol JSON.",
    )
    parser.add_argument("--stage", required=True, choices=["preflight", "rewrite", "audit"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cache-dir",
        default="/prodcpfs/user/panzhixin/hf_cache",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require generator/verifier and Phi tokenizer to already exist locally.",
    )
    parser.add_argument(
        "--overwrite-reports",
        action="store_true",
        help="Replace only derived preflight/audit reports; immutable source shards are never overwritten.",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _load_protocol(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return validate_llm_rewrite_protocol(value)


def _runtime_provenance(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("The frozen rewrite runtime requires transformers") from exc
    observed = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "cuda_runtime_version": torch.version.cuda,
    }
    required = protocol.get("execution", {}).get("required_runtime")
    if required is not None:
        if not isinstance(required, Mapping):
            raise ValueError("Rewrite required_runtime must be an object")
        for key, expected in required.items():
            if observed.get(key) != expected:
                raise RuntimeError(
                    f"Rewrite runtime {key} drifted: expected {expected!r}, "
                    f"got {observed.get(key)!r}"
                )
    return observed


def _paths(protocol: Mapping[str, Any]) -> Dict[str, Path]:
    root = _resolve(str(protocol["execution"]["output_root"]))
    return {
        "root": root,
        "preflight": root / "preflight.json",
        "shards": root / "shards",
        "all_results": root / "all_view_results.jsonl",
        "accepted_views": root / "accepted_views.jsonl",
        "audit_report": root / "audit_report.json",
        "human_template": root / "human_blind_audit_template.jsonl",
        "human_key": root / "human_blind_audit_key.json",
    }


def _validate_model_selection(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    selection_path = _resolve(str(protocol["model_selection"]["config"]))
    observed_sha = file_sha256(selection_path)
    if observed_sha != protocol["model_selection"]["sha256"]:
        raise ValueError(
            "Rewrite model-selection config SHA256 drifted: "
            f"expected {protocol['model_selection']['sha256']}, got {observed_sha}"
        )
    with selection_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    role_map = {
        "generator": "rewrite_generator",
        "verifier": "rewrite_verifier",
    }
    for protocol_role, selection_role in role_map.items():
        for protocol_key, selection_key in (
            ("repo_id", "repo_id"),
            ("revision", "revision"),
            ("tokenizer_revision", "tokenizer_revision"),
            ("torch_dtype", "torch_dtype"),
            ("deployment", "deployment"),
        ):
            if protocol[protocol_role][protocol_key] != selection[selection_role][selection_key]:
                raise ValueError(
                    f"Rewrite {protocol_role} {protocol_key} differs from frozen model selection"
                )
    task = selection["task_and_feature_model"]
    tokenization = protocol["tokenization"]
    expected_task = {
        "feature_model_id": task["repo_id"],
        "feature_model_revision": task["revision"],
        "tokenizer_revision": task["tokenizer_revision"],
    }
    for key, expected in expected_task.items():
        if tokenization[key] != expected:
            raise ValueError(f"Rewrite tokenization {key} differs from frozen task model")
    return {
        "path": str(selection_path),
        "sha256": observed_sha,
        "schema_version": selection.get("schema_version"),
    }


def _context(protocol_path: Path, protocol: Mapping[str, Any]) -> Dict[str, Any]:
    source_path = _resolve(str(protocol["source"]["manifest"]))
    source_sha = file_sha256(source_path)
    if source_sha != protocol["source"]["sha256"]:
        raise ValueError(
            f"Rewrite source SHA256 drifted: expected {protocol['source']['sha256']}, got {source_sha}"
        )
    rows = read_jsonl(source_path)
    selected = select_configured_sources(rows, protocol)
    return {
        "protocol_path": protocol_path,
        "protocol_sha256": file_sha256(protocol_path),
        "source_path": source_path,
        "source_sha256": source_sha,
        "selected": selected,
        "model_selection": _validate_model_selection(protocol),
        "runtime": _runtime_provenance(protocol),
    }


def _preflight_payload(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    return {
        "schema_version": "clir-semantic-rewrite-llm-preflight-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_tier": protocol["evidence_tier"],
        "formal_mechanism_claim_allowed": False,
        "protocol": str(protocol_path),
        "protocol_sha256": context["protocol_sha256"],
        "source_manifest": str(context["source_path"]),
        "source_sha256": context["source_sha256"],
        "selected_sources": [
            {
                "id": row["id"],
                "query_id": row["query_id"],
                "correctness": int(row["correctness"]),
                "source_response_sha256": text_sha256(row["response"]),
            }
            for row in context["selected"]
        ],
        "styles": [dict(style) for style in protocol["styles"]],
        "planned_views": len(context["selected"]) * len(protocol["styles"]),
        "model_selection": dict(context["model_selection"]),
        "generator_prompt_sha256": canonical_json_sha256(
            {
                "system": protocol["generator"]["system_prompt"],
                "user": protocol["generator"]["user_prompt_template"],
            }
        ),
        "verifier_prompt_sha256": canonical_json_sha256(
            {
                "system": protocol["verifier"]["system_prompt"],
                "user": protocol["verifier"]["user_prompt_template"],
            }
        ),
        "decoding": dict(protocol["decoding"]),
        "runtime": dict(context["runtime"]),
        "output_root": str(paths["root"]),
        "code": git_state(PROJECT_ROOT),
    }


def preflight(
    args: argparse.Namespace,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    report = _preflight_payload(protocol_path, protocol, context, paths)
    preview = {**report, "execute": bool(args.execute)}
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return
    if paths["preflight"].exists() and not args.overwrite_reports:
        raise FileExistsError(f"Refusing to overwrite rewrite preflight: {paths['preflight']}")
    atomic_write_json(paths["preflight"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _require_preflight(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    if not paths["preflight"].is_file():
        raise FileNotFoundError("Run --stage preflight --execute before rewrite")
    with paths["preflight"].open(encoding="utf-8") as handle:
        report = json.load(handle)
    current = _preflight_payload(protocol_path, protocol, context, paths)
    for key in (
        "protocol_sha256",
        "source_sha256",
        "selected_sources",
        "styles",
        "planned_views",
        "model_selection",
        "generator_prompt_sha256",
        "verifier_prompt_sha256",
        "decoding",
        "runtime",
    ):
        if report.get(key) != current[key]:
            raise ValueError(f"Rewrite preflight {key} drifted")
    return report


class LocalChatModel:
    """One frozen local causal LM reused for generation and self-verification."""

    def __init__(
        self,
        protocol: Mapping[str, Any],
        *,
        device: str,
        cache_dir: str,
        local_files_only: bool,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Activate the SWIFT environment before running rewrite") from exc
        model_config = protocol["generator"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["repo_id"],
            revision=model_config["tokenizer_revision"],
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_fast=True,
        )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The frozen local-LLM BF16 gate currently requires a CUDA device")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["repo_id"],
            revision=model_config["revision"],
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.generation_config = deepcopy(self.model.generation_config)
        self.generation_config.do_sample = False
        self.generation_config.temperature = protocol["decoding"].get("temperature")
        self.generation_config.top_p = protocol["decoding"].get("top_p")
        self.generation_config.top_k = protocol["decoding"].get("top_k")
        self.generation_config.num_beams = int(protocol["decoding"]["num_beams"])
        self.generation_config.repetition_penalty = float(
            protocol["decoding"]["repetition_penalty"]
        )
        self.max_input_tokens = int(protocol["decoding"]["max_input_tokens"])

    def complete(
        self, messages: Sequence[Mapping[str, str]], max_new_tokens: int
    ) -> Dict[str, Any]:
        prompt = self.tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_tokens = int(inputs["input_ids"].shape[1])
        if input_tokens > self.max_input_tokens:
            raise ValueError(
                f"Rewrite prompt has {input_tokens} tokens, limit is {self.max_input_tokens}"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        completion_ids = generated[0, input_tokens:]
        text = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {
            "text": text,
            "text_sha256": text_sha256(text),
            "input_tokens": input_tokens,
            "output_tokens": int(completion_ids.numel()),
            "elapsed_seconds": elapsed,
            "device": str(self.device),
            "torch_dtype": "bfloat16",
            "decoding": "greedy",
        }


def _load_feature_tokenizer(
    protocol: Mapping[str, Any], *, cache_dir: str, local_files_only: bool
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for rewrite tokenization") from exc
    tokenization = protocol["tokenization"]
    return AutoTokenizer.from_pretrained(
        tokenization["feature_model_id"],
        revision=tokenization["tokenizer_revision"],
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_fast=True,
    )


def _shard_paths(paths: Mapping[str, Path], source_id: str) -> tuple[Path, Path]:
    stem = artifact_stem(source_id)
    return paths["shards"] / f"{stem}.json", paths["shards"] / f"{stem}.complete.json"


def _load_valid_shard(
    paths: Mapping[str, Path],
    source_id: str,
    protocol_sha256: str,
    runtime: Mapping[str, Any],
) -> Dict[str, Any] | None:
    payload_path, marker_path = _shard_paths(paths, source_id)
    if not payload_path.exists() and not marker_path.exists():
        return None
    if not payload_path.is_file() or not marker_path.is_file():
        raise ValueError(f"Rewrite shard is incomplete for {source_id!r}")
    with marker_path.open(encoding="utf-8") as handle:
        marker = json.load(handle)
    expected_marker = {
        "schema_version": MARKER_SCHEMA,
        "source_id": source_id,
        "protocol_sha256": protocol_sha256,
        "payload_sha256": file_sha256(payload_path),
    }
    for key, expected in expected_marker.items():
        if marker.get(key) != expected:
            raise ValueError(f"Rewrite shard marker {key} drifted for {source_id!r}")
    with payload_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SHARD_SCHEMA:
        raise ValueError(f"Rewrite shard schema drifted for {source_id!r}")
    if payload.get("source_id") != source_id or payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Rewrite shard identity drifted for {source_id!r}")
    if payload.get("runtime") != dict(runtime):
        raise ValueError(f"Rewrite shard runtime drifted for {source_id!r}")
    if len(payload.get("view_results", [])) != 2:
        raise ValueError(f"Rewrite shard view count drifted for {source_id!r}")
    return payload


def rewrite(
    args: argparse.Namespace,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    _require_preflight(protocol_path, protocol, context, paths)
    healthy: Dict[str, Dict[str, Any]] = {}
    pending: list[Mapping[str, Any]] = []
    for source in context["selected"]:
        shard = _load_valid_shard(
            paths,
            str(source["id"]),
            context["protocol_sha256"],
            context["runtime"],
        )
        if shard is None:
            pending.append(source)
        else:
            healthy[str(source["id"])] = shard
    preview = {
        "stage": "rewrite",
        "execute": bool(args.execute),
        "source_shards": len(context["selected"]),
        "healthy_shards": len(healthy),
        "pending_shards": [source["id"] for source in pending],
        "planned_views": len(context["selected"]) * len(protocol["styles"]),
        "device": args.device,
        "local_files_only": args.local_files_only,
    }
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return
    if not pending:
        print(json.dumps({**preview, "status": "all_shards_already_complete"}, indent=2))
        return

    feature_tokenizer = _load_feature_tokenizer(
        protocol,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    for source in context["selected"]:
        decoded = feature_tokenizer.decode(
            source["output_token_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded != source["response"]:
            raise ValueError(f"Source Phi token roundtrip drifted for {source['id']!r}")
    backend = LocalChatModel(
        protocol,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    code = git_state(PROJECT_ROOT)
    completed: list[Dict[str, Any]] = []
    for source in pending:
        results = [
            rewrite_source_view(
                source,
                style,
                protocol,
                feature_tokenizer=feature_tokenizer,
                complete=backend.complete,
                protocol_sha256=context["protocol_sha256"],
                code=code,
            )
            for style in protocol["styles"]
        ]
        payload = {
            "schema_version": SHARD_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_id": source["id"],
            "query_id": source["query_id"],
            "source_correctness": int(source["correctness"]),
            "source_manifest_sha256": context["source_sha256"],
            "protocol_sha256": context["protocol_sha256"],
            "runtime": dict(context["runtime"]),
            "view_results": results,
            "code": code,
        }
        payload_path, marker_path = _shard_paths(paths, str(source["id"]))
        if payload_path.exists() or marker_path.exists():
            raise FileExistsError(f"Refusing to overwrite rewrite shard for {source['id']!r}")
        atomic_write_json(payload_path, payload)
        marker = {
            "schema_version": MARKER_SCHEMA,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source_id": source["id"],
            "protocol_sha256": context["protocol_sha256"],
            "payload": str(payload_path),
            "payload_sha256": file_sha256(payload_path),
            "view_count": len(results),
            "accepted_view_count": sum(bool(result["accepted"]) for result in results),
        }
        atomic_write_json(marker_path, marker)
        completed.append(marker)
        print(
            json.dumps(
                {
                    "source_id": source["id"],
                    "accepted": marker["accepted_view_count"],
                    "views": marker["view_count"],
                    "payload": str(payload_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(json.dumps({**preview, "completed": completed}, ensure_ascii=False, indent=2))


def _human_audit_records(
    results: Sequence[Mapping[str, Any]],
    sources_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    template: list[Dict[str, Any]] = []
    key: Dict[str, Any] = {}
    for result in results:
        if not result.get("accepted"):
            continue
        view = result["accepted_view"]
        lineage = view["rewrite_lineage"]
        source_text = str(sources_by_id[str(result["source_id"])]["response"])
        rewrite_text = view["response"]
        audit_id = canonical_json_sha256(
            {"view_id": view["id"], "certificate": lineage["certificate_sha256"]}
        )[:16]
        rewrite_first = int(audit_id, 16) % 2 == 0
        candidate_a, candidate_b = (
            (rewrite_text, source_text) if rewrite_first else (source_text, rewrite_text)
        )
        template.append(
            {
                "audit_id": audit_id,
                "problem": view["question"],
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "meaning_preserved": None,
                "final_answer_preserved": None,
                "numbers_operations_preserved": None,
                "introduced_correction": None,
                "added_or_removed_claims": None,
                "style_materially_changed": None,
                "decision": None,
                "issues": [],
            }
        )
        key[audit_id] = {
            "view_id": view["id"],
            "source_id": result["source_id"],
            "style_id": result["style_id"],
            "candidate_a_role": "rewrite" if rewrite_first else "source",
            "source_correctness": result["source_correctness"],
        }
    return template, key


def audit(
    args: argparse.Namespace,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    _require_preflight(protocol_path, protocol, context, paths)
    shards = [
        _load_valid_shard(
            paths,
            str(source["id"]),
            context["protocol_sha256"],
            context["runtime"],
        )
        for source in context["selected"]
    ]
    if any(shard is None for shard in shards):
        raise FileNotFoundError("Rewrite audit requires all four source shards")
    results = [result for shard in shards for result in shard["view_results"]]
    accepted = [result["accepted_view"] for result in results if result.get("accepted")]
    accepted_summary = validate_accepted_views(
        accepted,
        protocol,
        protocol_sha256=context["protocol_sha256"],
        require_complete=False,
    )
    rejection_counts: Counter[str] = Counter(
        reason for result in results for reason in result.get("rejection_reasons", [])
    )
    source_style_counts: Dict[str, set[str]] = defaultdict(set)
    source_rewrite_hashes: Dict[str, set[str]] = defaultdict(set)
    incorrect_repairs = 0
    for result in results:
        if result.get("accepted"):
            source_style_counts[str(result["source_id"])].add(str(result["style_id"]))
            source_rewrite_hashes[str(result["source_id"])].add(
                str(result["rewritten_response_sha256"])
            )
        gates = result.get("deterministic_gates")
        if isinstance(gates, Mapping):
            source_correct = int(gates["source_correctness_check"]["correctness"])
            rewrite_correct = int(gates["rewrite_correctness_check"]["correctness"])
            if source_correct == 0 and rewrite_correct == 1:
                incorrect_repairs += 1
    planned_views = int(protocol["quality_gate"]["planned_views"])
    required_accepted = int(protocol["quality_gate"]["required_accepted_views"])
    required_styles = int(protocol["quality_gate"]["required_styles_per_semantic_group"])
    automated_passed = (
        len(results) == planned_views
        and len(accepted) == required_accepted
        and incorrect_repairs
        <= int(protocol["quality_gate"]["maximum_incorrect_source_repairs"])
        and len(source_style_counts) == len(context["selected"])
        and all(len(styles) == required_styles for styles in source_style_counts.values())
        and (
            not protocol["quality_gate"].get(
                "require_distinct_rewrites_per_semantic_group", False
            )
            or (
                len(source_rewrite_hashes) == len(context["selected"])
                and all(
                    len(hashes) == required_styles
                    for hashes in source_rewrite_hashes.values()
                )
            )
        )
    )
    sources_by_id = {str(source["id"]): source for source in context["selected"]}
    human_template, human_key = _human_audit_records(results, sources_by_id)
    diagnostic_scope = protocol.get("diagnostic_scope")
    scale_authorization_eligible = not (
        isinstance(diagnostic_scope, Mapping)
        and diagnostic_scope.get("may_authorize_scale") is False
    )
    report = {
        "schema_version": "clir-semantic-rewrite-llm-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "passed_automated_gate_pending_human_blind_audit"
            if automated_passed
            else "failed_automated_gate"
        ),
        "automated_gate_passed": automated_passed,
        "human_blind_audit": {
            "required_before_scale": True,
            "status": "pending",
            "template_rows": len(human_template),
        },
        "scale_authorization_eligible": scale_authorization_eligible,
        "scale_allowed": False,
        "formal_mechanism_claim_allowed": False,
        "same_checkpoint_self_verification": True,
        "diagnostic_scope": (
            dict(diagnostic_scope) if isinstance(diagnostic_scope, Mapping) else None
        ),
        "protocol": str(protocol_path),
        "protocol_sha256": context["protocol_sha256"],
        "source_manifest_sha256": context["source_sha256"],
        "runtime": dict(context["runtime"]),
        "source_shards": len(shards),
        "planned_views": planned_views,
        "completed_view_results": len(results),
        "accepted_views": len(accepted),
        "rejected_views": len(results) - len(accepted),
        "incorrect_source_repairs": incorrect_repairs,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "accepted_summary": accepted_summary,
        "code": git_state(PROJECT_ROOT),
    }
    preview = {**report, "execute": bool(args.execute)}
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return
    derived_paths = (
        paths["all_results"],
        paths["accepted_views"],
        paths["audit_report"],
        paths["human_template"],
        paths["human_key"],
    )
    conflicts = [str(path) for path in derived_paths if path.exists()]
    if conflicts and not args.overwrite_reports:
        raise FileExistsError(
            "Refusing to overwrite rewrite audit outputs: " + ", ".join(conflicts)
        )
    atomic_write_jsonl(paths["all_results"], results)
    atomic_write_jsonl(paths["accepted_views"], accepted)
    atomic_write_jsonl(paths["human_template"], human_template)
    atomic_write_json(paths["human_key"], human_key)
    report["all_results_sha256"] = file_sha256(paths["all_results"])
    report["accepted_views_sha256"] = file_sha256(paths["accepted_views"])
    report["human_template_sha256"] = file_sha256(paths["human_template"])
    atomic_write_json(paths["audit_report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not automated_passed:
        raise RuntimeError("LLM semantic rewrite automated gate failed; inspect audit report")


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol_config).resolve()
    protocol = _load_protocol(protocol_path)
    paths = _paths(protocol)
    context = _context(protocol_path, protocol)
    if args.stage == "preflight":
        preflight(args, protocol_path, protocol, context, paths)
    elif args.stage == "rewrite":
        rewrite(args, protocol_path, protocol, context, paths)
    else:
        audit(args, protocol_path, protocol, context, paths)


if __name__ == "__main__":
    main()
