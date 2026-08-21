#!/usr/bin/env python3
"""Freeze all-64 JP-to-H heads and score the 96-row blind set label-free."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_jp_h_frozen_probe_v1 import (  # noqa: E402
    atomic_torch_save,
    load_frozen_base,
    supervised_token_matrix,
)
from src.clir_data import clir_collate, read_jsonl  # noqa: E402
from src.clir_frozen_h_probe import fit_linear_probe, score_linear_probe  # noqa: E402
from src.clir_h_temporal_smoother import (  # noqa: E402
    smooth_persisted_probabilities,
)
from src.clir_hidden_states import (  # noqa: E402
    OnlineHiddenStateConfig,
    load_online_hidden_state_extractor,
    validate_online_rows,
)
from src.clir_jp_h_blind_scoring import (  # noqa: E402
    build_prediction_row,
    sanitize_lineage_row,
    validate_blind_scoring_protocol,
    validate_prediction_row,
)
from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    git_state,
)


DEFAULT_PROTOCOL = ROOT / "configs/jp_h_blind_validation_v1/scoring_protocol_v1.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verify_artifact(spec: Mapping[str, Any], *, name: str) -> Path:
    path = resolve(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {name}: {path}")
    observed = file_sha256(path)
    if observed != spec["sha256"]:
        raise ValueError(
            f"Frozen {name} hash drifted: expected {spec['sha256']}, got {observed}"
        )
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def verify_declared_inputs(protocol: Mapping[str, Any]) -> dict[str, Path]:
    """Verify the allowlisted scoring inputs; sealed labels are not traversed."""

    paths = {
        name: verify_artifact(spec, name=name)
        for name, spec in protocol["inputs"].items()
    }
    package = _load_json(paths["blind_package_report"])
    if (
        package.get("status") != "passed"
        or package.get("rows") != 96
        or package.get("queries") != 96
        or package.get("selected_exact_token_alignment_failures") != 0
        or package.get("forbidden_field_leaks") != 0
        or package.get("pilot_test_accessed") is not False
        or package.get("final_test_accessed") is not False
    ):
        raise ValueError("Blind acquisition package is not a clean 96-row exact-token set")
    cache_report = _load_json(paths["old_jp_feature_cache_report"])
    if (
        cache_report.get("status") != "passed"
        or cache_report.get("rows") != 64
        or cache_report.get("model_dim") != 768
        or cache_report.get("base_checkpoint_unchanged") is not True
        or cache_report.get("reward_score_bit_identical_rows") != 64
        or cache_report.get("token_value_bit_identical_rows") != 64
    ):
        raise ValueError("Old frozen JP feature cache failed its integrity contract")
    return paths


def load_blind_inputs(
    lineage_path: Path,
    *,
    expected_rows: int,
    expected_tokens: int,
) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(lineage_path)
    rows = [sanitize_lineage_row(row) for row in raw_rows]
    rows.sort(key=lambda row: int(row["selection_order"]))
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} blind rows, got {len(rows)}")
    if [row["selection_order"] for row in rows] != list(range(expected_rows)):
        raise ValueError("Blind selection order must be the contiguous range 0..95")
    if len({row["item_id"] for row in rows}) != expected_rows:
        raise ValueError("Blind item IDs are not unique")
    if len({row["query_id"] for row in rows}) != expected_rows:
        raise ValueError("Blind query IDs are not unique")
    total_tokens = sum(len(row["output_token_ids"]) for row in rows)
    if total_tokens != expected_tokens:
        raise ValueError(
            f"Blind output-token count drifted: expected {expected_tokens}, got {total_tokens}"
        )
    return rows


def verify_tokenizer_decode(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    revision: str,
    trust_remote_code: bool,
    cache_dir: str,
) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment failure path
        raise RuntimeError("transformers is required for blind scoring") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    for row in rows:
        decoded = tokenizer.decode(
            row["output_token_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded != row["response"]:
            raise ValueError(
                f"Exact output-token decode drifted for blind item {row['item_id']}"
            )


def _load_old_training_matrix(
    protocol: Mapping[str, Any], paths: Mapping[str, Path]
) -> tuple[Tensor, Tensor, Mapping[str, Tensor]]:
    cache = torch.load(
        paths["old_jp_feature_cache"], map_location="cpu", weights_only=False
    )
    if (
        cache.get("schema_version")
        != "clir-jp-h-frozen-token-feature-cache-v1"
        or cache.get("model_dim") != 768
        or len(cache.get("records", {})) != 64
    ):
        raise ValueError("Old JP token-feature cache schema or coverage drifted")
    rows = [
        *read_jsonl(paths["old_fold0_train48"]),
        *read_jsonl(paths["old_fold0_dev16"]),
    ]
    if len(rows) != 64 or len({str(row["id"]) for row in rows}) != 64:
        raise ValueError("All-64 head training rows are not 64 unique trajectories")
    features, targets = supervised_token_matrix(rows, cache["records"])
    readout = protocol["readout"]
    if features.shape != (int(readout["supervised_tokens"]), 768):
        raise ValueError("All-64 supervised feature matrix shape drifted")
    if targets.numel() != int(readout["supervised_tokens"]):
        raise ValueError("All-64 supervised target length drifted")
    if int(targets.sum()) != int(readout["positive_tokens"]):
        raise ValueError("All-64 positive-token count drifted")
    return features, targets, cache["records"]


def _head_from_checkpoint(path: Path, *, expected_seed: int) -> nn.Linear:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version")
        != "clir-jp-h-blind-all64-linear-head-checkpoint-v1"
        or checkpoint.get("seed") != expected_seed
        or checkpoint.get("model_dim") != 768
        or checkpoint.get("probe_parameter_count") != 769
        or checkpoint.get("base_state_dict_included") is not False
        or checkpoint.get("new_hallucination_labels_used") is not False
    ):
        raise ValueError(f"Frozen all-64 head checkpoint drifted for seed {expected_seed}")
    head = nn.Linear(768, 1, dtype=torch.float32)
    head.load_state_dict(checkpoint["state_dict"])
    head.requires_grad_(False)
    head.eval()
    return head


def prepare_heads(
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    paths: Mapping[str, Path],
    output_root: Path,
    device: torch.device,
    code: Mapping[str, Any],
    resume: bool,
) -> tuple[dict[int, nn.Linear], list[dict[str, Any]], Path]:
    head_root = output_root / "final_heads"
    manifest_path = head_root / "head_manifest_v1.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError("Frozen all-64 heads already exist; use --resume")
        manifest = _load_json(manifest_path)
        if (
            manifest.get("schema_version") != "clir-jp-h-blind-head-manifest-v1"
            or manifest.get("scoring_protocol_sha256") != protocol_sha256
            or manifest.get("status") != "completed"
        ):
            raise ValueError("Frozen all-64 head manifest drifted")
        records = list(manifest.get("heads", []))
        if [record.get("seed") for record in records] != protocol["readout"]["seeds"]:
            raise ValueError("Frozen all-64 head seed coverage drifted")
        heads: dict[int, nn.Linear] = {}
        for record in records:
            seed = int(record["seed"])
            path = resolve(record["path"])
            if file_sha256(path) != record["sha256"]:
                raise ValueError(f"Frozen all-64 head hash drifted for seed {seed}")
            heads[seed] = _head_from_checkpoint(path, expected_seed=seed)
        return heads, records, manifest_path

    features, targets, _ = _load_old_training_matrix(protocol, paths)
    head_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    heads = {}
    readout = protocol["readout"]
    for seed in map(int, readout["seeds"]):
        head, history = fit_linear_probe(
            features,
            targets,
            seed=seed,
            epochs=int(readout["epochs"]),
            learning_rate=float(readout["learning_rate"]),
            weight_decay=float(readout["weight_decay"]),
            max_grad_norm=float(readout["max_grad_norm"]),
            device=device,
        )
        seed_root = head_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = seed_root / "probe.pt"
        checkpoint = {
            "schema_version": "clir-jp-h-blind-all64-linear-head-checkpoint-v1",
            "scoring_protocol_sha256": protocol_sha256,
            "base_checkpoint_sha256": protocol["inputs"]["jp_checkpoint"]["sha256"],
            "old_feature_cache_sha256": protocol["inputs"]["old_jp_feature_cache"][
                "sha256"
            ],
            "old_train48_sha256": protocol["inputs"]["old_fold0_train48"]["sha256"],
            "old_dev16_sha256": protocol["inputs"]["old_fold0_dev16"]["sha256"],
            "seed": seed,
            "model_dim": 768,
            "probe_parameter_count": 769,
            "training_rows": 64,
            "supervised_tokens": int(targets.numel()),
            "positive_tokens": int(targets.sum()),
            "base_state_dict_included": False,
            "new_hallucination_labels_used": False,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in head.state_dict().items()
            },
            "code": dict(code),
        }
        atomic_torch_save(checkpoint, checkpoint_path)
        history_path = seed_root / "training_history.jsonl"
        atomic_write_jsonl(history_path, history)
        record = {
            "seed": seed,
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": file_sha256(checkpoint_path),
            "training_history_path": str(history_path.relative_to(ROOT)),
            "training_history_sha256": file_sha256(history_path),
            "initial_train_bce": history[0]["pre_update_train_bce"],
            "final_train_bce": history[-1]["post_update_train_bce"],
        }
        records.append(record)
        head = head.to("cpu")
        head.requires_grad_(False)
        head.eval()
        heads[seed] = head
        print(
            f"[heads] seed={seed} final_bce={record['final_train_bce']:.8f} "
            f"sha256={record['sha256']}",
            flush=True,
        )
    manifest = {
        "schema_version": "clir-jp-h-blind-head-manifest-v1",
        "status": "completed",
        "scoring_protocol_sha256": protocol_sha256,
        "training_rows": 64,
        "supervised_tokens": int(targets.numel()),
        "positive_tokens": int(targets.sum()),
        "heads": records,
        "new_hallucination_labels_used": False,
        "code": dict(code),
    }
    atomic_write_json(manifest_path, manifest)
    del features, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return heads, records, manifest_path


def _feature_payload(
    path: Path,
    *,
    item: Mapping[str, Any],
    protocol_sha256: str,
) -> Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "clir-jp-h-blind-jp-feature-v1"
        or payload.get("scoring_protocol_sha256") != protocol_sha256
        or payload.get("item_id") != item["item_id"]
        or payload.get("output_token_ids_sha256")
        != item["output_token_ids_sha256"]
    ):
        raise ValueError(f"Persisted JP feature metadata drifted for {item['item_id']}")
    features = payload.get("token_features")
    expected_shape = (len(item["output_token_ids"]), 768)
    if (
        not isinstance(features, Tensor)
        or features.shape != expected_shape
        or features.dtype != torch.float32
        or not torch.isfinite(features).all()
    ):
        raise ValueError(f"Persisted JP feature tensor drifted for {item['item_id']}")
    return features.contiguous()


@torch.no_grad()
def extract_jp_features(
    item: Mapping[str, Any],
    *,
    extractor: Any,
    jp_model: nn.Module,
) -> Tensor:
    batch = clir_collate(
        [
            {
                "row_index": int(item["selection_order"]),
                "id": item["item_id"],
                "query_id": item["query_id"],
                "prompt_token_ids": list(item["prompt_token_ids"]),
                "output_token_ids": list(item["output_token_ids"]),
            }
        ]
    )
    batch = extractor.materialize(batch, include_condition=True)
    parameter_dtype = next(jp_model.parameters()).dtype
    for name in ("hidden_states", "condition_states", "condition_embedding"):
        if name in batch:
            batch[name] = batch[name].to(dtype=parameter_dtype)
    outputs = jp_model(
        batch["hidden_states"],
        mask=batch["mask"],
        condition_states=batch.get("condition_states"),
        condition_mask=batch.get("condition_mask"),
        condition_embedding=batch.get("condition_embedding"),
        condition_embedding_mask=batch.get("condition_embedding_mask"),
    )
    length = len(item["output_token_ids"])
    features = outputs["token_features"][0, :length].detach().cpu().contiguous()
    if (
        features.shape != (length, 768)
        or features.dtype != torch.float32
        or not torch.isfinite(features).all()
    ):
        raise ValueError(f"Frozen JP features drifted for {item['item_id']}")
    del outputs, batch
    return features


def score_features(
    features: Tensor,
    *,
    heads: Mapping[int, nn.Linear],
    head_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records_by_seed = {int(record["seed"]): record for record in head_records}
    predictions: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        logits, probabilities = score_linear_probe(heads[seed], features)
        smoothed_logits, smoothed_probabilities = smooth_persisted_probabilities(
            probabilities, window_tokens=3
        )
        predictions.append(
            {
                "seed": seed,
                "head_sha256": records_by_seed[seed]["sha256"],
                "raw_logits": logits,
                "raw_probabilities": probabilities,
                "smoothed_logits": smoothed_logits,
                "smoothed_probabilities": smoothed_probabilities,
            }
        )
    return predictions


def _feature_path(output_root: Path, item: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "jp_token_features"
        / f"{int(item['selection_order']):03d}_{item['item_id']}.pt"
    )


def _row_checkpoint_path(output_root: Path, item: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "row_checkpoints"
        / f"{int(item['selection_order']):03d}_{item['item_id']}.json"
    )


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    paths: Mapping[str, Path],
    output_root: Path,
    heads: Mapping[int, nn.Linear],
    head_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    cache_dir: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed: dict[str, dict[str, Any]] = {}
    feature_records: dict[str, dict[str, Any]] = {}
    for item in rows:
        prediction_path = _row_checkpoint_path(output_root, item)
        feature_path = _feature_path(output_root, item)
        if resume and prediction_path.is_file():
            prediction = _load_json(prediction_path)
            validate_prediction_row(
                prediction,
                expected_item=item,
                protocol_sha256=protocol_sha256,
            )
            if not feature_path.is_file():
                raise FileNotFoundError(
                    f"Prediction exists without JP feature payload: {feature_path}"
                )
            observed_feature_sha = file_sha256(feature_path)
            if observed_feature_sha != prediction["jp_feature_sha256"]:
                raise ValueError(f"JP feature hash drifted for {item['item_id']}")
            _feature_payload(
                feature_path, item=item, protocol_sha256=protocol_sha256
            )
            completed[str(item["item_id"])] = prediction
            feature_records[str(item["item_id"])] = {
                "schema_version": "clir-jp-h-blind-jp-feature-record-v1",
                "selection_order": int(item["selection_order"]),
                "item_id": str(item["item_id"]),
                "output_token_ids_sha256": str(item["output_token_ids_sha256"]),
                "tokens": len(item["output_token_ids"]),
                "shape": [len(item["output_token_ids"]), 768],
                "dtype": "float32",
                "path": str(feature_path.relative_to(ROOT)),
                "sha256": observed_feature_sha,
            }
    if len(completed) == len(rows):
        print(f"[score] resumed {len(completed)}/{len(rows)} completed rows", flush=True)
        ordered_predictions = [completed[str(item["item_id"])] for item in rows]
        ordered_features = [feature_records[str(item["item_id"])] for item in rows]
        return ordered_predictions, ordered_features

    representation = protocol["representation"]
    extractor_config = OnlineHiddenStateConfig(
        model_id=str(representation["task_model"]),
        model_revision=str(representation["task_model_revision"]),
        tokenizer_revision=str(representation["tokenizer_revision"]),
        torch_dtype=str(representation["torch_dtype"]),
        trust_remote_code=bool(representation["trust_remote_code"]),
        layer_count=int(representation["layer_count"]),
        per_layer_hidden_size=int(representation["per_layer_hidden_size"]),
        feature_dim=int(representation["all_layer_feature_width"]),
    )
    validate_online_rows(rows, extractor_config)
    extractor = load_online_hidden_state_extractor(
        extractor_config,
        device=device,
        cache_dir=cache_dir,
        local_files_only=True,
        check_finite=True,
    )
    jp_model, _ = load_frozen_base(paths["jp_checkpoint"], device)
    if any(parameter.requires_grad for parameter in jp_model.parameters()):
        raise AssertionError("Frozen JP model unexpectedly has trainable parameters")

    for item in rows:
        item_id = str(item["item_id"])
        if item_id in completed:
            continue
        feature_path = _feature_path(output_root, item)
        prediction_path = _row_checkpoint_path(output_root, item)
        if feature_path.is_file():
            if not resume:
                raise FileExistsError(f"JP feature already exists: {feature_path}")
            features = _feature_payload(
                feature_path, item=item, protocol_sha256=protocol_sha256
            )
        else:
            features = extract_jp_features(
                item, extractor=extractor, jp_model=jp_model
            )
            payload = {
                "schema_version": "clir-jp-h-blind-jp-feature-v1",
                "scoring_protocol_sha256": protocol_sha256,
                "item_id": item_id,
                "query_id": str(item["query_id"]),
                "selection_order": int(item["selection_order"]),
                "output_token_ids_sha256": str(item["output_token_ids_sha256"]),
                "tokens": len(item["output_token_ids"]),
                "feature_width": 768,
                "storage_dtype": "float32",
                "base_checkpoint_sha256": protocol["inputs"]["jp_checkpoint"][
                    "sha256"
                ],
                "new_hallucination_labels_used": False,
                "token_features": features,
            }
            atomic_torch_save(payload, feature_path)
        feature_sha = file_sha256(feature_path)
        prediction = build_prediction_row(
            item,
            score_features(
                features, heads=heads, head_records=head_records
            ),
            protocol_sha256=protocol_sha256,
            jp_feature_sha256=feature_sha,
        )
        validate_prediction_row(
            prediction,
            expected_item=item,
            protocol_sha256=protocol_sha256,
        )
        atomic_write_json(prediction_path, prediction)
        completed[item_id] = prediction
        feature_records[item_id] = {
            "schema_version": "clir-jp-h-blind-jp-feature-record-v1",
            "selection_order": int(item["selection_order"]),
            "item_id": item_id,
            "output_token_ids_sha256": str(item["output_token_ids_sha256"]),
            "tokens": len(item["output_token_ids"]),
            "shape": [len(item["output_token_ids"]), 768],
            "dtype": "float32",
            "path": str(feature_path.relative_to(ROOT)),
            "sha256": feature_sha,
        }
        print(
            f"[score] completed={len(completed)}/{len(rows)} "
            f"order={int(item['selection_order']):03d} item_id={item_id} "
            f"tokens={len(item['output_token_ids'])}",
            flush=True,
        )
        del features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del jp_model, extractor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ordered_predictions = [completed[str(item["item_id"])] for item in rows]
    ordered_features = [feature_records[str(item["item_id"])] for item in rows]
    return ordered_predictions, ordered_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--cache-dir", default="/prodcpfs/user/panzhixin/hf_cache"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = _load_json(protocol_path)
    validate_blind_scoring_protocol(protocol)
    protocol_sha256 = file_sha256(protocol_path)
    output_root = resolve(protocol["execution"]["output_root"])
    preflight = {
        "schema_version": "clir-jp-h-blind-scoring-preflight-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "output_root": str(output_root),
        "device": args.device,
        "rows": protocol["execution"]["expected_rows"],
        "seeds": protocol["readout"]["seeds"],
        "sealed_annotation_paths_opened": False,
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    code = git_state(ROOT)
    if code["dirty"]:
        raise RuntimeError("Blind scoring requires a clean committed worktree")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Blind scoring requested CUDA but CUDA is unavailable")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Blind-scoring output exists; use --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    completion_path = resolve(protocol["execution"]["completion_report"])
    predictions_path = resolve(protocol["execution"]["predictions"])
    if completion_path.is_file() and args.resume:
        completion = _load_json(completion_path)
        if (
            completion.get("status") == "completed_predictions_frozen"
            and completion.get("scoring_protocol_sha256") == protocol_sha256
            and predictions_path.is_file()
            and file_sha256(predictions_path) == completion.get("predictions_sha256")
        ):
            print(json.dumps(completion, ensure_ascii=False, indent=2))
            return
        raise ValueError("Existing blind-scoring completion report failed validation")

    paths = verify_declared_inputs(protocol)
    rows = load_blind_inputs(
        paths["blind_private_lineage"],
        expected_rows=int(protocol["execution"]["expected_rows"]),
        expected_tokens=int(protocol["execution"]["expected_total_output_tokens"]),
    )
    representation = protocol["representation"]
    verify_tokenizer_decode(
        rows,
        model_id=str(representation["task_model"]),
        revision=str(representation["tokenizer_revision"]),
        trust_remote_code=bool(representation["trust_remote_code"]),
        cache_dir=args.cache_dir,
    )
    device = torch.device(args.device)
    heads, head_records, head_manifest_path = prepare_heads(
        protocol,
        protocol_sha256=protocol_sha256,
        paths=paths,
        output_root=output_root,
        device=device,
        code=code,
        resume=args.resume,
    )
    predictions, feature_records = score_rows(
        rows,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        paths=paths,
        output_root=output_root,
        heads=heads,
        head_records=head_records,
        device=device,
        cache_dir=args.cache_dir,
        resume=args.resume,
    )
    if len(predictions) != 96 or len(feature_records) != 96:
        raise AssertionError("Blind scoring did not produce exactly 96 durable rows")
    atomic_write_jsonl(predictions_path, predictions)
    feature_manifest_path = output_root / "jp_token_feature_manifest_v1.jsonl"
    atomic_write_jsonl(feature_manifest_path, feature_records)
    predictions_sha256 = file_sha256(predictions_path)
    completion = {
        "schema_version": "clir-jp-h-blind-scoring-completion-v1",
        "status": "completed_predictions_frozen",
        "evidence_tier": "blind_validation_pre_unseal_predictions",
        "scoring_protocol": str(protocol_path.relative_to(ROOT)),
        "scoring_protocol_sha256": protocol_sha256,
        "execution_code": code,
        "rows": len(predictions),
        "queries": len({row["query_id"] for row in predictions}),
        "total_output_tokens": sum(row["token_count"] for row in predictions),
        "seeds": protocol["readout"]["seeds"],
        "heads": head_records,
        "head_manifest": str(head_manifest_path.relative_to(ROOT)),
        "head_manifest_sha256": file_sha256(head_manifest_path),
        "predictions": str(predictions_path.relative_to(ROOT)),
        "predictions_sha256": predictions_sha256,
        "feature_manifest": str(feature_manifest_path.relative_to(ROOT)),
        "feature_manifest_sha256": file_sha256(feature_manifest_path),
        "jp_feature_rows": len(feature_records),
        "base_checkpoint_sha256_before": protocol["inputs"]["jp_checkpoint"][
            "sha256"
        ],
        "base_checkpoint_sha256_after": file_sha256(paths["jp_checkpoint"]),
        "base_checkpoint_unchanged": (
            file_sha256(paths["jp_checkpoint"])
            == protocol["inputs"]["jp_checkpoint"]["sha256"]
        ),
        "new_hallucination_labels_opened": False,
        "selection_correctness_fed_to_model": False,
        "smoother_enters_reward_score": False,
        "pilot_test_accessed": False,
        "final_test_accessed": False,
    }
    if completion["base_checkpoint_unchanged"] is not True:
        raise RuntimeError("Read-only JP checkpoint changed during blind scoring")
    atomic_write_json(completion_path, completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
