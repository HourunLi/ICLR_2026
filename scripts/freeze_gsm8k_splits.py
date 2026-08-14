#!/usr/bin/env python3
"""Freeze deterministic query-level GSM8K memberships for CLIR experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clir_real_data import load_protocol, protocol_hashes
from src.clir_stage_a import atomic_write_json, build_gsm8k_split_manifest


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phi35_gsm8k_pilot_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "splits" / "gsm8k_phi35_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-primary-size", type=int, default=6000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--development-size", type=int, default=32)
    parser.add_argument("--pilot-test-size", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite frozen split manifest: {output}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the real-data requirements before freezing GSM8K splits") from exc

    protocol = load_protocol(args.protocol_config)
    hashes = protocol_hashes(protocol)
    dataset_cfg = protocol["dataset"]
    common = {
        "path": dataset_cfg["repo_id"],
        "name": dataset_cfg["subset"],
        "revision": dataset_cfg["revision"],
        "cache_dir": args.cache_dir,
    }
    train = load_dataset(split="train", **common)
    test = load_dataset(split="test", **common)
    manifest = build_gsm8k_split_manifest(
        train,
        test,
        dataset={
            "repo_id": dataset_cfg["repo_id"],
            "subset": dataset_cfg["subset"],
            "revision": dataset_cfg["revision"],
            "train_rows": len(train),
            "test_rows": len(test),
        },
        protocol_version=protocol["protocol_version"],
        protocol_sha256=hashes["protocol_sha256"],
        acquisition_protocol_sha256=hashes["acquisition_protocol_sha256"],
        label_protocol_sha256=hashes["label_protocol_sha256"],
        seed=args.seed,
        train_primary_size=args.train_primary_size,
        validation_size=args.validation_size,
        development_size=args.development_size,
        pilot_test_size=args.pilot_test_size,
    )
    atomic_write_json(output, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest_sha256": manifest["manifest_sha256"],
                "memberships": {
                    name: len(query_ids)
                    for name, query_ids in manifest["memberships"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
