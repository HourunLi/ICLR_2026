#!/usr/bin/env python3
"""Launch the frozen 12-cell dual-prior pilot with exclusive GPU assignment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_hallucination_annotation import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/dual_prior_evidence_v1/training_protocol_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def discover_gpus(explicit: str | None) -> list[str]:
    if explicit is not None:
        values = [value.strip() for value in explicit.split(",") if value.strip()]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        values = [
            value.strip()
            for value in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if value.strip()
        ]
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"GPU list must be non-empty and unique, got {values}")
    return values


def validate_completed_result(
    path: Path,
    *,
    cell: str,
    seed: int,
    protocol_sha256: str,
    commit: str,
) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "clir-dual-prior-cell-result-v1":
        raise ValueError(f"Unexpected completed result schema: {path}")
    if result.get("cell") != cell or int(result.get("seed", -1)) != seed:
        raise ValueError(f"Completed result identity drifted: {path}")
    if result.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Completed result protocol drifted: {path}")
    code = result.get("code")
    if not isinstance(code, Mapping) or code.get("commit") != commit or code.get("dirty"):
        raise ValueError(f"Completed result code state differs from this matrix: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated physical GPU identifiers; defaults to CUDA_VISIBLE_DEVICES or nvidia-smi.",
    )
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-dual-prior-standalone-training-protocol-v1":
        raise ValueError("Unexpected dual-prior training protocol schema")
    code = git_state(ROOT)
    if protocol["execution"]["clean_committed_worktree_required"] and code["dirty"]:
        raise RuntimeError("Dual-prior matrix requires a clean committed worktree")
    protocol_sha = file_sha256(protocol_path)
    gpus = discover_gpus(args.gpus)
    frozen_max = int(protocol["execution"]["max_parallel_cells"])
    requested_max = frozen_max if args.max_parallel is None else int(args.max_parallel)
    if requested_max <= 0 or requested_max > frozen_max:
        raise ValueError(f"--max-parallel must be in [1, {frozen_max}]")
    max_parallel = min(requested_max, len(gpus))
    seeds = [int(value) for value in protocol["matched_training"]["seeds"]]
    cells = list(protocol["cells"])
    jobs = [(seed, cell) for seed in seeds for cell in cells]
    output_root = resolve(protocol["execution"]["output_root"])
    interpreter = Path(protocol["execution"]["recommended_interpreter"])
    if not interpreter.is_file():
        raise FileNotFoundError(f"Frozen interpreter does not exist: {interpreter}")

    preflight = {
        "schema_version": "clir-dual-prior-matrix-preflight-v1",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "code": code,
        "jobs": [{"seed": seed, "cell": cell} for seed, cell in jobs],
        "gpus": gpus,
        "max_parallel": max_parallel,
        "output_root": str(output_root),
        "execute": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus[:max_parallel]:
        gpu_queue.put(gpu)

    def run_job(seed: int, cell: str) -> dict[str, Any]:
        cell_root = output_root / f"seed_{seed}" / cell
        result_path = cell_root / "cell_result.json"
        if result_path.is_file():
            validate_completed_result(
                result_path,
                cell=cell,
                seed=seed,
                protocol_sha256=protocol_sha,
                commit=str(code["commit"]),
            )
            return {
                "seed": seed,
                "cell": cell,
                "status": "already_completed",
                "result_sha256": file_sha256(result_path),
            }
        protected_outputs = [
            cell_root / name
            for name in (
                "model.pt",
                "metrics.jsonl",
                "run.json",
                "train_scored.jsonl",
                "dev_scored.jsonl",
                "evaluation.json",
            )
        ]
        partial = [str(path) for path in protected_outputs if path.exists()]
        if partial:
            raise FileExistsError(
                f"Refusing to overwrite partial cell {cell}/seed {seed}: {partial}"
            )
        gpu = gpu_queue.get()
        try:
            cell_root.mkdir(parents=True, exist_ok=True)
            log_path = cell_root / "launcher.log"
            command = [
                str(interpreter),
                str(ROOT / protocol["execution"]["runner"]),
                "--protocol",
                str(protocol_path),
                "--cell",
                cell,
                "--seed",
                str(seed),
                "--execute",
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"START seed={seed} cell={cell} gpu={gpu}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Cell failed seed={seed} cell={cell} gpu={gpu}; see {log_path}"
                )
            validate_completed_result(
                result_path,
                cell=cell,
                seed=seed,
                protocol_sha256=protocol_sha,
                commit=str(code["commit"]),
            )
            print(f"DONE seed={seed} cell={cell} gpu={gpu}", flush=True)
            return {
                "seed": seed,
                "cell": cell,
                "gpu": gpu,
                "status": "completed",
                "log": str(log_path),
                "result_sha256": file_sha256(result_path),
            }
        finally:
            gpu_queue.put(gpu)

    completed_jobs: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(run_job, seed, cell): (seed, cell)
            for seed, cell in jobs
        }
        for future in as_completed(futures):
            seed, cell = futures[future]
            try:
                completed_jobs.append(future.result())
            except BaseException as exc:
                failures.append(f"seed={seed} cell={cell}: {type(exc).__name__}: {exc}")

    launcher_report = {
        **preflight,
        "schema_version": "clir-dual-prior-matrix-launch-v1",
        "status": "completed" if not failures else "failed",
        "completed_jobs": sorted(
            completed_jobs, key=lambda row: (int(row["seed"]), str(row["cell"]))
        ),
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    launcher_report_path = output_root / "matrix_launch.json"
    atomic_write_json(launcher_report_path, launcher_report)
    if failures:
        raise RuntimeError("Dual-prior matrix failed:\n" + "\n".join(failures))

    subprocess.run(
        [
            str(interpreter),
            str(ROOT / protocol["execution"]["summarizer"]),
            "--protocol",
            str(protocol_path),
        ],
        cwd=ROOT,
        check=True,
    )
    print(json.dumps(launcher_report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
