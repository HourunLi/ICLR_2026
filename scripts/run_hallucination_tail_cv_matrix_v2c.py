#!/usr/bin/env python3
"""Run the frozen T0/T2 four-fold, three-seed tail CV matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clir_real_data import file_sha256  # noqa: E402
from src.clir_stage_a import atomic_write_json, git_state  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "configs/hallucination_localization_v2/tail_cv_protocol_v2c.json"
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def matrix_jobs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    folds = [int(fold) for fold in protocol["cross_validation"]["folds"]]
    seeds = [int(seed) for seed in protocol["matched_training"]["seeds"]]
    cells = list(protocol["cells"])
    jobs = [
        {"fold": fold, "seed": seed, "cell": cell}
        for fold in sorted(folds)
        for seed in seeds
        for cell in cells
        if not (fold == 0 and seed == 42)
    ]
    expected = int(protocol["pipeline_gate"]["new_cells"])
    if len(jobs) != expected:
        raise ValueError(f"Tail CV matrix expected {expected} new cells, got {len(jobs)}")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Tail CV matrix requires a clean committed worktree")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != "clir-hallucination-tail-cross-validation-v2c":
        raise ValueError("Unknown tail CV training protocol")
    jobs = matrix_jobs(protocol)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU must be provided")
    if args.max_parallel <= 0 or args.max_parallel > len(gpus):
        raise ValueError("--max-parallel must be positive and no greater than GPU count")
    preflight = {
        "schema_version": "clir-hallucination-tail-cv-matrix-preflight-v2c",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "code": state,
        "jobs": jobs,
        "job_count": len(jobs),
        "gpus": gpus,
        "max_parallel": args.max_parallel,
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    output_root = resolve(protocol["execution"]["output_root"])
    report_path = output_root / "matrix_run_v2c.json"
    logs_root = output_root / "launcher_logs"
    if report_path.exists() or logs_root.exists():
        raise FileExistsError(f"Refusing to reuse tail CV matrix output {output_root}")
    logs_root.mkdir(parents=True, exist_ok=False)

    pending = list(jobs)
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    failed: dict[str, Any] | None = None
    while pending or running:
        while pending and len(running) < args.max_parallel:
            used_gpus = {str(job["gpu"]) for job in running}
            free_gpus = [gpu for gpu in gpus if gpu not in used_gpus]
            if not free_gpus:
                break
            job = {**pending.pop(0), "gpu": free_gpus[0]}
            label = f"fold_{job['fold']}_seed_{job['seed']}_{job['cell']}"
            log_path = logs_root / f"{label}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "scripts/run_hallucination_localization_pilot_v2.py"),
                "--protocol",
                str(protocol_path),
                "--fold",
                str(job["fold"]),
                "--seed",
                str(job["seed"]),
                "--cell",
                job["cell"],
                "--execute",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append(
                {
                    **job,
                    "label": label,
                    "log": str(log_path.relative_to(ROOT)),
                    "command": command,
                    "process": process,
                    "log_handle": log_handle,
                    "started_unix": time.time(),
                }
            )
        time.sleep(1.0)
        still_running: list[dict[str, Any]] = []
        for job in running:
            returncode = job["process"].poll()
            if returncode is None:
                still_running.append(job)
                continue
            job["log_handle"].close()
            record = {
                key: value
                for key, value in job.items()
                if key not in {"process", "log_handle"}
            }
            record["returncode"] = returncode
            record["elapsed_seconds"] = time.time() - float(job["started_unix"])
            completed.append(record)
            if returncode != 0 and failed is None:
                failed = record
        running = still_running
        if failed is not None:
            for job in running:
                job["process"].terminate()
            for job in running:
                job["process"].wait()
                job["log_handle"].close()
            break

    report = {
        "schema_version": "clir-hallucination-tail-cv-matrix-run-v2c",
        "status": "completed" if failed is None else "failed",
        "protocol": preflight["protocol"],
        "protocol_sha256": preflight["protocol_sha256"],
        "code": state,
        "requested_jobs": len(jobs),
        "completed_jobs": len(completed),
        "failed_job": failed,
        "jobs": completed,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed is not None:
        raise RuntimeError(f"Tail CV matrix failed at {failed['label']}; inspect {failed['log']}")


if __name__ == "__main__":
    main()
