"""Keep the Mac training machine full while draining the V5 remaining jobs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil

from run_v5_matrix import build_jobs


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "journal_results" / "v5-mac"


def _training_processes():
    result = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or ())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "Train_SINR.py" in command or "Train_D3QN.py" in command:
            result.append(process.pid)
    return result


def build_remaining_jobs(python, device="mps", output_root=DEFAULT_OUTPUT):
    common = dict(
        python=python,
        device=device,
        output_root=str(output_root),
        skip_shared_proposed=True,
        reuse_existing_e1=True,
    )
    e4 = build_jobs(
        SimpleNamespace(
            groups=("E4",),
            levels=("medium", "high"),
            **common,
        )
    )
    e5 = [
        job
        for job in build_jobs(
            SimpleNamespace(
                groups=("E5",),
                levels=("medium", "high"),
                **common,
            )
        )
        if job.job_id in {"E5/sa_d3qn/medium", "E5/sa_d3qn/high"}
    ]
    # Complete the two remaining E5 baselines before filling all twelve
    # slots with the longer E4 sensitivity sweep.
    return [*e5, *e4]


def _complete(job):
    return any(job.output_root.rglob("UAV_TD3_improved.npz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    jobs = build_remaining_jobs(args.python, args.device, output_root)
    pending = [job for job in jobs if not _complete(job)]
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running = {}
    attempted = set()
    failed = {}
    print(
        f"remaining_jobs={len(pending)} target_workers={args.workers}",
        flush=True,
    )

    while pending or running:
        for job_id, (process, log_stream) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log_stream.close()
            del running[job_id]
            if returncode != 0:
                failed[job_id] = returncode
            print(f"finished {job_id} rc={returncode}", flush=True)

        active = _training_processes()
        while pending and len(active) < args.workers:
            job = pending.pop(0)
            if _complete(job):
                print(f"skip-complete {job.job_id}", flush=True)
                continue
            attempted.add(job.job_id)
            log_path = log_dir / (job.job_id.replace("/", "__") + ".log")
            log_stream = log_path.open("w", encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                job.command,
                cwd=ROOT,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            running[job.job_id] = (process, log_stream)
            print(f"started {job.job_id} pid={process.pid} log={log_path}", flush=True)
            time.sleep(1)
            active = _training_processes()
        if pending or running:
            time.sleep(max(2, args.poll_seconds))

    print(f"attempted={len(attempted)} failed={failed}", flush=True)
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
