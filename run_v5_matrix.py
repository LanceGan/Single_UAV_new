"""Build and execute the V5 neural-network experiment matrix on Win/CUDA."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "journal_results" / "v5-win"


@dataclass(frozen=True)
class Job:
    job_id: str
    group: str
    command: tuple[str, ...]
    output_root: Path


def _common(python, script, device, level):
    return [
        python,
        script,
        "--data_level",
        level,
        "--total_steps",
        "2000",
        "--qos_enforcement",
        "continue",
        "--outage_max_s",
        "6",
        "--seed",
        "1",
        "--eval_interval",
        "25",
        "--device",
        device,
    ]


def _continuity_flags(blind=False):
    values = [
        "--allow_preflight_failure",
        "--teacher",
        "mapguide",
        "--guide_prob",
        "0.5",
        "--guide_min_prob",
        "0.5",
    ]
    if blind:
        values.append("--a2g_blind")
    return values


def _td3_ddpg_job(
    *,
    python,
    device,
    base,
    group,
    label,
    level,
    algorithm,
    sequence,
    blind=False,
    workers="3000",
):
    name = "ca_td3" if algorithm == "td3_bc" else "ddpg"
    if group == "E1":
        output = base / "E1" / name
        job_id = f"E1/{name}/{level}"
    else:
        output = base / group / label
        job_id = f"{group}/{label}/{level}"
    command = [
        *_common(python, "Train_SINR.py", device, level),
        "--episodes",
        str(workers),
        "--algorithm",
        algorithm,
        "--sequence_method",
        sequence,
        "--output_root",
        str(output),
        *_continuity_flags(blind),
    ]
    return Job(job_id, group, command, output)


def _d3qn_job(
    *,
    python,
    device,
    base,
    group,
    label,
    level,
    sequence,
    blind=False,
    episodes="3000",
):
    output = base / (group if group != "E1" else "E1") / label
    job_id = f"{group}/{label}/{level}"
    command = [
        *_common(python, "Train_D3QN.py", device, level),
        "--episodes",
        str(episodes),
        "--sequence_method",
        sequence,
        "--teacher",
        "mapguide",
        "--guide_prob",
        "0.5",
        "--guide_min_prob",
        "0.5",
        "--pretrain_demo_episodes",
        "40",
        "--pretrain_updates",
        "20000",
        "--output_root",
        str(output),
    ]
    if blind:
        command.append("--a2g_blind")
    return Job(job_id, group, command, output)


def _ppo_job(*, python, device, base, level):
    output = base / "E1" / "ppo"
    command = [
        *_common(python, "Train_PPO.py", device, level),
        "--episodes",
        "3000",
        "--sequence_method",
        "priga",
        "--teacher",
        "mapguide",
        "--pretrain_demo_episodes",
        "40",
        "--pretrain_updates",
        "20000",
        "--guide_prob",
        "0.5",
        "--guide_min_prob",
        "0.5",
        "--allow_preflight_failure",
        "--output_root",
        str(output),
    ]
    return Job(f"E1/ppo/{level}", "E1", command, output)


def _sac_job(*, python, device, base, level):
    output = base / "E1" / "sac"
    command = [
        *_common(python, "Train_SAC.py", device, level),
        "--episodes",
        "3000",
        "--sequence_method",
        "priga",
        "--teacher",
        "mapguide",
        "--pretrain_demo_episodes",
        "40",
        "--pretrain_updates",
        "20000",
        "--guide_prob",
        "0.5",
        "--guide_min_prob",
        "0.5",
        "--allow_preflight_failure",
        "--output_root",
        str(output),
    ]
    return Job(f"E1/sac/{level}", "E1", command, output)


def build_jobs(args):
    base = Path(args.output_root).resolve()
    jobs = []
    if "E1" in args.groups:
        for level in args.levels:
            if not args.reuse_existing_e1:
                jobs.append(
                    _td3_ddpg_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        group="E1",
                        label="ca_td3",
                        level=level,
                        algorithm="td3_bc",
                        sequence="priga",
                    )
                )
            jobs.extend(
                [
                    _td3_ddpg_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        group="E1",
                        label="ddpg",
                        level=level,
                        algorithm="ddpg_bc",
                        sequence="priga",
                    ),
                    _ppo_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        level=level,
                    ),
                    _sac_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        level=level,
                    ),
                ]
            )
            if not args.reuse_existing_e1 or level in ("medium", "high"):
                jobs.append(
                    _d3qn_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        group="E1",
                        label="d3qn",
                        level=level,
                        sequence="priga",
                    )
                )
    if "E2" in args.groups:
        for level in args.levels:
            for sequence in ("ga", "pso", "sa_distance", "astar", "eega", "yang"):
                jobs.append(
                    _td3_ddpg_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        group="E2",
                        label=sequence,
                        level=level,
                        algorithm="td3_bc",
                        sequence=sequence,
                    )
                )
    if "E3" in args.groups:
        for level in args.levels:
            if level == "low":
                continue
            jobs.append(
                _td3_ddpg_job(
                    python=args.python,
                    device=args.device,
                    base=base,
                    group="E3",
                    label="distance_tsp_visible",
                    level=level,
                    algorithm="td3_bc",
                    sequence="distance_tsp",
                )
            )
            jobs.append(
                _td3_ddpg_job(
                    python=args.python,
                    device=args.device,
                    base=base,
                    group="E3",
                    label="distance_tsp_blind",
                    level=level,
                    algorithm="td3_bc",
                    sequence="distance_tsp",
                    blind=True,
                )
            )
    if "E4" in args.groups:
        weights = (("w80-20", 0.8, 0.2), ("w50-50", 0.5, 0.5), ("w20-80", 0.2, 0.8))
        for level in args.levels:
            for budget in (3, 4, 5, 6, 8):
                for weight_name, w1, w2 in weights:
                    if weight_name != "w50-50" and budget != 6:
                        continue
                    if budget == 6 and weight_name == "w50-50":
                        continue
                    label = f"qos{budget}/{weight_name}"
                    job = _td3_ddpg_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        group="E4",
                        label=label,
                        level=level,
                        algorithm="td3_bc",
                        sequence="priga",
                    )
                    command = list(job.command)
                    command.extend(
                        [
                            "--objective_w1",
                            str(w1),
                            "--objective_w2",
                            str(w2),
                        ]
                    )
                    # Replace the common 6 s budget only for this scan.
                    budget_index = command.index("--outage_max_s") + 1
                    command[budget_index] = str(budget)
                    jobs.append(Job(job.job_id, "E4", tuple(command), job.output_root))
    if "E5" in args.groups:
        for level in args.levels:
            for sequence, label in (
                ("eega", "eega_td3"),
                ("distance_tsp", "distance_tsp_td3"),
                ("yang", "yang_td3"),
            ):
                jobs.append(
                    _td3_ddpg_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        group="E5",
                        label=label,
                        level=level,
                        algorithm="td3_bc",
                        sequence=sequence,
                        blind=True,
                    )
                )
            for sequence, label in (
                ("sa_service", "drl_sa"),
                ("sa_distance", "sa_d3qn"),
            ):
                jobs.append(
                    _d3qn_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        group="E5",
                        label=label,
                        level=level,
                        sequence=sequence,
                        blind=True,
                        episodes="5000",
                    )
                )
    # E1 CA-TD3 is the shared PRIGA+CA-TD3 run used by E2 and E5.
    if (
        not args.skip_shared_proposed
        and not args.reuse_existing_e1
        and any(group in args.groups for group in ("E2", "E4", "E5"))
    ):
        existing = {job.job_id for job in jobs}
        for level in args.levels:
            job_id = f"E1/ca_td3/{level}"
            if job_id not in existing:
                jobs.append(
                    _td3_ddpg_job(
                        python=args.python,
                        device=args.device,
                        base=base,
                        group="E1",
                        label="ca_td3",
                        level=level,
                        algorithm="td3_bc",
                        sequence="priga",
                    )
                )
    unique = {}
    for job in jobs:
        unique[job.job_id] = job
    return [unique[key] for key in sorted(unique)]


def _execute(job):
    if any(job.output_root.rglob("run_config.json")):
        return job.job_id, "skip-existing", ""
    path_parts = (job.output_root, *job.output_root.parents)
    log_base = next(
        (path for path in path_parts if path.name.startswith("v5-")),
        DEFAULT_OUTPUT,
    )
    log_dir = log_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (job.job_id.replace("/", "__") + ".log")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            job.command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return job.job_id, "ok" if process.returncode == 0 else f"fail:{process.returncode}", str(log_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=("E1", "E2", "E3", "E4", "E5"), default=("E1",))
    parser.add_argument("--levels", nargs="+", choices=("low", "medium", "high"), default=("low",))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--skip-shared-proposed",
        action="store_true",
        help="Do not duplicate the E1 PRIGA+CA-TD3 runs when only E5 baselines are assigned to this machine.",
    )
    parser.add_argument(
        "--reuse-existing-e1",
        action="store_true",
        help="Reuse E1 CA-TD3, D3QN Low, and PPO Low; schedule only the missing V5 E1 runs.",
    )
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    jobs = build_jobs(args)
    if args.stop_after > 0:
        jobs = jobs[: args.stop_after]
    manifest = [
        {
            "id": job.job_id,
            "group": job.group,
            "output_root": str(job.output_root),
            "command": job.command,
        }
        for job in jobs
    ]
    manifest_path = Path(args.output_root) / "plan.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "manifest": str(manifest_path), "execute": args.execute}, indent=2))
    if not args.execute:
        for job in jobs:
            print(" ".join(job.command))
        return
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_execute, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            outcomes.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    failed = [
        job_id
        for job_id, status, _ in outcomes
        if status.startswith("fail:")
    ]
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
