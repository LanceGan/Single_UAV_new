#!/usr/bin/env python3
"""Summarize the formal joint A* and PSO runs without scoring failed missions."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASTAR_ROOT = ROOT / "journal_results" / "v4-deterministic" / "astar-global-6"
PSO_ROOT = ROOT / "journal_results" / "v4-deterministic" / "pso-joint"
OUTPUT = ROOT / "journal_results" / "v4-summary"
LEVELS = ("low", "medium", "high")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _astar_record(level: str) -> dict:
    data = _read_json(ASTAR_ROOT / level / "metrics.json")
    planner = data["planner"]
    replay = data["world_replay"]
    valid = bool(planner.get("success") and planner.get("optimality_proven") and replay.get("success"))
    return {
        "method": "A*",
        "level": level,
        "valid": valid,
        "status": planner["status"],
        "termination_reason": replay.get("termination_reason"),
        "inspection_order": data.get("inspection_order", []),
        "task_time_s": replay.get("task_time_s") if valid else None,
        "transmission_delay_s": replay.get("transmission_delay_s") if valid else None,
        "objective_s": replay.get("objective_s") if valid else None,
        "outage_s": replay.get("outage_time_s"),
        "visited_inspection_points": replay.get("visited_inspection_points"),
        "uploaded_mb": replay.get("uploaded_mb"),
        "expected_upload_mb": replay.get("expected_upload_mb"),
        "planner_runtime_s": planner.get("runtime_s"),
        "expansions": planner.get("expansions"),
        "generated_labels": planner.get("generated_labels"),
        "optimality_proven": planner.get("optimality_proven"),
        "source": str((ASTAR_ROOT / level / "metrics.json").relative_to(ROOT)),
    }


def _pso_record(level: str) -> dict:
    data = _read_json(PSO_ROOT / level / "metrics.json")
    report = data["report"]
    valid = bool(
        report.get("success")
        and report.get("visited_inspection_points") == report.get("expected_inspection_points") == 6
        and float(report.get("uploaded_mb", 0.0)) >= float(report.get("expected_upload_mb", 1.0)) - 1e-7
        and float(report.get("outage_time_s", float("inf"))) <= float(report.get("outage_limit_s", 0.0))
    )
    return {
        "method": "PSO",
        "level": level,
        "valid": valid,
        "status": "success" if valid else "invalid",
        "termination_reason": report.get("termination_reason"),
        "inspection_order": data.get("order", []),
        "task_time_s": report.get("task_time_s") if valid else None,
        "transmission_delay_s": report.get("transmission_delay_s") if valid else None,
        "objective_s": (
            float(report["task_time_s"]) + float(report["transmission_delay_s"])
            if valid else None
        ),
        "outage_s": report.get("outage_time_s"),
        "visited_inspection_points": report.get("visited_inspection_points"),
        "uploaded_mb": report.get("uploaded_mb"),
        "expected_upload_mb": report.get("expected_upload_mb"),
        "planner_runtime_s": None,
        "expansions": None,
        "generated_labels": None,
        "optimality_proven": False,
        "source": str((PSO_ROOT / level / "metrics.json").relative_to(ROOT)),
    }


def build_summary() -> dict:
    records = [_astar_record(level) for level in LEVELS]
    records.extend(_pso_record(level) for level in LEVELS)
    return {
        "qos_constraint": "max_contiguous_outage",
        "outage_budget_s": 6.0,
        "definitions": {
            "proposed": "IGA sequence + standard TD3; TD3 unchanged",
            "astar": "single joint A* over visit order and trajectory on the 25 m lattice",
            "pso": "joint particle encoding of visit-order priorities and per-leg trajectory controls",
            "valid": "complete six-point mission, full upload, World replay success, and T_out^cont <= 6 s",
        },
        "records": records,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (OUTPUT / "global-baselines-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUTPUT / "global-baselines-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["records"][0]))
        writer.writeheader()
        writer.writerows(summary["records"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
