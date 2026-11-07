#!/usr/bin/env python3
"""Summarize the formal radio-map-free A* and PSO baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PURE_ROOT = ROOT / "journal_results" / "v4-deterministic" / "pure-heuristics-6"
OUTPUT = ROOT / "journal_results" / "v4-summary"
LEVELS = ("low", "medium", "high")
METHODS = (("A*", "astar"), ("PSO", "pso"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(method: str, slug: str, level: str) -> dict:
    source = PURE_ROOT / slug / level / "metrics.json"
    data = _read_json(source)
    valid = bool(data.get("success"))
    planner = data.get("planner", {})
    return {
        "method": method,
        "level": level,
        "valid": valid,
        "status": "success" if valid else "invalid",
        "termination_reason": data.get("termination_reason"),
        "inspection_order": data.get("inspection_order", []),
        "task_time_s": data.get("task_time_s") if valid else None,
        "transmission_delay_s": data.get("transmission_delay_s") if valid else None,
        "objective_s": data.get("objective_s") if valid else None,
        "outage_s": data.get("outage_time_s"),
        "visited_inspection_points": data.get("visited_inspection_points"),
        "uploaded_mb": data.get("uploaded_mb"),
        "expected_upload_mb": data.get("expected_upload_mb"),
        "path_distance_world_units": planner.get("path_distance_world_units"),
        "planner_runtime_s": planner.get("runtime_s"),
        "expansions": planner.get("expansions"),
        "generated_states": planner.get("generated_states"),
        "radio_maps_used_for_planning": False,
        "source": str(source.relative_to(ROOT)),
    }


def build_summary() -> dict:
    records = [
        _record(method, slug, level)
        for level in LEVELS
        for method, slug in METHODS
    ]
    return {
        "qos_constraint": "max_contiguous_outage",
        "outage_budget_s": 6.0,
        "definitions": {
            "astar": "distance-only end-to-end A* over UAV position and visited inspection set",
            "pso": "distance-only end-to-end PSO whose decoded permutation defines a piecewise-linear path",
            "radio_maps_used_for_planning": False,
            "upload_protocol": "hover at each inspection point until upload completes",
            "valid": "complete six-point mission, full upload, World replay success, and T_out^cont <= 6 s",
        },
        "records": records,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (OUTPUT / "global-baselines-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (OUTPUT / "global-baselines-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary["records"][0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary["records"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
