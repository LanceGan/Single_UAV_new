"""Summarize validated V5 Mac training runs without modifying experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "journal_results" / "v5-mac"
STALL_MINUTES = 15.0


def _finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _complete(item, expected_points, expected_upload):
    points = int(item.get("visited_inspection_points", 0) or 0)
    uploaded = _finite_number(item.get("uploaded_mb")) or 0.0
    return points >= expected_points and uploaded >= expected_upload - 1e-7


def _run_label(run_dir, output_root):
    return run_dir.relative_to(output_root).as_posix()


def summarize_run(run_dir, output_root, now=None):
    history_path = run_dir / "eval_history.json"
    config_path = run_dir / "run_config.json"
    if not history_path.exists() or not config_path.exists():
        return None

    now = time.time() if now is None else now
    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    history = history_payload.get("history", [])
    if not history:
        return None

    configured_episodes = int(config.get("episodes", 0) or 0)
    last_episode = int(history[-1].get("episode", 0) or 0)
    complete = configured_episodes > 0 and last_episode >= configured_episodes
    age_minutes = max(0.0, (now - history_path.stat().st_mtime) / 60.0)
    if complete:
        status = "completed"
    elif age_minutes > STALL_MINUTES:
        status = "stalled"
    else:
        status = "running"

    payloads = config.get("payloads_mb") or []
    expected_upload = sum(float(value) for value in payloads)
    expected_points = 6
    completed_evaluations = [
        item
        for item in history
        if _complete(item, expected_points, expected_upload)
    ]
    feasible_evaluations = [
        item
        for item in completed_evaluations
        if bool(item.get("success", False))
    ]
    best = history_payload.get("eval_best") or history_payload.get("best") or {}
    best_complete = bool(best) and _complete(best, expected_points, expected_upload)
    task_time = _finite_number(best.get("task_time_s"))
    transmission_delay = _finite_number(best.get("transmission_delay_s"))
    outage = _finite_number(best.get("outage_time_s"))
    outage_limit = _finite_number(best.get("outage_limit_s"))
    if outage_limit is None:
        outage_limit = _finite_number(config.get("outage_max_s"))
    qos_feasible = (
        best_complete
        and bool(best.get("success", False))
        and outage is not None
        and outage_limit is not None
        and outage <= outage_limit + 1e-9
    )

    return {
        "run": _run_label(run_dir, output_root),
        "status": status,
        "algorithm": config.get("algorithm", ""),
        "level": config.get("data_level", ""),
        "sequence_method": config.get("sequence_method", ""),
        "a2g_blind": bool(config.get("a2g_blind", False)),
        "objective_weights": [
            _finite_number(config.get("objective_w1")),
            _finite_number(config.get("objective_w2")),
        ],
        "qos_constraint": config.get("qos_constraint", ""),
        "qos_enforcement": config.get("qos_enforcement", ""),
        "outage_limit_s": outage_limit,
        "configured_episodes": configured_episodes,
        "last_episode": last_episode,
        "evaluations": len(history),
        "completed_evaluations": len(completed_evaluations),
        "feasible_evaluations": len(feasible_evaluations),
        "update_age_minutes": round(age_minutes, 2),
        "eval_best_episode": int(best.get("episode", -1) or -1),
        "eval_best_complete": best_complete,
        "eval_best_qos_feasible": qos_feasible,
        "task_time_s": task_time,
        "transmission_delay_s": transmission_delay,
        "objective_s": (
            task_time + transmission_delay
            if task_time is not None and transmission_delay is not None
            else None
        ),
        "contiguous_outage_s": outage,
        "visited_inspection_points": best.get("visited_inspection_points"),
        "uploaded_mb": _finite_number(best.get("uploaded_mb")),
        "expected_upload_mb": expected_upload,
        "inspection_order": best.get("inspection_order") or config.get("order"),
        "source_path": str(run_dir.resolve()),
    }


def collect(output_root=DEFAULT_OUTPUT, now=None):
    output_root = Path(output_root).resolve()
    records = []
    for history_path in sorted((output_root / "E3").rglob("eval_history.json")):
        record = summarize_run(history_path.parent, output_root, now=now)
        if record is not None and record["level"] in {"medium", "high"}:
            records.append(record)
    for history_path in sorted((output_root / "E4").rglob("eval_history.json")):
        record = summarize_run(history_path.parent, output_root, now=now)
        if record is not None and record["level"] in {"medium", "high"}:
            records.append(record)
    for history_path in sorted((output_root / "E5").rglob("eval_history.json")):
        record = summarize_run(history_path.parent, output_root, now=now)
        if record is not None and record["level"] in {"medium", "high"}:
            records.append(record)
    return records


def write_summary(records, output_root=DEFAULT_OUTPUT):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "status_summary.json"
    csv_path = output_root / "status_summary.csv"
    columns = list(records[0]) if records else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(
        json.dumps(
            {
                "generated_at_unix": int(time.time()),
                "stall_threshold_minutes": STALL_MINUTES,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = collect(args.output_root)
    json_path, csv_path = write_summary(records, args.output_root)
    print(f"{len(records)} runs -> {json_path} {csv_path}")


if __name__ == "__main__":
    main()
