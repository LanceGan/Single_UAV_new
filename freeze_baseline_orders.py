"""Freeze deterministic baseline orders and run MapGuide preflight."""

from __future__ import annotations

import json
from pathlib import Path

from mission_preflight import run_preflight
from sequence_optimizers import (
    distance_sa_order,
    distance_tsp_order,
    eega_order,
    service_sa_order,
    yang_order,
)
from sequence_runner import solve_ga_order
from training_config import load_fixed_payloads


ROOT = Path(__file__).resolve().parent
ORDER_ROOT = ROOT / "journal_results" / "v4-sequence-orders"
PREFLIGHT_ROOT = ROOT / "journal_results" / "v4-sequence-preflight"
LEVELS = ("low", "medium", "high")


def main() -> None:
    distance_order, distance_meta = distance_tsp_order()
    sa_distance_order, sa_distance_meta = distance_sa_order(seed=1)
    if list(sa_distance_order) != list(distance_order):
        raise RuntimeError("seed-1 SA-D3QN no longer matches the distance TSP order")

    orders = {
        "distance_tsp": {
            "order": [int(node) for node in distance_order],
            "levels": list(LEVELS),
            "meta": distance_meta,
        },
        "sa_d3qn": {
            "order": [int(node) for node in sa_distance_order],
            "levels": list(LEVELS),
            "meta": sa_distance_meta,
            "alias_of": "distance_tsp",
        },
        "levels": {},
    }
    for level in LEVELS:
        payloads = load_fixed_payloads(level)
        priga = [int(node) for node in solve_ga_order(payloads, iterations=50)]
        eega, eega_meta = eega_order(payloads, seed=1)
        yang, yang_meta = yang_order(payloads)
        drl_sa, drl_sa_meta = service_sa_order(payloads, seed=1)
        orders["levels"][level] = {
            "payloads_mb": payloads.tolist(),
            "priga": {"order": priga, "meta": {"method": "PRIGA seed-1 order"}},
            "distance_tsp": {"order": list(distance_order), "alias": "distance_tsp"},
            "eega": {"order": eega, "meta": eega_meta},
            "yang": {"order": yang, "meta": yang_meta},
            "drl_sa": {"order": drl_sa, "meta": drl_sa_meta},
            "sa_d3qn": {"order": list(distance_order), "alias_of": "distance_tsp"},
        }

    ORDER_ROOT.mkdir(parents=True, exist_ok=True)
    (ORDER_ROOT / "orders.json").write_text(
        json.dumps(orders, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {}
    for level in LEVELS:
        level_orders = orders["levels"][level]
        seen = {}
        summary[level] = {}
        for method in ("priga", "distance_tsp", "eega", "yang", "drl_sa", "sa_d3qn"):
            order = tuple(level_orders[method]["order"])
            if order in seen:
                summary[level][method] = {
                    **seen[order],
                    "reused_from": summary[level][seen[order]["method"]]["method"],
                }
                continue
            result = run_preflight(
                level=level,
                output_dir=PREFLIGHT_ROOT / level / method,
                order=order,
                outage_max_s=6,
            )
            record = {
                "method": method,
                "order": list(order),
                "success": bool(result["success"]),
                "time_s": float(result["time_s"]),
                "outage_s": float(result["outage_s"]),
                "visited_points": int(result["visited_points"]),
                "uploaded_mb": float(result["uploaded_mb"]),
            }
            seen[order] = record
            summary[level][method] = record

    (PREFLIGHT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
