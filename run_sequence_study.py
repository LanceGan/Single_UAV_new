"""Deterministic sequence-only study with radio-aware proxy metrics.

This script does not train a trajectory policy.  It compares how inspection
orders are selected and evaluates each resulting straight-line route on the
fixed A2G/G2A maps.  End-to-end metrics must still be obtained by replaying the
selected order through the common trajectory learner.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
from pathlib import Path

import numpy as np

from fixed_radio_maps import FixedRadioMaps
from sequence_optimizers import (
    distance_astar_order,
    distance_pso_order,
    distance_sa_order,
    distance_tsp_order,
    eega_order,
    ga_proxy_order,
    priga_order,
    yang_order,
)
from sequence_runner import load_planning_nodes
from training_config import load_fixed_payloads


ROOT = Path(__file__).resolve().parent
MAPS = FixedRadioMaps(ROOT / "results" / "datas")
SPEED_MPS = 50.0
SAMPLE_SPACING_M = 25.0
OUTAGE_THRESHOLD = 0.5
DEFAULT_METHODS = ("priga", "ga", "pso", "sa", "astar", "eega", "yang")
ALL_METHODS = (*DEFAULT_METHODS, "distance_tsp")


def _a2g_rate(rate_map, point):
    snr_db = rate_map.a2g_snr_db(point)
    return float(math.log2(1.0 + 10.0 ** (snr_db / 10.0)))


def _path_order(method, payloads, seed, ga_iterations):
    nodes = load_planning_nodes()
    if method == "priga":
        return priga_order(
            payloads,
            nodes_km=nodes,
            seed=seed,
            generations=ga_iterations,
            outage_probability_threshold=OUTAGE_THRESHOLD,
        )
    if method == "ga":
        return ga_proxy_order(nodes, seed=seed)
    if method == "pso":
        return distance_pso_order(nodes, seed=seed)
    if method == "sa":
        return distance_sa_order(nodes, seed=seed)
    if method == "astar":
        return distance_astar_order(nodes)
    if method == "eega":
        return eega_order(payloads, nodes_km=nodes, seed=seed)
    if method == "yang":
        return yang_order(payloads, nodes_km=nodes)
    if method == "distance_tsp":
        return distance_tsp_order(nodes)
    raise ValueError(f"unsupported sequence method: {method}")


def route_proxy(order, payloads_mb, maps=None):
    """Evaluate a fixed inspection order on straight-line canonical legs."""
    maps = maps if maps is not None else MAPS
    nodes = load_planning_nodes()
    route = (0, *[int(node) for node in order], len(nodes) - 1)
    all_rates, weighted_rates = [], []
    leg_rows = []
    total_distance_m = 0.0
    estimated_tau_s = 0.0

    for source, destination in zip(route, route[1:]):
        start = nodes[source]
        end = nodes[destination]
        distance_m = float(np.linalg.norm(end - start)) * 1000.0
        samples = max(2, int(math.ceil(distance_m / SAMPLE_SPACING_M)) + 1)
        points = np.linspace(start, end, samples)
        rates = np.asarray([_a2g_rate(maps, point) for point in points], dtype=float)
        outages = np.asarray(
            [
                maps.g2a_outage_probability(point) >= OUTAGE_THRESHOLD
                for point in points
            ],
            dtype=bool,
        )
        slot_s = distance_m / SPEED_MPS / max(samples - 1, 1)
        current_run = 0
        max_run = 0
        for outage in outages:
            current_run = current_run + 1 if outage else 0
            max_run = max(max_run, current_run)
        payload_mb = 0.0 if source == 0 else float(payloads_mb[source - 1])
        mean_rate = float(np.mean(rates))
        upload_time_s = payload_mb / max(mean_rate, 1e-9)
        estimated_tau_s += upload_time_s
        total_distance_m += distance_m
        all_rates.extend(rates.tolist())
        weighted_rates.append((payload_mb, mean_rate))
        leg_rows.append(
            {
                "source": int(source),
                "destination": int(destination),
                "distance_m": distance_m,
                "flight_time_s": distance_m / SPEED_MPS,
                "mean_a2g_rate": mean_rate,
                "payload_mb": payload_mb,
                "estimated_upload_time_s": upload_time_s,
                "max_contiguous_g2a_outage_s": float(max_run * slot_s),
                "total_g2a_outage_s": float(np.sum(outages) * slot_s),
            }
        )

    payload_total = float(np.sum(weighted_rates, axis=0)[0])
    weighted_rate = (
        float(sum(payload * rate for payload, rate in weighted_rates)) / payload_total
        if payload_total > 0.0
        else float(np.mean(all_rates))
    )
    estimated_t = total_distance_m / SPEED_MPS
    return {
        "inspection_order": [int(node) for node in order],
        "canonical_route": [int(node) for node in route],
        "flight_distance_m": total_distance_m,
        "estimated_t_s": estimated_t,
        "estimated_tau_s": estimated_tau_s,
        "estimated_t_plus_tau_s": estimated_t + estimated_tau_s,
        "mean_a2g_rate_length_weighted": float(np.mean(all_rates)),
        "mean_a2g_rate_payload_weighted": weighted_rate,
        "mean_leg_max_contiguous_g2a_outage_s": float(
            np.mean([leg["max_contiguous_g2a_outage_s"] for leg in leg_rows])
        ),
        "max_contiguous_g2a_outage_s": float(
            np.max([leg["max_contiguous_g2a_outage_s"] for leg in leg_rows])
        ),
        "total_g2a_outage_s": float(
            np.sum([leg["total_g2a_outage_s"] for leg in leg_rows])
        ),
        "legs": leg_rows,
    }


def _run_task(task):
    method, level, seed, ga_iterations = task
    payloads = load_fixed_payloads(level)
    order, order_meta = _path_order(method, payloads, seed, ga_iterations)
    proxy = route_proxy(order, payloads)
    return {
        "method": method,
        "level": level,
        "seed": int(seed),
        "payloads_mb": payloads.tolist(),
        "payload_total_mb": float(payloads.sum()),
        **order_meta,
        **proxy,
        "metric_scope": "predicted straight-line sequence proxy; not World replay",
    }


def _write_outputs(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda item: (item["level"], item["method"]))
    payload = {
        "protocol": {
            "method": "sequence-only proxy study",
            "speed_mps": SPEED_MPS,
            "sample_spacing_m": SAMPLE_SPACING_M,
            "outage_probability_threshold": OUTAGE_THRESHOLD,
            "note": "Tau is a straight-line upload estimate; end-to-end values require CA-TD3 replay.",
        },
        "records": ordered,
    }
    (output_dir / "sequence-study.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    columns = [
        "level",
        "method",
        "inspection_order",
        "estimated_t_s",
        "estimated_tau_s",
        "estimated_t_plus_tau_s",
        "mean_a2g_rate_length_weighted",
        "mean_a2g_rate_payload_weighted",
        "mean_leg_max_contiguous_g2a_outage_s",
        "max_contiguous_g2a_outage_s",
        "flight_distance_m",
    ]
    with (output_dir / "sequence-study.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for item in ordered:
            row = {key: item[key] for key in columns}
            row["inspection_order"] = "-".join(str(node) for node in row["inspection_order"])
            writer.writerow(row)
    print(
        json.dumps(
            {
                "records": len(ordered),
                "levels": sorted({item["level"] for item in ordered}),
                "methods": sorted({item["method"] for item in ordered}),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=DEFAULT_METHODS,
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=("low", "medium", "high"),
        default=("low", "medium", "high"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ga-iterations", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "journal_results" / "v5-sequence",
    )
    args = parser.parse_args()
    tasks = [
        (method, level, args.seed, args.ga_iterations)
        for level in args.levels
        for method in args.methods
    ]
    jobs = int(args.jobs) if int(args.jobs) > 0 else min(len(tasks), 7)
    if jobs == 1:
        results = [_run_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(_run_task, tasks))
    _write_outputs(results, args.output.resolve())


if __name__ == "__main__":
    main()
