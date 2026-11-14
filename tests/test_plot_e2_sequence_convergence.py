from __future__ import annotations

import csv
import json
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / ".."
    / "P3"
    / "figs"
    / "v5"
    / "source_data"
)
FIGURE_ROOT = SOURCE.parent


def _load_json(name):
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def test_e2_convergence_covers_seven_methods_and_three_levels():
    payload = _load_json("e2_sequence_convergence_v5.json")
    selected = payload["selected"]

    assert len(selected) == 21
    assert {row["level"] for row in selected} == {"Low", "Medium", "High"}
    assert len({row["method"] for row in selected}) == 7
    assert all(row["iteration"] == row["budget"] for row in selected)
    assert all(
        sorted(int(node) for node in row["incumbent_order"].split("-"))
        == [1, 2, 3, 4, 5, 6]
        for row in selected
    )


def test_e2_proxy_is_recomputed_per_payload_level():
    payload = _load_json("e2_sequence_convergence_v5.json")
    selected = {
        (row["level"], row["method"]): row
        for row in payload["selected"]
    }
    low = selected[("Low", "GA (additional)")]
    high = selected[("High", "GA (additional)")]

    assert low["incumbent_order"] == high["incumbent_order"]
    assert low["proxy_t_plus_tau_s"] != high["proxy_t_plus_tau_s"]


def test_e2_csv_records_keep_original_iteration_budgets():
    path = SOURCE / "e2_sequence_convergence_v5.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    observed_max = {}
    for row in rows:
        key = (row["level"], row["method"])
        observed_max[key] = max(
            observed_max.get(key, 0), int(row["iteration"])
        )
    assert len(rows) > 20_000
    assert all(value == int(METHOD_BUDGETS[key[1]]) for key, value in observed_max.items())


def test_eeap_pso_orders_are_ready_for_ca_td3_replay():
    payload = _load_json("e2_eeap_pso_orders_v5.json")
    levels = payload["levels"]

    assert set(levels) == {"Low", "Medium", "High"}
    for record in levels.values():
        assert sorted(record["inspection_order"]) == [1, 2, 3, 4, 5, 6]
        assert record["generations"] == 200
        assert record["two_opt_interval"] == 20
        assert "payload-capacity limit" in record["removed_constraints"]
        assert "onboard-energy limit" in record["removed_constraints"]
        assert "proxy" in record["metric_scope"]


def test_retired_e2_convergence_panels_are_not_reintroduced():
    for suffix in (
        "a",
        "b",
        "c",
    ):
        stem = FIGURE_ROOT / f"e2_sequence_convergence_v5_panel_{suffix}"
        assert not stem.with_suffix(".pdf").exists()
        assert not stem.with_suffix(".svg").exists()
        assert not stem.with_suffix(".png").exists()
    legend = FIGURE_ROOT / "e2_sequence_convergence_v5_legend"
    assert not legend.with_suffix(".pdf").exists()


METHOD_BUDGETS = {
    "PRIGA (proposed)": 500,
    "EEGA-style GA (Algorithm in EEGA study)": 200,
    "DRL-SA sequence (Algorithm in DRL-SA study)": 4000,
    "SA-D3QN sequence (Algorithm in SA-D3QN study)": 4000,
    "GA (additional)": 200,
    "PSO (additional)": 250,
    "EEAP-PSO (Algorithm in MCO-A* study; constraints removed)": 200,
}
