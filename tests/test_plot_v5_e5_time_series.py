from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "P3" / "figs" / "v5" / "source_data"


def test_e5_time_series_cover_seven_pipelines_and_three_levels():
    payload = np.load(SOURCE / "e5_time_series_v5.npz")
    names = {
        key.rsplit("_time_s", 1)[0]
        for key in payload.files
        if key.endswith("_time_s")
    }

    assert len(names) == 21
    for name in names:
        assert payload[f"{name}_time_s"].shape == payload[
            f"{name}_current_contiguous_outage_s"
        ].shape
        assert payload[f"{name}_objective_s"].shape[0] == payload[
            f"{name}_time_s"
        ].shape[0] - 1


def test_current_outage_resets_and_is_not_a_cumulative_series():
    payload = json.loads(
        (SOURCE / "e5_time_series_v5.json").read_text(encoding="utf-8")
    )
    assert "resets to zero" in payload["protocol"]["contiguous_outage"]
    assert "cumulative transmission delay" in payload["protocol"]["objective"]

    arrays = np.load(SOURCE / "e5_time_series_v5.npz")
    reset_found = False
    for key in arrays.files:
        if not key.endswith("_current_contiguous_outage_s"):
            continue
        values = arrays[key]
        if len(values) > 1 and np.any(values[1:] < values[:-1]):
            reset_found = True
            break
    assert reset_found


def test_e5_time_series_has_independent_panels_and_legend():
    figure_dir = SOURCE.parent
    for panel in "abcdef":
        assert (
            figure_dir / f"e5_time_series_v5_panel_{panel}.pdf"
        ).is_file()
    assert (figure_dir / "e5_time_series_v5_legend.pdf").is_file()


def test_e5_csv_keeps_the_final_current_outage_sample_and_table_maximum():
    source = SOURCE / "e5_time_series_v5.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (SOURCE / "e5_method_payload_matrix_v5.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        expected = {
            (row["method"], row["level"]): float(row["contiguous_outage_s"])
            for row in csv.DictReader(stream)
        }

    maxima = {}
    for row in rows:
        key = (row["method"], row["level"])
        maxima[key] = max(
            maxima.get(key, 0.0),
            float(row["current_contiguous_outage_s"]),
        )

    assert maxima.keys() == expected.keys()
    assert all(
        abs(maxima[key] - value) < 1e-12
        for key, value in expected.items()
    )
