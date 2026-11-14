import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "P3" / "figs" / "v5" / "source_data"


def test_e5_low_time_series_cover_seven_pipelines():
    with np.load(SOURCE / "e5_low_realtime_v5.npz") as payload:
        names = {
            key.rsplit("_time_s", 1)[0]
            for key in payload.files
            if key.endswith("_time_s")
        }

        assert len(names) == 7
        for name in names:
            time_shape = payload[f"{name}_time_s"].shape
            assert time_shape == payload[
                f"{name}_g2a_outage_probability"
            ].shape
            assert time_shape == payload[
                f"{name}_current_contiguous_outage_s"
            ].shape
            assert payload[f"{name}_positions_km"].shape == (time_shape[0], 2)


def test_outage_probability_uses_the_g2a_map_without_complementing():
    payload = json.loads(
        (SOURCE / "e5_low_realtime_v5.json").read_text(encoding="utf-8")
    )
    assert "arr_0" in payload["protocol"]["g2a_outage_probability"]
    assert "not complemented" in payload["protocol"]["g2a_outage_probability"]
    assert "resets to zero" in payload["protocol"]["contiguous_outage"]
    assert payload["protocol"]["scope"].startswith("Low payload tier only")

    with np.load(SOURCE / "e5_low_realtime_v5.npz") as arrays:
        for key in arrays.files:
            if key.endswith("_g2a_outage_probability"):
                probability = arrays[key]
                assert np.all(probability >= 0.0)
                assert np.all(probability <= 1.0)
                assert np.ptp(probability) > 0.0


def test_current_outage_resets_and_is_not_a_cumulative_series():
    with np.load(SOURCE / "e5_low_realtime_v5.npz") as arrays:
        reset_found = False
        for key in arrays.files:
            if not key.endswith("_current_contiguous_outage_s"):
                continue
            values = arrays[key]
            if len(values) > 1 and np.any(values[1:] < values[:-1]):
                reset_found = True
                break
    assert reset_found


def test_e5_low_has_two_independent_panels_and_shared_legend():
    figure_dir = SOURCE.parent
    for panel in "ab":
        assert (
            figure_dir / f"e5_low_realtime_v5_panel_{panel}.pdf"
        ).is_file()
    assert (figure_dir / "e5_low_realtime_v5_legend.pdf").is_file()


def test_e5_low_csv_peaks_match_the_retained_complete_table():
    source = SOURCE / "e5_low_realtime_v5.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (SOURCE / "e5_method_payload_matrix_v5.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        table_rows = list(csv.DictReader(stream))
        expected = {
            (row["method"], row["level"]): float(row["contiguous_outage_s"])
            for row in table_rows
        }

    assert len(table_rows) == 21
    maxima = {}
    for row in rows:
        key = (row["method"], row["level"])
        maxima[key] = max(
            maxima.get(key, 0.0),
            float(row["current_contiguous_outage_s"]),
        )

    low_expected = {
        key: value for key, value in expected.items() if key[1] == "Low"
    }
    assert maxima.keys() == low_expected.keys()
    assert all(
        abs(maxima[key] - value) < 1e-12
        for key, value in low_expected.items()
    )
