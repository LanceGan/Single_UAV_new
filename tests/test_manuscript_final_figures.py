from __future__ import annotations

import re
from pathlib import Path


MANUSCRIPT = (
    Path(__file__).resolve().parents[1] / ".." / "P3" / "journal.tex"
).resolve()


def _manuscript() -> str:
    return MANUSCRIPT.read_text(encoding="utf-8")


def test_e5_complete_table_contains_all_21_method_tier_rows():
    text = _manuscript()
    start = text.index(r"\label{tab:e5-complete}")
    end = text.index(r"\end{tabular}", start)
    rows = [
        line
        for line in text[start:end].splitlines()
        if line.rstrip().endswith(r"\\") and "&" in line
    ]

    assert len(rows) == 22  # Header plus 21 method-tier outcomes.
    assert "DRM-HO (PRIGA+CA-TD3) & Low" in text[start:end]
    assert "DRL-SA \\cite{ref1} & High" in text[start:end]
    assert "SA-D3QN \\cite{ref14} & High" in text[start:end]


def test_final_manuscript_retains_six_inspection_order_methods():
    text = _manuscript()
    for label in (
        "Proposed PRIGA",
        "Yang-style",
        "EEGA-style",
        "DRL-SA",
        "SA-D3QN",
        "EEAP-PSO",
    ):
        assert label in text

    for panel in "abcdef":
        assert (
            f"e2_inspection_order_trajectories_v5_panel_{panel}.pdf"
            in text
        )


def test_removed_final_figure_families_are_not_active_in_manuscript():
    text = _manuscript()
    for stale in (
        "e3_a2g_ablation_v5_panel_",
        "e4_weight_trajectories_v5",
        "e4_weight_realtime_comm_v5",
        "e5_time_series_v5",
        "e5_end_to_end_convergence_v5",
        "e5_method_payload_matrix_v5",
    ):
        assert stale not in text


def test_e5_current_outage_is_described_as_resetting():
    text = _manuscript()
    caption_start = text.index(r"\label{fig:e5-time-series}")
    caption = text[max(0, caption_start - 2000) : caption_start]

    assert "current contiguous outage" in caption
    assert "resets to zero" in caption
    assert not re.search(r"cumulative outage series", caption)


def test_e1_table_uses_dual_map_astar_instead_of_sac():
    text = _manuscript()
    assert (
        "Dual-map A* & Medium & 189.5 & 127.0 & 316.5 & 4.0 & 6/6 & \\checkmark"
        in text
    )
    assert "SAC" not in text
