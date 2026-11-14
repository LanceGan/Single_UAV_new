from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT.parent / "P3"
PLOT_SOURCES = (
    ROOT / "plot_v5_e1_all_trajectories.py",
    ROOT / "plot_v5_inspection_order_trajectories.py",
    ROOT / "plot_v5_e3_ablation_trajectories.py",
    P3 / "figs" / "v5" / "plot_e4_payload_generalization.py",
)
TRAJECTORY_STEMS = (
    [f"e1_a2g_trajectories_v5_panel_{panel}" for panel in "abcd"]
    + [f"e1_g2a_trajectories_v5_panel_{panel}" for panel in "abcd"]
    + [f"e2_inspection_order_trajectories_v5_panel_{panel}" for panel in "abcdef"]
    + [f"e3_a2g_trajectories_v5_panel_{panel}" for panel in "abc"]
    + [
        "e4_low_qos_trajectories_v5_panel_a",
        "e4_low_weight_trajectories_v5_panel_b",
    ]
)


def test_final_trajectory_sources_use_internal_legends():
    for source in PLOT_SOURCES:
        text = source.read_text(encoding="utf-8")
        assert "add_internal_legend(" in text
        assert "figure.legend(" not in text


def test_legend_style_matches_the_conference_figure():
    helper = (ROOT / "trajectory_legend.py").read_text(encoding="utf-8")
    assert 'loc="upper left"' in helper
    assert "frameon=True" in helper
    assert "framealpha=0.75" in helper
    assert "Times New Roman" in helper


def test_trajectory_colorbars_are_aligned_and_have_simplified_ticks():
    for source in (
        ROOT / "plot_v5_e1_all_trajectories.py",
        ROOT / "plot_v5_e3_ablation_trajectories.py",
        P3 / "figs" / "v5" / "plot_e4_payload_generalization.py",
    ):
        text = source.read_text(encoding="utf-8")
        assert "align_colorbar_to_axis(" in text
        assert "simplify_colorbar_ticks(" in text


def test_final_trajectory_collision_reports_exist():
    qa_dir = P3 / "figs" / "v5" / "qa"
    for stem in TRAJECTORY_STEMS:
        report = json.loads(
            (qa_dir / f"{stem}.collision-audit.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["auditable"] is True, stem
