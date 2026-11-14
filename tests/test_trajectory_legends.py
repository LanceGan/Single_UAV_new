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


def test_final_trajectory_assets_have_no_collision_failures():
    qa_dir = P3 / "figs" / "v5" / "qa"
    for stem in TRAJECTORY_STEMS:
        report = json.loads(
            (qa_dir / f"{stem}.collision-audit.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["summary"]["fail"] == 0, stem
