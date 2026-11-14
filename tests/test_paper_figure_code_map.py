from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_maps_every_manuscript_figure_to_its_generator():
    for figure_number in range(1, 12):
        assert f"| Fig. {figure_number} |" in README

    expected_sources = (
        "../P3/figs/system_model.tex",
        "plot_algorithm_framework.py",
        "plot_v5_e1_training_validation.py",
        "plot_v5_e1_all_trajectories.py",
        "plot_v5_e3_ablation_trajectories.py",
        "plot_v5_inspection_order_trajectories.py",
        "../P3/figs/v5/plot_e4_payload_generalization.py",
        "../P3/figs/v5/plot_e5_all_level_trajectories.py",
        "plot_v5_e5_time_series.py",
    )
    for source in expected_sources:
        assert source in README


def test_readme_documents_schematic_reproducibility_gaps():
    assert "Microsoft PowerPoint 2021" in README
    assert "没有找到完全一致的已跟踪生成脚本" in README
    assert "Fig. 2 当前论文资产与维护版生成器之间存在版式差异" in README
