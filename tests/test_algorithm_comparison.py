import json

import numpy as np
import pytest

from plot_algorithm_comparison import (
    load_actor_rewards,
    load_series,
    plot_objective_comparison,
    plot_reward_comparison,
    plot_success_comparison,
    summarize,
)
from plot_formal_results import _generate_d3qn_figures, _style, actor_reward_series


def test_method_display_name_hides_bc_suffix_in_figures():
    from evaluate_policy import method_display_name

    assert method_display_name("td3_bc") == "TD3"
    assert method_display_name("ddpg_bc") == "DDPG"
    assert method_display_name("td3") == "TD3"
    assert method_display_name("d3qn") == "D3QN"


def test_method_title_qualifies_sequence_baselines():
    from plot_formal_results import method_title

    assert method_title("td3_bc", {"sequence_method": "GA with exact verification/fallback"}) == "TD3"
    assert method_title("td3_bc", {"sequence_method": "exact distance TSP over all permutations"}) == "Distance-TSP+TD3"
    assert method_title("td3_bc", {"sequence_method": "simulated annealing on G2A flight+outage seconds"}) == "DRL-SA (TD3 trajectory)"
    assert method_title("td3_bc", {}) == "TD3"


def _history_rows(successes):
    rows = []
    for index, ok in enumerate(successes, start=1):
        rows.append({
            "episode": 25 * index,
            "task_time_s": 200.0 + index,
            "transmission_delay_s": 80.0,
            "total_return": 1.0 if ok else -5.0,
            "visited_inspection_points": 6 if ok else 1,
            "success": bool(ok),
        })
    return rows


def _write_history(tmp_path, name, successes, best_key="eval_best"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    rows = _history_rows(successes)
    feasible = [row for row in rows if row["success"]]
    best = (min(feasible, key=lambda row: row["task_time_s"] + row["transmission_delay_s"])
            if feasible else None)
    (run_dir / "eval_history.json").write_text(json.dumps({best_key: best, "history": rows}))
    return run_dir


def test_summarize_flags_methods_without_feasible_evaluations(tmp_path):
    ok = summarize("TD3", load_series(_write_history(tmp_path, "ok", [False, True, True])))
    bad = summarize("DDPG", load_series(_write_history(tmp_path, "bad", [False] * 3)))

    assert ok["successes"] == 2
    assert ok["best_feasible"]["objective_s"] == pytest.approx(282.0)
    assert ok["best_feasible"]["episode"] == 50
    assert bad["successes"] == 0 and bad["best_feasible"] is None


def test_comparison_plots_render_mixed_and_all_failed_runs(tmp_path):
    _style()
    entries = [
        ("TD3", load_series(_write_history(tmp_path, "ok", [False, True]))),
        ("DDPG", load_series(_write_history(tmp_path, "bad", [False, False]))),
    ]

    plot_objective_comparison(entries, [("A*", 214.0), ("MapGuide", 273.0)],
                              tmp_path / "objective", "Low load")
    plot_success_comparison(entries, tmp_path / "success", "Low load")

    for stem in ("objective", "success"):
        assert (tmp_path / f"{stem}.png").is_file()
        assert (tmp_path / f"{stem}.pdf").is_file()


def test_actor_reward_series_drops_guided_episodes_and_clips_to_display_window():
    rewards = np.arange(1.0, 7.0)  # episodes 1..6
    guided = np.array([False, True, False, False, True, False])

    episodes, actor = actor_reward_series(rewards, guided, limit=4)

    np.testing.assert_array_equal(episodes, [1, 3, 4])
    np.testing.assert_allclose(actor, [1.0, 3.0, 4.0])


def test_load_actor_rewards_reads_both_archive_layouts(tmp_path):
    continuous = tmp_path / "continuous"
    continuous.mkdir()
    np.savez(continuous / "UAV_TD3_improved.npz",
             ep_rewards=np.array([1.0, 2.0, 3.0]),
             ep_guided=np.array([False, True, False]))
    discrete = tmp_path / "discrete"
    discrete.mkdir()
    np.savez(discrete / "training_history.npz",
             episode_rewards=np.array([5.0, 6.0]),
             episode_guided=np.array([True, False]))

    episodes_c, rewards_c = load_actor_rewards(continuous)
    episodes_d, rewards_d = load_actor_rewards(discrete)

    np.testing.assert_array_equal(episodes_c, [1, 3])
    np.testing.assert_allclose(rewards_c, [1.0, 3.0])
    np.testing.assert_array_equal(episodes_d, [2])
    np.testing.assert_allclose(rewards_d, [6.0])


def test_reward_comparison_renders_with_short_series(tmp_path):
    _style()
    reward_entries = [
        ("TD3", np.arange(1, 251), np.random.default_rng(0).normal(size=250)),
        ("DDPG", np.arange(1, 51), np.random.default_rng(1).normal(size=50)),
    ]

    plot_reward_comparison(reward_entries, tmp_path / "rewards", "Low load")

    assert (tmp_path / "rewards.png").is_file()
    assert (tmp_path / "rewards.pdf").is_file()


def test_d3qn_run_figures_render_without_checkpoints(tmp_path):
    run_dir = tmp_path / "d3qn"
    (run_dir / "models").mkdir(parents=True)
    rows = _history_rows([False, False])
    (run_dir / "eval_history.json").write_text(
        json.dumps({"selection": "s", "best": rows[0], "history": rows})
    )
    np.savez(run_dir / "training_history.npz", episode_rewards=np.array([-1.0, -2.0]))
    config = {"algorithm": "d3qn", "data_level": "low", "variant": "A"}
    (run_dir / "run_config.json").write_text(json.dumps(config))
    _style()

    summary = _generate_d3qn_figures(run_dir, config)

    assert summary["successful_evaluations"] == 0
    assert summary["eval_best_episode"] == 25
    assert summary["eval_best"] is None  # no checkpoint on disk -> trajectory skipped
    figures = run_dir / "paper_figures"
    assert (figures / "objective_vs_episode.png").is_file()
    assert (figures / "reward_vs_episode.png").is_file()
