import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_load_run_context_accepts_ddpg_configuration(tmp_path):
    from evaluate_policy import STATE_SCHEMA, load_run_context

    data_dir = Path(__file__).resolve().parents[1] / "results" / "datas"
    map_hashes = {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("Radio_datas.npz", "Radio_datas_A2G.npz")
    }
    config = {
        "algorithm": "ddpg",
        "data_level": "low",
        "episodes": 1000,
        "map_hashes": map_hashes,
        "net_width": 256,
        "order": [1, 6, 2, 3, 5, 4],
        "outage_max_s": 100.0,
        "payloads_mb": [90.0, 90.0, 60.0, 60.0, 90.0, 60.0],
        "state_dim": 8,
        "uav_h": 1.0,
        "state_schema": STATE_SCHEMA,
        "variant": "baseline",
    }
    (tmp_path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    context = load_run_context(tmp_path)
    config["state_schema"] = "legacy"
    (tmp_path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="state schema"):
        load_run_context(tmp_path)

    assert context.algorithm == "ddpg"


def test_checkpoint_paths_use_algorithm_specific_training_names(tmp_path):
    from evaluate_policy import checkpoint_paths

    paths = checkpoint_paths(tmp_path, "ddpg", 1000)

    assert paths["final"].name == "ddpg_ddpg_actor1000.pth"
    assert paths["training_best"].name == "ddpg_ddpg_actorbest.pth"


@pytest.mark.parametrize(
    "best_episode,expected",
    [(45, True), (885, False), (None, None)],
)
def test_training_provenance_flags_guidance_eligible_best_episode(best_episode, expected):
    from evaluate_policy import training_provenance

    provenance = training_provenance(
        {"exploration_strategy": "guided", "guided_warmup_episodes": 200},
        best_episode,
    )

    assert provenance["exploration_strategy"] == "guided"
    assert provenance["guided_warmup_episodes"] == 200
    assert provenance["training_best_in_guided_period"] is expected


def test_evaluator_import_does_not_initialize_cuda():
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"

    result = subprocess.run(
        [sys.executable, "-c", "import evaluate_policy; print('cpu-safe')"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cpu-safe"


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"terminal": True}, (True, "success")),
        ({"outage_s": 100.5}, (False, "qos_violation")),
        ({"step": 1000}, (False, "timeout")),
        ({"uploaded_mb": 449.0}, (False, "incomplete_upload_at_target")),
    ],
)
def test_classify_termination_prioritizes_mission_constraints(overrides, expected):
    from evaluate_policy import classify_termination

    values = {
        "terminal": False,
        "step": 500,
        "max_steps": 1000,
        "outage_s": 10.0,
        "outage_max_s": 100.0,
        "visited_points": 6,
        "expected_points": 6,
        "uploaded_mb": 450.0,
        "expected_upload_mb": 450.0,
    }
    values.update(overrides)

    assert classify_termination(**values) == expected


def test_load_run_context_requires_a_complete_td3_configuration(tmp_path):
    from evaluate_policy import load_run_context

    with pytest.raises(FileNotFoundError, match="run_config.json"):
        load_run_context(tmp_path)

    (tmp_path / "run_config.json").write_text(
        json.dumps({"algorithm": "td3", "state_dim": 8}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required fields"):
        load_run_context(tmp_path)


def test_load_actor_rejects_checkpoint_with_wrong_state_dimension(tmp_path):
    import numpy as np
    import torch

    from evaluate_policy import load_actor
    from td3_actor import Actor

    wrong_actor = Actor(7, 2, 256, torch.tensor([np.pi, 0.25]))
    checkpoint = Path(tmp_path) / "wrong.pth"
    torch.save(wrong_actor.state_dict(), checkpoint)

    with pytest.raises(ValueError, match="incompatible"):
        load_actor(checkpoint, state_dim=8, net_width=256)



def _report(success, visited=6, uploaded=450.0, time_s=150.0, delay_s=100.0):
    return {"success": success, "visited_inspection_points": visited, "uploaded_mb": uploaded,
            "task_time_s": time_s, "transmission_delay_s": delay_s}


def test_selection_key_prefers_feasible_then_lower_objective_then_progress():
    from evaluate_policy import selection_key

    ranked = sorted(
        [
            _report(False, visited=2, uploaded=150.0),
            _report(True, time_s=160.0, delay_s=100.0),
            _report(False, visited=5, uploaded=380.0),
            _report(True, time_s=140.0, delay_s=100.0),
        ],
        key=selection_key,
    )

    assert [(r["success"], r["task_time_s"], r["visited_inspection_points"]) for r in ranked] == [
        (True, 140.0, 6), (True, 160.0, 6), (False, 150.0, 5), (False, 150.0, 2)]


def test_checkpoint_paths_include_eval_best_only_when_saved(tmp_path):
    from evaluate_policy import checkpoint_paths

    assert "eval_best" not in checkpoint_paths(tmp_path, "td3", 1000)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "td3_td3_actoreval_best.pth").write_bytes(b"")
    assert checkpoint_paths(tmp_path, "td3", 1000)["eval_best"].name == "td3_td3_actoreval_best.pth"


def test_checkpoint_paths_for_demonstration_variants_use_base_agent_file_names(tmp_path):
    from evaluate_policy import checkpoint_paths

    paths = checkpoint_paths(tmp_path, "td3_bc", 1000)
    assert paths["final"].name == "td3_bc_td3_actor1000.pth"


def test_standard_agent_best_episode_is_never_marked_guided():
    from evaluate_policy import training_provenance

    provenance = training_provenance(
        {"exploration_strategy": "guided", "guided_warmup_episodes": 400, "guide_prob": 0.0}, 45)
    assert provenance["training_best_in_guided_period"] is False
