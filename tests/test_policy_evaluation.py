import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_load_run_context_accepts_ddpg_configuration(tmp_path):
    from evaluate_policy import load_run_context

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
    }
    (tmp_path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    context = load_run_context(tmp_path)

    assert context.algorithm == "ddpg"


def test_checkpoint_paths_use_algorithm_specific_training_names(tmp_path):
    from evaluate_policy import checkpoint_paths

    paths = checkpoint_paths(tmp_path, "ddpg", 1000)

    assert paths["final"].name == "ddpg_ddpg_actor1000.pth"
    assert paths["training_best"].name == "ddpg_ddpg_actorbest.pth"


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

