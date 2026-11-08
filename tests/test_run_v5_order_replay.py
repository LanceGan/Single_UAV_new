import numpy as np
import torch

from run_v5_order_replay import ActorMapGuideController


class _World:
    def __init__(self):
        self.Traverse_order = 0
        self.target_dis = 10.0
        self.t = 0


class _Actor:
    def __call__(self, _state):
        return torch.tensor([[0.0, 0.1]], dtype=torch.float32)


def test_actor_mapguide_controller_switches_after_equal_stall_threshold(monkeypatch):
    world = _World()
    monkeypatch.setattr(
        "run_v5_order_replay.MapGuide",
        lambda _world, use_a2g: type(
            "Guide",
            (),
            {"action": lambda self: np.asarray([1.0, 0.25])},
        )(),
    )
    controller = ActorMapGuideController(
        world,
        _Actor(),
        stall_steps=2,
    )
    state = np.zeros(8)

    np.testing.assert_allclose(controller(state), [0.0, 0.1])
    world.t = 1
    np.testing.assert_allclose(controller(state), [0.0, 0.1])
    world.t = 2
    np.testing.assert_allclose(controller(state), [1.0, 0.25])
    assert controller.takeover_step == 2
    assert controller.actor_steps == 2
    assert controller.guide_steps == 1


def test_actor_mapguide_controller_resets_timer_when_target_changes(monkeypatch):
    world = _World()
    monkeypatch.setattr(
        "run_v5_order_replay.MapGuide",
        lambda _world, use_a2g: type(
            "Guide",
            (),
            {"action": lambda self: np.asarray([1.0, 0.25])},
        )(),
    )
    controller = ActorMapGuideController(world, _Actor(), stall_steps=3)
    state = np.zeros(8)

    world.t = 5
    world.Traverse_order = 1
    world.target_dis = 9.0
    controller(state)
    assert controller.takeover_step is None
