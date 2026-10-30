import math

import numpy as np
import pytest

from World import World
from training_config import build_output_paths


def _world(variant, **overrides):
    kwargs = dict(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6], variant=variant)
    kwargs.update(overrides)
    return World(**kwargs)


def _bearing(world):
    delta = world.target_loc - np.array([world.UAVs[0].x, world.UAVs[0].y])
    return math.atan2(delta[1], delta[0])


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="variant"):
        _world("C")


@pytest.mark.parametrize("variant,dim", [("baseline", 8), ("A", 8), ("AB", 24)])
def test_state_dimension_follows_variant(variant, dim):
    world = _world(variant)
    state, _ = world.reset()
    assert world.state_dim == dim and state.shape == (dim,)


def test_baseline_heading_is_absolute():
    world = _world("baseline")
    world.reset()
    start = np.array([world.UAVs[0].x, world.UAVs[0].y])
    world.step_inside(np.array([0.0, 0.25]), None, 0)
    np.testing.assert_allclose([world.UAVs[0].x, world.UAVs[0].y], start + [0.25, 0.0])


@pytest.mark.parametrize("variant", ["A", "AB"])
def test_relative_heading_zero_flies_straight_at_target(variant):
    world = _world(variant)
    world.reset()
    before = np.linalg.norm(world.target_loc - [world.UAVs[0].x, world.UAVs[0].y])
    world.step_inside(np.array([0.0, 0.25]), None, 0)
    after = np.linalg.norm(world.target_loc - [world.UAVs[0].x, world.UAVs[0].y])
    assert before - after == pytest.approx(0.25)


@pytest.mark.parametrize("variant", ["baseline", "A", "AB"])
def test_absolute_action_conversion_round_trips(variant):
    world = _world(variant)
    world.reset()
    for heading in (-3.0, -1.0, 0.5, 3.1):
        absolute = np.array([heading, 0.2])
        action = world.absolute_to_action(absolute)
        assert -math.pi <= action[0] <= math.pi
        start = np.array([world.UAVs[0].x, world.UAVs[0].y])
        world.UAVs[0].x, world.UAVs[0].y = start
        world.step_inside(action, None, 0)
        moved = np.array([world.UAVs[0].x, world.UAVs[0].y]) - start
        np.testing.assert_allclose(moved, 0.2 * np.array([math.cos(heading), math.sin(heading)]), atol=1e-9)
        world.UAVs[0].x, world.UAVs[0].y = start
        world.reset()


def test_neighborhood_samples_use_target_frame_and_one_step_radius(monkeypatch):
    world = _world("AB")
    sampled = []
    monkeypatch.setattr(world, "get_date_rate_A2G", lambda loc: sampled.append(np.asarray(loc).reshape(-1)[:2]) or 0.0)
    monkeypatch.setattr(world, "get_g2a_outage_probability", lambda loc: 0.0)
    world.reset()
    sampled.clear()
    world.reset_state()
    position = np.array([world.UAVs[0].x, world.UAVs[0].y])
    offsets = np.array(sampled[:8]) - position
    np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), world.dist_max)
    first = math.atan2(offsets[0][1], offsets[0][0])
    assert first == pytest.approx(_bearing(world))
    for k in range(8):
        angle = math.atan2(offsets[k][1], offsets[k][0])
        assert math.cos(angle - first - k * math.pi / 4) == pytest.approx(1.0)


def test_neighborhood_features_are_normalized():
    world = _world("AB")
    state, _ = world.reset()
    assert np.all((state[8:] >= 0.0) & (state[8:] <= 1.0))


def test_output_paths_separate_variants(tmp_path):
    paths = build_output_paths(tmp_path, "low", "td3", variant="AB")
    assert paths["root"] == tmp_path / "AB" / "low" / "td3"


@pytest.mark.parametrize("variant", ["A", "AB"])
def test_relabeled_mapguide_demonstration_completes_mission(variant):
    """Demonstrations stored in the policy frame must replay to the same successful mission."""
    import contextlib
    import io

    from map_guidance import MapGuide
    from training_config import load_fixed_payloads

    with contextlib.redirect_stdout(io.StringIO()):
        world = _world(variant, t=1000, data_sizes=load_fixed_payloads("low"),
                       traverse_sequence=[1, 6, 2, 3, 5, 4], outage_max_s=100.0)
        state, t = world.reset()
        guide = MapGuide(world)
        done = terminal = False
        while not done:
            action = world.absolute_to_action(guide.action())
            state, _, done, t, terminal = world.step_inside(action, state, t)
    assert terminal and t * world.delta_T == pytest.approx(196.5)


def test_boundary_crossing_clips_each_coordinate_and_keeps_penalty():
    world = _world("baseline")
    state, _ = world.reset()
    world.UAVs[0].x, world.UAVs[0].y = 0.1, 5.0
    heading = math.atan2(0.15, -0.2)  # 0.25 step: 0.2 west (crosses x=0), 0.15 north
    world.step_inside(np.array([heading, 0.25]), state, 0)
    assert world.UAVs[0].x == pytest.approx(0.0)
    assert world.UAVs[0].y == pytest.approx(5.15)
    assert world.fa == 1
    assert world.reward_summary()["boundary"] < 0.0


def test_boundary_slide_does_not_lock_a_repeated_action():
    world = _world("baseline")
    state, _ = world.reset()
    world.UAVs[0].x, world.UAVs[0].y = 0.0, 5.0
    heading = math.atan2(0.15, -0.2)
    positions = []
    for t in range(3):
        state, *_ = world.step_inside(np.array([heading, 0.25]), state, t)
        positions.append((world.UAVs[0].x, world.UAVs[0].y))
    assert len(set(positions)) == 3


def test_stall_diagnostics_track_consecutive_motionless_slots():
    world = _world("baseline")
    state, _ = world.reset()
    for t, distance in enumerate([0.0, 0.0, 0.2, 0.0, 0.0, 0.0]):
        state, *_ = world.step_inside(np.array([0.0, distance]), state, t)
    assert world.stall_steps == 5
    assert world.max_stall_run == 3
    world.reset()
    assert world.stall_steps == world.max_stall_run == world.fa == 0
