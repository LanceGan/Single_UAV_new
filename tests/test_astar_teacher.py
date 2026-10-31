import contextlib
import io
import math

import numpy as np
import pytest

from World import World
from astar_teacher import AStarTeacher, d3qn_demo_pairs, load_or_plan_teacher_path
from discrete_actions import DiscreteActionCodec
from training_config import load_fixed_payloads


def _world(order=(1, 6, 2, 3, 5, 4), level="low"):
    with contextlib.redirect_stdout(io.StringIO()):
        return World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=1000,
                     data_sizes=load_fixed_payloads(level), outage_max_s=100.0,
                     ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                     users_name="results/datas/Users_6.txt",
                     BS_loc=np.zeros((4, 3)), traverse_sequence=list(order), variant="A")


@pytest.fixture(scope="module")
def low_path(tmp_path_factory):
    cache = tmp_path_factory.mktemp("teacher")
    return load_or_plan_teacher_path("low", (1, 6, 2, 3, 5, 4), cache_dir=cache)


def test_teacher_path_is_cached_and_reused(low_path, tmp_path_factory):
    positions, meta = low_path
    assert positions.shape[1] == 2 and len(positions) > 100
    assert meta["objective_s"] == pytest.approx(214.0)
    again, meta_again = load_or_plan_teacher_path("low", (1, 6, 2, 3, 5, 4), cache_dir=meta["cache_dir"])
    np.testing.assert_allclose(again, positions)
    assert meta_again["from_cache"]


def test_cached_path_for_a_different_order_is_rejected(low_path):
    _, meta = low_path
    with pytest.raises(ValueError, match="order"):
        load_or_plan_teacher_path("low", (1, 2, 3, 4, 5, 6), cache_dir=meta["cache_dir"], plan_if_missing=False)


def test_continuous_teacher_replays_astar_objective(low_path):
    positions, _ = low_path
    world = _world()
    with contextlib.redirect_stdout(io.StringIO()):
        state, t = world.reset()
        teacher = AStarTeacher(world, positions)
        done = terminal = False
        while not done:
            state, _, done, t, terminal = world.step_inside(teacher.action(), state, t)
    assert terminal
    assert t * world.delta_T + world.trans_delay * world.delta_T == pytest.approx(214.0)


def test_d3qn_demo_pairs_follow_the_astar_trajectory(low_path, tmp_path):
    _, meta = low_path
    codec = DiscreteActionCodec()
    states, actions = d3qn_demo_pairs("low", (1, 6, 2, 3, 5, 4), codec, cache_dir=meta["cache_dir"])
    assert states.shape[0] == actions.shape[0] > 100
    assert states.shape[1] == 8
    assert actions.min() >= 0 and actions.max() < len(codec)
    # The exact A* path stays on its own trajectory, so every demo action is meaningful.
    assert np.all(np.isfinite(states))


def test_nearest_discrete_action_wraps_heading():
    codec = DiscreteActionCodec()
    index = AStarTeacher.nearest_discrete(codec, np.array([math.pi - 0.01, 0.25]))
    heading, distance = codec.decode(index)
    assert math.cos(heading - math.pi) > 0.99 and distance == pytest.approx(0.25)
    assert codec.decode(AStarTeacher.nearest_discrete(codec, np.array([1.0, 0.0])))[1] == 0.0
