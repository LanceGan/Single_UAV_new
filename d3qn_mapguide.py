"""MapGuide demonstrations represented in D3QN's 33-action space."""

import contextlib
import io

import numpy as np

from World import World
from discrete_actions import DiscreteActionCodec
from map_guidance import MapGuide
from training_config import load_fixed_payloads


def build_world(level, order):
    return World(
        length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=1000,
        uav_h=1.0, data_sizes=load_fixed_payloads(level), outage_max_s=100.0,
        ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
        users_name="results/datas/Users_6.txt",
        BS_loc=np.array([[4.77, 4.56, 0.25], [14.89, 4.89, 0.25],
                         [4.63, 13.78, 0.25], [13.48, 14.89, 0.25]]),
        traverse_sequence=list(order), variant="A",
    )


def discrete_mapguide_action(world, guide, codec):
    absolute = guide.action()
    relative = world.absolute_to_action(absolute)
    return codec.nearest(relative)


def collect_mapguide_demonstration(level, order, codec=None, world=None):
    """Collect one successful discrete MapGuide trajectory and full transitions."""
    codec = codec or DiscreteActionCodec()
    with contextlib.redirect_stdout(io.StringIO()):
        world = world or build_world(level, order)
        state, step = world.reset()
    guide = MapGuide(world)
    transitions = {name: [] for name in
                   ("states", "actions", "rewards", "next_states", "dones")}
    uploaded = 0.0
    total_return = 0.0
    done = False
    terminal = False
    while not done:
        action_index = discrete_mapguide_action(world, guide, codec)
        action = codec.decode(action_index)
        with contextlib.redirect_stdout(io.StringIO()):
            next_state, reward, done, step, terminal = world.step_inside(action, state, step)
        transitions["states"].append(np.asarray(state, dtype=np.float32))
        transitions["actions"].append(int(action_index))
        transitions["rewards"].append(float(reward))
        transitions["next_states"].append(np.asarray(next_state, dtype=np.float32))
        transitions["dones"].append(bool(done))
        uploaded += float(world.last_uploaded_data)
        total_return += float(reward)
        state = next_state
    packed = {
        "states": np.asarray(transitions["states"], dtype=np.float32),
        "actions": np.asarray(transitions["actions"], dtype=np.int64),
        "rewards": np.asarray(transitions["rewards"], dtype=np.float32),
        "next_states": np.asarray(transitions["next_states"], dtype=np.float32),
        "dones": np.asarray(transitions["dones"], dtype=bool),
    }
    report = {
        "success": bool(terminal),
        "steps": int(step),
        "task_time_s": float(step * world.delta_T),
        "visited_inspection_points": int(world.Traverse_order),
        "uploaded_mb": float(round(uploaded, 8)),
        "outage_time_s": float(world.out_time),
        "return": total_return,
    }
    if not report["success"]:
        raise RuntimeError(f"discrete MapGuide demonstration is infeasible: {report}")
    return packed, report
