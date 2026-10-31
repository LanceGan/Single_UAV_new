import json

import numpy as np

from World import World
from astar_planner import PlannerResult, PlannerState, SpatialGraph
from run_astar import replay_node_path, save_level_artifacts


def test_graph_path_replays_as_variant_a_actions_in_real_world(tmp_path):
    users = tmp_path / "users.txt"
    users.write_text("2.62 2.65 0\n", encoding="utf-8")
    world = World(
        length=20,
        width=20,
        uav_num=1,
        dist_max=0.25,
        delta_t=0.5,
        t=1000,
        data_sizes=np.array([0.01]),
        ini_loc=[2.62, 2.65],
        end_loc=[2.62, 2.65],
        users_name=str(users),
        BS_loc=np.zeros((4, 3)),
        traverse_sequence=[1],
        variant="A",
    )
    graph = SpatialGraph.build(
        length=20,
        width=20,
        spacing=0.25,
        special_points={"start": (2.62, 2.65)},
    )
    start = graph.special_nodes["start"]

    report, trace = replay_node_path(world, graph, (start, start, start))

    assert report["success"]
    assert report["task_time_s"] == 1.0
    assert report["transmission_delay_s"] == 0.5
    assert trace["actions"].shape == (3, 2)
    assert np.allclose(trace["positions"], [[2.62, 2.65]] * 3)


def test_artifact_writer_records_capped_status_without_claiming_optimality(tmp_path):
    graph = SpatialGraph.build(
        length=0.25,
        width=0.25,
        spacing=0.25,
        special_points={"start": (0.12, 0.12)},
    )
    start = graph.special_nodes["start"]
    state = PlannerState(start, 0, 0.0, 0, 0)
    result = PlannerResult(
        status="expansion_limit",
        node_path=(start,),
        state_path=(state,),
        planner_cost_s=0.0,
        expansions=0,
        generated_labels=1,
        runtime_s=0.01,
        optimality_proven=False,
        message="maximum expansions reached before a feasible path was found",
    )
    report = {
        "success": False,
        "termination_reason": "planner_expansion_limit",
        "task_time_s": 0.0,
        "transmission_delay_s": 0.0,
        "objective_s": 0.0,
        "outage_time_s": 0.0,
        "flight_distance_m": 0.0,
    }
    trace = {
        "positions": np.array([[0.12, 0.12]]),
        "time_s": np.array([0.0]),
        "outage_s": np.array([0.0]),
        "pending_mb": np.array([0.0]),
        "visited_points": np.array([0]),
        "actions": np.array([[0.0, 0.0]]),
    }

    save_level_artifacts(
        output_dir=tmp_path,
        level="low",
        order=(1,),
        payloads_mb=np.array([1.0]),
        graph=graph,
        planner_result=result,
        replay_report=report,
        trace=trace,
        draw_background=False,
    )

    metrics = json.loads((tmp_path / "low" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["planner"]["status"] == "expansion_limit"
    assert metrics["planner"]["optimality_proven"] is False
    for name in ("trajectory.json", "trajectory.npz", "trajectory.png", "trajectory.pdf"):
        assert (tmp_path / "low" / name).is_file()
