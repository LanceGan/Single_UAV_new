import math

import numpy as np

from astar_planner import SpatialGraph
from joint_astar import JointAStarPlanner, JointPlannerState, joint_transition


def test_joint_transition_collects_an_unvisited_point_and_tracks_outage():
    positions = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=float)
    state = JointPlannerState(node=0, visited_mask=0, remaining_mb=0.0,
                              outage_streak_slots=0, elapsed_slots=0,
                              target_index=0)

    result = joint_transition(
        state,
        destination=1,
        positions=positions,
        inspection_points=np.array([[0.5, 0.0]], dtype=float),
        payloads_mb=np.array([3.0]),
        end_position=np.array([2.0, 0.0]),
        snr_db_by_node=np.array([10.0, 10.0]),
        outage_by_node=np.array([False, True]),
        qos_slots=4,
    )

    assert result is not None
    assert result.newly_visited == 0
    assert result.state.visited_mask == 1
    assert result.state.remaining_mb == 3.0
    assert result.state.outage_streak_slots == 1


def test_joint_astar_outputs_an_order_that_emerges_from_the_search():
    graph = SpatialGraph.build(
        length=1.5,
        width=1.5,
        spacing=0.25,
        special_points={
            "start": (0.12, 0.12),
            "a": (0.12, 1.38),
            "b": (1.38, 0.12),
            "end": (1.38, 1.38),
        },
    )
    points = np.array([
        graph.positions[graph.special_nodes["a"]],
        graph.positions[graph.special_nodes["b"]],
    ])
    planner = JointAStarPlanner(
        graph=graph,
        start_node=graph.special_nodes["start"],
        inspection_points=points,
        payloads_mb=np.array([0.01, 0.01]),
        end_position=graph.positions[graph.special_nodes["end"]],
        snr_db_by_node=np.full(len(graph.positions), 40.0),
        outage_by_node=np.zeros(len(graph.positions), dtype=bool),
        arrival_radius=0.2,
        qos_slots=12,
        horizon_slots=200,
        max_expansions=200_000,
        max_labels=500_000,
        max_runtime_s=20.0,
    )

    result = planner.plan()

    assert result.success
    assert result.optimality_proven
    assert sorted(result.inspection_order) == [1, 2]
    assert result.state_path[-1].visited_mask == 3
    assert result.state_path[-1].remaining_mb == 0.0
    assert math.isfinite(result.planner_cost_s)
