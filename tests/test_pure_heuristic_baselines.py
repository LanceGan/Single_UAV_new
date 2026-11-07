import numpy as np
import pytest

from astar_planner import SpatialGraph
from pure_heuristic_baselines import (
    PureAStarTrajectoryPlanner,
    _route_distance,
    solve_pure_pso,
)


def test_pure_astar_builds_a_complete_geometric_trajectory():
    graph = SpatialGraph.build(
        length=3.0,
        width=1.0,
        spacing=0.5,
        special_points={
            "start": (0.1, 0.5),
            "a": (0.6, 0.5),
            "b": (2.4, 0.5),
            "end": (2.9, 0.5),
        },
    )
    planner = PureAStarTrajectoryPlanner(
        graph=graph,
        start_node=graph.special_nodes["start"],
        inspection_nodes=[graph.special_nodes["a"], graph.special_nodes["b"]],
        end_node=graph.special_nodes["end"],
        arrival_radius=0.25,
    )

    result = planner.plan()

    assert result.success
    assert result.optimality_proven
    assert sorted(result.inspection_order) == [1, 2]
    assert result.node_path[0] == graph.special_nodes["start"]
    assert np.linalg.norm(
        graph.positions[result.node_path[-1]] - graph.positions[graph.special_nodes["end"]]
    ) <= 0.25 + 1e-12


def test_pure_pso_returns_a_permutation_and_geometry_only_cost():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )

    result = solve_pure_pso(points=points, seed=3, swarm=12, iterations=20)

    assert result.success
    assert sorted(result.inspection_order) == [1, 2, 3]
    assert result.path_distance == pytest.approx(
        _route_distance(result.inspection_order, points)
    )
    assert result.path_distance >= 3.0
