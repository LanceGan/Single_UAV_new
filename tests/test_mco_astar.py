import numpy as np

from astar_planner import SpatialGraph
from mco_astar import MCOAStarPlanner


def _graph_and_points():
    graph = SpatialGraph.build(length=4.0, width=2.0, spacing=1.0)
    return graph, graph.grid_node_at((0.0, 0.0)), graph.grid_node_at((4.0, 0.0))


def test_mco_astar_uses_shortest_path_when_outage_streak_fits():
    graph, start, goal = _graph_and_points()
    outage = np.zeros(len(graph.positions), dtype=bool)
    outage[:5] = True
    planner = MCOAStarPlanner(graph=graph, outage_by_node=outage, qos_slots=4)

    result = planner.plan(start_node=start, goal_node=goal)

    assert result.success
    assert result.cost_slots == 4
    assert max(result.outage_streak_slots_path) == 4


def test_mco_astar_detours_when_shortest_path_violates_outage_budget():
    graph, start, goal = _graph_and_points()
    outage = np.zeros(len(graph.positions), dtype=bool)
    outage[:5] = True
    planner = MCOAStarPlanner(graph=graph, outage_by_node=outage, qos_slots=3)

    result = planner.plan(start_node=start, goal_node=goal)

    assert result.success
    assert result.cost_slots == 6
    assert max(result.outage_streak_slots_path) <= 3
    assert result.node_path != tuple(range(5))


def test_mco_astar_rejects_a_goal_inside_the_upload_guard_ring():
    graph, start, goal = _graph_and_points()
    planner = MCOAStarPlanner(
        graph=graph,
        outage_by_node=np.zeros(len(graph.positions), dtype=bool),
        qos_slots=12,
    )

    result = planner.plan(
        start_node=start,
        goal_node=goal,
        blocked_nodes=(goal,),
    )

    assert not result.success
    assert result.status == "blocked_goal"
