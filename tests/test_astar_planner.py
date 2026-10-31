import math

import numpy as np
import pytest

from astar_planner import (
    AStarPlanner,
    DominanceLabel,
    DominanceTable,
    PlannerState,
    SpatialGraph,
    arrival_disk_heuristic,
    transition_slot,
)


def _tiny_graph():
    return SpatialGraph.build(
        length=0.5,
        width=0.5,
        spacing=0.25,
        special_points={"start": (0.12, 0.12), "target": (0.38, 0.38)},
    )


def test_graph_has_only_hover_cardinal_and_special_connections_within_one_step():
    graph = _tiny_graph()
    grid_origin = graph.grid_node_at((0.0, 0.0))
    grid_center = graph.grid_node_at((0.25, 0.25))
    start = graph.special_nodes["start"]

    assert grid_origin in graph.neighbors[grid_origin]
    assert graph.grid_node_at((0.25, 0.0)) in graph.neighbors[grid_origin]
    assert graph.grid_node_at((0.0, 0.25)) in graph.neighbors[grid_origin]
    assert grid_center not in graph.neighbors[grid_origin]
    assert start in graph.neighbors[start]
    assert grid_origin in graph.neighbors[start]
    assert grid_center in graph.neighbors[start]
    assert all(
        np.linalg.norm(graph.positions[neighbor] - graph.positions[start]) <= 0.25 + 1e-12
        for neighbor in graph.neighbors[start]
    )
    assert tuple(graph.positions[start]) == pytest.approx((0.12, 0.12))


def test_arrival_heuristic_measures_remaining_distance_to_50m_disk():
    assert arrival_disk_heuristic((0.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert arrival_disk_heuristic((0.6, 0.0), (1.0, 0.0)) == 0.0


def test_transition_charges_delay_only_when_slot_starts_with_pending_payload():
    state = PlannerState(node=0, leg=0, remaining_mb=1.0, outage_slots=0, elapsed_slots=0)
    result = transition_slot(
        state,
        destination=1,
        positions=np.array([[0.0, 0.0], [0.25, 0.0]]),
        targets=np.array([[2.0, 0.0]]),
        payloads_mb=np.array([]),
        snr_db_by_node=np.array([100.0, 100.0]),
        outage_by_node=np.array([False, False]),
    )

    assert result is not None
    assert result.pending_at_start
    assert result.uploaded_mb == pytest.approx(1.0)
    assert result.cost_s == pytest.approx(1.0)
    assert result.state.remaining_mb == 0.0


def test_same_slot_final_upload_can_enter_target_but_new_payload_waits_until_next_slot():
    state = PlannerState(node=0, leg=0, remaining_mb=0.25, outage_slots=0, elapsed_slots=4)
    result = transition_slot(
        state,
        destination=1,
        positions=np.array([[0.0, 0.0], [0.25, 0.0]]),
        targets=np.array([[0.25, 0.0], [1.0, 0.0]]),
        payloads_mb=np.array([7.0]),
        snr_db_by_node=np.array([100.0, 100.0]),
        outage_by_node=np.array([False, False]),
    )

    assert result is not None and result.arrived and not result.success
    assert result.state.leg == 1
    assert result.state.remaining_mb == pytest.approx(7.0)
    assert result.uploaded_mb == pytest.approx(0.25)
    assert result.cost_s == pytest.approx(1.0)


def test_target_entry_with_post_upload_payload_is_pruned():
    state = PlannerState(node=0, leg=0, remaining_mb=10.0, outage_slots=0, elapsed_slots=0)
    result = transition_slot(
        state,
        destination=1,
        positions=np.array([[0.0, 0.0], [0.25, 0.0]]),
        targets=np.array([[0.25, 0.0], [1.0, 0.0]]),
        payloads_mb=np.array([2.0]),
        snr_db_by_node=np.array([-100.0, -100.0]),
        outage_by_node=np.array([False, False]),
    )

    assert result is None


def test_qos_is_strict_but_success_on_slot_1000_is_allowed():
    qos_edge = PlannerState(node=0, leg=0, remaining_mb=0.0, outage_slots=200, elapsed_slots=0)
    common = dict(
        destination=1,
        positions=np.array([[0.0, 0.0], [0.25, 0.0]]),
        targets=np.array([[0.25, 0.0]]),
        payloads_mb=np.array([]),
        snr_db_by_node=np.array([0.0, 0.0]),
        outage_by_node=np.array([False, True]),
    )
    assert transition_slot(qos_edge, **common) is None

    last_slot = PlannerState(node=0, leg=0, remaining_mb=0.0, outage_slots=0, elapsed_slots=999)
    common["outage_by_node"] = np.array([False, False])
    result = transition_slot(last_slot, **common)
    assert result is not None and result.success
    assert result.state.elapsed_slots == 1000

    already_at_horizon = PlannerState(node=0, leg=0, remaining_mb=0.0, outage_slots=0, elapsed_slots=1000)
    assert transition_slot(already_at_horizon, **common) is None


def test_dominance_requires_actual_payload_cost_qos_and_elapsed_all_not_worse():
    table = DominanceTable(bucket_mb=5.0)
    better = DominanceLabel(
        PlannerState(node=3, leg=1, remaining_mb=6.0, outage_slots=8, elapsed_slots=20),
        cost_s=13.0,
    )
    assert table.accept(better)

    dominated = DominanceLabel(
        PlannerState(node=3, leg=1, remaining_mb=7.0, outage_slots=9, elapsed_slots=21),
        cost_s=14.0,
    )
    assert not table.accept(dominated)

    less_payload_but_more_outage = DominanceLabel(
        PlannerState(node=3, leg=1, remaining_mb=5.5, outage_slots=10, elapsed_slots=20),
        cost_s=13.0,
    )
    assert table.accept(less_payload_but_more_outage)


def test_astar_is_deterministic_and_reports_expansion_cap_honestly():
    graph = _tiny_graph()
    snr = np.full(len(graph.positions), 100.0)
    outage = np.zeros(len(graph.positions), dtype=bool)
    kwargs = dict(
        graph=graph,
        start_node=graph.special_nodes["start"],
        targets=np.array([[0.38, 0.38]]),
        payloads_mb=np.array([]),
        snr_db_by_node=snr,
        outage_by_node=outage,
        max_expansions=100,
        max_labels=1000,
        max_runtime_s=5.0,
    )

    first = AStarPlanner(**kwargs).plan()
    second = AStarPlanner(**kwargs).plan()
    capped = AStarPlanner(**{**kwargs, "max_expansions": 0}).plan()

    assert first.status == "success"
    assert first.node_path == second.node_path
    assert first.expansions == second.expansions
    assert capped.status == "expansion_limit"
    assert not capped.success
    assert not capped.optimality_proven



def test_capped_search_reports_its_most_advanced_partial_path():
    graph = SpatialGraph.build(
        length=1.0, width=1.0, spacing=0.25,
        special_points={"start": (0.12, 0.12), "target": (0.88, 0.88)},
    )
    planner = AStarPlanner(
        graph=graph,
        start_node=graph.special_nodes["start"],
        targets=np.array([[0.88, 0.88]]),
        payloads_mb=np.array([]),
        snr_db_by_node=np.full(len(graph.positions), 100.0),
        outage_by_node=np.zeros(len(graph.positions), dtype=bool),
        max_expansions=3,
        max_labels=1000,
        max_runtime_s=5.0,
    )
    result = planner.plan()
    assert result.status == "expansion_limit"
    assert len(result.node_path) > 1
    assert result.node_path[0] == graph.special_nodes["start"]


def _corridor_planner(heuristic="mission", snr=10.0, outage=None, payloads=(6.0,), **limits):
    graph = SpatialGraph.build(
        length=1.5, width=0.5, spacing=0.25,
        special_points={"start": (0.12, 0.12), "a": (0.75, 0.38), "end": (1.38, 0.12)},
    )
    count = len(graph.positions)
    return AStarPlanner(
        graph=graph,
        start_node=graph.special_nodes["start"],
        targets=np.array([[0.75, 0.38], [1.38, 0.12]]),
        payloads_mb=np.asarray(payloads, dtype=float),
        snr_db_by_node=np.full(count, snr),
        outage_by_node=np.zeros(count, dtype=bool) if outage is None else outage(graph),
        arrival_radius=0.2,
        heuristic=heuristic,
        **{"max_expansions": 200_000, "max_labels": 1_000_000, "max_runtime_s": 30.0, **limits},
    )


def test_mission_heuristic_counts_remaining_legs_and_pending_uploads():
    planner = _corridor_planner()
    start = PlannerState(planner.start_node, 0, 0.0, 0, 0)
    after_first = PlannerState(planner.graph.special_nodes["a"], 1, 6.0, 0, 3)
    leg_only = arrival_disk_heuristic(planner.graph.positions[start.node], planner.targets[0],
                                      arrival_radius=0.2)
    assert planner._heuristic(start) > leg_only
    rate_slot = planner.max_upload_mb_per_slot
    upload_slots = math.ceil(6.0 / rate_slot)
    assert planner._heuristic(after_first) >= 2 * upload_slots * planner.delta_t - 1e-9


@pytest.mark.parametrize("snr,payloads", [(10.0, (6.0,)), (0.0, (4.0,)), (20.0, (30.0,))])
def test_mission_heuristic_preserves_optimal_cost(snr, payloads):
    def patchy_outage(graph):
        return graph.positions[:, 1] < 0.2

    informed = _corridor_planner("mission", snr=snr, payloads=payloads, outage=patchy_outage).plan()
    uninformed = _corridor_planner("zero", snr=snr, payloads=payloads, outage=patchy_outage).plan()
    assert informed.success and uninformed.success
    assert informed.planner_cost_s == pytest.approx(uninformed.planner_cost_s)
    assert informed.expansions <= uninformed.expansions


def test_heuristic_never_exceeds_optimal_cost_from_start():
    planner = _corridor_planner("mission", snr=5.0, payloads=(9.0,))
    result = planner.plan()
    assert result.success
    start = PlannerState(planner.start_node, 0, 0.0, 0, 0)
    assert planner._heuristic(start) <= result.planner_cost_s + 1e-9


def test_dominance_compares_labels_across_payload_buckets():
    table = DominanceTable(bucket_mb=1.0)
    better = DominanceLabel(PlannerState(node=3, leg=1, remaining_mb=2.2, outage_slots=4, elapsed_slots=10), 8.0)
    worse = DominanceLabel(PlannerState(node=3, leg=1, remaining_mb=7.9, outage_slots=5, elapsed_slots=11), 9.0)
    assert table.accept(better)
    assert not table.accept(worse)


def test_sequential_plan_solves_each_leg_and_is_never_better_than_joint():
    def patchy_outage(graph):
        return graph.positions[:, 1] < 0.2

    joint = _corridor_planner(snr=5.0, payloads=(9.0,), outage=patchy_outage).plan()
    sequential = _corridor_planner(snr=5.0, payloads=(9.0,), outage=patchy_outage).plan_sequential()
    assert joint.success and sequential.success
    assert sequential.planner_cost_s >= joint.planner_cost_s - 1e-9
    assert sequential.state_path[-1].leg == 2
    assert not sequential.optimality_proven
    assert sequential.node_path[0] == sequential.state_path[0].node
    assert len(sequential.node_path) == len(sequential.state_path)


def test_plan_can_stop_after_one_leg_from_a_given_state():
    planner = _corridor_planner(snr=10.0, payloads=(6.0,))
    first = planner.plan(stop_leg=1)
    assert first.success and first.state_path[-1].leg == 1
    assert first.state_path[-1].remaining_mb == pytest.approx(6.0)
    second = planner.plan(initial_state=first.state_path[-1])
    assert second.success and second.state_path[0] == first.state_path[-1]


def test_sequential_plan_reports_failing_leg_honestly():
    planner = _corridor_planner(snr=10.0, payloads=(6.0,), max_expansions=2)
    result = planner.plan_sequential()
    assert result.status == "expansion_limit"
    assert "leg 1" in result.message
