import numpy as np
import pytest

from sequence_optimizers import (
    _path_cost,
    distance_matrix_m,
    distance_tsp_order,
    g2a_edge_seconds,
    sa_order,
)


class _StubMaps:
    """g2a_outage_probability returns 1.0 inside the blocked x-band, else 0."""

    def __init__(self, blocked=None):
        self.blocked = blocked  # (x_min_km, x_max_km) or None

    def g2a_outage_probability(self, point_km):
        if self.blocked is None:
            return 0.0
        x = float(np.asarray(point_km).reshape(2)[0])
        return 1.0 if self.blocked[0] <= x <= self.blocked[1] else 0.0


def _square_nodes():
    # start, four inner nodes on a line with a detour, end
    return np.array([
        [0.0, 0.0],
        [0.4, 0.0],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.6, 0.0],
        [1.0, 0.0],
    ])


def test_distance_tsp_is_exact_and_ignores_everything_but_geometry():
    order, meta = distance_tsp_order(_square_nodes())

    assert order == [2, 1, 4, 3]  # straight sweep along the line
    assert meta["distance_m"] == pytest.approx(1000.0)
    assert meta["uses_g2a"] is False and meta["uses_a2g"] is False
    assert meta["uses_payloads"] is False


def test_distance_tsp_on_repository_nodes_is_deterministic_permutation():
    first, _ = distance_tsp_order()
    second, _ = distance_tsp_order()

    assert first == second
    assert sorted(first) == [1, 2, 3, 4, 5, 6]


def test_g2a_edge_seconds_counts_outage_fraction():
    clear_flight, clear_outage = g2a_edge_seconds([0, 0], [1, 0], _StubMaps())
    blocked_flight, blocked_outage = g2a_edge_seconds([0, 0], [1, 0], _StubMaps(blocked=(-1, 2)))

    assert clear_flight == pytest.approx(1000.0 / 50.0)
    assert clear_outage == 0.0
    assert blocked_flight == pytest.approx(clear_flight)
    assert blocked_outage == pytest.approx(blocked_flight)


def test_sa_reaches_exact_optimum_on_clear_map_and_is_seed_deterministic():
    nodes = _square_nodes()
    exact, exact_meta = distance_tsp_order(nodes)

    sa_first, meta_first = sa_order(nodes, maps=_StubMaps(), seed=3)
    sa_second, _ = sa_order(nodes, maps=_StubMaps(), seed=3)

    assert sa_first == sa_second
    assert sorted(sa_first) == [1, 2, 3, 4]
    # With no outage anywhere the SA objective reduces to flight seconds.
    assert meta_first["objective_s"] == pytest.approx(exact_meta["distance_m"] / 50.0)
    assert sa_first == exact
    assert meta_first["uses_a2g"] is False and meta_first["uses_payloads"] is False


def test_sa_detours_around_outage_corridor():
    # Start (0,0), two nodes left of a blocked band, one node right of it, end right.
    nodes = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [0.15, 0.4],
        [0.9, 0.2],
        [1.0, 0.0],
    ])
    blocked = _StubMaps(blocked=(0.3, 0.7))

    order, meta = sa_order(nodes, maps=blocked, seed=5, sweeps=3000)

    # Any order must cross the band once, but crossing it more than once is
    # strictly worse; the optimum visits both left nodes before the right one.
    assert order[-1] == 3
    assert set(order[:2]) == {1, 2}
    assert meta["objective_s"] > 0.0
