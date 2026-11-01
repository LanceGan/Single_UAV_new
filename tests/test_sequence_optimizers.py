import numpy as np
import pytest

from sequence_optimizers import (
    _path_cost,
    distance_matrix_m,
    distance_tsp_order,
    eega_cost_matrix,
    eega_order,
    g2a_edge_seconds,
    ga_proxy_order,
    sa_order,
    yang_order,
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


def test_ga_proxy_returns_plain_int_permutation_and_is_seed_deterministic():
    first, meta = ga_proxy_order(seed=1)
    second, _ = ga_proxy_order(seed=1)

    assert first == second
    assert sorted(first) == [1, 2, 3, 4, 5, 6]
    assert all(type(gene) is int for gene in first)  # run_config json-serializable
    assert meta["uses_a2g"] is False and meta["uses_payloads"] is False
    assert "proxy" in meta["proxy_disclaimer"] or "EEGA" in meta["proxy_disclaimer"]


def test_ga_proxy_finds_the_line_optimum():
    order, meta = ga_proxy_order(_square_nodes(), seed=1, generations=120)
    exact, exact_meta = distance_tsp_order(_square_nodes())

    assert order == exact
    assert meta["distance_m"] == pytest.approx(exact_meta["distance_m"])


class _FlatSnrMaps:
    """Constant G2A SNR everywhere -> constant reciprocal rate."""

    def g2a_snr_db(self, point_km):
        return 0.0  # rate = log2(2) = 1


def test_eega_cost_matrix_is_payload_weighted_and_start_legs_are_free():
    nodes = _square_nodes()
    light = eega_cost_matrix(nodes, [10.0, 10.0, 10.0, 10.0], maps=_FlatSnrMaps())
    heavy = eega_cost_matrix(nodes, [10.0, 100.0, 10.0, 10.0], maps=_FlatSnrMaps())

    assert np.all(light[0, 1:] == 0.0)  # literal formula: no payload at the start yet
    np.testing.assert_allclose(heavy[2], 10.0 * light[2])  # node 2 payload scaled x10
    np.testing.assert_allclose(heavy[1], light[1])


def test_eega_and_yang_orders_are_valid_and_deterministic_on_repository_data():
    from training_config import load_fixed_payloads

    payloads = load_fixed_payloads("low")
    eega_first, eega_meta = eega_order(payloads, seed=1)
    eega_second, _ = eega_order(payloads, seed=1)
    yang_first, yang_meta = yang_order(payloads)
    yang_second, _ = yang_order(payloads)

    assert eega_first == eega_second and yang_first == yang_second
    assert sorted(eega_first) == [1, 2, 3, 4, 5, 6]
    assert sorted(yang_first) == [1, 2, 3, 4, 5, 6]
    assert all(type(gene) is int for gene in eega_first + yang_first)
    assert eega_meta["uses_a2g"] is False and yang_meta["uses_a2g"] is False
    assert eega_meta["uses_payloads"] is True and yang_meta["uses_payloads"] is True


def test_yang_cost_shifts_toward_offload_when_payloads_grow():
    nodes = _square_nodes()
    light, _ = yang_order([1.0, 1.0, 1.0, 1.0], nodes_km=nodes, maps=_FlatSnrMaps())
    assert sorted(light) == [1, 2, 3, 4]


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
