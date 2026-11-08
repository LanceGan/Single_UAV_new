import numpy as np

from run_sequence_study import route_proxy
from sequence_optimizers import distance_astar_order, distance_pso_order


def _toy_nodes():
    return np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.5],
            [4.0, 0.5],
        ],
        dtype=float,
    )


def test_distance_astar_returns_complete_permutation():
    order, meta = distance_astar_order(_toy_nodes())

    assert sorted(order) == [1, 2, 3, 4, 5, 6]
    assert meta["optimality_proven"] is True
    assert meta["uses_payloads"] is False


def test_distance_pso_is_seeded_and_returns_complete_permutation():
    first, first_meta = distance_pso_order(
        _toy_nodes(), seed=7, particles=20, iterations=20
    )
    second, second_meta = distance_pso_order(
        _toy_nodes(), seed=7, particles=20, iterations=20
    )

    assert first == second
    assert sorted(first) == [1, 2, 3, 4, 5, 6]
    assert first_meta["distance_m"] == second_meta["distance_m"]


def test_route_proxy_reports_rates_and_outage_for_finite_order():
    order = [1, 2, 3, 4, 5, 6]
    # The proxy only calls these two map methods, so a small deterministic
    # stand-in avoids depending on the full radio-map files in a unit test.
    class Maps:
        def a2g_snr_db(self, _point):
            return 10.0

        def g2a_outage_probability(self, point):
            return 0.6 if point[0] > 0.5 else 0.0

    payloads = np.arange(1.0, 7.0)
    result = route_proxy(order, payloads, maps=Maps())

    assert result["estimated_t_s"] > 0.0
    assert result["estimated_tau_s"] > 0.0
    assert result["mean_a2g_rate_length_weighted"] > 0.0
    assert result["mean_a2g_rate_payload_weighted"] > 0.0
    assert result["max_contiguous_g2a_outage_s"] > 0.0
    assert len(result["legs"]) == 7
