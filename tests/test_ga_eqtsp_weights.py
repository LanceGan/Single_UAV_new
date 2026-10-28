import numpy as np
import pytest

from GA_EQTSP import GA, power


def test_edge_weight_reflects_source_payload_and_g2a_outage():
    ga = GA.__new__(GA)
    energy = np.ones((3, 3))
    throughput = np.full((3, 3), 10.0)
    outage = np.zeros((3, 3))
    outage[1, 2] = 5.0

    weights = ga.compute_weight_mec(3, energy, np.array([0.0, 200.0, 50.0]), throughput, outage)

    assert weights[1, 2] > weights[2, 1]
    assert weights[1, 2] > weights[1, 0]


def test_node_payloads_add_zero_payloads_for_start_and_end():
    assert GA.build_node_payloads(np.array([50, 60, 70, 80, 90, 100])).tolist() == [0, 50, 60, 70, 80, 90, 100, 0]


def test_forbidden_edge_with_zero_payload_stays_infinite_not_nan():
    ga = GA.__new__(GA)
    weights = ga.compute_weight_mec(2, np.array([[np.inf, 1.0], [np.inf, np.inf]]), np.array([0.0, 0.0]), np.ones((2, 2)), np.zeros((2, 2)))
    assert np.isinf(weights[1, 0])
    assert not np.isnan(weights[1, 0])


def test_edge_weight_matches_flight_upload_and_outage_time_sum():
    ga = GA.__new__(GA)
    energy = np.full((2, 2), np.inf)
    energy[0, 1] = power(18.2) * 2.0
    throughput = np.full((2, 2), 10.0)
    outage = np.zeros((2, 2)); outage[0, 1] = 3.0
    weights = ga.compute_weight_mec(2, energy, np.array([100.0, 0.0]), throughput, outage)
    assert weights[0, 1] == pytest.approx(15.0)


def test_ga_mutation_preserves_fixed_start_and_end():
    ga = GA.__new__(GA)
    for _ in range(20):
        result = ga.ga_mutate([0, 1, 2, 3, 4, 5, 6, 7])
        assert result[0] == 0
        assert result[-1] == 7


def test_ga_crossover_preserves_fixed_start_and_end():
    ga = GA.__new__(GA)
    for _ in range(20):
        left, right = ga.ga_cross([0, 1, 2, 3, 4, 5, 6, 7], [0, 6, 5, 4, 3, 2, 1, 7])
        assert left[0] == right[0] == 0
        assert left[-1] == right[-1] == 7
