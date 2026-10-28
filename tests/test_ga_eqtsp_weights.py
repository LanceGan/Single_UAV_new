import numpy as np

from GA_EQTSP import GA


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
