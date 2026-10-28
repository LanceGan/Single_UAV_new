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
