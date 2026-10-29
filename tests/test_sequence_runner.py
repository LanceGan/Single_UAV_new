import numpy as np

from sequence_runner import node_payloads_for_tsp, load_planning_nodes


def test_sequence_runner_adds_start_and_end_zero_payloads():
    payloads = node_payloads_for_tsp(np.array([50, 60, 70, 80, 90, 100]))
    assert payloads.tolist() == [0, 50, 60, 70, 80, 90, 100, 0]


def test_planner_uses_world_inspection_coordinates():
    nodes = load_planning_nodes()
    actual = np.loadtxt("results/datas/Users_6.txt")[:, :2] / 10.0
    np.testing.assert_allclose(nodes[1:-1], actual)
