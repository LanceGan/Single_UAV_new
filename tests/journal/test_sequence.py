import math
from dataclasses import replace

from journal.sequence import EdgeEstimator
from journal.scenario import load_scenario


class ConstantRadio:
    def __init__(self, rate, outage=False):
        self.rate, self.outage = rate, outage

    def a2g_rate_mb_s(self, _xy): return self.rate
    def g2a_outage(self, _xy): return self.outage


def test_larger_source_payload_increases_same_edge_cost():
    estimate = EdgeEstimator(load_scenario("low"), ConstantRadio(rate=10)).estimate
    assert estimate(0, 1, 250).cost > estimate(0, 1, 50).cost


def test_qos_infeasible_edge_is_rejected():
    scenario = replace(load_scenario("low"), outage_max_s=1.0)
    edge = EdgeEstimator(scenario, ConstantRadio(rate=10, outage=True)).estimate(0, 1, 100)
    assert not edge.feasible and math.isinf(edge.cost)
