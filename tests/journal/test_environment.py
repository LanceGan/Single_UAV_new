from dataclasses import replace

import numpy as np
import pytest

from journal.environment import InspectionEnv
from journal.scenario import load_scenario


class ConstantRadio:
    def __init__(self, rate_mb_s=0.0, outage=False):
        self.rate_mb_s = rate_mb_s
        self.outage = outage

    def a2g_rate_mb_s(self, _xy):
        return self.rate_mb_s

    def g2a_outage(self, _xy):
        return self.outage


def make_env(*, rate=0.0, outage=False, payload=(10.0,), outage_max_s=80.0):
    base = load_scenario("low")
    scenario = replace(base, inspection_xy=base.inspection_xy[:1], payload_mb=np.asarray(payload), outage_max_s=outage_max_s)
    return InspectionEnv(scenario, ConstantRadio(rate, outage), sequence=[0])


def test_upload_ledger_uses_slot_duration():
    env = make_env(rate=4.0, payload=(10.0,))
    env.reset()
    env.mark_payload_collected(0)
    env.step(np.array([0.0, 0.0]))
    assert env.remaining_mb == pytest.approx(8.0)


def test_terminal_leg_requires_final_upload():
    env = make_env(rate=0.0, payload=(10.0,))
    env.reset()
    env.mark_payload_collected(0)
    _, _, terminated, _, info = env.arrive_at_endpoint()
    assert terminated and not info["metrics"]["feasible"]


def test_outage_budget_terminates_infeasible_episode():
    env = make_env(outage=True, outage_max_s=0.5)
    env.reset()
    env.step(np.array([0.0, 0.0]))
    _, _, terminated, _, info = env.step(np.array([0.0, 0.0]))
    assert terminated and info["metrics"]["qos_violation"]
