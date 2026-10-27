"""Common fixed-task environment with explicit upload and QoS ledgers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from journal.scenario import Scenario


@dataclass
class Metrics:
    feasible: bool = True
    qos_violation: bool = False
    outage_s: float = 0.0
    transmission_delay_s: float = 0.0
    flight_distance: float = 0.0


class InspectionEnv:
    action_bounds = (float(-np.pi), 0.25)

    def __init__(self, scenario: Scenario, radio, sequence: list[int]) -> None:
        self.scenario, self.radio, self.sequence = scenario, radio, sequence
        self.action_bounds = (-float(np.pi), scenario.speed_max * scenario.delta_t)
        self.reset()

    def reset(self):
        self.position = np.asarray(self.scenario.start_xy, dtype=float)
        self.remaining_mb = 0.0
        self.current_payload_index: int | None = None
        self.metrics = Metrics()
        self.done = False
        self.reward_totals = {name: 0.0 for name in ("progress", "idle", "boundary", "arrival_upload", "completion", "timeout", "qos")}
        return self._state(), {"metrics": self.metrics.__dict__.copy()}

    def _state(self) -> np.ndarray:
        return np.array([*self.position, self.remaining_mb, self.metrics.outage_s], dtype=float)

    def mark_payload_collected(self, inspection_index: int) -> None:
        if self.current_payload_index is not None or self.remaining_mb > 0:
            raise RuntimeError("previous payload has not been uploaded")
        self.current_payload_index = inspection_index
        self.remaining_mb = float(self.scenario.payload_mb[inspection_index])
        self.metrics.transmission_delay_s = 0.0

    def _info(self):
        return {"metrics": self.metrics.__dict__.copy(), "reward_terms": self.reward_totals.copy()}

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("episode is already terminated")
        heading, distance = np.asarray(action, dtype=float)
        distance = float(np.clip(distance, 0.0, self.action_bounds[1]))
        self.position += distance * np.array([np.cos(heading), np.sin(heading)])
        self.metrics.flight_distance += distance
        terms = {name: 0.0 for name in self.reward_totals}
        terms["progress"] = distance / self.action_bounds[1] if self.action_bounds[1] else 0.0
        if distance == 0.0:
            terms["idle"] = -0.05
        if self.remaining_mb > 0:
            self.remaining_mb = max(0.0, self.remaining_mb - self.radio.a2g_rate_mb_s(self.position) * self.scenario.delta_t)
            self.metrics.transmission_delay_s += self.scenario.delta_t
        if self.radio.g2a_outage(self.position):
            self.metrics.outage_s += self.scenario.delta_t
        if self.metrics.outage_s > self.scenario.outage_max_s:
            self.metrics.feasible = False
            self.metrics.qos_violation = True
            terms["qos"] = -1.0
            self.done = True
        for name, value in terms.items():
            self.reward_totals[name] += value
        return self._state(), float(sum(terms.values())), self.done, False, self._info()

    def arrive_at_endpoint(self):
        terms = {name: 0.0 for name in self.reward_totals}
        if self.remaining_mb > 0:
            self.metrics.feasible = False
            terms["arrival_upload"] = -1.0
        else:
            terms["completion"] = 1.0
        self.done = True
        for name, value in terms.items():
            self.reward_totals[name] += value
        return self._state(), float(sum(terms.values())), True, False, self._info()
