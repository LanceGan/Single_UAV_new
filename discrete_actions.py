"""Discrete target-relative actions for the Variant-A D3QN baseline."""

import math

import numpy as np


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class DiscreteActionCodec:
    """Map 33 action indices to ``[relative heading, distance]`` actions."""

    HEADING_COUNT = 16
    DISTANCES = (0.125, 0.25)
    ACTION_COUNT = 1 + HEADING_COUNT * len(DISTANCES)

    def __init__(self):
        headings = [_wrap_angle(2.0 * math.pi * index / self.HEADING_COUNT)
                    for index in range(self.HEADING_COUNT)]
        self._actions = np.asarray(
            [[0.0, 0.0]]
            + [[heading, distance] for distance in self.DISTANCES for heading in headings],
            dtype=np.float64,
        )

    @property
    def actions(self):
        return self._actions.copy()

    def __len__(self):
        return self.ACTION_COUNT

    def decode(self, action_index):
        action_index = int(action_index)
        if action_index < 0 or action_index >= len(self):
            raise IndexError(f"action index {action_index} is outside [0, {len(self)})")
        return self._actions[action_index].copy()

    def encode(self, action):
        """Snap a continuous ``[relative heading, distance]`` action to the nearest index.

        Distances below half of the smallest step mean no discrete move is a
        better match than hovering; headings snap on the circle, so pi and -pi
        resolve to the same action.
        """
        heading, distance = float(action[0]), float(action[1])
        if distance < self.DISTANCES[0] / 2.0:
            return 0
        distance_index = min(
            range(len(self.DISTANCES)), key=lambda index: abs(distance - self.DISTANCES[index])
        )
        step = 2.0 * math.pi / self.HEADING_COUNT
        heading_index = int(round(_wrap_angle(heading) / step)) % self.HEADING_COUNT
        return 1 + distance_index * self.HEADING_COUNT + heading_index

    def to_absolute(self, action_index, target_bearing):
        """Return an absolute heading for diagnostics; World Variant A decodes it itself."""
        heading, distance = self.decode(action_index)
        return np.asarray([_wrap_angle(float(target_bearing) + heading), distance], dtype=float)
