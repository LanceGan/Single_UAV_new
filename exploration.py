"""Goal-directed, action-specific exploration for UAV navigation."""

import math

import numpy as np


def exploration_scales(episode, total_episodes, target_distance):
    fraction = min(max(episode / max(0.6 * total_episodes, 1), 0.0), 1.0)
    angle_noise = max(0.05, 0.3 + (0.05 - 0.3) * fraction)
    distance_noise = max(0.01, 0.05 + (0.01 - 0.05) * fraction)
    if target_distance <= 1.0:
        angle_noise = min(angle_noise, 0.05)
        distance_noise = min(distance_noise, 0.01)
    return angle_noise, distance_noise


def guided_action(current_pos, target_pos, angle_noise, max_distance, rng):
    delta = np.asarray(target_pos, dtype=float) - np.asarray(current_pos, dtype=float)
    heading = math.atan2(delta[1], delta[0]) + rng.normal(0.0, angle_noise)
    return np.array([np.clip(heading, -math.pi, math.pi), max_distance])


def upload_guided_action(current_pos, angle_noise, max_distance, location_score, rng):
    current_pos = np.asarray(current_pos, dtype=float)
    headings = np.linspace(-math.pi, math.pi, 16, endpoint=False)
    scores = [location_score(current_pos + max_distance * np.array([math.cos(h), math.sin(h)])) for h in headings]
    heading = headings[int(np.argmax(scores))] + rng.normal(0.0, angle_noise)
    return np.array([np.clip(heading, -math.pi, math.pi), max_distance])
