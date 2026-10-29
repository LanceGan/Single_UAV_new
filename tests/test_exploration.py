import math

import numpy as np

from exploration import guided_action, exploration_scales, upload_guided_action


def test_exploration_scales_decay_and_switch_to_precise_control_near_target():
    early_angle, early_distance = exploration_scales(episode=1, total_episodes=1000, target_distance=10.0)
    late_angle, late_distance = exploration_scales(episode=1000, total_episodes=1000, target_distance=10.0)
    near_angle, near_distance = exploration_scales(episode=1, total_episodes=1000, target_distance=0.5)
    assert early_angle > late_angle >= 0.05
    assert early_distance > late_distance >= 0.01
    assert near_angle <= 0.05
    assert near_distance <= 0.01


def test_guided_action_points_toward_target_and_has_legal_distance():
    action = guided_action(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0, 0.25, np.random.default_rng(1))
    assert action[0] == 0.0
    assert action[1] == 0.25


def test_upload_guided_action_selects_high_rate_low_outage_direction():
    def score(location):
        return location[0]
    action = upload_guided_action(np.array([0.0, 0.0]), 0.0, 0.25, score, np.random.default_rng(1))
    assert action[0] == 0.0
    assert action[1] == 0.25
