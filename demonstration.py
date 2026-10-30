"""Demonstration-assisted actor updates shared by TD3 and DDPG."""

import math

import torch


def linear_decay(initial, episode, horizon):
    """Value that falls linearly from ``initial`` to zero over ``horizon`` episodes."""
    if horizon <= 0:
        return 0.0
    return initial * max(0.0, 1.0 - episode / horizon)


def behavior_cloning_loss(pred, demo_action, demo_mask, max_distance):
    """Mean imitation error over demonstrated samples only.

    Heading error uses 1 - cos so that -pi and pi are the same direction;
    distance error is normalized by the action bound to share the same scale.
    """
    count = demo_mask.sum()
    if count.item() == 0:
        return pred.sum() * 0.0
    heading = 1.0 - torch.cos(pred[:, :1] - demo_action[:, :1])
    distance = ((pred[:, 1:2] - demo_action[:, 1:2]) / max_distance) ** 2
    return ((heading + distance) * demo_mask).sum() / count
