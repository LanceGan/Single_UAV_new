import math

import numpy as np
import pytest
import torch
from torch import nn

from D3QN import (
    D3QNAgent,
    DiscreteReplayBuffer,
    DuelingQNetwork,
    double_dqn_targets,
    dqfd_margin_loss,
)
from discrete_actions import DiscreteActionCodec


def test_action_codec_has_hover_and_32_unique_bounded_moves():
    codec = DiscreteActionCodec()
    actions = codec.actions

    assert actions.shape == (33, 2)
    assert len({tuple(action) for action in actions}) == 33
    np.testing.assert_allclose(actions[0], [0.0, 0.0])
    assert np.all(actions[:, 0] >= -math.pi)
    assert np.all(actions[:, 0] <= math.pi)
    assert set(np.round(actions[1:, 1], 3)) == {0.125, 0.25}
    assert np.count_nonzero(np.isclose(actions[:, 1], 0.125)) == 16
    assert np.count_nonzero(np.isclose(actions[:, 1], 0.25)) == 16


def test_action_headings_are_relative_to_target_bearing():
    codec = DiscreteActionCodec()

    np.testing.assert_allclose(codec.decode(1), [0.0, 0.125], atol=1e-8)
    np.testing.assert_allclose(codec.decode(2), [math.pi / 8.0, 0.125], atol=1e-8)
    np.testing.assert_allclose(codec.to_absolute(1, math.pi / 3.0), [math.pi / 3.0, 0.125])
    np.testing.assert_allclose(codec.to_absolute(9, math.pi / 3.0), [-2.0 * math.pi / 3.0, 0.125])
    with pytest.raises(IndexError):
        codec.decode(33)


def test_encode_round_trips_and_snaps_to_nearest_action():
    codec = DiscreteActionCodec()

    for index in range(len(codec)):
        assert codec.encode(codec.decode(index)) == index
    # Below half of the smallest step the only faithful discrete action is hovering.
    assert codec.encode([1.3, 0.05]) == 0
    assert codec.encode([0.0, 0.0625]) == 1
    # Nearest sixteenth of a turn, including both signs of the pi seam.
    assert codec.encode([math.pi / 8.0 + 0.02, 0.125]) == 2
    assert codec.encode([math.pi, 0.125]) == 9
    assert codec.encode([-math.pi, 0.125]) == 9
    # Nearest legal step length, clipping anything past the 0.25 bound.
    assert codec.encode([0.0, 0.2]) == 17
    assert codec.encode([0.0, 1.0]) == 17


def test_replay_demo_partition_mixes_fraction_and_survives_online_wraparound():
    replay = DiscreteReplayBuffer(
        state_dim=2, capacity=8, seed=0, demo_capacity=4, demo_fraction=0.25
    )
    for _ in range(4):
        replay.add(np.zeros(2), 7, 1.0, np.zeros(2), False, demo=True)
    for _ in range(20):  # wraps the online region without touching demonstrations
        replay.add(np.ones(2), 1, 0.0, np.ones(2), False)

    assert len(replay) == 8 + 4
    _, actions, _, _, _, demo_mask = replay.sample(8, "cpu")
    demo_rows = demo_mask.bool().squeeze(1)
    assert demo_mask.sum().item() == 2  # round(8 * 0.25)
    assert set(actions[demo_rows].squeeze(1).tolist()) == {7}
    assert set(actions[~demo_rows].squeeze(1).tolist()) == {1}


def test_replay_serves_demonstrations_alone_before_any_online_transition():
    replay = DiscreteReplayBuffer(
        state_dim=2, capacity=8, seed=0, demo_capacity=4, demo_fraction=0.25
    )
    replay.add(np.zeros(2), 3, 1.0, np.zeros(2), True, demo=True)

    _, actions, _, _, _, demo_mask = replay.sample(4, "cpu")

    assert demo_mask.sum().item() == 4
    assert set(actions.squeeze(1).tolist()) == {3}


def test_dqfd_margin_loss_penalizes_only_demos_that_lack_the_margin():
    q_values = torch.tensor([[5.0, 1.0, 0.0], [0.0, 2.0, 1.5]])
    actions = torch.tensor([[0], [1]])

    # First demo clears the margin (5 >= 2); the second is short by 0.5 (2.5 - 2).
    both = dqfd_margin_loss(q_values, actions, torch.ones((2, 1)), margin=1.0)
    first_only = dqfd_margin_loss(
        q_values, actions, torch.tensor([[1.0], [0.0]]), margin=1.0
    )
    neither = dqfd_margin_loss(q_values, actions, torch.zeros((2, 1)), margin=1.0)

    assert both.item() == pytest.approx(0.25)
    assert first_only.item() == pytest.approx(0.0)
    assert neither.item() == pytest.approx(0.0)


def test_learn_adds_weighted_margin_term_for_demonstrations():
    rng = np.random.default_rng(3)
    replay = DiscreteReplayBuffer(
        state_dim=8, capacity=64, seed=3, demo_capacity=64, demo_fraction=0.5
    )
    for index in range(32):
        replay.add(
            rng.normal(size=8),
            index % 33,
            float(rng.normal()),
            rng.normal(size=8),
            index % 7 == 0,
            demo=index % 2 == 0,
        )
    agent = D3QNAgent(
        8, 33, hidden_dim=32, batch_size=16, seed=5, device="cpu", margin=1.0
    )

    agent.bc_weight = 0.7
    weighted = agent.learn(replay)
    agent.bc_weight = 0.0
    unweighted = agent.learn(replay)

    assert weighted["margin_loss"] > 0.0
    assert weighted["loss"] == pytest.approx(
        weighted["td_loss"] + 0.7 * weighted["margin_loss"], rel=1e-5
    )
    assert unweighted["loss"] == pytest.approx(unweighted["td_loss"], rel=1e-6)


def test_dueling_aggregation_centers_advantage_for_each_sample():
    values = torch.tensor([[2.0], [-4.0]])
    advantages = torch.tensor([[1.0, 3.0, 8.0], [6.0, 3.0, 0.0]])

    q_values = DuelingQNetwork.aggregate(values, advantages)

    torch.testing.assert_close(q_values, torch.tensor([[-1.0, 1.0, 6.0], [-1.0, -4.0, -7.0]]))
    torch.testing.assert_close(q_values.mean(dim=1, keepdim=True), values)


class _FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values[: states.shape[0]]


def test_double_dqn_uses_online_argmax_and_target_value_with_done_mask():
    online = _FixedQ([[1.0, 9.0, 3.0], [8.0, 2.0, 1.0]])
    target = _FixedQ([[100.0, 5.0, 50.0], [7.0, 60.0, 80.0]])
    rewards = torch.tensor([[2.0], [4.0]])
    dones = torch.tensor([[0.0], [1.0]])

    result = double_dqn_targets(
        rewards, dones, torch.zeros((2, 8)), online, target, gamma=0.5
    )

    # Online selects indices 1 and 0. Target evaluates those as 5 and 7;
    # the second bootstrap is removed because that transition ended an episode.
    torch.testing.assert_close(result, torch.tensor([[4.5], [4.0]]))
    assert not result.requires_grad


def test_agent_learning_updates_online_network_and_clips_gradients():
    rng = np.random.default_rng(7)
    replay = DiscreteReplayBuffer(state_dim=8, capacity=64, seed=3)
    for index in range(64):
        replay.add(
            rng.normal(size=8),
            index % 33,
            float(rng.normal()),
            rng.normal(size=8),
            index % 11 == 0,
        )
    agent = D3QNAgent(
        state_dim=8,
        action_dim=33,
        hidden_dim=32,
        batch_size=16,
        target_update_interval=2,
        gradient_clip=0.25,
        seed=5,
        device="cpu",
    )
    before = [parameter.detach().clone() for parameter in agent.online_network.parameters()]

    first = agent.learn(replay)
    second = agent.learn(replay)

    assert np.isfinite(first["loss"]) and np.isfinite(second["loss"])
    assert second["gradient_norm"] <= 0.25 + 1e-6
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.online_network.parameters())
    )
    assert all(parameter.grad is None for parameter in agent.target_network.parameters())
    for online_parameter, target_parameter in zip(
        agent.online_network.parameters(), agent.target_network.parameters()
    ):
        torch.testing.assert_close(online_parameter, target_parameter)


def test_epsilon_zero_is_deterministic_and_epsilon_one_explores():
    agent = D3QNAgent(8, 33, hidden_dim=16, seed=9, device="cpu")
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        agent.online_network.advantage[-1].bias.copy_(torch.arange(33, dtype=torch.float32))

    assert agent.select_action(np.zeros(8), epsilon=0.0) == 32
    sampled = {agent.select_action(np.zeros(8), epsilon=1.0) for _ in range(100)}
    assert sampled <= set(range(33))
    assert len(sampled) > 1
