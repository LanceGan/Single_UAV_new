import math

import numpy as np
import torch

from PPO import PPOAgent, generalized_advantage_estimate
from Train_PPO import create_parser


def _agent():
    return PPOAgent(
        state_dim=8,
        action_dim=2,
        max_action=np.array([math.pi, 0.25], dtype=np.float32),
        net_width=32,
        actor_lr=3e-4,
        critic_lr=1e-3,
        seed=3,
        device="cpu",
    )


def test_ppo_actions_are_bounded_and_log_probs_are_finite():
    agent = _agent()
    state = np.zeros(8, dtype=np.float32)
    deterministic = agent.deterministic_action(state)
    sampled, log_prob, value = agent.sample_action(state)

    for action in (deterministic, sampled):
        assert -math.pi <= action[0] <= math.pi
        assert 0.0 <= action[1] <= 0.25
    assert np.isfinite(log_prob)
    assert np.isfinite(value)


def test_ppo_update_changes_actor_and_returns_finite_losses():
    agent = _agent()
    rng = np.random.default_rng(1)
    states = rng.random((32, 8)).astype(np.float32)
    actions = np.column_stack((
        rng.uniform(-math.pi, math.pi, 32),
        rng.uniform(0.0, 0.25, 32),
    )).astype(np.float32)
    old_log_probs = rng.normal(size=32).astype(np.float32)
    returns = rng.normal(size=32).astype(np.float32)
    advantages = rng.normal(size=32).astype(np.float32)
    before = [parameter.detach().clone() for parameter in agent.actor.parameters()]

    metrics = agent.update(
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        returns=returns,
        advantages=advantages,
        epochs=2,
        batch_size=16,
    )

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.actor.parameters())
    )
    assert all(np.isfinite(value) for value in metrics.values())


def test_generalized_advantage_matches_single_step_case():
    returns, advantages = generalized_advantage_estimate(
        rewards=[1.0],
        values=[0.5],
        next_values=[0.0],
        dones=[1.0],
        gamma=0.995,
        lam=0.95,
    )
    np.testing.assert_allclose(advantages, [0.5])
    np.testing.assert_allclose(returns, [1.0])


def test_ppo_parser_defaults_match_fixed_training_budget():
    args = create_parser().parse_args([])
    assert args.episodes == 3000
    assert args.outage_max_s == 6.0
    assert args.sequence_method == "iga"
    assert args.teacher == "mapguide"
    assert args.pretrain_updates == 5000
    assert args.learning_rate == 3e-4
    assert args.gamma == 0.995
