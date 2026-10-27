import numpy as np

from tests.journal.test_environment import make_env


def test_step_reward_equals_named_reward_terms():
    env = make_env(rate=1.0)
    env.reset()
    _, reward, _, _, info = env.step(np.array([0.0, 0.1]))

    assert reward == sum(info["reward_terms"].values())
    assert set(info["reward_terms"]) == {
        "progress", "idle", "boundary", "arrival_upload", "completion", "timeout", "qos"
    }
