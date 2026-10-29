from training_transitions import replay_terminal_mask


def test_all_episode_endings_are_replay_terminals():
    assert replay_terminal_mask(done=True) is True  # QoS, timeout, upload failure, or success
    assert replay_terminal_mask(done=False) is False
