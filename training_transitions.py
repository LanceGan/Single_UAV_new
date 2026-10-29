"""Replay-buffer transition semantics shared by TD3 and DDPG training."""


def replay_terminal_mask(done):
    """Every completed episode is terminal for critic bootstrapping."""
    return bool(done)
