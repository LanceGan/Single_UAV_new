from reward_logging import log_reward_terms


class Writer:
    def __init__(self):
        self.calls = []

    def add_scalar(self, name, value, step):
        self.calls.append((name, value, step))


def test_log_reward_terms_writes_each_named_term_and_total():
    writer = Writer()
    log_reward_terms(writer, {"progress": 20.0, "idle": -3.0}, episode=4)

    assert ("reward_terms/progress", 20.0, 4) in writer.calls
    assert ("reward_terms/idle", -3.0, 4) in writer.calls
    assert ("reward_terms/total", 17.0, 4) in writer.calls
