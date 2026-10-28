import numpy as np

from training_config import load_fixed_payloads


def test_training_config_loads_each_fixed_payload_level():
    assert np.all((load_fixed_payloads("low") >= 50) & (load_fixed_payloads("low") < 100))
    assert np.all((load_fixed_payloads("medium") >= 100) & (load_fixed_payloads("medium") < 200))
    assert np.all((load_fixed_payloads("high") >= 200) & (load_fixed_payloads("high") <= 300))
