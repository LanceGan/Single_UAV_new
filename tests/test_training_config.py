import numpy as np

from training_config import build_output_paths, load_fixed_payloads


def test_training_config_loads_each_fixed_payload_level():
    assert np.all((load_fixed_payloads("low") >= 50) & (load_fixed_payloads("low") < 100))
    assert np.all((load_fixed_payloads("medium") >= 100) & (load_fixed_payloads("medium") < 200))
    assert np.all((load_fixed_payloads("high") >= 200) & (load_fixed_payloads("high") <= 300))


def test_training_outputs_are_isolated_by_data_level(tmp_path):
    paths = build_output_paths(tmp_path, "low")
    assert paths["root"] == tmp_path / "low"
    assert paths["model_prefix"].parent == tmp_path / "low" / "models"
