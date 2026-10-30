import numpy as np

from training_config import build_output_paths, load_fixed_payloads


def test_training_config_loads_each_fixed_payload_level():
    assert np.all((load_fixed_payloads("low") >= 50) & (load_fixed_payloads("low") < 100))
    assert np.all((load_fixed_payloads("medium") >= 100) & (load_fixed_payloads("medium") < 200))
    assert np.all((load_fixed_payloads("high") >= 200) & (load_fixed_payloads("high") <= 300))


def test_each_level_has_three_high_and_three_low_payloads():
    for level, high_value, low_value in (("low", 90.0, 60.0), ("medium", 180.0, 120.0), ("high", 290.0, 210.0)):
        payloads = load_fixed_payloads(level)
        assert np.count_nonzero(payloads == high_value) == 3
        assert np.count_nonzero(payloads == low_value) == 3


def test_training_outputs_are_isolated_by_data_level(tmp_path):
    paths = build_output_paths(tmp_path, "low", "ddpg")
    assert paths["root"] == tmp_path / "low" / "ddpg"
    assert paths["model_prefix"].parent == tmp_path / "low" / "ddpg" / "models"


def test_algorithm_profiles_separate_standard_and_demonstration_variants():
    import pytest
    from training_config import resolve_algorithm

    assert resolve_algorithm("td3", None, None, None, None) == {
        "base": "td3", "guide_prob": 0.0, "bc_weight": 0.0,
        "bc_min_weight": 0.0, "demo_fraction": 0.0}
    assert resolve_algorithm("ddpg", None, None)["guide_prob"] == 0.0
    assert resolve_algorithm("td3_bc", None, None, None, None) == {
        "base": "td3", "guide_prob": 0.5, "bc_weight": 1.0,
        "bc_min_weight": 0.25, "demo_fraction": 0.25}
    assert resolve_algorithm("ddpg_bc", 0.3, 2.0, 0.2, 0.4) == {
        "base": "ddpg", "guide_prob": 0.3, "bc_weight": 2.0,
        "bc_min_weight": 0.2, "demo_fraction": 0.4}
    with pytest.raises(ValueError, match="td3_bc"):
        resolve_algorithm("td3", 0.5, None, None, None)
    with pytest.raises(ValueError, match="ddpg_bc"):
        resolve_algorithm("ddpg", None, 1.0, None, None)
    with pytest.raises(ValueError, match="td3_bc"):
        resolve_algorithm("td3", None, None, 0.25, None)
