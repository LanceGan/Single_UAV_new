import numpy as np
import pytest

from mission_preflight import run_preflight


@pytest.mark.parametrize("level,total", [("low", 450.0), ("medium", 900.0), ("high", 1500.0)])
def test_full_mission_completes_in_real_world(level, total):
    result = run_preflight(level)
    assert result["success"], result
    assert result["visited_points"] == 6
    assert result["uploaded_mb"] == total
    assert result["outage_s"] <= 100
    assert result["time_s"] <= 500
    assert result["reward_accounting_error"] < 1e-7
    assert result["max_pending_data_at_visit"] == 0.0
    assert np.isfinite(result["return"])
