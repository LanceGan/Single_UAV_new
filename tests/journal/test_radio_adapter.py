import numpy as np
import pytest

from journal.radio_adapter import RadioAdapter
from journal.scenario import load_scenario


@pytest.fixture
def adapter():
    return RadioAdapter(load_scenario("low"))


def test_adapter_uses_legacy_module_parameters(adapter):
    assert adapter.a2g_module.PB == 0.1
    assert adapter.g2a_module.PB == 0.1


def test_seeded_queries_are_repeatable(adapter):
    point = np.array([2.62, 2.65])
    adapter.seed(7)
    first = adapter.a2g_rate_mb_s(point)
    adapter.seed(7)
    second = adapter.a2g_rate_mb_s(point)

    assert first == pytest.approx(second)


def test_g2a_outage_returns_boolean(adapter):
    assert isinstance(adapter.g2a_outage(np.array([2.62, 2.65])), bool)
