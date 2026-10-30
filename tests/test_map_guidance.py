from types import SimpleNamespace

import numpy as np

from map_guidance import MapGuide


def test_pending_upload_can_escape_target_protection_ring():
    """A source inside the guard ring must not be removed from its own route graph."""
    shape = (21, 21)
    world = SimpleNamespace(
        radio_maps=SimpleNamespace(
            a2g_snr=np.zeros(shape),
            g2a_outage=np.zeros(shape),
        ),
        BandWidth=1.0,
        dist_max=0.25,
        delta_T=0.5,
        target_loc=np.array([1.5, 1.0]),
        distance=0.5,
        transmit=False,
        data_size=10.0,
        Traverse_order=1,
        UAVs=[SimpleNamespace(x=0.9, y=1.0)],
        length=2.0,
        width=2.0,
    )

    action = MapGuide(world).action()

    assert np.all(np.isfinite(action))
    assert 0.0 < action[1] <= world.dist_max
    endpoint = np.array([world.UAVs[0].x, world.UAVs[0].y]) + action[1] * np.array(
        [np.cos(action[0]), np.sin(action[0])]
    )
    assert np.linalg.norm(endpoint - world.target_loc) > np.linalg.norm(
        np.array([world.UAVs[0].x, world.UAVs[0].y]) - world.target_loc
    )
