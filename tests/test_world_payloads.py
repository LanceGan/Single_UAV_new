import numpy as np
import pytest

from World import World


def test_world_uses_per_inspection_payload_vector():
    payloads = np.array([51.0, 62.0, 73.0, 84.0, 95.0, 106.0])
    world = World(
        length=20,
        width=20,
        uav_num=1,
        dist_max=0.25,
        delta_t=0.5,
        t=100,
        data_size=300,
        data_sizes=payloads,
        ini_loc=[2.62, 2.65],
        end_loc=[18.95, 11.12],
        users_name="results/datas/Users_6.txt",
        BS_loc=np.array([[4.77, 4.56, 0.25], [14.89, 4.89, 0.25], [4.63, 13.78, 0.25], [13.48, 14.89, 0.25]]),
        traverse_sequence=[1, 2, 3, 4, 5, 6],
    )

    assert world.payload_for_user(1) == 51.0
    assert world.payload_for_user(6) == 106.0


def test_arrival_loads_payload_for_departed_inspection_point(monkeypatch):
    payloads = np.array([51.0, 62.0, 73.0, 84.0, 95.0, 106.0])
    world = World(
        length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
        data_sizes=payloads, ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
        users_name="results/datas/Users_6.txt",
        BS_loc=np.zeros((4, 3)), traverse_sequence=[1, 2, 3, 4, 5, 6],
    )
    monkeypatch.setattr(world, "get_date_rate_A2G", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_date_rate_G2A", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_g2a_outage", lambda _location: False)
    world.reset()
    world.distance = 100.0

    state, _, _, t, _ = world.step_inside(np.array([0.0, 0.0]), np.zeros(5), 0)

    assert t == 1
    assert world.data_size == 51.0
    assert world.transmit is False
    assert state[4] == 51.0


def test_upload_uses_data_rate_times_slot_duration(monkeypatch):
    payloads = np.array([51.0, 62.0, 73.0, 84.0, 95.0, 106.0])
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  data_sizes=payloads, ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(world, "get_date_rate_A2G", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_date_rate_G2A", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_g2a_outage", lambda _location: False)
    world.reset(); world.distance = 100.0
    _, _, _, t, _ = world.step_inside(np.array([0.0, 0.0]), np.zeros(5), 0)
    world.distance = 0.1
    world.step_inside(np.array([0.0, 0.0]), np.zeros(5), t)

    rate = world.BandWidth * np.log2(1 + 10 ** (10.0 / 10.0))
    assert world.data_size == pytest.approx(51.0 - rate * world.delta_T)


def test_world_terminates_when_g2a_outage_budget_is_exceeded(monkeypatch):
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  outage_max_s=0.5, ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(world, "get_date_rate_A2G", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_g2a_outage", lambda _location: True)
    world.reset()
    _, _, done, t, terminal = world.step_inside(np.array([0.0, 0.0]), np.zeros(5), 0)
    assert not done and not terminal and world.out_time == 0.5
    _, _, done, _, terminal = world.step_inside(np.array([0.0, 0.0]), np.zeros(5), t)
    assert done and not terminal and world.out_time == 1.0


def test_world_records_named_reward_terms():
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    reward = world.get_reward(0.0, 0.0, 1.0, 1.0, 0.1, 10.0, 10.0)

    assert reward == pytest.approx(sum(world.last_reward_terms.values()))
    assert set(world.last_reward_terms) == {"progress", "idle", "boundary", "communication"}
