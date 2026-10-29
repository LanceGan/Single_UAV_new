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
    assert state[4] == pytest.approx(51.0 / 106.0)


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


def test_world_accumulates_reward_terms_within_episode():
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    world.reset()
    world.get_reward(0.0, 0.0, 1.0, 1.0, 0.1, 10.0, 10.0)
    world.get_reward(0.0, 0.0, 1.0, 1.0, 0.1, 10.0, 10.0)

    assert world.reward_term_totals["progress"] == pytest.approx(2.0)


def test_reward_summary_returns_a_copy_of_episode_totals():
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    world.reset()
    summary = world.reward_summary()
    summary["progress"] = 99.0
    assert world.reward_term_totals["progress"] == 0.0


def test_arrival_with_pending_upload_does_not_advance_to_next_target(monkeypatch):
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(world, "get_date_rate_A2G", lambda _location: -100.0)
    monkeypatch.setattr(world, "get_date_rate_G2A", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_g2a_outage", lambda _location: False)
    world.reset(); world.distance = 100.0; world.transmit = False; world.data_size = 10.0

    _, _, done, _, terminal = world.step_inside(np.array([0.0, 0.0]), np.zeros(5), 0)
    assert done and not terminal
    assert world.Traverse_order == 0
    assert world.data_size > 0


def test_sparse_reward_terms_are_accumulated():
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    world.reset()
    world.add_reward_term("arrival_upload", -30.0)
    assert world.reward_summary()["arrival_upload"] == pytest.approx(-30.0)


def test_idle_penalty_is_bounded_when_no_payload_is_pending():
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    world.reset()
    world.transmit = True
    world.get_reward(0.0, 0.0, 1.0, 1.0, 0.0, 10.0, 10.0)
    assert world.last_reward_terms["idle"] == pytest.approx(-0.1)


def test_progress_reward_penalizes_motion_away_from_target():
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    world.get_reward(0.0, 0.0, 1.1, 1.0, -0.1, 10.0, 10.0)
    assert world.last_reward_terms["progress"] == pytest.approx(-1.0)


def test_uploaded_data_receives_bounded_immediate_reward(monkeypatch):
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(world, "get_date_rate_A2G", lambda _location: 100.0)
    monkeypatch.setattr(world, "get_date_rate_G2A", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_g2a_outage_probability", lambda _location: 0.0)
    state, _ = world.reset()
    world.transmit = False
    world.data_size = 1.0
    world.step_inside(np.array([0.0, 0.0]), state, 0)
    assert world.last_uploaded_data == pytest.approx(1.0)
    assert world.reward_summary()["upload"] == pytest.approx(world.UPLOAD_REWARD_PER_MB)


def test_world_defaults_to_fixed_maps_and_100_second_qos_budget():
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    assert world.outage_max_s == 100.0
    assert world.get_date_rate_A2G(np.array([[0.0, 0.0]])) == world.radio_maps.a2g_snr_db(np.array([0.0, 0.0]))


def test_state_exposes_qos_usage_and_penalizes_each_outage_slot(monkeypatch):
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(world, "get_date_rate_A2G", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_date_rate_G2A", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_g2a_outage_probability", lambda _location: 0.8)
    monkeypatch.setattr(world, "get_g2a_outage", lambda _location: True)
    state, _ = world.reset()
    next_state, _, _, _, _ = world.step_inside(np.array([0.1, 0.1]), state, 0)
    assert len(next_state) == 8
    assert next_state[5] == pytest.approx(0.5 / 100.0)
    expected_snr = (10.0 - world.a2g_snr_min) / (world.a2g_snr_max - world.a2g_snr_min)
    assert next_state[6] == pytest.approx(expected_snr)
    assert next_state[7] == pytest.approx(0.8)
    assert np.all((next_state >= 0.0) & (next_state <= 1.0))
    assert world.last_reward_terms["communication"] == pytest.approx(-5.0)


def test_final_payload_upload_allows_terminal_completion(monkeypatch):
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  data_sizes=np.array([1.0, 60, 60, 60, 60, 60]),
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1])
    monkeypatch.setattr(world, "get_date_rate_A2G", lambda _location: 100.0)
    monkeypatch.setattr(world, "get_date_rate_G2A", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_g2a_outage", lambda _location: False)
    state, _ = world.reset(); world.distance = 100.0
    state, _, done, t, terminal = world.step_inside(np.array([0.0, 0.0]), state, 0)
    assert not done and not terminal and world.Traverse_order == 1
    _, _, done, _, terminal = world.step_inside(np.array([0.0, 0.0]), state, t)
    assert done and terminal


def test_reward_scales_make_qos_violation_more_costly_than_two_arrivals():
    world = World(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    assert 2 * world.ARRIVAL_REWARD < world.QOS_TERMINATION_PENALTY + 200 * world.OUTAGE_STEP_PENALTY
