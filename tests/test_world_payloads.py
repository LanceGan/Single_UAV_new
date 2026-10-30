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
    world.add_reward_term("arrival", -30.0)
    assert world.reward_summary()["arrival"] == pytest.approx(-30.0)


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
    assert np.all((next_state[2:4] >= -1.0) & (next_state[2:4] <= 1.0))
    assert np.all((next_state[[0, 1, 4, 5, 6, 7]] >= 0.0) & (next_state[[0, 1, 4, 5, 6, 7]] <= 1.0))
    assert "communication" not in world.reward_summary()


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


def test_endpoint_cannot_report_success_after_exceeding_qos(monkeypatch):
    world = World(length=20, width=20, uav_num=1, delta_t=0.5, t=1000,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1])
    state, _ = world.reset()
    world.Traverse_order = 1
    world.target_loc = np.array(world.end_loc)
    world.UAVs[0].x, world.UAVs[0].y = world.end_loc
    world.out_time = world.outage_max_s
    monkeypatch.setattr(world, "get_g2a_outage_probability", lambda _: 1.0)
    _, _, done, _, success = world.step_inside(np.zeros(2), state, 0)
    assert done and not success
    assert world.reward_summary()["completion"] == 0.0


def test_reset_clears_previous_success_flag():
    world = World(users_name="results/datas/Users_6.txt", traverse_sequence=[1], BS_loc=np.zeros((4, 3)))
    world.terminal = True
    world.reset()
    assert not world.terminal


def _world(**overrides):
    kwargs = dict(length=20, width=20, uav_num=1, dist_max=0.25, delta_t=0.5, t=100,
                  ini_loc=[2.62, 2.65], end_loc=[18.95, 11.12],
                  users_name="results/datas/Users_6.txt", BS_loc=np.zeros((4, 3)),
                  traverse_sequence=[1, 2, 3, 4, 5, 6])
    kwargs.update(overrides)
    return World(**kwargs)


def _constant_radio(monkeypatch, world, snr_db=10.0, outage_probability=0.0):
    monkeypatch.setattr(world, "get_date_rate_A2G", lambda _location: snr_db)
    monkeypatch.setattr(world, "get_date_rate_G2A", lambda _location: 10.0)
    monkeypatch.setattr(world, "get_g2a_outage_probability", lambda _location: outage_probability)


def test_reward_terms_sum_to_step_reward(monkeypatch):
    world = _world()
    _constant_radio(monkeypatch, world)
    state, _ = world.reset()
    _, reward, _, _, _ = world.step_inside(np.array([0.3, 0.25]), state, 0)
    assert reward == pytest.approx(sum(world.reward_summary().values()))


def test_full_speed_progress_outweighs_time_cost(monkeypatch):
    world = _world()
    _constant_radio(monkeypatch, world)
    state, _ = world.reset()
    heading = np.arctan2(*(world.target_loc - np.array(world.initial_loc))[::-1])
    world.step_inside(np.array([heading, world.dist_max]), state, 0)
    terms = world.reward_summary()
    assert terms["progress"] == pytest.approx(world.PROGRESS_WEIGHT * world.delta_T * world.REWARD_SCALE, rel=1e-6)
    assert terms["progress"] + terms["time"] == pytest.approx(world.delta_T * world.REWARD_SCALE, rel=1e-6)


def test_pending_upload_slot_is_charged_as_transmission_delay(monkeypatch):
    world = _world()
    _constant_radio(monkeypatch, world, snr_db=-100.0)
    state, _ = world.reset()
    world.step_inside(np.zeros(2), state, 0)
    assert world.reward_summary()["delay"] == 0.0
    world.transmit = False; world.data_size = 10.0; world.current_payload = 10.0
    world.step_inside(np.zeros(2), state, 1)
    assert world.reward_summary()["delay"] == pytest.approx(-world.delta_T * world.REWARD_SCALE)


def test_completed_leg_upload_is_worth_fixed_reward_regardless_of_payload(monkeypatch):
    world = _world()
    _constant_radio(monkeypatch, world, snr_db=100.0)
    state, _ = world.reset()
    world.transmit = False; world.data_size = 2.0; world.current_payload = 8.0
    world.step_inside(np.zeros(2), state, 0)
    assert world.reward_summary()["upload"] == pytest.approx(
        world.UPLOAD_LEG_REWARD * 2.0 / 8.0 * world.REWARD_SCALE)


def test_all_infeasible_terminations_share_one_penalty(monkeypatch):
    expected = -world_penalty()
    qos = _world(outage_max_s=0.4)
    _constant_radio(monkeypatch, qos, outage_probability=1.0)
    state, _ = qos.reset()
    qos.step_inside(np.zeros(2), state, 0)
    assert qos.reward_summary()["qos"] == pytest.approx(expected)

    timeout = _world(t=1)
    _constant_radio(monkeypatch, timeout)
    state, _ = timeout.reset()
    timeout.step_inside(np.zeros(2), state, 0)
    assert timeout.reward_summary()["timeout"] == pytest.approx(expected)

    early = _world()
    _constant_radio(monkeypatch, early, snr_db=-100.0)
    state, _ = early.reset()
    early.distance = 100.0; early.transmit = False; early.data_size = 10.0; early.current_payload = 10.0
    early.step_inside(np.zeros(2), state, 0)
    assert early.reward_summary()["incomplete_upload"] == pytest.approx(expected)


def world_penalty():
    return World.INFEASIBLE_PENALTY * World.REWARD_SCALE


def test_infeasible_penalty_exceeds_worst_feasible_objective():
    """Giving up early must never beat finishing, even with the slowest feasible mission."""
    world = _world(t=1000)
    horizon_s = world.T * world.delta_T
    worst_feasible_cost = horizon_s + horizon_s + world.outage_max_s  # T + tau + T_out
    milestones = 7 * world.ARRIVAL_REWARD + 6 * world.UPLOAD_LEG_REWARD + world.COMPLETION_REWARD
    assert world.INFEASIBLE_PENALTY > worst_feasible_cost - milestones


def test_state_encodes_target_relative_to_uav():
    world = _world()
    state, _ = world.reset()
    expected = (world.target_loc - np.array(world.initial_loc)) / np.array([world.length, world.width])
    np.testing.assert_allclose(state[2:4], expected)


def test_outage_slots_are_not_rewarded_per_slot(monkeypatch):
    world = _world()
    _constant_radio(monkeypatch, world, outage_probability=1.0)
    state, _ = world.reset()
    _, reward, _, _, _ = world.step_inside(np.zeros(2), state, 0)
    assert world.out_time == world.delta_T
    assert reward == pytest.approx(sum(world.reward_summary().values()))
    assert set(world.reward_summary()) == {"time", "delay", "progress", "boundary", "upload", "arrival",
                                            "completion", "incomplete_upload", "timeout", "qos"}


def test_zero_payload_point_needs_no_upload_and_never_divides_by_zero(monkeypatch):
    world = _world(data_sizes=np.array([0.0, 60, 60, 60, 60, 60]))
    _constant_radio(monkeypatch, world, snr_db=10.0)
    state, _ = world.reset(); world.distance = 100.0
    state, _, done, t, _ = world.step_inside(np.zeros(2), state, 0)
    assert not done and world.transmit and world.data_size == 0.0
    world.step_inside(np.zeros(2), state, t)
    assert world.reward_summary()["delay"] == 0.0
    assert world.reward_summary()["upload"] == 0.0


def test_upload_reward_is_safe_when_payload_size_is_unknown(monkeypatch):
    world = _world()
    _constant_radio(monkeypatch, world, snr_db=100.0)
    state, _ = world.reset()
    world.transmit = False; world.data_size = 2.0; world.current_payload = 0.0
    world.step_inside(np.zeros(2), state, 0)
    assert np.isfinite(world.reward_summary()["upload"])
