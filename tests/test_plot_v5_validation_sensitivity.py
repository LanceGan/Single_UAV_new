from plot_v5_validation_sensitivity import load_e3, load_e4_qos, load_e4_weights


def test_e3_ablation_preserves_high_blind_failure():
    rows = load_e3()
    assert len(rows) == 6
    high_blind = next(
        row
        for row in rows
        if row["level"] == "High" and row["method"] == "A2G-blind"
    )
    assert high_blind["complete"] is False
    assert high_blind["objective_s"] is None
    assert high_blind["visited_points"] == 2


def test_e4_qos_matrix_is_complete_and_preserves_missing_objectives():
    rows = load_e4_qos()
    assert len(rows) == 15
    medium_8 = next(
        row
        for row in rows
        if row["level"] == "Medium" and row["setting"] == "qos=8s, w50-50"
    )
    assert medium_8["complete"] is False
    assert medium_8["objective_s"] is None
    assert medium_8["visited_points"] == 4


def test_e4_weight_matrix_uses_shared_balanced_six_second_run():
    rows = load_e4_weights()
    assert len(rows) == 9
    medium_balanced = next(
        row
        for row in rows
        if row["level"] == "Medium" and row["setting"] == "qos=6s, w=50/50"
    )
    assert medium_balanced["objective_s"] == 352.0
    assert medium_balanced["qos_feasible"] is True
