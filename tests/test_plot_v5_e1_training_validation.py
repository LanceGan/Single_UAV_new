from plot_v5_e1_training_validation import (
    load_low_histories,
    smooth_series,
)


def test_moving_average_preserves_leading_missing_values():
    result = smooth_series([1, 2, 3, 4], 2)
    assert result[0] != result[0]
    assert result[1:].tolist() == [1.5, 2.5, 3.5]


def test_low_histories_have_all_five_executors():
    rewards, evaluations = load_low_histories()
    assert {row["method"] for row in rewards} == {
        "CA-TD3",
        "D3QN",
        "PPO",
        "DDPG",
        "SAC",
    }
    assert len(rewards) == 5 * 3000
    assert len(evaluations) == 5 * 120
