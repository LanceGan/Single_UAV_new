import numpy as np
import pytest

from pso_trajectory import (
    GENES_PER_LEG,
    build_leg_plans,
    decode_order,
    fitness_from_report,
    leg_endpoints,
    split_particle,
)


def test_leg_plans_geometry_offsets_are_perpendicular_and_hover_sits_outside_disk():
    order = [1, 6, 2, 3, 5, 4]
    genes = np.zeros((len(order) + 1) * GENES_PER_LEG)
    genes[3] = 42.4  # first leg hover gene

    plans = build_leg_plans(order, genes)

    assert len(plans) == len(order) + 1
    first = plans[0]
    a, b = leg_endpoints(order)[0]
    # Zero offsets -> control points on the straight segment at 1/4, 1/2, 3/4.
    for fraction, vertex in zip((0.25, 0.5, 0.75), first["vertices"][:3]):
        np.testing.assert_allclose(vertex, a + fraction * (b - a), atol=1e-9)
    hover_vertex = first["vertices"][first["hover_index"]]
    assert np.linalg.norm(hover_vertex - b) == pytest.approx(0.8, abs=1e-9)
    assert first["hover_slots"] == 42
    assert np.allclose(first["vertices"][-1], b)

    # A perpendicular offset moves the control point off the chord by its value.
    genes2 = genes.copy()
    genes2[1] = 2.0  # middle control point of leg 0
    shifted = build_leg_plans(order, genes2)[0]["vertices"][1]
    chord_mid = a + 0.5 * (b - a)
    assert np.linalg.norm(shifted - chord_mid) == pytest.approx(2.0, abs=1e-9)


def test_fitness_ranks_every_failure_below_every_success():
    success_slow = {"success": True, "task_time_s": 400.0, "transmission_delay_s": 250.0,
                    "visited_inspection_points": 6, "uploaded_mb": 450.0}
    near_miss = {"success": False, "task_time_s": 100.0, "transmission_delay_s": 10.0,
                 "visited_inspection_points": 6, "uploaded_mb": 450.0}

    assert fitness_from_report(success_slow) < fitness_from_report(near_miss)
    worse_fail = {"success": False, "task_time_s": 50.0, "transmission_delay_s": 0.0,
                  "visited_inspection_points": 1, "uploaded_mb": 50.0}
    assert fitness_from_report(near_miss) < fitness_from_report(worse_fail)


def test_particle_jointly_decodes_a_complete_order_and_trajectory_controls():
    point_count = 6
    control_dim = (point_count + 1) * GENES_PER_LEG
    particle = np.zeros(point_count + control_dim)
    particle[:point_count] = [0.4, -0.8, 0.0, 1.2, -0.2, 0.7]
    particle[point_count:] = np.arange(control_dim)

    order, controls = split_particle(particle, point_count)

    assert order == decode_order(particle[:point_count], point_count)
    assert sorted(order) == [1, 2, 3, 4, 5, 6]
    np.testing.assert_array_equal(controls, np.arange(control_dim))
