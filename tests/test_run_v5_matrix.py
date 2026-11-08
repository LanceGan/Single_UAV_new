from types import SimpleNamespace

from run_v5_matrix import build_jobs


def _args(groups, levels=("low", "medium", "high")):
    return SimpleNamespace(
        groups=tuple(groups),
        levels=tuple(levels),
        python="python",
        device="cuda",
        output_root="/tmp/v5-plan",
        skip_shared_proposed=False,
        reuse_existing_e1=False,
    )


def test_v5_matrix_cell_counts_and_dependencies():
    assert len(build_jobs(_args(("E1",)))) == 15
    assert len(build_jobs(_args(("E2",)))) == 21
    assert len(build_jobs(_args(("E3",)))) == 4
    assert len(build_jobs(_args(("E4",)))) == 21
    assert len(build_jobs(_args(("E5",)))) == 18


def test_mac_e5_can_reuse_proposed_without_duplication():
    args = _args(("E5",))
    args.skip_shared_proposed = True
    jobs = build_jobs(args)

    assert len(jobs) == 15
    assert all(job.group == "E5" for job in jobs)


def test_minimal_e1_schedules_only_missing_runs():
    args = _args(("E1",))
    args.reuse_existing_e1 = True
    jobs = build_jobs(args)

    assert len(jobs) == 11
    methods = sorted(job.job_id.split("/")[1] for job in jobs)
    assert methods.count("d3qn") == 2
    assert methods.count("ppo") == 3
    assert methods.count("ddpg") == 3
    assert methods.count("sac") == 3
    assert "ca_td3" not in methods


def test_every_job_invokes_a_training_entrypoint():
    jobs = build_jobs(_args(("E1", "E2", "E3", "E4", "E5")))
    scripts = {
        "Train_SINR.py",
        "Train_D3QN.py",
        "Train_PPO.py",
        "Train_SAC.py",
    }
    assert len(jobs) == 70
    assert all(job.command[1] in scripts for job in jobs)
    assert all("--qos_enforcement" in job.command for job in jobs)
    assert all("--total_steps" in job.command for job in jobs)
