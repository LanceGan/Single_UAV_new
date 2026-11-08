from types import SimpleNamespace

from run_v5_matrix import build_jobs


def _args(groups, levels=("low", "medium", "high")):
    return SimpleNamespace(
        groups=tuple(groups),
        levels=tuple(levels),
        python="python",
        device="cuda",
        output_root="/tmp/v5-plan",
    )


def test_v5_matrix_cell_counts_and_dependencies():
    assert len(build_jobs(_args(("E1",)))) == 15
    assert len(build_jobs(_args(("E2",)))) == 21
    assert len(build_jobs(_args(("E3",)))) == 4
    assert len(build_jobs(_args(("E4",)))) == 21
    assert len(build_jobs(_args(("E5",)))) == 18


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
