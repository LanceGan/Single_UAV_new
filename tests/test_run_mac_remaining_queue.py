from run_mac_remaining_queue import build_remaining_jobs


def test_mac_remaining_queue_contains_e4_and_missing_sa_d3qn(tmp_path):
    jobs = build_remaining_jobs(
        python="python",
        device="mps",
        output_root=tmp_path,
    )

    assert len(jobs) == 14
    assert sum(job.group == "E4" for job in jobs) == 12
    assert {
        job.job_id for job in jobs if job.group == "E5"
    } == {"E5/sa_d3qn/medium", "E5/sa_d3qn/high"}
