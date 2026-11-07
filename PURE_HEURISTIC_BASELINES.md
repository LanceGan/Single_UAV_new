# Radio-Map-Free A*/PSO Baseline Handoff

## Definition

The formal A* and PSO baselines are pure geometric heuristics. They do not read
the G2A map, the A2G map, payload values, or the PRIGA sequence.

- **Pure A*** searches over `(UAV grid position, visited inspection set)` and
  minimizes geometric distance only.
- **Pure PSO** optimizes a complete inspection permutation under a geometric
  path-length objective. The decoded permutation defines a piecewise-linear
  end-to-end trajectory.

Both methods use the same replay protocol:

> After collecting a payload, hover at that inspection point until the buffer
> clears; then continue along the precomputed geometric trajectory.

Final success, outage, upload, `T`, and `tau` are measured only by the shared
World replay.

## Results

| Method | Candidate path | Low | Medium | High |
|---|---:|---|---|---|
| Pure A* | 5.216 km | QoS violation, 6.5 s, 1/6 | same | same |
| Pure PSO | 4.451 km | QoS violation, 6.5 s, 1/6 | same | same |

Both methods construct complete candidate geometric trajectories, but neither
has a feasible `T + tau` under the 6 s maximum-contiguous-outage constraint.
The paths fail immediately after the first inspection point because no
radio-map information is available to avoid G2A outage regions or shape A2G
upload-aware detours.

## Files

- `pure_heuristic_baselines.py`: A*/PSO solvers and route replay.
- `run_pure_heuristics.py`: three-level formal run.
- `tests/test_pure_heuristic_baselines.py`: solver tests.
- `journal_results/v4-deterministic/pure-heuristics-6/`: metrics and traces.
- `journal_results/v4-summary/global-baselines-summary.json/csv`: unified summary.

Run:

```bash
python3 run_pure_heuristics.py \
  --levels low medium high \
  --outage-max-s 6 \
  --output journal_results/v4-deterministic/pure-heuristics-6

python3 summarize_global_baselines.py
```

## Writing Rules

Allowed:

- "Pure A* and PSO construct complete geometric candidate trajectories."
- "Neither uses any radio-map, payload, or PRIGA information."
- "Both violate the 6 s outage constraint after the first point."
- "Their candidate path lengths are 5.216 and 4.451 km, respectively."

Not allowed:

- treating candidate path length as the mission objective;
- reporting a feasible `T + tau` for a failed mission;
- claiming A*/PSO uses G2A or A2G maps;
- ranking the infeasible heuristics against DRM-HO.
