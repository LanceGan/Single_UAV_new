# MCO-A* Baseline Handoff

> **Current paper scope:** the formal A* and PSO baselines are now the
> radio-map-free end-to-end heuristics in `pure_heuristic_baselines.py`.
> This document preserves the earlier MCO-A* adaptation as an archival,
> optional reference and should not be presented as the paper's pure A*
> baseline.

## 1. What Was Added

The supplied ICCT-2025 source implements a radio-map-driven hierarchical
framework with two separable parts:

- MCO-A* for a G2A connectivity-constrained trajectory;
- EEAP-PSO with dynamic programming and 2-Opt for multi-trip cargo allocation.

Only MCO-A* is suitable for direct comparison with the trajectory executor in
this repository. The EEAP-PSO layer optimizes warehouse returns, payload weight,
battery energy, and total energy in joules; it does not model the proposed
single mission, heterogeneous uplink data, A2G offloading, or `T + tau`.

The implemented baseline is therefore:

> **MCO-A* trajectory baseline with the fixed PRIGA sequence and an
> A2G-unaware upload-hub rule**

It is not a complete reproduction of RM-HO. This distinction must remain
visible in all paper tables, captions, and prose.

## 2. Adaptation Contract

| Item | Implementation |
|---|---|
| Visit order | Fixed to the PRIGA order used by the proposed method |
| Trajectory search | MCO-A* with state `(node, contiguous G2A outage slots)` |
| Control lattice | 25 m cardinal moves, one 0.5 s slot per move |
| Hard QoS | `T_out^cont <= 6 s` |
| G2A information | Visible to the planner |
| A2G information | Hidden from the planner |
| Upload | Safe hover hub chosen by geographic detour only |
| Final metrics | Real `World` replay only |
| Optimality claim | Shortest feasible path per planned segment on the adapted lattice only |

The source algorithm uses a 10 m, 8-neighbor continuous-time grid. It was
mapped to the repository's 25 m, 0.5 s action lattice so that planner and World
replay use the same physical moves and outage accounting. The source's
probability-weighted outage time was also replaced by the present binary outage
slot definition:

```text
outage_t = 1[P_out(q_t) >= 0.5]
current_run_{t+1} = outage_{t+1} (current_run_t + Delta t)
T_out^cont = max_t current_run_t
```

## 3. Files

- `mco_astar.py`: planner and deterministic fixed-order guide.
- `run_mco_astar.py`: three-level World replay and artifact writer.
- `tests/test_mco_astar.py`: outage-budget and guard-ring tests.
- `journal_results/v4-deterministic/mco-astar-6/`: formal 6 s artifacts.

Run:

```bash
python3 run_mco_astar.py \
  --levels low medium high \
  --outage-max-s 6 \
  --output journal_results/v4-deterministic/mco-astar-6
```

## 4. Results

| Level | Valid | Order | T (s) | tau (s) | T+tau (s) | `T_out^cont` (s) | Visits | Upload |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Low | Yes | `[1,6,2,3,5,4]` | 287.5 | 170.0 | 457.5 | 6.0 | 6/6 | 450/450 MB |
| Medium | Yes | `[1,6,5,4,2,3]` | 459.5 | 329.5 | 789.0 | 5.5 | 6/6 | 900/900 MB |
| High | No | `[1,6,5,4,2,3]` | 500.0 | 334.5 | Not ranked | 5.5 | 5/6 | 1182.11/1500 MB |

The high-load run reaches the 500 s World horizon with one inspection point
unvisited. Its `834.5 s` partial objective must not be entered into a feasible
ranking or cited as a completed-mission target.

At the low tier, MCO-A* is `87.1%` slower than DRM-HO in `T + tau`
(`457.5` versus `244.5 s`). At the medium tier it is `124.1%` slower
(`789.0` versus `352.0 s`). At the high tier, DRM-HO completes the mission
while MCO-A* times out.

## 5. Writing Boundary

Allowed:

- "MCO-A* is adapted to the common inspection and upload environment."
- "The baseline sees the G2A map but not the A2G map."
- "MCO-A* plans the shortest feasible trajectory per segment under the adapted
  continuous-outage state."
- "It is weaker because it cannot route toward high-rate uplink regions."
- "The high-load MCO-A* run did not finish within the 500 s horizon."

Not allowed:

- "We reproduced the complete RM-HO framework."
- "MCO-A* is globally optimal for the inspection mission."
- "The MCO-A* partial objective is a valid high-load result."
- "EEAP-PSO was compared end to end."

## 6. Verification

```bash
python3 -m pytest -q \
  tests/test_mco_astar.py \
  tests/test_map_guidance.py \
  tests/test_preflight.py \
  tests/test_v4_paper_figures.py

python3 /path/to/nature-figure/scripts/validate_figure.py \
  plot_v4_paper_figures.py --strict

python3 /path/to/nature-figure/scripts/audit_pdf_text.py \
  paper_figures/v4/layer1_methods_v4.pdf --min-pt 5

python3 /path/to/nature-figure/scripts/audit_figure_collisions.py \
  paper_figures/v4/layer1_methods_v4.pdf --strict
```
