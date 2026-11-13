# Deferred Experiments

## Outage-Probability Threshold Robustness

Status: active. User selected the original threshold grid
`kappa_out = 0.5/0.3/0.2` at the fixed 6 s budget. Incomplete missions at the
two stricter thresholds are allowed and must be reported as failures rather
than replaced with target values.

Purpose:

> Test whether DRM-HO conclusions are specific to the empirical outage-event
> threshold `kappa_out = 0.5`.

Primary protocol:

| Parameter | Value |
|---|---|
| Data level | Low |
| Outage budget | `T_out^max = 6 s` |
| Objective weights | `(w1, w2) = (0.5, 0.5)` |
| Outage thresholds | `0.5, 0.3, 0.2` |
| Sequence | PRIGA, recomputed under each threshold |
| Trajectory | CA-TD3 |
| Episodes | 3000 |
| Seed | 1 |
| Evaluation | Same deterministic `eval_best` protocol |

Reuse:

- `kappa_out = 0.5`: reuse
  `journal_results/v4-qos/w50-50/qos6/A/low/td3_bc`.
- `kappa_out = 0.3`: new training.
- `kappa_out = 0.2`: new training.

Total new DRL runs: **2**.

Interpretation:

- Lowering `kappa_out` classifies more slots as outage.
- For any fixed trajectory, the realized `T_out^cont` must be non-decreasing as
  `kappa_out` decreases.
- After retraining, `T + tau_total` need not be strictly monotonic because PRIGA
  and CA-TD3 may choose different orders and detours.
- Report feasibility, `T`, `tau_total`, `T + tau_total`, realized
  `T_out^cont`, the resulting order, and feasible-evaluation count.

Implementation prerequisite:

Completed: `outage_probability_threshold` is now propagated through:

- World outage-slot accounting;
- longest-contiguous-outage termination;
- CA-TD3 state and hard constraint;
- MapGuide outage raster and endpoint safety checks;
- PRIGA edge-cost outage term and sequence selection;
- A* demonstration planning and cache identity;
- mission preflight, evaluation replay, and `run_config.json`.

Legacy configurations without the field default to `kappa_out = 0.5`.

Formal preflight result at Low load, `T_out^max = 6 s`:

| `kappa_out` | PRIGA order | Max contiguous outage | Visited | Uploaded | Result |
|---:|---|---:|---:|---:|---|
| 0.3 | `[1,6,2,3,5,4]` | 6.5 s | 2/6 | 150/450 MB | infeasible |
| 0.2 | `[1,6,2,3,5,4]` | 6.5 s | 1/6 | 25.903/450 MB | infeasible |

The threshold-aware PRIGA order remains unchanged, while the existing MapGuide
demonstration route crosses the 6 s hard limit. A threshold-aware A* feasibility
search also finds no feasible path for the same order and budget. Therefore the
0.3/0.2 TD3 runs must not be launched under the original 6 s protocol. Relaxing
the budget, changing the sequence, changing the teacher, or reporting the 6 s
case as a feasibility boundary is a scientific-protocol decision that must be
recorded explicitly.

A follow-up preflight boundary check at the same 6 s budget found:

| `kappa_out` | Preflight result |
|---:|---|
| 0.50 | success |
| 0.45 | success, 6/6 visited, 450 MB uploaded, 3.5 s max outage |
| 0.42 | success, 6/6 visited, 450 MB uploaded, 4.0 s max outage |
| 0.41 | success, 6/6 visited, 450 MB uploaded, 4.5 s max outage |
| 0.40 | infeasible, 2/6 visited, 6.5 s max outage |

User decision: retain `0.30/0.20` even though their preflight missions are
incomplete. Training records `preflight_success=false` and
`preflight_allowed_failure=true`; final tables must show visited points,
uploaded data, realized outage, and failure status alongside any objective
value.

Retraining is required. Post-hoc evaluation of the existing `0.5` models under
`0.3/0.2` is a useful diagnostic but is not the full threshold-robustness
experiment.

Reporting destination:

- Start as robustness/SI evidence.
- Promote to the main text only if the `0.2` result materially changes the
  conclusion or reveals a substantive failure boundary.
