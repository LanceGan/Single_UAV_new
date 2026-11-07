# Deferred Experiments

## Outage-Probability Threshold Robustness

Status: deferred until the current controlled sequence ablation and the optional
Medium/High DRL matrix are complete.

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

Do not hard-code only the evaluation threshold. Parameterize `kappa_out`
through:

- World outage-slot accounting;
- longest-contiguous-outage termination;
- CA-TD3 state and hard constraint;
- PRIGA edge-cost outage term;
- run config, state schema, metrics, plots, and QA.

Retraining is required. Post-hoc evaluation of the existing `0.5` models under
`0.3/0.2` is a useful diagnostic but is not the full threshold-robustness
experiment.

Reporting destination:

- Start as robustness/SI evidence.
- Promote to the main text only if the `0.2` result materially changes the
  conclusion or reveals a substantive failure boundary.
