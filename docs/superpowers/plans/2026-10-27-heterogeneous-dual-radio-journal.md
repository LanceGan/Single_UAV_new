# Heterogeneous Dual-Radio Journal Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, fixed-scenario UAV inspection experiment that supports three fixed heterogeneous data vectors and fairly compares TD3, DDPG, DQN, A*, and conference baselines under the original objective and fixed QoS constraint.

**Architecture:** Preserve the existing map generation and spatial assets as the physical scenario source. Add an isolated `journal/` package that adapts the original radio-map modules, introduces explicit data/QoS accounting, and exposes a shared environment to continuous and discrete controllers. Sequence planning is data-aware IGA; all controllers consume the same selected sequence and the same fixed low/medium/high data vector.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, pytest, existing SciPy/Matplotlib project dependencies.

**Spec:** `docs/superpowers/specs/2026-10-27-heterogeneous-data-dual-radio-journal-design.md`

## Global Constraints

- Reuse the repository's existing A2G/G2A map code, GBS layout, inspection locations, start/end locations and numerical simulation parameters; do not rebuild the physical scenario.
- Paper equations govern objective/constraints; existing code governs numerical parameter values where they differ.
- Objective remains `T + tau_total`; QoS remains the fixed hard constraint `T_out <= T_out_max`.
- Exactly three persisted data vectors: one each for `[50,100)`, `[100,200)`, and `[200,300]` MB; do not sample data during training or evaluation.
- Train TD3, DDPG and DQN separately for each fixed vector; all methods share that vector and fixed spatial scenario.
- Upload each point's payload after arrival and before the next waypoint; upload the final point's payload on the terminal leg.
- Never overwrite existing conference scripts, result files, maps or checkpoints; new output belongs under `journal_results/`.
- Record all reward terms, QoS feasibility and raw per-run metrics; evaluate learned policies with exploration disabled.

## Review Focus

- A last inspection payload must be uploaded before the endpoint is accepted (Task 3 test `test_terminal_leg_requires_final_upload`).
- A QoS-over-budget transition must not be reported as a feasible result (Task 3 test `test_outage_budget_terminates_infeasible_episode`).
- Exactly the stored vector—not a newly sampled vector—must be used by all methods in a load scenario (Task 1 test `test_fixed_vector_is_loaded_verbatim`).
- The scaled A2G data decrement must contain `delta_t`, including at a partial final time slot (Task 3 test `test_upload_ledger_uses_slot_duration`).
- DQN and A* must use the same waypoint sequence and hard constraints as continuous DRL (Tasks 6 and 7 fairness tests).

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `journal/scenario.py` | Load legacy spatial inputs, persist/load the three fixed data vectors, expose scenario metadata. |
| `journal/radio_adapter.py` | Query existing `radio_map_A2G.py` and `radio_map_G2A.py` through one seed-controlled interface. |
| `journal/environment.py` | Shared UAV transition, upload ledger, QoS ledger, reward terms and evaluation metrics. |
| `journal/sequence.py` | Data-aware edge estimator, IGA, distance TSP and G2A-only sequence baselines. |
| `journal/agents/td3_agent.py` | Adapt the existing TD3 implementation to the shared environment. |
| `journal/agents/ddpg_agent.py` | Adapt the existing DDPG implementation to the shared environment. |
| `journal/agents/dqn_agent.py` | DQN and fixed discrete action codec. |
| `journal/planners/astar.py` | Budget-augmented grid A* controller. |
| `journal/evaluate.py` | Deterministic evaluation and raw metric serialization. |
| `journal/run_experiments.py` | Explicit CLI for the three data vectors and all methods. |
| `tests/journal/` | Unit, integration and fairness tests. |
| `journal_results/` | New run artifacts only; ignored by Git except small fixture metadata. |

### Task 1: Fixed legacy scenario and three payload vectors

**Files:**
- Create: `journal/__init__.py`, `journal/scenario.py`, `journal/data_vectors.json`, `tests/journal/test_scenario.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces `Scenario` with `start_xy`, `end_xy`, `inspection_xy`, `gbs_xyz`, `delta_t`, `speed_max`, `outage_max_s` and `payload_mb: np.ndarray`.
- Produces `load_scenario(load_level: Literal["low", "medium", "high"]) -> Scenario`.

- [ ] **Step 1: Write failing scenario tests**

```python
def test_fixed_vector_is_loaded_verbatim():
    scenario = load_scenario("low")
    assert scenario.payload_mb.tolist() == stored_low_vector
    assert len(scenario.payload_mb) == len(scenario.inspection_xy)

def test_payload_vectors_stay_in_declared_intervals():
    assert np.all((low >= 50) & (low < 100))
    assert np.all((medium >= 100) & (medium < 200))
    assert np.all((high >= 200) & (high <= 300))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/journal/test_scenario.py -v`

Expected: FAIL because `journal.scenario` does not exist.

- [ ] **Step 3: Implement `load_scenario` and persist exactly three vectors**

Read node coordinates and legacy numerical settings from the existing repository modules/files; do not duplicate or regenerate the spatial map. Generate one six-element vector per interval using `numpy.random.default_rng(20261027)`, write the generated values and seed to `journal/data_vectors.json`, then load only this file in future runs. Add `journal_results/` to `.gitignore`.

- [ ] **Step 4: Run scenario tests**

Run: `pytest tests/journal/test_scenario.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add journal tests/journal .gitignore
git commit -m "feat: add fixed heterogeneous task scenarios"
```

### Task 2: Legacy radio-map adapter and reproducible queries

**Files:**
- Create: `journal/radio_adapter.py`, `tests/journal/test_radio_adapter.py`

**Interfaces:**
- Consumes: `Scenario` from Task 1 and original `radio_map_A2G.py` / `radio_map_G2A.py`.
- Produces `RadioAdapter.a2g_rate_mb_s(xy) -> float`, `RadioAdapter.g2a_outage(xy) -> bool`, and `RadioAdapter.seed(seed: int) -> None`.

- [ ] **Step 1: Write failing adapter tests**

```python
def test_adapter_uses_legacy_module_parameters(adapter):
    assert adapter.a2g_module.PB == 0.1
    assert adapter.g2a_module.PB == 0.1

def test_seeded_queries_are_repeatable(adapter):
    adapter.seed(7); first = adapter.a2g_rate_mb_s(np.array([2.62, 2.65]))
    adapter.seed(7); second = adapter.a2g_rate_mb_s(np.array([2.62, 2.65]))
    assert first == pytest.approx(second)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/journal/test_radio_adapter.py -v`

Expected: FAIL because `RadioAdapter` does not exist.

- [ ] **Step 3: Implement the adapter without changing legacy map values**

Convert environment coordinates to the legacy map coordinate convention in exactly one place. Derive A2G rate from the legacy SNR return and existing bandwidth convention; derive G2A outage from the legacy map threshold. Seed NumPy immediately before query batches so fading-derived values are deterministic per run. Preserve legacy powers, noise terms and geometry unchanged.

- [ ] **Step 4: Run adapter tests**

Run: `pytest tests/journal/test_radio_adapter.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add journal/radio_adapter.py tests/journal/test_radio_adapter.py
git commit -m "feat: adapt legacy dual radio maps"
```

### Task 3: Shared environment, correct ledgers and calibrated reward terms

**Files:**
- Create: `journal/environment.py`, `tests/journal/test_environment.py`, `tests/journal/test_reward_terms.py`

**Interfaces:**
- Consumes: `Scenario`, `RadioAdapter`, fixed waypoint sequence.
- Produces `InspectionEnv.reset() -> tuple[np.ndarray, dict]`, `step(action) -> tuple[np.ndarray, float, bool, bool, dict]`, and `info["reward_terms"]`, `info["metrics"]`.

- [ ] **Step 1: Write failing ledger/QoS tests**

```python
def test_upload_ledger_uses_slot_duration(env_with_constant_maps):
    env = env_with_constant_maps(rate_mb_s=4.0, delta_t=0.5, payload_mb=[10])
    env.reset(); env.mark_payload_collected(0); env.step(move_action)
    assert env.remaining_mb == pytest.approx(8.0)

def test_terminal_leg_requires_final_upload(env_with_constant_maps):
    env = env_with_constant_maps(rate_mb_s=0.0, payload_mb=[10])
    _, _, terminated, _, info = env.arrive_at_endpoint()
    assert terminated and not info["metrics"]["feasible"]

def test_outage_budget_terminates_infeasible_episode(env_with_constant_maps):
    env = env_with_constant_maps(outage=True, outage_max_s=0.5, delta_t=0.5)
    env.reset(); env.step(move_action); *_, terminated, _, info = env.step(move_action)
    assert terminated and info["metrics"]["qos_violation"]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/journal/test_environment.py tests/journal/test_reward_terms.py -v`

Expected: FAIL because `InspectionEnv` does not exist.

- [ ] **Step 3: Implement physics, data and QoS ledgers**

Use continuous `[heading, distance]` actions with the original action bounds. Add the selected payload only upon arrival at each inspection node; decrement it by `a2g_rate_mb_s * delta_t`; accumulate `T_out` by `delta_t` for G2A outage slots; fail immediately when it exceeds the fixed legacy threshold. The endpoint is valid only after the last payload is cleared.

- [ ] **Step 4: Add reward-term instrumentation before choosing scales**

Return named dense and sparse terms: `progress`, `idle`, `boundary`, `arrival_upload`, `completion`, `timeout`, and `qos`. Require `reward == sum(reward_terms.values())`. Start with dimensionless terms normalized by `dist_max`, `payload_mb.sum()`, and `outage_max_s`; log the per-episode sum of each term to JSONL. Do not optimize constants until the logs show no single shaping term overwhelms terminal/constraint terms.

- [ ] **Step 5: Run environment and reward-term tests**

Run: `pytest tests/journal/test_environment.py tests/journal/test_reward_terms.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add journal/environment.py tests/journal/test_environment.py tests/journal/test_reward_terms.py
git commit -m "feat: add constrained inspection environment"
```

### Task 4: Data-aware sequence optimizers

**Files:**
- Create: `journal/sequence.py`, `tests/journal/test_sequence.py`

**Interfaces:**
- Consumes: `Scenario`, `RadioAdapter` and `InspectionEnv` timing convention.
- Produces `EdgeEstimate(distance_m, upload_rate_mb_s, outage_s, payload_mb, feasible, cost)`; `DataAwareIGA.solve(scenario) -> list[int]`; `DistanceTSP.solve(scenario) -> list[int]`; `G2ASA.solve(scenario) -> list[int]`.

- [ ] **Step 1: Write failing edge/ordering tests**

```python
def test_larger_source_payload_increases_same_edge_cost(edge_estimator):
    small = edge_estimator.estimate(i=0, j=1, payload_mb=50)
    large = edge_estimator.estimate(i=0, j=1, payload_mb=250)
    assert large.cost > small.cost

def test_qos_infeasible_edge_is_rejected(edge_estimator):
    edge = edge_estimator.estimate(i=0, j=1, payload_mb=100)
    assert not edge.feasible and math.isinf(edge.cost)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/journal/test_sequence.py -v`

Expected: FAIL because `EdgeEstimator` does not exist.

- [ ] **Step 3: Implement an edge estimator consistent with Task 3**

Sample each straight candidate leg using the environment time step. Let `C_ij` be average A2G upload rate and use the heterogeneous extension of paper Eq. (17): `cost = d_ij * t_ij * I_i / max(C_ij, epsilon)`. Mark an edge infeasible if its accumulated outage exceeds `T_out_max` or it cannot upload `I_i` before the leg ends. Reuse the legacy GA crossover/mutation logic only after feeding it this edge matrix.

- [ ] **Step 4: Implement and test baselines**

Implement distance-only TSP and G2A-aware SA with the same fixed start/end constraints. Ensure every returned sequence is a permutation of all inspection nodes and has no direct start-to-end shortcut.

- [ ] **Step 5: Run sequence tests**

Run: `pytest tests/journal/test_sequence.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add journal/sequence.py tests/journal/test_sequence.py
git commit -m "feat: add data-aware sequence optimization"
```

### Task 5: Continuous-action TD3 and DDPG experiments

**Files:**
- Create: `journal/agents/__init__.py`, `journal/agents/td3_agent.py`, `journal/agents/ddpg_agent.py`, `journal/train_continuous.py`, `tests/journal/test_continuous_agents.py`

**Interfaces:**
- Consumes: `InspectionEnv`, sequence from Task 4, fixed vector level and training seed.
- Produces `train_continuous(algorithm: Literal["td3", "ddpg"], level: str, seed: int) -> RunArtifact`.

- [ ] **Step 1: Write failing fairness tests**

```python
@pytest.mark.parametrize("algorithm", ["td3", "ddpg"])
def test_continuous_agents_receive_identical_environment_contract(algorithm, env):
    artifact = train_one_epoch(algorithm, env, seed=3)
    assert artifact.scenario_hash == env.scenario_hash
    assert artifact.action_bounds == env.action_bounds
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/journal/test_continuous_agents.py -v`

Expected: FAIL because `train_one_epoch` does not exist.

- [ ] **Step 3: Adapt existing TD3 and DDPG implementations**

Reuse network/replay logic from `TD3.py` and `DDPG.py`; remove hard-coded CUDA device selection, use the shared environment, fixed sequence and standardized seeds. TD3 and DDPG must share total environment steps, replay warm-up, action bounds, checkpoints, reward logs and deterministic evaluation. Save outputs under `journal_results/<level>/<algorithm>/seed_<n>/`.

- [ ] **Step 4: Run continuous-agent tests**

Run: `pytest tests/journal/test_continuous_agents.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add journal/agents journal/train_continuous.py tests/journal/test_continuous_agents.py
git commit -m "feat: add TD3 and DDPG journal runners"
```

### Task 6: DQN discrete-action baseline

**Files:**
- Create: `journal/agents/dqn_agent.py`, `journal/action_codec.py`, `journal/train_dqn.py`, `tests/journal/test_dqn.py`

**Interfaces:**
- Consumes: `InspectionEnv`, shared IGA sequence and `DiscreteActionCodec(headings=16, distances=(0.0, dist_max/2, dist_max))`.
- Produces `DQNAgent.select_action(state) -> int` and `codec.decode(index) -> np.ndarray`.

- [ ] **Step 1: Write failing DQN tests**

```python
def test_discrete_action_codec_preserves_continuous_bounds(codec, env):
    for index in range(codec.n_actions):
        heading, distance = codec.decode(index)
        assert -np.pi <= heading <= np.pi
        assert 0.0 <= distance <= env.action_bounds[1]

def test_dqn_uses_the_iga_sequence(dqn_runner, iga_sequence):
    assert dqn_runner.sequence == iga_sequence
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/journal/test_dqn.py -v`

Expected: FAIL because DQN modules do not exist.

- [ ] **Step 3: Implement DQN with a frozen 48-action codec**

Use 16 evenly spaced headings in `[-pi, pi)` and three distances `(0, dist_max/2, dist_max)`. Use replay buffer, target network and epsilon schedule; match TD3/DDPG environment-step budget and fixed IGA sequence. Persist the codec in run metadata.

- [ ] **Step 4: Run DQN tests**

Run: `pytest tests/journal/test_dqn.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add journal/agents/dqn_agent.py journal/action_codec.py journal/train_dqn.py tests/journal/test_dqn.py
git commit -m "feat: add discrete DQN baseline"
```

### Task 7: Budget-augmented A* and deterministic evaluation

**Files:**
- Create: `journal/planners/__init__.py`, `journal/planners/astar.py`, `journal/evaluate.py`, `tests/journal/test_astar.py`, `tests/journal/test_evaluate.py`

**Interfaces:**
- Consumes: `InspectionEnv` transition semantics and sequence from Task 4.
- Produces `AStarPlanner.plan(env, sequence) -> TrajectoryResult` and `evaluate(policy_or_plan, env) -> Metrics`.

- [ ] **Step 1: Write failing A*/evaluation tests**

```python
def test_astar_rejects_path_that_exceeds_qos_budget(constrained_grid_env):
    assert AStarPlanner().plan(constrained_grid_env, sequence) is None

def test_evaluation_is_deterministic_without_exploration(eval_env, fixed_policy):
    assert evaluate(fixed_policy, eval_env) == evaluate(fixed_policy, eval_env)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/journal/test_astar.py tests/journal/test_evaluate.py -v`

Expected: FAIL because planner/evaluator do not exist.

- [ ] **Step 3: Implement A* state and cost**

Use a 100 m grid derived from the existing 2 km map, with A* state `(grid_x, grid_y, remaining_data_bucket, outage_bucket, leg_index)`. One transition uses the Task 3 ledger, is pruned on QoS violation, and uses time plus upload-delay cost. Concatenate feasible leg paths for the supplied sequence. Persist grid/data/QoS bucket sizes in metrics.

- [ ] **Step 4: Implement common deterministic evaluation**

Evaluate TD3/DDPG/DQN with exploration disabled and A* directly. Compute `T`, `tau_total`, `T + tau_total`, `T_out`, feasibility, A2G average rate, flight distance, incomplete-upload rate and wall-clock runtime from the same ledger.

- [ ] **Step 5: Run A*/evaluation tests**

Run: `pytest tests/journal/test_astar.py tests/journal/test_evaluate.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add journal/planners journal/evaluate.py tests/journal/test_astar.py tests/journal/test_evaluate.py
git commit -m "feat: add constrained A-star evaluation"
```

### Task 8: Experiment CLI, artifacts and end-to-end checks

**Files:**
- Create: `journal/run_experiments.py`, `journal/plot_results.py`, `tests/journal/test_experiment_runner.py`, `README_journal.md`

**Interfaces:**
- Consumes: all task interfaces above.
- Produces `python -m journal.run_experiments --level {low,medium,high} --method {...} --seed N` and JSON/NPZ/PNG results under `journal_results/`.

- [ ] **Step 1: Write failing runner tests**

```python
def test_runner_writes_raw_metrics_without_touching_legacy_results(tmp_path):
    path = run_experiment(level="low", method="astar", output_root=tmp_path)
    assert (path / "metrics.json").exists()
    assert not (repo_root / "results" / "trajectory" / "new_file.npz").exists()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/journal/test_experiment_runner.py -v`

Expected: FAIL because `run_experiment` does not exist.

- [ ] **Step 3: Implement the runner and reporting contract**

Expose methods `proposed_td3`, `iga_ddpg`, `iga_dqn`, `iga_astar`, `tsp_td3`, `sa_td3`, `tsp_astar`. Save input vector, sequence, seed, legacy module hashes, reward-term totals and all evaluation metrics. Plot low/medium/high results separately with error bars across learning seeds only.

- [ ] **Step 4: Run the full fast test suite**

Run: `pytest tests/journal -v`

Expected: PASS.

- [ ] **Step 5: Run smoke experiments**

Run: `python -m journal.run_experiments --level low --method iga_astar --seed 1 --smoke`

Expected: a feasible/infeasible result with complete metric JSON and no legacy-file modification.

- [ ] **Step 6: Commit**

```bash
git add journal tests/journal README_journal.md .gitignore
git commit -m "feat: add reproducible journal experiment runner"
```

## Self-Review

- **Spec coverage:** Tasks 1-2 preserve the fixed legacy scenario and data vectors; Task 3 enforces data/QoS semantics and reward calibration; Task 4 implements heterogeneous IGA; Tasks 5-7 add TD3, DDPG, DQN and A*; Task 8 provides reproducibility and reporting.
- **Type consistency:** `Scenario`, `RadioAdapter`, `InspectionEnv`, sequence lists and `Metrics` are introduced before later consumers.
- **Review focus coverage:** Each listed failure mode is tied to a named test in its owning task.
- **Proportion:** The plan defines interfaces, tests and fixed choices without transcribing implementation bodies.
