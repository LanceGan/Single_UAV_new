# D3QN and A-Star Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run fair D3QN and constrained A-star baselines on the three fixed payload levels.

**Architecture:** D3QN gets its own action codec, network/agent, replay, training CLI and deterministic evaluation while reusing World. A-star is a separate planner over a 25 m graph with data/QoS augmented labels and shared metric/plot outputs; the two subsystems share only existing scenario/config readers.

**Tech Stack:** Python, NumPy, PyTorch, SciPy/heapq, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-10-31-d3qn-astar-baselines-design.md`

## Global Constraints

- Fixed maps, positions, payload vectors, IGA orders and QoS=100 s.
- Objective remains T+tau; T_out is only a hard constraint.
- Variant A semantics and 25 m maximum step.
- D3QN and A-star outputs use new directories and never overwrite formal-v5-restart.

## Review Focus

- The action codec must contain exactly 33 unique bounded actions.
- Double-DQN target must select with online network and evaluate with target network.
- D3QN evaluation must be epsilon-free and guide-free.
- A-star must reject every state over the QoS budget or entering a target with post-upload pending data, while allowing a same-slot final upload.
- A-star must use the 50 m arrival disk and the 1,000-slot horizon exactly.
- A-star path metrics must be recomputed consistently from its transition ledger.

---

### Task 1: D3QN baseline

**Files:**
- Create: `discrete_actions.py`, `D3QN.py`, `Train_D3QN.py`, `evaluate_d3qn.py`
- Create: `tests/test_d3qn.py`, `tests/test_d3qn_training.py`

- [ ] Test 33-action uniqueness/bounds and relative-heading semantics.
- [ ] Test dueling aggregation, Double-DQN target and one learning update.
- [ ] Implement training/eval/checkpoint/config outputs.
- [ ] Run a short Low smoke and full tests.
- [ ] Launch three 3000-episode runs after verification.

### Task 2: QoS-constrained A-star baseline

**Files:**
- Create: `astar_planner.py`, `run_astar.py`
- Create: `tests/test_astar_planner.py`, `tests/test_astar_integration.py`

- [ ] Test 25 m graph edges, 50 m arrival disk, upload/delay ledger, QoS and 1,000-slot pruning.
- [ ] Test same-slot final upload, post-upload target rejection, dominance and deterministic repeatability.
- [ ] Implement augmented A-star with dominance pruning and metrics.
- [ ] Replay candidate paths in World, then run Low/Medium/High and generate trajectory/metric artifacts with honest search status.

### Task 3: Integration and reporting

**Files:**
- Modify: `plot_formal_results.py` or add a focused comparison plotter.
- Create: `tests/test_baseline_reporting.py`

- [ ] Verify all methods use identical vectors/orders/map hashes.
- [ ] Generate comparable trajectory and metric summaries.
- [ ] Run full suite, independent review and Chinese commits.

