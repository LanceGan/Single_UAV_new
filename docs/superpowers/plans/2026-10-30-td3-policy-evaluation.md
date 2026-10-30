# TD3 Autonomous Policy Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic, guide-free TD3 evaluation that emits trajectory figures, raw traces, and complete mission metrics.

**Architecture:** Add a focused `evaluate_policy.py` entrypoint that reconstructs the formal training environment from `run_config.json`, loads `TD3.Actor` directly on CPU, performs deterministic rollouts, and renders the final trajectory on both fixed radio maps. Keep model selection, rollout, metric construction, and plotting as testable functions.

**Tech Stack:** Python, NumPy, PyTorch, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-10-30-td3-policy-evaluation-design.md`

## Global Constraints

- No MapGuide or exploration noise in evaluation.
- Do not modify training, reward, model, or existing result files.
- Reuse the exact formal-v2 run configuration and fixed radio maps.
- Run Actor inference on CPU so concurrent DDPG training is not disturbed.

## Review Focus

- A checkpoint whose tensor shapes disagree with `run_config.json` must fail clearly.
- A missing checkpoint or run configuration must fail before rollout.
- Success requires endpoint completion, all six visits, full upload, and QoS compliance.
- Timeout, QoS violation, and premature arrival must have distinct termination reasons.
- Reward-term totals must account for the reported return within floating-point tolerance.

---

### Task 1: Deterministic evaluator and report

**Files:**
- Create: `evaluate_policy.py`
- Create: `tests/test_policy_evaluation.py`

**Interfaces:**
- Produces: `load_run_context(run_dir)`, `evaluate_checkpoint(context, checkpoint)`, `plot_trajectory(...)`, and CLI artifacts under `evaluation/`.

- [x] **Step 1: Write failing tests for success classification, termination reasons, and checkpoint/config validation.**
- [x] **Step 2: Run `conda run -n pytorch python -m pytest -q tests/test_policy_evaluation.py` and verify RED.**
- [x] **Step 3: Implement the minimal evaluator, JSON/NPZ serialization, and dual-map plotting.**
- [x] **Step 4: Run the targeted tests and then the complete `tests` suite.**
- [x] **Step 5: Run the evaluator against formal-v2 final and best TD3 checkpoints and inspect the generated figures and metrics.**
- [x] **Step 6: Commit source, tests, design, and plan with a Chinese commit message; leave generated experiment artifacts uncommitted.**

