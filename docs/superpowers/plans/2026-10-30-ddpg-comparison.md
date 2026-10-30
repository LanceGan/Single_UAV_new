# DDPG Evaluation and TD3 Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate completed DDPG checkpoints under the same autonomous protocol as TD3 and produce an evidence-based comparison.

**Architecture:** Generalize `evaluate_policy.py` around the algorithm recorded in `run_config.json`; both actors share the same checkpoint-compatible bounded network. Preserve the existing output schema so TD3 and DDPG metrics are directly comparable.

**Tech Stack:** Python, NumPy, PyTorch, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-10-30-ddpg-comparison-design.md`

## Global Constraints

- No MapGuide or exploration noise during either algorithm's evaluation.
- Use the same maps, payload vector, inspection order, QoS threshold, horizon, and metric definitions.
- Do not alter trained checkpoints, training rewards, or the World dynamics.

## Review Focus

- Reject unsupported algorithms while accepting both TD3 and DDPG configurations.
- Resolve algorithm-specific checkpoint filenames correctly.
- Preserve CPU-only import and inference.
- Keep TD3 output behavior backward compatible.
- Clearly distinguish a guided-period training best from autonomous evaluation success.

---

### Task 1: Generalize evaluation and run DDPG comparison

**Files:**
- Modify: `evaluate_policy.py`
- Modify: `tests/test_policy_evaluation.py`

- [x] Write failing tests for DDPG configuration acceptance and checkpoint naming.
- [x] Implement algorithm-aware context, labels, report metadata, and checkpoint paths.
- [x] Run targeted and full tests.
- [x] Evaluate DDPG final and training_best, inspect both figures, and compare against TD3 metrics.
- [x] Commit code, tests, spec, and plan with a Chinese message.

