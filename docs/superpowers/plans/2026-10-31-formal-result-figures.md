# Formal Result Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate individual paper-ready trajectory, objective, and reward figures for all four formal-v5-restart runs.

**Architecture:** Add one reusable CLI that validates eval_history/NPZ data, extracts feasibility-aware series, loads eval_best through the existing evaluator, and writes PNG/PDF/JSON artifacts under each run.

**Tech Stack:** Python, NumPy, Matplotlib, PyTorch, pytest

**Spec:** `docs/superpowers/specs/2026-10-31-formal-result-figures-design.md`

## Global Constraints

- Failed early terminations must not be presented as feasible low T or T+tau values.
- Use eval_best checkpoint and deterministic, guide-free rollout for trajectories.
- Do not modify model checkpoints or training outputs.
- Produce both PNG and PDF for every figure.

## Review Focus

- DDPG has zero feasible evaluations and must still render without an empty-array crash.
- Moving-average episode alignment must be correct.
- T+tau must equal the two source fields exactly.
- All four run directories must produce the same artifact set.

---

### Task 1: Figure generator and four-run execution

**Files:**
- Create: `plot_formal_results.py`
- Create: `tests/test_formal_result_plots.py`

- [x] Write and verify failing tests for feasibility masking and moving-average alignment.
- [x] Implement extraction, plotting, trajectory evaluation, and CLI.
- [x] Run targeted and full tests.
- [x] Generate and visually inspect all four artifact sets.
- [x] Commit with a Chinese message.

