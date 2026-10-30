# Persistent Demonstration Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep a persistent BC anchor and non-evicting demonstration replay, then launch the four requested 3000-episode formal runs.

**Architecture:** Extend ReplayBuffer with separate online and demonstration rings and deterministic mixed sampling. Extend algorithm profiles and the BC schedule with a 0.25 floor for *_bc agents while standard agents remain zero-demonstration.

**Tech Stack:** Python, NumPy, PyTorch, pytest, PowerShell background processes

**Spec:** `docs/superpowers/specs/2026-10-30-persistent-demonstration-replay-design.md`

## Global Constraints

- Variant A, seed 1, QoS limit 100 s, fixed maps and fixed payload vectors.
- Formal outputs go to `journal_results/formal-v5` and never overwrite formal-v4.
- Pure actor evaluation remains guide-free and noise-free.
- Four runs launch only after targeted tests, full suite, and TD3/DDPG short smokes pass.

## Review Focus

- Online buffer wraparound must not overwrite demonstrations.
- Mixed batches must contain 25% demonstrations when both pools are populated.
- Empty-pool fallback must still return a full batch.
- Standard agents must reject nonzero BC floor or demo fraction.
- BC weight must equal 0.25 at and after the decay horizon.

---

### Task 1: Persistent demonstration replay and formal launch

**Files:**
- Modify: `ReplayBuffer.py`, `demonstration.py`, `training_config.py`, `Train_SINR.py`
- Modify: `tests/test_demonstration.py`, `tests/test_training_config.py`

- [x] Write and run failing tests for BC floor, non-evicting demonstrations, mixed ratio, fallback, and profile separation.
- [x] Implement the minimal buffer, schedule, configuration, and logging changes.
- [x] Run targeted tests and full suite.
- [x] Run short TD3-BC and DDPG-BC integration smokes.
- [x] Commit with a Chinese message.
- [x] Launch four hidden formal-v5 processes and report PID/log/result paths.

