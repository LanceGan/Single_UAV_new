#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-mps}"
BASE_ARGS=(
  --data_level low
  --episodes 5000
  --a2g_blind
  --outage_max_s 6
  --teacher mapguide
  --guide_prob 0.5
  --guide_min_prob 0.5
  --guided_warmup_episodes 400
  --demo_fraction 0.25
  --demo_weight 1.0
  --demo_margin 0.8
  --bc_weight 1.0
  --bc_min_weight 0.25
  --bc_decay_episodes 600
  --n_step 10
  --action_repeat 2
  --pretrain_demo_episodes 40
  --pretrain_updates 20000
  --learning_rate 1e-4
  --epsilon_decay_steps 300000
  --seed 1
  --eval_interval 25
  --device "${DEVICE}"
)

mkdir -p journal_results/v4-baselines

"${PYTHON}" Train_D3QN.py \
  "${BASE_ARGS[@]}" \
  --output_root journal_results/v4-baselines/drl-sa \
  --sequence_method sa_service \
  2>&1 | tee journal_results/v4-baselines/drl-sa.console.log

"${PYTHON}" Train_D3QN.py \
  "${BASE_ARGS[@]}" \
  --output_root journal_results/v4-baselines/sa-d3qn \
  --sequence_method sa_distance \
  2>&1 | tee journal_results/v4-baselines/sa-d3qn.console.log
