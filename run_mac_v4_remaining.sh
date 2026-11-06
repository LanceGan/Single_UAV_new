#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-mps}"
OUT_ROOT="journal_results/v4-baselines"
LAYER1_DIR="journal_results/v4-models/layer1/d3qn/low/d3qn"
mkdir -p "${OUT_ROOT}" journal_results/v4-models

if [[ ! -f "${LAYER1_DIR}/eval_history.json" ]]; then
  "${PYTHON}" Train_D3QN.py \
    --data_level low \
    --episodes 3000 \
    --output_root journal_results/v4-models/layer1/d3qn \
    --sequence_method iga \
    --outage_max_s 6 \
    --teacher mapguide \
    --guide_prob 0.5 \
    --guide_min_prob 0.5 \
    --guided_warmup_episodes 400 \
    --demo_fraction 0.25 \
    --demo_weight 1.0 \
    --demo_margin 0.8 \
    --bc_weight 1.0 \
    --bc_min_weight 0.25 \
    --bc_decay_episodes 600 \
    --n_step 10 \
    --action_repeat 2 \
    --pretrain_demo_episodes 40 \
    --pretrain_updates 20000 \
    --learning_rate 1e-4 \
    --epsilon_decay_steps 300000 \
    --seed 1 \
    --eval_interval 25 \
    --device "${DEVICE}" \
    2>&1 | tee journal_results/v4-models/layer1-d3qn.console.log
else
  printf 'Skipping completed layer-1 D3QN: %s\n' "${LAYER1_DIR}"
fi

for method in distance-tsp eega yang; do
  "${PYTHON}" Train_SINR.py \
    --variant A \
    --data_level low \
    --algorithm td3_bc \
    --teacher mapguide \
    --guide_min_prob 0.2 \
    --episodes 3000 \
    --seed 1 \
    --eval_interval 25 \
    --outage_max_s 6 \
    --a2g_blind \
    --device "${DEVICE}" \
    --sequence_method "${method//-/_}" \
    --output_root "${OUT_ROOT}/${method}" \
    2>&1 | tee "${OUT_ROOT}/${method}.console.log"
done

"${PYTHON}" Train_SINR.py \
  --variant A \
  --data_level low \
  --algorithm td3_bc \
  --teacher mapguide \
  --guide_min_prob 0.2 \
  --episodes 3000 \
  --seed 1 \
  --eval_interval 25 \
  --outage_max_s 6 \
  --device "${DEVICE}" \
  --sequence_method distance_tsp \
  --output_root journal_results/v4-baselines/distance-tsp-a2g-visible \
  2>&1 | tee journal_results/v4-baselines/distance-tsp-a2g-visible.console.log
