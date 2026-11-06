#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUT="journal_results/v4-deterministic/pso-joint"
mkdir -p "${OUT}"

for level in low medium high; do
  "${PYTHON}" -u pso_trajectory.py \
    --data_level "${level}" \
    --seed 1 \
    --swarm 40 \
    --iterations 60 \
    --outage_max_s 6 \
    --output "${OUT}" \
    2>&1 | tee "${OUT}/${level}.console.log"
done
