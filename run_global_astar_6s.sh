#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUT="journal_results/v4-deterministic/astar-global-6"
mkdir -p "${OUT}"

for level in low medium high; do
  "${PYTHON}" -u run_astar.py \
    --levels "${level}" \
    --mode global \
    --outage-max-s 6 \
    --output-dir "${OUT}" \
    --max-expansions 20000000 \
    --max-labels 20000000 \
    --max-runtime-s 14400 \
    --progress-every 100000 \
    2>&1 | tee "${OUT}/${level}.console.log"
done
