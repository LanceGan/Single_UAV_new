#!/bin/zsh

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1

KAPPAS=(0.30 0.20)
RUN_ROOT="${ROOT}/journal_results/v4-kappa"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}/status"

pids=()
for kappa in "${KAPPAS[@]}"; do
  tag="${kappa/./p}"
  output_root="${RUN_ROOT}/kappa${kappa}"
  log_path="${LOG_DIR}/kappa${kappa}.log"
  status_path="${RUN_ROOT}/status/kappa${kappa}.exit"
  (
    "${PYTHON}" "${ROOT}/Train_SINR.py" \
      --variant A \
      --data_level low \
      --algorithm td3_bc \
      --teacher mapguide \
      --seed 1 \
      --episodes 3000 \
      --ga_iterations 500 \
      --outage_max_s 6 \
      --outage_probability_threshold "${kappa}" \
      --allow_preflight_failure \
      --objective_w1 0.5 \
      --objective_w2 0.5 \
      --device mps \
      --output_root "${output_root}" \
      > "${log_path}" 2>&1
    print -r -- "$?" > "${status_path}"
  ) &
  pids+=("$!")
  print -r -- "kappa=${kappa} pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

for kappa in "${KAPPAS[@]}"; do
  status_path="${RUN_ROOT}/status/kappa${kappa}.exit"
  status="$(<"${status_path}")"
  print -r -- "kappa=${kappa} exit=${status}"
  [[ "${status}" == "0" ]] || failed=1
done

exit "${failed}"
