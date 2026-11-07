#!/bin/zsh
set -u
set -o pipefail

cd "$(dirname "$0")"

root="journal_results/v4-sequence-ablation/a2g-aware"
log_root="$root/logs"
mkdir -p "$log_root"

common=(
  --variant A
  --data_level low
  --algorithm td3_bc
  --teacher mapguide
  --guide_prob 0.5
  --guide_min_prob 0.2
  --guided_warmup_episodes 400
  --bc_weight 1.0
  --bc_min_weight 0.25
  --bc_decay_episodes 600
  --demo_fraction 0.25
  --episodes 3000
  --seed 1
  --eval_interval 25
  --outage_max_s 6
  --objective_w1 0.5
  --objective_w2 0.5
  --device mps
)

methods=(distance-tsp eega yang)
selectors=(distance_tsp eega yang)
statuses=()

for index in {1..3}; do
  method="${methods[$index]}"
  selector="${selectors[$index]}"
  log_path="$log_root/${method}.log"
  echo "Starting ${method} at $(date '+%Y-%m-%d %H:%M:%S')"
  PYTHONUNBUFFERED=1 python3 Train_SINR.py \
    "${common[@]}" \
    --sequence_method "$selector" \
    --output_root "$root/$method" \
    2>&1 | tee "$log_path"
  status=${pipestatus[1]}
  statuses+=("$method=$status")
  echo "Finished ${method} at $(date '+%Y-%m-%d %H:%M:%S') with status ${status}"
done

printf '%s\n' "${statuses[@]}" > "$root/queue_status.txt"
if printf '%s\n' "${statuses[@]}" | grep -q '=[^0]$'; then
  echo "sequence ablation queue completed with failures"
  exit 1
fi

touch "$root/COMPLETE"
echo "sequence ablation queue completed successfully"
