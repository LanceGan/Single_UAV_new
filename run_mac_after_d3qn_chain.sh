#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:-}"
if [[ -n "${wait_pid}" ]] && kill -0 "${wait_pid}" 2>/dev/null; then
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 60
  done
fi

exec "$(dirname "$0")/run_mac_v4_remaining.sh"
