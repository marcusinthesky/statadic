#!/usr/bin/env bash
# Retry only diagnosed transient bundle-server failures during explicit setup.
set -uo pipefail
log=$(mktemp)
trap 'rm -f "$log"' EXIT
for delay in 10 30 90 0; do
  "$@" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  if (( status == 0 )); then exit 0; fi
  if (( delay == 0 )) || ! rg -qi 'HTTP[^[:digit:]]*(429|5[0-9][0-9])|\b(429|5[0-9][0-9])\b.*HTTP' "$log"; then
    exit "$status"
  fi
  wait=$((delay + RANDOM % 5))
  echo "Transient TeX bundle failure; retrying in ${wait}s." >&2
  sleep "$wait"
done
