#!/usr/bin/env bash
# Capacity preparation is confined to disposable GitHub-hosted Linux runners.
set -euo pipefail
mkdir -p build/ci
memory() {
  date --utc --iso-8601=seconds
  free -m
  # Report numeric process usage without exposing command-line arguments.
  ps -eo pid,ppid,rss,comm --sort=-rss | head -n 12 || true
  local group
  group=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
  for metric in memory.current memory.peak memory.max memory.events; do
    if [[ -r "/sys/fs/cgroup${group}/${metric}" ]]; then
      echo "${metric}: $(cat "/sys/fs/cgroup${group}/${metric}")"
    fi
  done
}
diagnose() {
  date --utc --iso-8601=seconds
  memory
  df -h / /nix /tmp 2>/dev/null || df -h /
  df -i / /nix /tmp 2>/dev/null || df -i /
  du -sh /nix/store ~/.cache/uv ~/.cache/mathlib src/lean/.lake 2>/dev/null || true
}
case "${1:-diagnose}" in
  prepare)
    [[ "${GITHUB_ACTIONS:-}" == true && "${RUNNER_ENVIRONMENT:-}" == github-hosted ]] || {
      echo 'Refusing SDK removal outside a GitHub-hosted runner' >&2
      exit 1
    }
    diagnose | tee build/ci/capacity-before.txt
    # No STATADIC command uses the preinstalled Android, .NET, Haskell or CodeQL SDK.
    sudo rm -rf /usr/local/lib/android /usr/share/dotnet /opt/ghc /opt/hostedtoolcache/CodeQL
    diagnose | tee build/ci/capacity-prepared.txt
    ;;
  diagnose) diagnose | tee "build/ci/capacity-${2:-current}.txt" ;;
  run)
    label=${2:?A log label is required}
    [[ $label =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid log label' >&2; exit 2; }
    shift 2
    (( $# )) || { echo 'A command is required' >&2; exit 2; }
    # A lost runner can skip even always() steps. Stream telemetry while the
    # command is alive so its last measurements survive in the Actions log.
    (
      sleeper=''
      trap 'if [[ -n $sleeper ]]; then kill "$sleeper" 2>/dev/null || true; fi; exit 0' TERM INT
      while true; do
        { echo "Runner resources: ${label}"; memory; } | tee -a "build/ci/memory-${label}.log"
        sleep 30 &
        sleeper=$!
        wait "$sleeper"
      done
    ) &
    monitor=$!
    command_pid=''
    cleanup() {
      if [[ -n $command_pid ]]; then
        kill -- "-$command_pid" 2>/dev/null || kill "$command_pid" 2>/dev/null || true
        wait "$command_pid" 2>/dev/null || true
      fi
      kill "$monitor" 2>/dev/null || true
      wait "$monitor" 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    # Give the whole command pipeline its own process group for cancellation.
    setsid --wait bash -o pipefail -c 'log=$1; shift; "$@" 2>&1 | tee "$log"' \
      bash "build/ci/${label}.log" "$@" &
    command_pid=$!
    status=0
    wait "$command_pid" || status=$?
    command_pid=''
    exit "$status"
    ;;
  *) echo 'Usage: ci-capacity.sh [prepare|diagnose [label]|run label command ...]' >&2; exit 2 ;;
esac
