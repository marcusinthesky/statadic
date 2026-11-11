#!/usr/bin/env bash
# Capacity preparation is confined to disposable GitHub-hosted Linux runners.
set -euo pipefail
mkdir -p build/ci
diagnose() {
  date --utc --iso-8601=seconds
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
  *) echo 'Usage: ci-capacity.sh [prepare|diagnose [label]]' >&2; exit 2 ;;
esac
