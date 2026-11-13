#!/usr/bin/env bash
# Orchestration only: Lake, Batteries, Mathlib and import-graph own the analyses.
set -euo pipefail
cd "$(dirname "$0")/.."
shopt -s globstar nullglob
sources=(Example.lean Example/**/*.lean packages/StatadicFormal/StatadicFormal.lean
  packages/StatadicFormal/StatadicFormal/**/*.lean lakefile.lean tools/**/*.lean)
# Complete indexes make both default roots cover all library sources.
case "${1:-lint}" in
  lint)
    bash scripts/quality.sh indexes
    LAKE_NO_CACHE=true lake --no-cache --wfail build Example StatadicFormal +tools.quality.Style
    lake lint
    lake env bash scripts/quality.sh text-style
    ;;
  indexes|indexes-fix)
    lake build mathlib/mk_all
    for spec in '. Example' 'packages/StatadicFormal StatadicFormal'; do
      read -r directory library <<< "$spec"
      executable="$PWD/.lake/packages/mathlib/.lake/build/bin/mk_all"
      if [[ "$1" == indexes-fix ]]; then
        (cd "$directory" && "$executable" --lib "$library" --module) || true
      fi
      (cd "$directory" && "$executable" --lib "$library" --module --check)
    done
    ;;
  fmt)
    lake env bash scripts/quality.sh text-style --fix
    ;;
  text-style)
    flags=()
    if [[ "${2:-}" == --fix ]]; then flags=(--fix); fi
    lean --run tools/quality/Style.lean "${flags[@]}" "${sources[@]}"
    ;;
  builtin) lake lint --builtin-only ;;
  report) lake lint --code-quality ;;
  imports)
    lake build
    lake build +ImportGraph.Lean.WithImportModules +ImportGraph.Imports.Unused
    lake env lean --run .lake/packages/importGraph/MainUnusedTransitiveImports.lean Example StatadicFormal
    ;;
  *) echo "unknown quality action: $1" >&2; exit 2 ;;
esac
