#!/usr/bin/env bash
# Prepare ignored installed GNU/Linux executables without modifying uv caches.
set -euo pipefail
: "${PROJECT_RUNTIME_LINKER:?Enter the pinned devenv shell first}"
: "${PROJECT_RUNTIME_LIBRARIES:?Enter the pinned devenv shell first}"
for executable in "$@"; do
  [[ -f "$executable" ]] || continue
  interpreter=$(patchelf --print-interpreter "$executable" 2>/dev/null) || continue
  # Static executables and musl binaries do not use the GNU loader.
  [[ "$interpreter" == *ld-linux* ]] || continue
  rpath=$(patchelf --print-rpath "$executable")
  if [[ "$interpreter" == "$PROJECT_RUNTIME_LINKER" && "$rpath" == "$PROJECT_RUNTIME_LIBRARIES"* ]]; then
    continue
  fi
  temporary=$(mktemp "${executable}.XXXXXX")
  trap 'rm -f "$temporary"' EXIT
  cp --dereference --preserve=mode "$executable" "$temporary"
  chmod u+w "$temporary"
  patchelf --set-interpreter "$PROJECT_RUNTIME_LINKER" \
    --set-rpath "$PROJECT_RUNTIME_LIBRARIES${rpath:+:$rpath}" "$temporary"
  mv -f "$temporary" "$executable"
  trap - EXIT
  printf 'Prepared native runtime: %s\n' "$executable"
done
