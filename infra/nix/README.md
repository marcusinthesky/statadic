---
description: Pinned system tools and native package definitions.
---

# Nix support

The root devenv.nix imports the environment defined here. Package definitions
under pkgs/ pin Schematter, the matched Lean toolchain, and optional AgentSync.
AgentSync is built with Cargo inside Nix; projects need no npm wrapper or
separate Rust installation. Setup prepares only the remaining workflow caches.
The bootstrap installs the CLI matching the root Devenv lock.

Keep Lean, Mathlib, Architect, manifests and release hashes in step when updating.

Lean's verified binary archive is extracted directly into its Nix output,
avoiding an unpack-and-copy peak across the builder/store filesystem boundary.
The package retains the upstream compiler integration, executable wrappers and
automatic ELF patching. The shell exposes its pinned native libraries for uv's
binary Python wheels. Lua 5.4 regenerates static documentation artwork only.

`ci-capacity.sh prepare` removes explicitly listed unused SDKs only on disposable
GitHub-hosted runners. Diagnostic mode is read-only and records disk/inode usage
and major cache sizes under ignored `build/ci/`. CI retains these diagnostics on
failure; x86-64 and ARM64 validation do not cancel one another.
