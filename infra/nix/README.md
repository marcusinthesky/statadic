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

The global library path supplies GCC runtime libraries and zlib, leaving libc to
each executable's loader. Exporting Nix glibc there can crash host-loaded wheel
executables on Ubuntu. `PROJECT_RUNTIME_LIBRARIES` retains the full Nix runtime
for `prepare-runtime.sh`, which also selects its matching loader.

`ci-capacity.sh prepare` removes explicitly listed unused SDKs only on disposable
GitHub-hosted runners. Diagnostic mode is read-only and records disk/inode usage
and major cache sizes under ignored `build/ci/`. `run LABEL COMMAND ...` retains
the command's combined output and exit status, and streams memory, process and
cgroup usage every 30 seconds. These measurements remain in the Actions console
if a lost runner cannot upload its final artifacts. CI retains diagnostics on
failure; x86-64 and ARM64 validation do not cancel one another.

Workflows restore a Nix store per architecture, profile and locked environment,
then save the prepared store before running validation. Tool downloads are saved
at the same boundary. A later test failure therefore leaves setup reusable.
The store save collects unreferenced paths toward a 3 GiB target; live environment
roots remain protected. No publishing credentials are needed for these caches.

The `python-ci` profile supplies the pinned Python, uv, Just and native runtime
without Lean or TeX. Regression, collection and shard jobs prepare only Python
inputs. The default development shell retains the complete toolchain. CI saves
JAX's persistent compilation cache separately for each architecture, CPU identity,
locked environment and shard, with a 256 MiB limit per cache. Cached compilation
still executes each test; local Testmon remains available for affected tests.

`ci_shards.py` plans four complete Python shards, retains canonical node outcomes
and verifies their disjoint, exhaustive coverage alongside the remaining root
checks. A passing aggregate is bound to the checked-out sources, environment and
workflow run. PR release preflight can consume that same-run receipt; standalone
release checks still execute the full checkout gate.

Hosted scientific tests set `STATADIC_CI_MEMORY_BOUNDED=1` to release in-memory
JAX compilation caches between test modules. The persistent compilation cache is
preserved. This bounds cache accumulation in long suites; resource logs still
determine whether a particular runner shutdown involved memory pressure.
