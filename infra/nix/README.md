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

Hosted workflows temporarily bypass Nix store archive restoration and saving
after a restore failed with permission errors. The pinned installer and Nix
binary substitutes still prepare the environment; uv, Mathlib and TeX download
caches remain available. A cache-key match alone does not establish that an
archive restored successfully.

The `python-ci` profile supplies the pinned Python, uv, Just and native runtime
without Lean or TeX. Regression, collection and shard jobs prepare only Python
inputs. The default development shell retains the complete toolchain. Hosted
CI temporarily disables JAX's persistent compilation cache after observed cache
eviction and lock overhead. Local Testmon remains available for affected tests.
The early regression gate runs the recorded package failures and the studies
precision regression in float32 and float64, and also prepares
exact locked wheel artifacts and verifies an isolated offline installation,
before the larger test matrix starts. The complete-toolchain job prepares the
same inputs for its independent publishing replication check.

`ci_shards.py` plans four complete Python shards per architecture, retains
canonical node outcomes and verifies their disjoint, exhaustive coverage in the
package workflow. A passing package receipt is bound to the checked-out sources,
environment and workflow run. Repository status checks require this package gate
and the shared x86 workspace checks; they do not emit a full repository receipt.
The legacy scientific/documentation preflight still accepts a separately produced
complete repository receipt, or executes its full checkout gate by default.
Hosted package publication uses the separate package candidate.

Timing history is a scheduling hint, independent of a passing check receipt.
The pytest plugin checkpoints on completed tests when at least 15 seconds have
elapsed since the previous write, and whenever a file completes. A separate job
retains measurements even when a shard assertion
or an unrelated check fails. It verifies checkout, architecture, workflow and
plan identity, accepts only fully completed files, and preserves older costs
for unfinished files. Failed assertions can contribute measured costs; they
cannot produce a successful validation receipt. Malformed timing hints are
ignored. No manually seeded Testmon database is needed by hosted CI.

To merge plan and shard checkpoints within the same workflow without preparing
Python packages:

```sh
python3 infra/nix/ci_shards.py timings \
  --inputs build/ci/inputs \
  --previous build/ci/timing/durations.json \
  --output build/ci/timing/durations.json
```

The input directory contains `plan/plan.json` and `shard-N/timings.json`.
The previous and output paths may be identical. The hosted workflow keeps
architecture-specific artifact names and cache namespaces.
The checkout, architecture, `GITHUB_RUN_ID`, environment and `STATADIC_CI_SEED`
must match the producing run. Version 2 plans retain `GITHUB_RUN_ATTEMPT` as
provenance and accept compatible earlier attempts within that same run. Future
attempts and changed inputs remain invalid. Legacy version 1 receipts retain
strict attempt matching. Partial reruns replace their own same-run artifacts;
missing shard outputs cannot create passing receipts.

The reusable package workflow owns package quality, fast-test shards, installed
reference/precision checks and canonical distributions on both architectures. It
has read-only repository permissions. Installed reports are reconciled with the
authored reference catalogue; their full float64 execution also supplies the
float64 precision subset. The existing reference status names depend on this gate
without rerunning its tests.

Workspace application, Lean, TeX and source checks run once on x86. Package quality
is not repeated in this lane. The former ARM workspace status requires this
shared gate without repeating its execution, preserving existing required check
names. Pages owns the single strict, float64 site build
and its deployment artifact. Local `just check` still validates the full site.
Publication independently verifies a retained candidate for the exact tagged
commit. Testmon is not a publication authority.

`Extended validation` runs weekly and on manual dispatch on x86 and ARM. It
executes the complete notebook script precision matrix and the bounded generated
linear/GLM tests with the existing exploration profile. PRs retain notebook cell
tests, the canonical site export, targeted precision checks and the exact FWL and
spatial regressions. Exploration does not replace scientific qualification.

Shards retain the 25 slowest durations and a traceback after 180 seconds without
test completion. `last-started.json` and streamed node-start messages identify
active tests if a runner stops before the final report. Job timeouts bound the
feedback cycle without marking slow scientific work as successful.

Production keeps two workers with xdist's `load` scheduling. Compare alternatives
using an existing plan in the prepared Python CI environment. Replay requires
the plan's complete identity, including its interpreter and workflow run:

```sh
uv run --project src/python --offline --frozen --all-packages --group reference \
  python infra/nix/ci_shards.py replay \
  --plan build/ci/plan --shard 0 --output build/ci/replay-0 \
  --workers 2 --dist loadfile
```

Repeat with `--dist load`, then with one worker if resource measurements warrant
it. Hold the checkout, seed, float64 CPU settings and memory-bounded setting
fixed. Compare cache settings on the same runner using
`ci-capacity.sh run` for memory and elapsed-time diagnostics. Replay results are
diagnostic only: they emit no passing shard receipt, and nondefault scheduling
cannot replace production timing history. Change the default only after comparing
runtime, peak memory and test outcomes across representative shards.

Hosted scientific tests set `STATADIC_CI_MEMORY_BOUNDED=1` to release in-memory
JAX compilation caches between test modules. This bounds cache accumulation in
long suites; resource logs still
determine whether a particular runner shutdown involved memory pressure.

An early regression failure remains a failure even when a bounded diagnostic
child is run to collect more evidence. Correct a production numerical defect at
its source and require every required CI check to pass before merging; reruns
alone do not resolve an observed failure.
