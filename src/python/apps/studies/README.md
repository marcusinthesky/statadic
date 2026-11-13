# Statistical studies

This internal uv application evaluates STATADIC methods without becoming a runtime
wheel dependency. [Spatial graph evidence](spatial-graph-evidence.md) gives PySAL
comparisons, projection-gradient timings and held-out size and power.
Read [ARCHITECTURE.md](ARCHITECTURE.md) before adding a method.
Run commands from the repository root through the pinned devenv environment:

```bash
just python::studies::check
just python::studies::ci-regressions
just python::studies::run build/studies/mmd-smoke --study mmd.finite-bias --profile smoke
just python::studies::resume build/studies/mmd-smoke
just python::studies::summarize build/studies/mmd-smoke
just python::studies::verify build/studies/mmd-smoke
just python::studies::export build/studies/mmd-smoke build/studies/mmd-publication
```

A smoke campaign defaults to 32 independent trials per scenario. An evaluation
campaign defaults to 10,000. Smoke runs cannot qualify a method. Every campaign
freezes its source identities, versions, precision, complete study definition,
seed and trial budget before execution. Run float32/float64 in separate processes
using `JAX_ENABLE_X64=0` and `JAX_ENABLE_X64=1` respectively.
`ci-regressions` runs the fixed studies regression in separate float32 and
float64 processes. The root `just ci-regressions` includes it in the early
hosted gate alongside the recorded package regressions.

`run --start I --stop J` executes a disjoint replicate range. Independent shards
must use identical study, profile, trials, seed and environment. `merge DEST SRC...`
accepts disjoint shards only. `resume` verifies existing records before continuing.
Completed trials are immutable atomic records. An interrupted computation remains
pending; a declared procedure failure remains an attempted failure.

The OLS study compares classical and HC0–HC3 covariance on fixed-design data at
n=64 and 256. Separate scenarios use t/F and normal/chi2 reference laws; selecting
robust covariance does not establish an exact finite-sample reference law.
The MMD study compares U/V reductions using finite-support population truths,
including null/alternative and equal/unequal sample sizes. Their registration does
not establish coverage of the rest of STATADIC. Two further studies evaluate sample
covariance/constant-correlation shrinkage risk and Sharpe point estimates under
Gaussian, skewed and dependent laws. `check --repository PATH` validates the
complete public-symbol catalogue and reports missing scientific studies and source
reviews. Canonical citations with verified metadata still need convention-specific
source review before receiving `verified-method-source` status.

## Retained installed-wheel campaigns

Paired summaries require frozen, explicit comparison groups with a shared
estimand, scale and kernel convention. Shared metric names or coincident null
truths do not establish comparability. Analysis version 2 omits undeclared pairs;
older publication summaries require a new export destination. Raw trial records
and historical campaigns are never rewritten.

Prepare an isolated local environment once, using the pinned devenv shell. This
builds STATADIC, evidence and studies wheels offline with the active locked interpreter
and no build isolation. STATADIC uses the same `--sdist --wheel` route as the release
build, with canonical `SOURCE_DATE_EPOCH=315532800` (1980 ZIP epoch), retaining
both archives. Exact wheel archive identity matters; equal unpacked Python files
with different ZIP timestamps do not satisfy the release gate. It exports hashed
third-party requirements from `uv.lock`,
excluding all workspace/editable packages, then installs those requirements and
only the three retained wheels into a new venv. No campaign starts automatically.
The ordinary uv cache is reused; third-party wheel archives are not duplicated.
Offline preparation requires the locked dependencies in that cache. Use the same
`UV_CACHE_DIR` as the pinned workspace setup, rather than a new empty cache.
Missing dependencies fail preparation with retained logs; populate the pinned
workspace dependency cache before trying a fresh preparation directory.

```bash
just python::studies::installed-prepare build/studies/installed-20261104 1
just python::studies::installed-run build/studies/installed-20261104 \
  build/studies/installed-mmd-smoke --study mmd.finite-bias --profile smoke --trials 2
just python::studies::installed-status build/studies/installed-20261104 \
  build/studies/installed-mmd-smoke
just python::studies::installed-resume build/studies/installed-20261104 \
  build/studies/installed-mmd-smoke
```

The last preparation argument selects x64 (`1`) or float32 (`0`) before any JAX
import. Run creates a suite; repeat `--study` or explicitly choose `--available`.
Evaluation uses at least 10,000 trials per scenario and 999 inner draws. Preparation
retains wheels, requirements, lock/project metadata, catalogue/public inventory,
source-reference records, installation versions, clean import paths and command
logs. It verifies all three installed wheel payloads, frozen file hashes and
installed versions before each invocation. Third-party dependencies are hash-checked
during initial installation; later same-version third-party payload edits are not
detected. Workspace wheel payloads are byte-checked on every invocation. Package
build inputs are fingerprinted before/after all three builds, rejecting concurrent
source additions, removals or edits. Changed or failed preparations require
a new directory. Preparation failures leave logs and never publish a completed
manifest. A preparation is bound to its absolute local path and pinned Python;
keep that venv and its Nix runtime available.

These commands invoke the retained interpreter directly; they never run `uv sync`
in the venv. After workspace edits, run/resume continues to use the retained
studies wheel, frozen catalogue and campaign definitions. For a terminal workflow
that also avoids the current workspace command wrapper:

```bash
build/studies/installed-20261104/venv/bin/python -I -m statadic_studies.installation \
  resume build/studies/installed-20261104 build/studies/installed-mmd-smoke
```

Resume accepts only frozen selections. Evidence reports the retained wheel and
source capsule, not later workspace sources. A completed selected smoke campaign
is neither release qualification nor evidence that all scientific methods were
covered. Source-checkout campaigns remain `imported-source-only`.

## Persistent local suites

Run a selected suite from a local terminal; no service or cloud provisioning is
required. The runner freezes the selected study definitions, complete catalogue
snapshot (including gaps), environment, wheel identity and budgets before trials.
It prepares every campaign manifest, then executes one CPU child process at a time
with the inherited locked interpreter. Precision is explicitly selected before
JAX imports. Logs and append-only state events stay inside the suite directory.

```bash
just python::studies::suite build/studies/local-smoke --repository . \
  --study mmd.finite-bias --profile smoke --trials 2 --x64 1
just python::studies::suite build/studies/local-smoke --resume --x64 1
just python::studies::suite build/studies/local-smoke --status
# Deliberate full registered-design campaign: potentially expensive.
just python::studies::suite build/studies/local-evaluation --repository . \
  --available --profile evaluation --x64 1
```

Repeat `--study` to select several designs, or explicitly choose `--available` for
all currently registered designs. Evaluation suites require at least 10,000
replicates per scenario and 999 inner draws. Smoke execution never qualifies a
method. A completed suite reports selection completion separately from scientific
qualification and never claims that catalogue gaps were evaluated.

Resume rejects changed implementation/engine bytes, dependencies, precision or
study definitions. It verifies completed publications before skipping them and
executes genuinely missing trials. Interruptions remain pending; numerical
procedure failures remain attempted outcomes. Failed child processes stop the
suite with retained logs and events. Interrupted partial publication directories
are retained separately before publication restarts. Use a fresh suite directory
for a changed source revision or design. `--wheel` supports the existing isolated,
noneditable installed-wheel consumer; source-only suites remain source-only.

## Panel regression diagnostics

When the early random-panel precision regression fails, a bounded diagnostic
rerun saves its rounded inputs and execution metadata under
`build/ci/panel-diagnostic/x64-0/` or `x64-1/`. The logged solver state belongs
to the diagnostic rerun; the original failure remains authoritative. Replay a
retained input file with an explicitly selected precision:

```bash
just python::studies::panel-replay build/ci/panel-diagnostic/x64-0/inputs.npz 0
```

The command reports the captured and current environments, then checks direct
and JIT fits and covariance validity. Different hardware or library versions can
change the result.
