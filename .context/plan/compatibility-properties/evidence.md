# Compatibility and property-testing execution evidence

Execution begins at `cdf44d0`, following the accepted plan of 2026-11-11.
The earlier reporting/estimand migration is already in `f21bddb`; it is not
reimplemented. The repository-local review transcript was removed at the owner's
request. The second transcript remains outside this repository.

## Confirmed defects

- With JAX 0.11.0, `from_dlpack(copy=True)` aliases a 64-byte-aligned NumPy buffer
  exported through Arrow. Deliberately offset storage does not alias. Before
  the fix, the hardened ownership regression reports one failure and one pass.
  An explicit JAX array copy completes independent ownership before return.
- The three JAX-version-sensitive failures exercise exception wording and exact
  floating random bytes. They now compare direct QR capability and STATADIC's key
  schedule, precision, shapes and statistical baselines instead.
- A generated reporting case exposed a remaining contradictory interval state:
  `interval_status="available"`, no endpoints, and a nonempty unavailable reason.
  It was accepted. The constructor now rejects an available status with an
  unavailable reason; the minimal case is retained as a literal example.

## Verification so far

- Ownership and JAX regression owner files: **48 passed**.
- Budget helper and subprocess validation support: **58 passed**.
- Initial corrected API properties and strategy self-tests: **23 passed**.
- The AST migration retained all **331** existing literal base budgets. Profiles
  now act on each explicit tier, rather than only the global default setting.
- Four installed-wheel environments ultimately passed **46 targeted tests each**:
  Python 3.13.14 and 3.14.10, lowest-direct and highest resolutions.
- Lowest runtime versions: JAX/jaxlib 0.11.0, jaxtyping 0.3.3, lineax 0.1.0,
  optax 0.2.4, optimistix 0.1.0, qpax 0.1.0, polars 1.3.0, pyarrow 24.0.0;
  matplotlib 3.9.2 on Python 3.13 and 3.10.5 on Python 3.14.
- Highest runtime versions: JAX/jaxlib 0.11.2, jaxtyping 0.3.11, lineax 0.1.1,
  optax 0.2.8, optimistix 0.1.0, qpax 0.1.4, polars 2.1.0, pyarrow 26.0.1,
  matplotlib 3.11.2. The scientific contributor lock stays at its existing
  versions; the compatibility tools add packaging 26.3 in the contributor closure.

These targeted runs do not qualify the full fast suite or every interior
combination of dependency ranges. Python 3.15, JAX 0.12 and GPUs remain deferred.
Final verification and record disposition follow below when completed.

## Retained compatibility checks

[Complete resolutions and selected nodes](compatibility.json) retain four
endpoint environments with **46 passing targeted tests each** and a CPU
float32 environment with **22 passing targeted tests**. These are individual
environment results, not one combined full-suite qualification.

A clean base-wheel install imported **266 modules**. Only the explicitly optional
`api.tables` and contributor `testing.fixtures` imports required absent pyarrow
and pytest. No pytest, pyarrow, polars or matplotlib was installed. A nonexistent
test-path probe confirmed the Just/tox launcher returns a nonzero pytest result
(exit 4), rather than reporting a successful run with no tests.

A collection-only audit compared actual decorated settings with the pre-change
source: **480 parameterized property nodes retained their default budgets**.
No full test suite was executed for this audit.

Two disjoint named-file external-reference batches passed **60** and **34** tests,
with no skips or xfails. They cover all external-reference files changed by the
earlier final-review migration, plus the changed shared linear reference and
precision witnesses. The `statadic-reference` cap was active.

The exploratory API search passed **29 properties** with fivefold budgets and no
unresolved counterexamples. A final two-test exploratory rerun passed after
making integer-weight replication keep its expanded row count fixed; both tests
also passed in the CPU float32 wheel environment.

Package strict typing reported **0 errors** and the isolated shape lane passed
**7 tests**. Focused layout validation passed for **268 source modules and 190
test files**. Explicit repository formatting and synchronization completed.

The first handoff check required an explicit API inventory refresh. Review of
the generated diff found only the reporting and table source hashes changed;
exports, signatures, statements and method mappings did not change.

The affected library lane passed **3,746 tests**, with **7 skips**. The publishing
replication then exposed uv cache separation: compatibility-only uv had shadowed
the pinned repository uv in the shared environment. The launcher now uses an
isolated, locked, no-dev tool environment. The ordinary workspace was restored
with locked synchronization and its exact publishing runtime cached during
explicit preparation. Offline publishing then passed **19 tests**. A final
isolated-launcher probe passed both aligned/offset wheel regressions and
confirmed no uv executable was left in the ordinary workspace environment.

## Alpha-review correction

The earlier diagnosis of the seed-42 spatial-panel failure as an existing
precision assumption was incomplete. The added average-score check in `f21bddb`
can reject the good float32 candidate, activating an older fallback that chooses
the zero-variance fit without comparing likelihoods. A read-only reproduction at
`5c663e9` returned log likelihood -148.50116, variance ratio zero, valid/converged
true and nonfinite covariance. A positive-variance search reached -90.31139 with
ratio 5.335. The proposed float64-only smoke skip was withdrawn.

The score check now has a floor of 16 working-precision epsilons in both initial
selection and recovery. The score is per observation in normalized spatial and
log-ratio coordinates; the floor does not depend on an arbitrary likelihood
offset. A boundary candidate must match or improve the retained likelihood
within the existing comparison tolerance. A better unconverged candidate stays
unconverged instead of being relabelled as boundary success.

Metric availability now rejects an unavailable reason when a value is available.
All 8 metric, 8 interval and 32 test-row states run through parametrization under
every profile. Renderer checks use four distinct reasons. Deterministic
axis-aligned fits exercise exactly zero SSE and tiny positive SSE; the existing
Gaussian likelihood rule is unchanged, with no near-zero cutoff.

Python 3.13–3.14 is the declared range; compatibility evidence remains targeted.
Documentation and the changelog explicitly distinguish random-key/shape/dtype
stability from floating-draw bitwise reproducibility across JAX patch releases.
Historical study reproduction requires its exact frozen environment.

The three named library owner files passed **201 tests** after these repairs.
The named studies file passed **26 tests**, including fresh-process float32 and
float64 success, the -90.31 likelihood, positive variance ratio and finite
covariance. Strict package typing reported **0 errors**, and isolated shapes
passed **7 tests**. Final handoff verification follows below once complete.
The previous full check reached strict Sphinx and failed on a shortened
standard-library `Path` annotation; the explicit standard-library exception now
covers that spelling as well as `pathlib.Path`.
