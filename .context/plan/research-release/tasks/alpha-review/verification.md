# Verification record

Candidate: `17fdc5410de31356df72d6fc6459853ca39260f9`.
Production source is unchanged by the review. Review record writes are separate
from candidate implementation evidence.

## Executed in this programme

- Read root/package guidance and repository-workflow/review-agent skills.
- Read Git status, HEAD, net diff and live metadata for all eight stack PRs.
- Created this task and nine child records through pinned `just task-new`.
- Ran the locked `audit.py`. The first invocation used one excess parent level
  and failed before producing a snapshot; corrected the review helper root and
  reran successfully. It wrote the four CSV/JSON review snapshots only.
- Ran `probes.py` through pinned devenv and
  `uv run --project src/python --offline --frozen --package statadic python`.
  Exit zero means the witness collection and independent smooth controls ran;
  it does not mean the witnessed defects passed their intended contracts.
- Ran `just python::statadic::test-fresh tests/api/test_inputs.py
  tests/api/test_inference.py tests/core/test_equations.py -n 0
  --junitxml=/home/marcussky/Git/statadic/build/alpha-review/shared-contracts.xml`.
  Result: **48 passed in 6.24 seconds**, exit zero. Named files only; no full suite.
- Ran `just python::statadic::dead-code`, retaining ignored output at
  `build/alpha-review/dead-code.txt`; exit 3 reports candidates for contextual
  review, not confirmed dead code. No deletion was performed.

## Bounded witness results

CPU, JAX 0.11.0, float64; no CUDA-enabled jaxlib. Candidate as above.
Lock SHA-256: `6e38bbedfe3d63efd14dd48a7c04952f24a79168786272f94800b120787680cb`.
Executed probe SHA-256:
`809cd84af174a0db25c2a8fc35df3502d0b7f5d856905478b56b0bf511492afb`.

| Witness | Observed result | Meaning |
| --- | --- | --- |
| Extreme normal contrast confidence | `(-inf, inf)`, `valid=True`. | Reproduced F001. |
| Writable NumPy-backed Arrow grouping | Four groups became one; repeated inference valid → unavailable. | Reproduced F002. |
| Amplified tolerance-scale negative meat | Covariance approximately `-10`, identified and numerically valid. | Reproduced inherited F003. |
| Angular energy with unrelated Euclidean witness | Zero rooted distance between distinct raw point measures accepted. | Runtime witness for inherited F004; source/type inspection establishes erased binding. |
| Chunked map, 17 rows and chunk 7 | Values, changed inputs, gradient `2x` and HVP `2v` passed. | Bounded JIT/vmap/lax.map remainder controls. |
| Smooth covariance objective `exp(t)/40` | Value, grad, JVP of grad, Hessian, JIT and vmap passed. | Interior-only derivative evidence; no solver/rank-boundary guarantee. |

No performance superiority, Monte Carlo qualification or human source review
follows from these probes. Source files were not repaired.

## Linear and conditional wave

- Ran `linear-probes.py` through the same pinned locked STATADIC project, float64 CPU,
  exit zero. Confirmed F009's fixed and estimated policy changes, F010's
  `KeyError('z')`, and F011's identified added rank one with returned df two.
  The script records adverse observations; exit zero does not mean contracts pass.
- Ran `just python::statadic::test-fresh tests/api/test_regression.py
  tests/api/regression/test__conditional.py tests/api/test_diagnostics.py
  tests/inference/test_linear_diagnostics.py -n 0
  --junitxml=/home/marcussky/Git/statadic/build/alpha-review/linear-conditional.xml`.
  Result: **35 passed, 1 failed in 15.52 seconds**, exit one. Failure is F012's
  obsolete `isinstance(model, Regression)` assertion at line 455.
- Ran the three named graph-consumer files from F015 through `test-fresh -n 0`,
  JUnit `build/alpha-review/grammar-consumers.xml`. Result: **3 collection errors**,
  exit two, 1.26 seconds; no contained tests executed. All errors import the
  deliberately removed composer name.
- Ran `jax-probes.py` through the pinned locked STATADIC project, x64 enabled, exit
  zero. Ignored outputs: `build/alpha-review/jax-probes.json` and
  `build/alpha-review/jax-compile.log`. Confirmed float32 hurdle gate promotion
  (F016), infinite deletion diagnostics with valid flag (F017), and two
  `Compiling jit(cond)` log events on each of two same-shape native Gamma calls.
  Compilation logs establish repeated compilation in this bounded eager path,
  not timing or memory superiority. No long benchmark ran.

## Moments and measurement harness

- Extended `audit.py` to join actual per-method study metrics from the locked
  studies project. Reran successfully: 83 designs; variant declarations include
  bias/MSE 165, size 123, power 55, coverage 101. Wrote `study-designs.json` and
  refreshed review snapshots only. These remain declarations, not executions.
- Ran the named performance-app files `test_measurement.py`, `test_campaign.py`
  and `test_numerical.py` via its locked project. Result: **27 passed in 0.38s**,
  exit zero, JUnit `build/alpha-review/performance-harness.xml`. No native timing
  campaign or full app suite was run.
- Ran `moments-probes.py` through the pinned STATADIC project; initial and extended
  runs exited zero. Extended ignored output: `build/alpha-review/moments-probes.json`.
  Reproduced F018/F019, F020's retained `(64,4,2)` float64 leaf, and F021's
  explicit float32 rescaling failure. The main fixtures are float64; the IV unit
  witness explicitly selects float32. Private CUE Hessian was zero versus the
  independent closed-form value one; this is a first-order envelope limitation.
- Ran `just python::statadic::test-fresh tests/estimation/test_iv.py
  tests/api/regression/test__iv.py tests/estimation/gel/test_fit.py
  tests/inference/test_gmm.py tests/api/test_moments_reporting.py -n 0
  --junitxml=/home/marcussky/Git/statadic/build/alpha-review/moments.xml`.
  Result: **45 passed, 1 deselected in 8.23s**, exit zero. The slow-marked case
  was excluded by the standard recipe. No complete suite or campaign ran.

## Dependent-model witnesses

- Ran `dependent-probes.py` through pinned STATADIC, exit zero; ignored output at
  `build/alpha-review/dependent-probes.json`. Direct imports of both public graph
  selection/resampling modules fail on the removed composer (F015).
- Float32/float64 log-logistic RMST was zero against the independent lower bound
  0.5 under eager, JIT and vmap (F022). This is a demonstrated adverse result,
  not a passing prediction test or qualified integration campaign.
- An initial selected dependent-test invocation named a nonexistent
  `test_graph_likelihood.py`, exited four and ran no tests. Corrected it to
  `test_graph_likelihood_reference.py` and ran it with AFT model, survival fit
  and survival-composition files: **37 passed in 15.80s**, one spreg bounded-
  optimizer tolerance warning, exit zero. JUnit `build/alpha-review/dependent.xml`.
- `graph-binding-probe.py` initially hit the expected float64 input guard in a
  default float32 process; reran with explicit x64 and exited zero. It simulated
  only the known import rename in memory, then reproduced F023. It is evidence
  conditional on F015, not proof that the current public module imports.
- `survival-input-probe.py` exited zero and reproduced F025 at preparation only:
  independent design used duplicated outcome x values 100–105, while zero-lag
  spatial design used node x values 0–5. Keys were identical and reordered.

## Measures and transformation witnesses

- `measures-probes.py`, pinned STATADIC float64 CPU, exited zero. Reproduced the
  cosine-ground nonstationary barycentre with `converged=True` (F026) and the
  all-NaN `grad(sum(eig_clip(I_2)))` against its smooth identity derivative (F027).
  These were analytical controls, not performance/statistical qualification.
- Extended measure probes confirmed invalid weighted probability lanes under eager
  and JIT (F029); output `build/alpha-review/measures-probes.json`.
- Named weighted-MMD, packed-transform, Wasserstein-barycentre and two spectral
  control selections: **47 passed in 7.29s**, exit zero, JUnit
  `build/alpha-review/measures.xml`. These tests do not cover the adverse witnesses.
- `transport-probes.py` first hit a fixture rank guard; corrected point supports
  to rank two. Successful runs reproduced extreme-site self-distance NaNs (F030),
  mutable-mass eager/JIT disagreement (F031), and optimized full-gradient work
  inside the actual sanitized coordinate loop (F035). Ignored outputs:
  `build/alpha-review/transport-probes.json`, `coordinate-optimized-hlo.txt` and
  `nonnegative-qp-optimized-hlo.txt` in the same directory. No timing claim follows.

## Delivery identity controls

- `delivery-probes.py`, pinned locked docs project with x64 enabled, exited zero.
  Mocked optax version metadata left campaign identity unchanged and the release
  identity comparison accepted it. Real threefry/rbg contexts produced different
  same-seed draws while the recorded identity remained equal (F032).
- Mocked CLI orchestration dispatched the abbreviated overridden repository and
  wheel (F033). No installed environment, dependency, suite or release was changed.
  Ignored output: `build/alpha-review/delivery-probes.json`.
- Updated inspection inventories: 379 changed paths had explicit bounded
  inspection records at this checkpoint; 1,054 remained pending primary review.
  This includes unregistered prior evidence reads and substantial test/docs/history
  scope; assignment is not inspection completion. Counts will be refreshed once
  the closure assignments report exact paths.

## Architecture and downstream closure

- Ran `devenv shell -- just python::architecture`: exit zero. Tach internal and
  external boundaries passed (the latter listed unused documentation dependencies
  as advisories); workspace/interface/hidden-import checks passed; **13 architecture
  unit tests passed in 1.279s**; the architecture script's Ruff lint/format checks
  passed. This validates declared dependency rules, not scientific cohesion by itself.
- `migration-probes.py` first failed because the isolated function namespace
  omitted its `Path` annotation name; corrected that helper only. The rerun in
  the locked studies project exited zero. It confirmed removed imports in the
  distribution embedded program, docs extension test and manuscript generator
  (F036). The actual extracted `formal()` body copied the retained macro file
  despite canonical `fresh_output()` rejecting stale certification (F037).
  Only temporary output was created; no full downstream build was run.
- Extended `measures-probes.py` and reran successfully in the locked STATADIC project.
  Checked concatenation changed masses `(.75,.25)` to `(.5,.5)`, mean `.5` to
  zero, while retaining `NonzeroPooledMean` (F038). Existing adverse measure
  witnesses remained reproducible. No production file changed.
- Ran `just python::statadic::test-fresh tests/api/test_composition_ux_hardening.py
  -n 0 --junitxml=/home/marcussky/Git/statadic/build/alpha-review/composition-hardening.xml`.
  Result: **7 passed, 2 failed in 2.80s**, exit one. Both failures are the
  undefined `SurvivalEquation` calls in F040; no production fix was applied.
- Ran `sinkhorn-probes.py` through the locked studies project, exit zero. The
  independent scalar two-atom coupling solve gave product-mass-KL value
  `0.362719935719978`; native eager/JIT agreed. Analytical coordinate gradients,
  independent mass-logit finite-difference gradients/Hessian, HVP, changed-input
  compiled reuse, vmap and unconverged-NaN controls passed. Scope: positive
  interior masses, epsilon one, float64 CPU, 256 fixed updates. No float32,
  zero-mass boundary, general convergence, performance or memory claim follows.

## Execution queue

Only the coordinator runs tests or bounded reproducers. Commands, input source
hashes, exit status, output location, precision and evidence limits are appended
after execution. No proposed command is a passing check.

## Required handoff

Executed one coordinated `devenv shell -- just check` after successful explicit
`just sync`. It exited one in `infra/nix/check.sh`. YAML/TOML/JSON, merge/case/
private-key checks, Python architecture, Nix formatting and Nix anti-pattern checks
passed. Markdown failed on pre-existing MD012 at
`docs/reference/conditional-laws.md:74` and `docs/survival-models.md:122` beneath
the package. The review CSVs also exposed a default-CRLF generator error in the
whitespace hook. Corrected `audit.py` to emit LF, regenerated all four review
CSVs, and reran `prek run text-check --all-files`: **passed**. Production files
were not repaired. The coordinated check itself was not rerun or called passing.

Not reached by that invocation: workflow check, Lean checks/certification, Python
lint/type/shape/Testmon/build/app/doc stages, LaTeX checks and AgentSync checks.
Separate successful architecture/named-test commands above retain their own
scope. A later explicit record sync/read-only workflow check validates final
review metadata without claiming those language stages ran.

Final explicit sync and read-only workflow check both passed: 303 views and
291 sources. Final scoped Ruff lint/format, rumdl on 26 review Markdown files,
and staged/unstaged whitespace diff checks passed. Git confirmed no changes under
Python, Lean, LaTeX, infrastructure or agent source directories; the only existing
record changed outside this review directory is its synchronized parent view.

Selected JUnit executions total **219 package-test passes, three failures and
three collection errors**, plus **27 performance-harness passes**. These are
invocation totals, not necessarily unique tests. Thirteen architecture unit tests
passed separately. Adverse probes establish their recorded failures; the smooth
controls and two-atom Sinkhorn assertions provide bounded positive evidence.
No full suite, long scientific/performance campaign or fresh Lean certification
was run.

Review helpers received scoped Ruff formatting/lint; review Markdown passed
scoped rumdl validation. Formatting and unused-import/lambda cleanup changed
helper bytes after some executions; the first probe hash above identifies its
executed checkpoint. The current helpers remain replayable against the frozen
candidate, and are not represented as byte-identical historical executables.

All 1,433 changed paths now join explicit inspection modes in the log; all 47
catalogued procedures/237 variants join editorial dispositions. Independent
challenges corrected packed-utility retention and procedure-to-variant finding
overpropagation. The review outcome is **hold candidate and implement the repair
backlog**, with scientific qualification and human fidelity gates kept distinct.
