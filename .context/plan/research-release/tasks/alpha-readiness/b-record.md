# Inference, moments and Gaussian integration implementation evidence

Authored on 2026-11-09. This record describes Agent B's corrective work from
`850797b`, following the frozen review of `17fdc54`. The coordinator owns source
freeze, serialized execution, canonical inventories and release decisions. This
record grants no numerical, statistical, performance, publication or human
source-fidelity approval.

## Implemented contracts

| Work | Actual change | Independent or adverse control |
| --- | --- | --- |
| R04/F001 | Shared normal critical values use the representable lower tail; coefficient and host contrast intervals require finite bounds, retaining point/SE evidence independently. Constants follow caller precision. | SciPy inverse-survival critical values at extreme representable confidence; unavailable bounds do not erase the retained estimate. |
| R04/F003 | Parameter covariance checks the final composed sampling covariance for PSD availability, including amplification by the parameter map. | A negative meat direction that passes a scale-relative input tolerance becomes an invalid amplified covariance lane. |
| R04/F011 | Breusch–Godfrey requires every requested lag restriction to increase rank. | A collinear added lag fails the requested reference instead of using the requested degrees of freedom. |
| R04/F017 | Influence retains scalar full-fit availability and separate observation masks for leverage, internal/external studentization, Cook distance and DFFITS, plus observation-by-parameter DFBETAS masks. Deleted variance must be distinguishably positive. | A degenerate deletion leaves unaffected rows and full-fit/leverage quantities intact. A owns host reporting of these masks. |
| R07 | Retired profiled legacy J entry points; active studies use the retained actual-fit, actual-weight J interface. GEL checks parameter identification in its frozen retained moment coordinates against the original derivative scale. | Finite-budget GMM retains its actual last solve weight; the projected-flat GEL witness cannot authorize a chi-square reference. |
| R08 | A narrow shared affine mean-equation solve supports IV without manufacturing an observation-by-moment-by-parameter derivative tensor or a fictitious affine fit. IV retains actual row scores, aggregate Jacobian, actual/effective weight, parameter map, dimensions and solver evidence. Native first-stage summaries replace host arithmetic. | Independent NumPy weighted solve/sandwich, actual-weight provenance, retained leaf shapes, extreme instrument-unit 2SLS controls and descriptive first-stage controls. |
| R10 | Full-weight KKP uses the published Gaussian quadratic-form covariance terms, with the zero-diagonal graph requirement explicit. | Exact Gaussian fourth-moment quadrature supplies an independent small-dimensional tau matrix; optional spreg parity is a separate convention check. |
| S007–S010/S018 | Removed dead legacy-J helpers and duplicate active profiled-study definitions. Panel cluster inference delegates the existing parameter-cluster owner; time HAC streams lag products and uses entity/time row lookup instead of an entity-by-time-by-parameter cube. | Existing independent covariance controls remain the acceptance targets. No measured speed or RSS reduction is asserted. |

`normal_critical_value`, `AffineMeanSolve`, `solve_affine_mean_equation`,
`IVMomentFit` and `iv_first_stage_diagnostics` are explicit new public primitives
approved by the coordinator for their actual callers. Retired public symbols are
`LinearHansenJResult`, `LinearHansenJKernelResult`, `linear_hansen_j` and
`linear_hansen_j_kernel`; historical data is retained. The active profiled plain
and centered J study definitions map to the corresponding actual-fit J definitions.

Extreme instrument-unit controls establish the tested 2SLS point/covariance
boundary. They do not establish arbitrary-unit invariance for every efficient or
iterated IV variability gate. Private CUE objective higher derivatives and
derivatives of finite optimization iterates are not promoted to fitted-solution
derivatives by these changes. Optional simplifications whose concrete duplicate
or active consumer was not established are not claimed complete.

Central panel execution exposed two follow-up defects in this lane. Both streamed
HAC loop indices now accept scalar integer tracers and eager integer indices;
integer row-position generation matches its storage dtype. The independent
weighted Fama–MacBeth seed `12000` then exposed an `int32` period-count division
that rounded its debias correction to float32 before multiplying a float64
covariance. Panel, group and Fama–MacBeth corrections now calculate in the actual
floating sampling precision, preserving integer counts and degrees of freedom.
An independent eight-period sample-covariance witness and the explicit external
seed retain the regression. Tolerances and formulas are unchanged. The build-only
`fmb-tail/probe.py` retains independent point/covariance/tail decompositions and
reconstructs the legacy rounded correction; B did not execute it.

## Explicit optional Gaussian proposal

`GaussianProposal` and `gaussian_tilt_proposal` implement a frozen sequential
normal importance proposal. Preparation uses deterministic marginal-bound
ordering and maintained L-BFGS on Botev's saddle residuals. An independently
recomputed residual is retained separately from actual optimizer termination.
This is not a certified minimax solver, a deterministic integration-error bound
or a rare-tail efficiency theorem.

`IntegrationDesign` has an optional proposal with the existing no-proposal
default. `gaussian_log_survival` evaluates the current Gaussian target with the
frozen order/tilts and exact importance ratio; it analytically integrates the
terminal coordinate. Proposal dimensions and permutation validity are explicit.
Caller precision, random keys, draw prefixes and the target Gaussian are
preserved. No covariance ridge or probability clipping is added. Derivatives
refer to the fixed-proposal approximation; differentiating refitted preparation
is outside the contract.

`prepare_spatial_survival_proposals` eagerly prepares a separate conditional
censored proposal per period at explicit pilot parameters. It shares the exact
selected-inverse-row covariance and Schur-complement construction with the
likelihood, preserves full-graph marginalization and supports different censor
dimensions. Refinement preserves proposals while extending draw prefixes;
training and validation retain distinct keys. Proposal preparation explicitly
rejects positive entry. Default spatial survival remains unchanged.

New controls compare changed target parameters under a frozen proposal against
independent scalar Gaussian quadrature, including finite-difference gradient and
Hessian controls. Native controls include distinct period censor dimensions and
a complete censored rho-zero fit, observed-information and refinement comparison
with the independent ordinary lognormal AFT path. Passing low-dimensional
controls cannot close the required high-censor workflow gate.

`SolverResult`, `GaussianProposal` and `IntegrationDesign` use structural pytree
unflattening so Equinox/JAX can construct Boolean/None filter trees without
invoking numerical constructor validation. Solver and proposal fields remain
dynamic; integration replicate/draw budgets remain static. Direct constructors
retain strict runtime contracts. Mask, exact leaf roundtrip, JIT/vmap and explicit
invalid-constructor controls distinguish these two uses.

The executable build-only pilot
`build/alpha-readiness/gaussian-proposal-pilot.py` retains complete-graph and
full-graph/missing-outcome one-factor controls, independent scalar quadrature,
distinct training/validation keys, source/lock/runtime identity and timing/RSS.
It is marked `qualification: false` and was not executed by B. It does not refit,
compute standard errors, establish calibration or cover arbitrary graphs. Its
default small workload is a pilot before the 500-node, 20-period matrix.

The hard spatial-survival gate remains 10/50/80 percent censoring under both
graph semantics, per-period independent log-probability error at most `1e-3`,
and identified refit shifts at most `0.02` observed-information standard errors.
The optional numerical brick does not close this gate or repair the default
high-censor path by declaration.

## Exact implementation and regression manifest

Paths below are relative to `src/python/packages/statadic/`, except the studies row.
Some listed files also contain separately owned edits; this manifest describes
B's scoped changes, not exclusive ownership of every diff.

| Responsibility | Production paths | Focused test paths |
| --- | --- | --- |
| Shared inference | `src/statadic/core/equations.py`; `src/statadic/inference/parameters.py`; `src/statadic/inference/linear_diagnostics.py`; `src/statadic/api/inference.py` | `tests/contracts/test_review_contracts.py`; `tests/api/test_inference.py` |
| Moments/GEL | `src/statadic/estimation/gmm.py`; `src/statadic/estimation/gel/fit.py`; `src/statadic/inference/_common.py`; `src/statadic/inference/gmm/hypothesis.py` | `tests/estimation/test_gmm.py`; `tests/inference/test_gmm.py`; `tests/inference/test_gel.py` |
| IV | `src/statadic/estimation/iv.py`; `src/statadic/api/regression/_iv.py` | `tests/estimation/test_iv.py` |
| Panel covariance | `src/statadic/inference/panel_covariance.py` | `tests/inference/test_panel_covariance_validation.py`; `tests/estimation/test_panel_reference.py`; existing owner/reference tests remain coordinator-run targets. |
| KKP | `src/statadic/estimation/spatial/kkp.py` | `tests/estimation/spatial/test_kkp_reference.py` |
| Gaussian proposal | `src/statadic/optimize/integration/gaussian.py`; `src/statadic/optimize/integration/__init__.py`; `src/statadic/estimation/survival/spatial.py`; `src/statadic/estimation/survival/refinement.py`; `src/statadic/estimation/survival/__init__.py` | `tests/optimize/integration/test_gaussian.py`; `tests/estimation/survival/test_spatial.py` |
| Structural solver reconstruction | `src/statadic/core/results.py`, SolverResult only | `tests/core/test_results.py` |
| Studies migration | `src/python/apps/studies/src/statadic_studies/adapters/moments.py`; `src/python/apps/studies/src/statadic_studies/definitions/moments.py` | `src/python/apps/studies/tests/test_moment_inference.py` |

Root owns the shared GMM initializer removal, Tach edges, catalogue associations,
public API/primitive inventories and derived identities. The proposal and
survival initializer additions were explicitly delegated to B. No compatibility
aliases, historical-data deletions or Pricing Perspective production edits were
introduced by this lane.

## Verification and source review limits

B ran scoped Ruff formatting and lint through the materialized pinned devenv
profile and locked offline uv project. The standard devenv shell could not
connect to the sandboxed Nix daemon; the coordinator was informed. B ran no
numerical Python, test, scientific/performance campaign, suite, certification or
final `just check`. Root's serialized development outcomes and exact current
retry state belong to [progress](progress.md), not this implementation record.

The coordinator's final named panel selection retained
`build/alpha-readiness/panel-precision-final.xml` and its log: **59 passed in
19.71 seconds**, with unchanged tolerances. This closes the remaining precision
failure exposed after the earlier 106-pass/one-failure tracer-repair selection;
these are separate selections, not additive totals. The independently executed
`build/alpha-readiness/fmb-tail/result.json` records unchanged source hashes and
the actual old float32 correction `1.1428571939468384`, versus the sampling-dtype
float64 value `1.1428571428571428`. Reconstructing the old correction reproduces
the `1.25534418e-8` endpoint shift. The native Student-t critical value differs
from SciPy by only `6.21724894e-15`; repaired native/linearmodels interval bounds
differ by at most `1.55431224e-15` in this seed. This is a precise correction
regression and independent reference check, not statistical, performance or
full-release qualification. Current source/mapping, scientific, resource,
candidate-artifact and release gates remain unchanged.

The latest strict-test typing repair gives callbacks concrete argument types,
asserts optional proposals/keys before use, and casts only a deliberately invalid
Boolean fixture for the strict-constructor rejection test. It does not weaken
production annotations. Final strict typing and scientific regression outcomes
must come from central execution.

The [source correspondence record](source-correspondence.md) records 36 exact
variant recommendations and the actual primary passages read, with narrow
configuration/domain exclusions. It leaves canonical source dispositions and
all 153 human claim fidelity reviews untouched. Existing approved symbols were
audited separately: 25 broad sets required narrowing; nested DSR dependencies
were legitimate. Root applied canonical corrections.

B's independent read-only challenge of C's energy capability, weighted-mass,
simplex barycentre, spectral derivative and extreme one-dimensional transport
repair diffs found no additional confirmed target/formula defect within those
inspected paths. C independently challenged the proposal importance ratio,
saddle equations and conditional Gaussian bridge. These are bounded challenges,
not universal API, source-fidelity, transformation or release signoff.

## Read-only release cohesion challenge

The current alpha contract and 176-variant selection preserve the ordinary-AFT,
resumed dependent-model and downstream boundaries; source/study gaps and the
unimplemented large joint-Gaussian numerical adapter remain fail-closed. The
small Gaussian pilots and passed consumer rehearsal are explicitly bounded.
No additional native arithmetic defect was confirmed in this cohesion pass.

This review found that `apps/docs/scripts/release.py` selected every
`qualification` test even with an explicit scope, including Sharpe/equivalence
procedures outside the selected alpha. The supplemental arm below repairs that
coupling. No incorrectly passing release was observed.

The numerical evidence reader requires at least one successful case per named
file. The scoped long lane now uses stricter exact-node verification, and root
clears ambient `PYTEST_ADDOPTS`/`PYTEST_PLUGINS` before release children. No
deselected release execution was observed or run by B. These repairs change no
source approvals, scientific thresholds or resource gates.

## Supplemental qualification arm

`apps/docs/src/statadic_docs_labs/qualification_lanes.py` explicitly associates
canonical variant IDs with **27 exact supplemental parameter nodes** for the
current scope. Every selected variant must still retain nonempty valid canonical
study/scenario/method cells, whose campaign qualification remains independently
mandatory. This arm supplies no adapter for the large joint-Gaussian numerical
contract and cannot close that gap.

Retention and verification rederive the exact expected/executed nodes, reject
missing, extra, duplicate, failed, errored or skipped cases, and bind mapping,
owner/test/helper source, wheel and scope identities. A campaign-backed scope
with zero supplemental nodes records that explicit disposition without an XML
report or a bare pytest fallback. Root integrated execution/verification and
environment clearing; the unscoped all-method `-m qualification` default is
unchanged.

Six native qualification files changed only explicit parameter IDs:
`test_linear.py`, `test_covariance.py`, `test_panels.py`, `test_survival.py`,
`test_resampling.py` and `test_samples.py`. Formulas, data generators, budgets and
tolerances are unchanged. Root confirmed collection of all 27 selected nodes:
**zero scientific trials ran** in that collection check. Focused docs regression
tests were authored in `apps/docs/tests/test_qualification_lanes.py`; this record
does not infer their execution outcome.

Root separately reported the final affected-native selection **213 passed in
71.62 seconds**, two studies files **37 passed in 52.11 seconds**, strict typing
with **zero errors**, and **seven shape tests passed**. These are separate
development selections, not additive totals, scientific qualification or release
approval. Final artifact/docs verification and the existing source, mapping,
statistical, full spatial-survival and resource gates remain open. B ran no
additional tests, checks or campaigns.
