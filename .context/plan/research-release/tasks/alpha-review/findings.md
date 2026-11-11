# Findings and adjudication

Candidate: `17fdc5410de31356df72d6fc6459853ca39260f9`.
This record separates confirmed defects, design risks and missing evidence.
Worker hypotheses remain unconfirmed until the coordinator inspects decisive
source or executes an appropriate bounded witness. No alpha approval is implied.

## Adjudicated findings

| ID | Priority/category | Origin | Coordinator decision |
| --- | --- | --- | --- |
| F001 | P2 correctness | Introduced | Fix contrast critical values/availability before alpha. Reproduced. |
| F002 | P2 retained-state correctness | Introduced | Snapshot Arrow deferred columns before alpha. Reproduced. |
| F003 | P2 numerical contract | Inherited | Consolidate covariance assembly and validate composed PSD. Reproduced. |
| F004 | P2 mathematical composition | Inherited | Bind energy witness to generating ground; narrow capability until then. Reproduced runtime witness plus source challenge. |
| F005 | P2 evidence association | Current metadata | Narrow Fama–MacBeth symbol/source scope. Independently challenged. |
| F006 | P2 evidence denominator | Current design | Register distinct positive-Poisson/hurdle procedures or explicitly defer. Independently challenged. |
| F007 | P2 evidence provenance | Standalone reference harness | Bind retained outcomes to actual imported source and lock. Independently challenged. |
| F008 | P3 documentation | Current editorial drift | Date historical counts and correct claims of a current snapshot. Independently challenged. |
| F009 | P2 model specification | Introduced | Preserve declared Gamma/NB2 dispersion under estimator substitution. Reproduced. |
| F010 | P2 input integration | Introduced | Resolve supplied deferred variance columns for diagnostics. Reproduced. |
| F011 | P2 statistical availability | Introduced | Withhold unqualified BG calibration under deficient added-lag rank. Reproduced. |
| F012 | P2 test regression | Introduced | Remove obsolete Conditional-is-Regression expectation. Named test fails. |
| F013 | P2 getting started | Introduced | Match tutorial dependencies to documented installation. Source confirmed. |
| F014 | P2 evidence denominator | Current design | Register calibrated diagnostics as substantive variants. Source confirmed. |
| F015 | P1 public import/migration | Introduced | Migrate graph production/test consumers to Equation. Two public modules and three test modules fail import. |
| F016 | P2 precision | Introduced | Preserve float32 in generated hurdle gates. Reproduced. |
| F017 | P2 availability/design | Introduced | Define and validate deletion-based influence availability. Reproduced nonfinite diagnostics. |
| F018 | P2 calibration | Inherited | Migrate legacy affine-J to actual-fit gates, then retire duplicate path. Reproduced. |
| F019 | P2 identification | Inherited | Require parameter identification within retained GEL coordinates. Reproduced. |
| F020 | P2 memory/design | Introduced IV path | Remove retained expanded IV derivative tensor using aggregate solve evidence. Allocation verified. |
| F021 | P2 numerical units | Introduced IV path | Equilibrate the instrument-rank decision or declare the restrictive unit boundary. Reproduced. |
| F022 | P2 numerical prediction | Introduced survival path | Do not certify underflowed RMST quadrature as agreement. Reproduced in both precisions and transforms. |
| F023 | P2 composition | Introduced graph path | Remove or reject a learned binding fixed by bandwidth selection. Witness conditional on F015 migration. |
| F024 | P2 published-method fidelity | Inherited KKP option | Correct full-weight tau or defer that option. Primary equation/derivation mismatch confirmed. |
| F025 | P2 data authority | Introduced survival path | Resolve conflicting node/outcome predictor authority consistently. Preparation witness reproduced. |
| F026 | P2 convergence contract | Inherited barycentre path | Restrict mean-update convergence to its Euclidean objective. Nonstationary cosine result reproduced. |
| F027 | P2 transformation contract | Inherited PSD surface | Correct repeated-eigenvalue AD or narrow the blanket grad promise. Smooth interior NaN gradient reproduced. |
| F028 | P2 unnecessary allocation | Introduced graph path | Bypass dense identity filters for pure SLX. Source-confirmed; timing unmeasured. |
| F029 | P2 invalid-lane correctness | Introduced measure migration | Mask invalid probability lanes and preserve unavailable diagonals. Reproduced. |
| F030 | P2 transport arithmetic | Introduced quantile path | Mask zero-width intervals before overflow-prone powers. Reproduced. |
| F031 | P2 immutable state | Introduced carrier | Reject/copy writable supplied masses. Eager/JIT disagreement reproduced. |
| F032 | P2 execution identity | Current campaign/release lane | Bind computational dependency and effective JAX configuration identities. Independently confirmed. |
| F033 | P2 frozen-input enforcement | Introduced installed CLI | Reject abbreviations that override frozen repository/wheel. Dispatch probe confirmed. |
| F034 | P3 evidence inventory | Introduced performance harness | Re-enumerate source paths after workers to catch additions. Independently confirmed. |
| F035 | P2 computational work | Introduced QP change | Restore a row derivative in coordinate descent. Actual optimized solver retains two full products per coordinate. |
| F036 | P2 downstream migration | Introduced gates/examples | Migrate removed constructors in distribution qualification, docs extension tests and manuscript generation. Candidate export mismatches confirmed. |
| F037 | P2 manuscript evidence | Introduced generator | Use the existing formal freshness check before copying certified exposition. Actual stale-copy acceptance reproduced. |
| F038 | P2 weighted evidence | Introduced carrier integration | Preserve masses when concatenating checked clouds; do not mint a false pooled-mean certificate. Reproduced. |
| F039 | P3 review transparency | Introduced metadata projection | Render canonical pending-human status as well as source correspondence for seventeen affected claims. Independently source-confirmed. |
| F040 | P2 contract-test regression | Introduced tests | Migrate two undefined SurvivalEquation calls to the retained model contract. Named file has seven passes and two NameErrors. |
| F041 | P2 notebook migration | Introduced carrier integration | Retire the unmaintained kernel-mixture demo or migrate both inputs to PackedClouds and correct its claims. Source-confirmed; not executed. |

All findings refer to frozen candidate `17fdc54`. Priorities are review triage,
not proof of a production incident. F003/F004 are release risks discovered here,
not introduced regressions. None has been repaired by this programme.

## F001 — Infinite contrast bounds reported as valid

`src/python/packages/statadic/src/statadic/api/inference.py`, `InferenceResult.contrast`,
computes `ndtri((1 + confidence) / 2)` after accepting all `0 < confidence < 1`.
At `0.9999999999999999`, host addition rounds the argument to one. The bounded
float64 run returned estimate `0.9106176924781324`, SE `0.1123862125398818`,
CI `(-inf, inf)`, `valid=True`, no unavailable reason. This violates D03's
explicit availability contract. The independent challenger confirmed that the
shared coefficient-interval path uses the stable lower tail and a finite guard.

**Repair acceptance:** Share stable critical-value and availability logic with
coefficient intervals; retain point estimates and SEs when only calibration is
unavailable. Check ordinary and extreme confidence in both precisions against
an independent inverse-survival reference. No valid interval may have nonfinite
bounds. Add a meaningful contrast-specific regression case.

## F002 — Arrow backing-buffer mutation changes retained inference

`src/python/packages/statadic/src/statadic/api/inputs.py:140` retains `ArrowColumns(table)`
for deferred grouping labels. Arrow arrays can share writable NumPy storage;
`api/tables.py`'s immutability assumption does not freeze that storage. The
coordinator verified shared buffer addresses. After fitting with four groups,
mutating the caller's integer buffer changed retained labels from
`(0,0,1,1,2,2,3,3)` to eight zeros. Clustered inference changed from available to
unavailable on the same retained point fit. Mapping inputs already copy extras.

**Repair acceptance:** Snapshot deferred Arrow columns with ownership sufficient
to prevent later caller mutation. Preserve optional-table dependency boundaries
and deferred role validation. Test aliased primitive and string/chunked columns,
repeated clustering, retained parameters and direct explicit observations.
Measure preparation/retained memory separately; correctness precedes copy tuning.

## F003 — PSD tolerance is amplified into materially negative valid covariance

`src/python/packages/statadic/src/statadic/core/equations.py`, `parameter_covariance`,
checks input PSD at an input-relative tolerance and final finiteness, but omits
the composed PSD check implemented by `_parameter_covariance_from_limiting`.
With float64 `J=[[0],[1e-8]]`, `Omega=diag(1,-1e-15)`, `W=I`, count one, the
input tolerance accepts Omega; the coordinator obtained covariance approximately
`[[-10]]`, `identified=True`, `numerically_valid=True`. This contradicts the
carrier's contract. Base arithmetic is the same, so this is inherited.

**Repair acceptance:** Route the generic limiting result through the existing
validated assembly owner while retaining panel-count, rank and input checks.
Fail unavailable on materially indefinite composed output; keep singular PSD
support. Any projection of input Omega is a separate scientific decision.
Verify the analytical witness, rescaled controls and eager/JIT/batched behavior.
Do not replace native QR paths with normal equations.

## F004 — Energy metrization accepts an unrelated ground witness

`core/matrices.py:803`, `metrize_energy`, accepts an independent strong-negative-
type `DomainLaw`; the ground type never occurs in the input matrix provenance.
`discrepancy/mmd/energy/matrix.py:232` erases its actual ground and exponent in
the returned functional type. The coordinator produced angular energy between
point masses at `(1,0)` and `(2,0)`, then supplied a Euclidean-domain witness:
the returned separating-metric contract has zero distance between distinct raw
empirical measures. The function discards both witness arguments at runtime.

This is missing construction/witness binding, not a demand for runtime proof of
a population theorem. The two inspected Lean claims correctly require the same
generating map and an injective mean embedding. No invalid Lean theorem was found.

**Repair acceptance:** Bind the existing construction evidence to the qualified
ground, exponent and equality, or let a qualified producer perform refinement.
Reject mismatched-ground composition statically where promised; retain an
independent separating Euclidean control. Narrow the advertised capability until
that contract exists. Avoid a parallel matrix hierarchy.

## F005 — Fama–MacBeth source association includes unrelated estimators

`src/python/apps/studies/catalogue.json:10988` assigns the Fama–MacBeth variant
twelve symbols, including Between, first-difference, pooled, random-effects and
within estimators. Its approved source scope is only the arithmetic mean of
periodwise coefficients. The source decision ledger requires exact producers.
The checker validates membership/evidence presence but cannot establish scope.

The independent challenge found no demonstrated runtime approval bypass:
documentation renders status per variant and the campaign gate checks aggregate
mapping/review conditions. The established defect is inaccurate machine-readable
association and misleading symbol queries.

**Repair acceptance:** Narrow this variant's producers, audit all 32 approved
symbol sets against actual approved scope, and test that unrelated estimators do
not inherit that association. Keep source and statistical qualification separate.

## F006 — Hurdle and positive-Poisson procedures vanish from the denominator

`catalogue.json:545–617` accounts for `HurdlePoisson`/`HurdleResult` as an API
wrapper. The underlying positive-Poisson producer is grouped with likelihood
owners having only Gamma/NB/probit studies. There is no distinct scientific
variant for positive-Poisson or composed hurdle estimation/prediction. Conditional
documentation describes a new zero-truncated likelihood and four prediction
targets. Hand-arithmetic tests are useful evidence, but do not fill missing
source/size/bias/power coverage.

**Repair acceptance:** Author the scientific procedures/variants and explicit
gaps, or remove/defer the alpha feature. Define estimands, partial availability,
source scopes, analytic/reference controls and relevant campaigns. Do not claim
the current 237-variant denominator exhausts the API.

## F007 — Standalone reference evidence lacks implementation identity

`src/python/packages/statadic/scripts/check_references.py:111–120` records software
versions, platform, suite layout and outcomes but not HEAD, lock/source hashes,
actual imported package identity or before/after source stability. A same-version
code change can leave a retained green manifest with no binding to implementation.
Historical retained outcomes therefore cannot establish this candidate's parity.

This finding is scoped to the standalone reference harness. Scientific campaign
and release-wheel gates have stronger source/capsule/wheel identities and must
not be described as equally unbound.

**Repair acceptance:** Bind actual imported source or exact installed wheel,
lock and definitions; detect changes during execution; distinguish checkout and
standalone installed runs. Retain adverse historical evidence. Test mismatched
source at unchanged version and interrupted/failed execution.

## F008 — Obsolete counts presented as a current superseding snapshot

`.context/plan/research-experience/catalogue-audit.md:9–19` describes 43 procedures,
191 variants and 75 designs as current and superseding history. The audited
candidate has 47/237 and 83 registered studies, with new mapping gaps. Other
current-basis prose retains obsolete design totals. Historical counts are valid
when dated; the current/superseding label is the defect.

**Repair acceptance:** Date preserved snapshots and link reproducible current
counts with denominator limits. Do not overwrite historical campaign results.

## F009 — Estimator substitution changes the conditional law

`api/regression/_conditional.py:515–537` uses the law's dispersion only when it
constructs a default estimator. An injected Gamma/NB2 estimator supplies its own
policy; `_likelihood.py:158–169` checks family and names but accepts either fixed
or estimated schemas. The coordinator declared Gamma fixed dispersion `0.5` and
obtained valid fitted dispersion `2.0` with an injected fixed estimator. An
estimated-policy injection instead returned `0.18529343609311158` and added a
dispersion parameter. Three review lanes independently identified this mismatch.

**Repair acceptance:** Keep the law authoritative and reject incompatible
estimator policies/results, or remove the duplicate declaration with a clear
authoritative owner. Preserve substitution of matching solvers with different
tolerances. Test Gamma/NB2 fixed values and fixed/estimated mismatches in both
directions, including parameter and covariance dimensions.

## F010 — Deferred raw-input columns cannot serve a variance diagnostic

`api/diagnostics.py:355` directly calls `observations.numeric(name)`. An unused
numeric `z` supplied alongside a raw mapping fit remains in the retained source
rather than eager numeric columns. `BreuschPagan(variables=("z",))` then raises
`KeyError('z')`; equivalent explicit `Observations` works. The documented request
names numeric training columns, with no requirement that they enter the mean.
The coordinator reproduced the failure on 20 rows.

**Repair acceptance:** Resolve and validate named numeric training columns from
the retained fit-time snapshot with explicit role/row checks. Preserve deferred
validation of irrelevant unused columns. Verify mapping, Arrow, explicit
observations, missing/nonnumeric columns and caller mutation. Reuse the existing
input owner rather than add a parallel diagnostic-data system.

## F011 — Breusch–Godfrey reports calibration without full added-lag rank

`inference/linear_diagnostics.py:271` always uses chi-square degrees of freedom
equal to requested lags; validity does not require that many added restrictions.
With four rows, `X=[1, (0,-1,0,1)]`, `y=(-1,0,1,0)` and two lags, the independent
NumPy ranks are 2 and 3. STATADIC returns statistic `8/3`, df 2, p `0.26359714`,
`valid=True`, although only one added restriction is identified.

**Repair acceptance:** Require the rank assumptions of the declared calibration
or return unavailable with a specific reason. Any reduced-rank reference is an
explicit scientific choice needing its own argument. Check duplicate and zero
lag columns, full-rank reference parity, retained row order and JIT execution.

## F012 — A migrated logistic test asserts an obsolete hierarchy

`tests/api/test_regression.py:455` asserts that `_logit(...)` is a `Regression`;
the helper now returns `Conditional`, which derives from `Model` directly. The
selected run failed exactly there; fitting had completed. This is a test defect,
not evidence that Conditional should be forced into a single-design base.

**Repair acceptance:** Update the hierarchy assertion and retain behavioral,
native-fit and `Model` checks. Keep the two-design hurdle composition possible.
Consolidate duplicated migration helpers only when useful cases remain explicit.

## F013 — Minimal documented installation cannot run the first regression

`docs/start/installation.md:12–13` installs STATADIC without optional data extras;
`examples/first_regression.py:12` imports Polars unconditionally. The first
tutorial tells readers to run it after that installation. Development and
distribution-test environments include optional table dependencies and conceal
the mismatch. This conclusion is from dependency/source inspection; no new
clean-environment installation has been executed by this review.

The same issue affects `poisson_exposure.py`, which requires PyArrow. Installation
creates `.venv` without activation, while both tutorial commands use bare
`python`, so the documented interpreter can also be wrong.

**Repair acceptance:** Explicitly install `statadic[data]` for the tutorial, or use a
mapping for the first example and introduce Polars afterward. Activate the named
environment or invoke its interpreter consistently. Smoke the exact
documented installation and downloaded script in a fresh environment. Direct
PGFplots export need not acquire an unrelated plotting dependency.

## F014 — Calibrated linear diagnostics are omitted as scientific variants

`catalogue.json:6193–6227` classifies the diagnostic owner scientifically but
has no scientific procedure IDs. Its rationale cites package qualification
instead of studies-app registration. Eight calibrated configurations therefore
sit outside the 237-variant accounting denominator. Independent diagnostic
reference tests and Gaussian-null size tests do exist; this is an inventory
defect, not a claim that diagnostics are untested.

**Repair acceptance:** Register each retained substantive calibration and link
the existing evidence with exact scope and remaining power/source gaps. Keep
host/native parity distinct from independent numerical oracles. Specifically,
the reference-marked host test calling the same native Breusch–Pagan kernel is
a contract test; its marker cannot establish independent arithmetic coverage.

## F015 — Incomplete composer migration breaks public graph modules

`api/composition.py` defines/exports `Equation` and deliberately removes the old
composer `Regression`. Three current test modules still import the removed name:
`tests/api/test_spatial_graph_regression.py`, `test_graph_learning.py` and
`test_graph_resampling.py`. The coordinator's selected run produced three
collection `ImportError`s and executed no tests in those modules.
Production `api/graph_selection.py:10` and `api/graph_resampling.py:12` also
import the removed name. Direct central imports confirmed both failures.
`docs/tutorials/learned-panel-graphs.md` teaches the removed import and
`docs/reference/composition.md:460` falsely promises an alias.

**Repair acceptance:** Migrate all current consumers/examples/prose to Equation
and remove the alias promise. Preserve the intentional clean alpha break (D04,
D05). Collect and run all affected named graph tests and the documented example.
Do not restore an obsolete alias to hide incomplete migration.

## F016 — Generated hurdle gate ignores response precision

`api/regression/_conditional.py:343–344` leaves an intercept/categorical-only
gate design boolean, then derives zero-indicator dtype from that design and a
weak Python float. With x64 enabled and a float32 caller response, the central
probe returned float64 gate parameters and float32 count parameters. This breaks
caller-owned precision and can double part of the numerical state unexpectedly.

**Repair acceptance:** Normalize generated gate design and indicator using the
caller response/input precision, as existing ordinary regression preparation
does. Verify logit/probit gates, intercept/categorical designs and both precisions
inside an x64-capable process; do not change global precision inside the library.

## F017 — Influence validity includes undefined deletion-based quantities

`inference/linear_diagnostics.py:330` sets one validity flag before computing
deleted-row variance. Intercept-only `y=(0,0,0,1)` gives zero variance after
deleting the last row. The coordinator obtained `valid=True` alongside infinite
external studentization, DFFITS and DFBETAS. The related Lean identity explicitly
assumes strictly positive deleted variance; that hypothesis is not checked here.
The three-row variant instead produced very large finite values due to roundoff.

**Design decision required:** Mathematical extended-real limits may be useful
descriptive outputs, but cannot silently stand for finite qualified deletion
evidence. Preserve valid full-fit quantities and introduce appropriate per-row/
per-quantity availability, or document a deliberately narrower carrier contract
that downstream reporting handles honestly. A single blanket validity flag needs
an explicit meaning. Test zero, negative-roundoff and near-zero deleted variance,
independent leave-one-out controls, and unchanged full-fit diagnostics. The code
reviewer challenged whether infinity itself is erroneous; the coordinator keeps
this as an availability/design finding rather than a false algebra claim.

## F018 — Legacy affine-J bypasses actual-weight validity

`inference/gmm/hypothesis.py:151` fits iterated GMM, then recalculates inverse
variability at final scores for J instead of validating the retained generating
weight. Its eager wrapper accepts coefficient convergence alone. On
`g0=((0,1),(2,2),(1,4),(5,1))`, common `D=((1),(0))`, plain bandwidth zero,
one update and tolerance one, the coordinator observed legacy statistic
`2.9572261641435165`, p `0.08549449291875018`. The corresponding retained-fit
path correctly withheld inference because its weight was not efficient.

**Repair acceptance:** Migrate the active studies adapter to actual-fit Hansen-J
gates, preserving separate fixed-moment tests. Retire `linear_hansen_j`, its
kernel and duplicate result carriers after checking callers. Verify convergence
without weight fixed-point, efficient controls, singular variability and exact
identification. This is inherited arithmetic; the candidate changed naming/types.

## F019 — GEL subtracts a parameter absent from its frozen profile

`estimation/gel/fit.py:228` checks full mean-derivative rank before freezing the
stochastic subspace. Equal initial/final stochastic ranks do not show that the
projected parameter is identified. For `v=(1,2,3,4)`, `w=(-1,1,-1,1)`,
`g0=[v,w,0]`, derivative rows `[-v_i,0,v_i]`, the starting theta is `0.5` and
the retained-coordinate derivative norm is about `3e-15`. The coordinator still
obtained convergence, stable rank two, `reference_valid=True`, df one and
statistic `3.4482758620689653`. The profile is flat in the parameter being
subtracted. No reviewed Lean result licenses this profile/reference extension.

**Repair acceptance:** Check parameter rank in the frozen retained coordinates
under an appropriate numerical scale. Reject or explicitly defer unsupported
profile calibration. Test rotating/degenerate retained spaces and full-rank
controls; preserve independent fixed GEL. This is an inherited alpha risk.

## F020 — IV retains an avoidable observation-by-instrument-by-parameter tensor

`estimation/iv.py:196` forms `-Z[:,:,None] * X[:,None,:]` and retains it through
the affine fit. The bounded full-fit probe retained a `(64,4,2)` float64 leaf
of 4,096 bytes. Its storage scales as `8*n*m*p`; 100,000×100×50 consumes
4,000,000,000 bytes for that leaf alone, before designs, scores or solver state.
An aggregate derivative `-Z.T @ X/n` suffices for the solve; row scores can be
computed as `Z*(y-X beta)`. This is a concrete memory design concern, not a
measured reference-method superiority claim.

**Repair acceptance:** Reuse aggregate mean-Jacobian/weighted-solve evidence and
retain actual maps, generating/effective W, preliminary/final variability,
rank, criterion, stationarity and separate iteration convergence. Keep QR initial
weighting and correct sandwich covariance. Merely broadcasting a common Jacobian
into the existing generic fit still retains the tensor. Verify full fit and
inference against independent IV oracles and measure isolated comparable memory.

## F021 — Instrument units can erase otherwise resolvable 2SLS identification

`estimation/iv.py:184–185` uses raw `matrix_rank(Z)` before the unit-equilibrated
GMM solve. Float32 orthogonal instruments `[1,t]`, `t=(-2,-1,0,1,2)`, give valid
coefficients approximately `(1.02,2)`. Rescaling only the second instrument by
the exactly representable, invertible factor `2^-24` returns unavailable NaNs.
The span and 2SLS estimand are unchanged; no input information was rounded away.
The coordinator reproduced this. Raw unit-sensitive rank is a stricter boundary
than the downstream weighted solve, not inevitable solver failure.

**Repair acceptance:** Equilibrate the instrument-rank/QR admission consistently
with the retained numerical contract, preserving true deficient-rank rejection
and actual W. Test independent unit rescalings in float32/64, covariance and
invalid batched lanes. If retained as an explicit numerical limitation, narrow
the public promise and workload envelope instead of implying unit invariance.

## F022 — Repeated quadrature underflow becomes a false RMST prediction

`model/survival/aft.py:261–264` accepts successive log-logistic quadrature
orders when they agree, including repeated zero caused by underflow. With
parameters `(0,-20)`, intercept design and horizon one, every transformed
quadrature node underflows. The coordinator obtained RMST zero in float32 and
float64, eagerly, JIT-compiled and vmapped. Independent monotonicity gives
`integral_0^1 S(t) dt >= S(1) = 0.5`, so zero is demonstrably incorrect.
Current tests do not cover this small-shape regime.

**Repair acceptance:** Stabilize or change the bounded integration variable,
or return unavailable when underflow prevents resolution. Distinguish numerical
agreement from accuracy. Check survival bounds `h*S(h) <= RMST <= h`, zero
horizon, small/large shapes, independent high-precision integration and both
precisions/transformation modes. Preserve caller precision and finite-horizon
meaning even when the unrestricted mean does not exist.

## F023 — Fixing a bandwidth candidate leaves an invalid learned binding

`api/graph_selection.py:101–107` preserves a supplied GraphML estimator while
`_bind` replaces the selected bandwidth reference by a fixed number. GraphML
then rejects the now-unused learned binding. The coordinator reproduced this
at preparation/initialization: an unbound smooth parameter initialized normally;
after binding the fixed candidate it raised “Graph bindings must match all
references used by the model.” The probe temporarily simulated only F015's
import migration in its own process; the current candidate still fails import.

**Repair acceptance:** Remove the selected reference from candidate/refit learning
bindings while preserving other learned parameters, or explicitly reject this
composition at selection entry. Test fixed grid selection with jointly learned
other parameters and final refit. F015 must be repaired first; the probe created
no supported alias and edited no production source.

## F024 — KKP full weighting does not match the published covariance

`estimation/spatial/kkp.py:240–241` uses `2*tr(W'W)/N^2` off diagonal and
`tr((W'W)^2)/N` on the second diagonal. [KKP (2007), equation 26 and its
appendix derivation](https://econweb.umd.edu/~prucha/papers/JE140(2007a).pdf)
instead give denominator N and a factor two on that diagonal. The source lane
read both; the coordinator independently inspected the primary equation text.
PDF screenshots were unavailable. Normality is required for optimal weighting.

For row-normalized `W=((0,1,0),(.5,0,.5),(0,1,0))`, the expected first two
rows are `(2,5/3,0)` and `(5/3,17/6,0)`; implementation gives `(2,5/9,0)`
and `(5/9,17/12,0)`. Both full matrices are positive definite. This is inherited
and changes weighting. External spreg agreement does not establish source fidelity;
the related Lean claim proves commutation rather than this covariance formula.

**Repair acceptance:** Correct the formula with independently derived Gaussian
quadratic-form controls, or defer `full_weights=True`. Review downstream fitted
parameters/inference and preserve the partial-weight option's distinct method.
Investigate comparator conventions/shared defects rather than bending the
published formula to maintain a green external comparison.

## F025 — Independent survival can use different covariates from a zero-lag baseline

`api/survival/_model.py:274–329` copies outcome-table columns first and fills
predictors from full nodes only when absent. Spatial preparation always uses
full-node covariates. The central reordered-key preparation witness supplied
node x values 0–5 and duplicated outcome x values 100–105. Independent preparation
used the latter; spatial preparation with fixed rho zero used the former.
No optimizer was involved. The helper promises alignment of full-node covariates.

**Repair acceptance:** Define one authoritative predictor source across the
same PanelData contract or reject conflicting duplicated values. If outcome
overrides are intentional, expose and document that choice rather than silently
changing the model when dependence is added. Test numeric, categorical and
measure roles with reordered/subset outcomes, duplicates and zero-lag controls.

## F026 — Non-Euclidean barycentre reports convergence away from stationarity

`geometry/barycentre/wasserstein/measure.py` accepts general ground distances but
uses the ambient weighted mean as its fixed-assignment minimizer and certifies
stable assignments whenever p is two. That minimizer is justified for squared
Euclidean distance, not arbitrary grounds. For singleton sources `(1,0)` and
`(0,1)`, weights `(0.8,0.2)`, cosine ground and p two, the coordinator obtained
support `(0.8,0.2)`, `converged=True`, objective `0.11546363210587697`.
The independent derivative along its direction is `-0.2823529411764706`, so this
is not even stationary. The behavior is inherited. Related Lean barycentre
statements assume Euclidean quadratic geometry.

**Repair acceptance:** Restrict this solver/convergence certificate to its
qualified Euclidean objective, or supply a real fixed-assignment minimizer for
other grounds. Remove the exact solver's misleading redirect to an unsupported
alternative. Verify independent stationarity/objective controls and distinguish
fixed assignments from minimization of the declared ground objective.

## F027 — Smooth eigenvalue clipping has NaN gradients at repeated eigenvalues

`optimize/psd.py` promises all repairs compose under grad, but `eig_clip` uses
ordinary eigenvector reconstruction. The central float64 witness differentiated
`sum(eig_clip(I_2))`: all gradient entries were NaN. Near this positive identity
the clipping operation is the identity on symmetric matrices, so the expected
gradient is the all-ones matrix. This is an interior repeated-eigenvalue issue,
not an unavoidable nondifferentiability at zero.

**Repair acceptance:** Reuse an appropriate stable spectral derivative where
supported or narrow/defer the affected transformation promise. Independently
test repeated positive/negative eigenvalues and qualified JVP/HVP domains.
Dynamic nearest-correlation solver reverse differentiation and generalized
zero-eigenspace selections require separate explicit contracts. Keep the live
custom-JVP callbacks; a dead-code scanner cannot recognize all decorator uses.
Base comparison confirms that both this public arithmetic and the blanket
gradient promise are inherited; new internal derivative primitives do not repair
the public path automatically.

## F028 — Pure SLX builds spatial identity filters it does not need

`estimation/spatial/graph_likelihood.py:55–60` builds two `T*N*N` filter batches,
multiplies them by already assembled designs/outcomes and computes determinants
even when both lag/error role counts are zero. Pure SLX is pooled least squares
on the prepared local/WX design. Its eager path has no enclosing JIT establishing
elimination of these intermediates. This is source-confirmed allocation growth;
no speed or peak-memory comparison has been run.

**Repair acceptance:** Use direct concatenated designs/outcomes with zero
log-Jacobian for the zero-role case, preserving observed-outcome selection and
the independent pooled-design witness. Measure full-fit retained/workspace bytes
and warm/cold behavior separately. Keep genuine spatial filters where present.

## F029 — Invalid measures become finite discrepancies and zero diagonals

`discrepancy/mmd/weighted.py:102–128` and `energy/weighted.py:103–115` omit
PackedClouds validity masks. The first row of masses `((2,0),(1,0))` is invalid,
yet the central two-cloud witness returned finite off-diagonal MMD 16 and energy
8 under eager and JIT execution. The migration removed the old eager value guard;
the new structural carrier intentionally needs per-lane numerical validation.
Scalar MMD/transport already implement that policy.

Source review also found unconditional diagonal zeroing in `mmd/matrix.py:131`
and upper-triangle assembly in `energy/matrix.py:121–122`, which can turn an
invalid singleton into a branded zero matrix.

**Repair acceptance:** Combine both input validity flags in every weighted
branch, including the linear shortcut. Preserve NaN diagonals for unavailable
lanes. Check invalid counts/masses/active coordinates, valid neighboring lanes,
singleton collections, inactive padding and eager/JIT/vmap composition. Keep the
structural pytree and legitimate zero-mass padding behavior.

## F030 — A zero-width quantile interval overflows a zero self-distance

`discrepancy/transport/univariate.py:78–80` computes a gap power before multiplying
by interval width. Tied CDF endpoints temporarily pair opposite sites on an unused
zero-width interval. Squaring overflows, then zero times infinity is NaN. The
central self-distance witnesses at float32 `(-1e20,1e20)` and float64
`(-1e200,1e200)` returned NaN eagerly and under JIT; exact W2 squared is zero.

**Repair acceptance:** Mask unused intervals before forming overflow-prone
differences/powers, preserving tie ordering and piecewise derivative semantics.
Verify independent quantile/LP controls, extreme representable sites, repeated
CDF endpoints and zero-mass padding in both precisions.

## F031 — Mutable masses break retained empirical state and compiled reuse

`sample/packed.py:132–137` stores supplied masses without the immutable-JAX-array
guard used for coordinates/counts. A NumPy `(0.5,0.5)` buffer is retained. The
coordinator mutated it to `(1,0)`: eager mean changed from one to zero while the
cached closure JIT still returned one. This violates the immutable three-leaf
contract, rather than merely choosing a different estimator.

**Repair acceptance:** Reject mutable probability leaves consistently or own an
immutable copy at the eager boundary. Test mutation and stable compiled calls,
structural reconstruction, factories and caller precision; do not add host value
checks inside traced reconstruction.

## F032 — Campaign identity omits numerical dependencies and effective JAX policy

`statadic_studies/campaign.py:52–65` records six distribution versions, omitting
computational dependencies such as optax and optimistix. Resume compares this
incomplete identity; `statadic_docs_labs/campaign_gate.py:10–22` repeats it. Direct
supplied-wheel campaigns do not bind the stronger installed-preparation identity
or a dependency lock. Effective PRNG/matmul/compiler policy is also absent, while
adapters can use JAX's default key implementation. Thus identical source/wheel
and recorded versions need not describe the same numerical execution or draws.

**Repair acceptance:** Extend the existing canonical execution identity or bind
the retained preparation/lock closure, and freeze or record effective PRNG and
compiler settings. Check resume and release identity rejection after changes,
keeping historical evidence readable. Preserve installed payload/version checks;
do not misdescribe their documented same-version third-party payload limit.
Current incomplete source-review gates still block full release qualification;
this is an independent identity gap, not proof those gates were bypassed.

## F033 — CLI abbreviations override installed-run frozen inputs

`installation.py:423–425` rejects exact `--repository`/`--wheel` spellings, while
the downstream suite parser accepts abbreviations. User arguments follow the
frozen options, so `--repo` and `--w` override them. The central dispatch probe
used mocked orchestration and observed the overridden repository and wheel;
it created no preparation or campaign.

**Repair acceptance:** Disable abbreviation on the relevant parsers or validate
canonically parsed options. Cover full/abbreviated/equals spellings and frozen
precision/resume/status behavior. Retain permitted scientific selection options.

## F034 — Performance final inventory misses source additions

`statadic_performance/cli.py:127–132` freezes a path list once. Its final guard at
271–274 rehashes that list, catching edits/removals but omitting newly added
nested Python files. Both the coordinator and independent reviewer inspected
the complete guard. This establishes an incomplete source-stability check, not
a demonstrated numerical change or measured speed defect.

**Repair acceptance:** Re-enumerate the same measured/harness roots before final
comparison, preserving exact hash checks. Inject a between-worker added file in
a harness test; no benchmark is required. Reuse the installed-preparation fresh-
inventory pattern and define shared evidence-source scope explicitly.

## F035 — Scalar coordinate updates compute the entire quadratic gradient

`optimize/simplex.py:305–307` replaces a row dot product with
`jax.grad(quadratic_objective)(...)[coordinate_index]`. The central optimized
HLO for the actual sanitized nonnegative-QP solver retains two full length-16
dot outputs inside the inner coordinate loop, followed by indexing. The compiler
did not eliminate this work in the bounded probe. This changes work per coordinate
from a row reduction to matrix-vector products, and work per sweep from quadratic
to cubic in width. No wall-clock ratio was measured.

**Repair acceptance:** Use the justified coordinate derivative of the symmetric
quadratic, while retaining the canonical objective and independent KKT checks.
Inspect optimized work at several widths and measure matched full-solver timing/
memory after freeze. Do not assume that an AD expression preserves an existing
sufficient-statistic cost model.

## F036 — Qualification and manuscript consumers retain removed constructors

`src/python/packages/statadic/scripts/check_distribution.py:45` embeds imports of
`PoissonRegression` and `LogisticRegression` and constructs both at 157–162.
`src/python/apps/docs/tests/test_external_extensions.py:13` imports the latter;
`src/latex/projects/01_statadic/analysis/generate.py:44` imports the former. Candidate
`statadic.api.regression` exports neither; public import tests deliberately require
their removal. The central AST/namespace probe confirmed all four missing
imports. The distribution integration program, docs collection path and optional
manuscript regeneration cannot reach their substantive assertions. No complete
wheel build, docs suite or manuscript regeneration was executed here.

**Repair acceptance:** Migrate all three consumers to the declared conditional
laws and roles. Preserve Poisson exposure/offset and binomial trials/logit checks;
retain the independent distribution witnesses and saved-predictor round trips.
Keep the intentional clean alpha break. Add a focused check of the embedded
program's imports and run the existing downstream commands after migration.
Coordinate with F015/F012 rather than restoring aliases to satisfy stale tests.

## F037 — Manuscript copies formal exposition rejected by canonical freshness

`src/latex/projects/01_statadic/analysis/generate.py:358–361` copies the mathematical
macro file after checking only its existence. `src/lean/tools/bridge.py:407–432`
already owns checks for certification inputs, manuscript hashes, compiled Lean
source, public inventory and exported artifacts. No generator caller invokes it.
The coordinator extracted the actual `formal()` body, supplied its normal globals
and a temporary destination, and observed successful copying with SHA-256
`669b1516277e8ce15db42cbf495593cc9761215b6a076192229269a9395a517c` while
`fresh_output()` rejected the same checkout as stale. This probe bypassed F036's
unrelated top-level import to isolate the copy contract; it is not a successful
complete generation run.

**Repair acceptance:** Call the existing freshness owner before consuming formal
fragments. Bind formal/citation/retained-validation inputs in generation
provenance. Stage a complete generation before replacing retained output, because
the present sequential writes can leave mixed old/new fragments after failure.
Reject stale/tampered source and fragment controls; preserve historical artifacts
with their identities. Human fidelity approval remains a separate requirement.

## F038 — Checked concatenation drops atom masses and preserves false evidence

`sample/evidence.py:784–795` rebuilds coordinate rows with `pack(rows)` and
reuses the old evidence. With this stack's mass-bearing carrier, that replaces
nonuniform masses by uniform masses. For unit sites `(1,-1)` weighted
`(.75,.25)`, the central witness started with weighted mean `.5` and
`NonzeroPooledMean`. Concatenating the checked carrier with itself returned two
uniform `(.5,.5)` rows, weighted mean zero, and the same nonzero-mean evidence.
The concatenation body predates the change; its incorrect behavior is introduced
by integrating the new weighted carrier without migrating this consumer.
The existing preservation test uses uniform masses and checks metadata/counts.

**Repair acceptance:** Preserve every retained probability leaf and support
identity when concatenating; use the existing weighted factory or explicit
structural assembly. Test nonuniform weights, different padding widths, counts,
empty/invalid policies and evidence truth after composition. Do not revalidate
only the metadata or silently erase weights. Include this consumer in the
weighted-carrier migration audit alongside F029–F031.

## F039 — Claim pages omit recorded pending human review

`statadic_docs_labs/formal_extension.py:60–67` renders `source.review` for sourced
claims and top-level `claim.review` only for claims without a source. Seventeen
candidate records put the pending human status solely in the omitted top-level
field: MULTIPLICITY-001/002, RANDOMIZATION-003, EQUIVALENCE-001, HAC-003–007,
HC-001–004, CLUSTER-001/002, PANELCOV-003 and COVBRIDGE-001, all with the `STATADIC-`
prefix. The coordinator inspected the branch and HC-004 record. The base had
the same branch but no such lost pending statuses; the new records expose the
projection gap. The generic fidelity caveat remains accurate, so this is review
transparency drift, not a false theorem or an automatic alpha blocker.

**Repair acceptance:** Project both canonical claim-review state and source
correspondence without merging their meanings or upgrading an agent assertion to
human approval. Exercise sourced/unsourced and pending/reviewed combinations
through the existing rendering tests. Update MacKinnon dossier scope per claim:
its old no-asymptotic-results paragraph now contradicts inclusion of HC-004.

## F040 — Two new composition tests call an undefined model name

`tests/api/test_composition_ux_hardening.py:52` and `:167` call
`SurvivalEquation` without an import or definition; the file imports
`SurvivalRegression`. The named-file execution returned seven passes and two
NameErrors before these tests' scientific assertions. This differs from F015's
broken production imports and F012's obsolete inheritance expectation.

**Repair acceptance:** Express both tests through the retained survival/model
composition API, preserving their unit-effect renaming and keyed missing-spell
assertions. Run the complete named file. Do not create a compatibility alias
solely to make unbound test names resolve.

## F041 — Kernel-mixture demo retains the retired array calling convention

`notebooks/kernel_mixing_demo.py:382,394` passes a raw target array and candidate
list to `energy_distance_kernel_mixture`. The candidate function's carrier
contract is `PackedClouds`, and `balancing/mixture.py:102` uses
`target.expectation`. The notebook migrates the function/solver names but not
these arguments. The reviewer read the complete authored diff and relevant
unchanged calls; the coordinator checked the carrier integration in the measures
wave. This is source-confirmed, not an executed notebook failure.

**Disposition:** Retire this legacy, untested notebook and its README pointer
for alpha, retaining current mixture examples/studies and any independent oracle.
Its inherited optimality and unsynchronized/GPU timing narrative needs broader
revision than renaming inputs. If retaining it instead, migrate both carriers,
state the precise optimization claim and run the example with synchronized
measurements in its declared scope. No production deletion occurred here.

## Evidence and remaining scope

The first four witnesses executed through the pinned locked project; all existing
48 selected shared tests passed. Details are in [verification](verification.md).
No long campaign, fresh Lean certification or performance comparison has run.
Additional wave hypotheses remain pending and are not included as confirmed
findings until challenged. See [coverage](coverage.md) for unreviewed scope.
