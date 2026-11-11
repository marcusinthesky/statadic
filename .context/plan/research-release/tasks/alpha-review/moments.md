# Moments and IV adjudication

Three lanes completed full current/source-diff reading for the affine GMM/GEL,
native IV and host moment owners, focused tests and exact evidence mappings.
Independent challenges and central witnesses distinguish inherited calibration
risks from introduced IV defects. Full spatial GMM/IV remains in the dependent
slice. [Inspection records](reviews.json) contain exact paths.

## Alpha disposition

Keep fixed/iterated affine GMM and actual-weight covariance with their separate
point-fit, efficiency and inferential availability evidence. Native fits correctly
retain the actual weight and moment map; do not simplify them into inverse-bread
covariance. Keep fixed GEL independently of affine-profile calibration.

Repair or defer the inherited affine-J door (F018) and projected GEL reference
(F019). Prefer migrating the active legacy-kernel studies caller to the existing
retained-fit gate, then remove the eager wrapper, kernel and duplicate carriers
(S008). No public nonlinear GMM producer was found in this slice; do not imply
that an affine implementation covers it.

Keep native IV as a useful operation subject to F020/F021's memory and unit
admission repairs and the eight explicit mapping gaps. QR initial weighting,
actual W, preliminary/final variability, parameter evaluation points, criterion,
stationarity and separate iteration convergence must survive reorganization.

## Scientific evidence coverage

| Group | Variants | Study mapped | Source approved |
| --- | ---: | ---: | ---: |
| Affine GMM estimation | 4 | 4 | 4 |
| Actual-weight covariance | 2 | 2 | 2 |
| GMM tests | 4 | 4 | 1 |
| Fixed/affine GEL | 12 | 12 | 0 |
| Long-run covariance | 4 | 4 | 2 |
| Native IV | 17 | 9 | 0 |

All 43 are pending retained campaigns. Method-specific mappings declare bias/MSE
for 22, size for 22, power for 16 and coverage/width for 15; these overlap.
The eight gaps are model/HC1 for 2SLS, two-step and iterated IV, plus two-step/
iterated Hansen-J. The older generic affine-IV study does not test the native
IV implementation. No fresh retained moment-family qualification was established.

Independent references include actual linearmodels coefficient sequences/W/
covariance, deliberately inefficient-W controls, SciPy primal fixed EL/ET optima,
independent IV matrix algebra and scalar/arch Bartlett calculations. A nonzero
affine GEL optimum does not receive equivalent independent oracle coverage from
the fixed-probability tests. Kernel-reusing HAC/cluster portions test composition,
not all of the underlying arithmetic. Fourteen relevant historical external
cases lack current source binding (F007); no relevant precision-lane cases were
found in that retained manifest.

## Proof/source scope

Read sixteen principal claim mappings (`STATADIC-GMM-003–011`, `IV-001–003`,
`JTEST-001–003`, `GEL-001`) and eleven principal modules: AffineGMM;
GMM/AffineLimit, AffineInference, EfficientWeight, Sampling, ScoreCLT,
InstrumentFailure, Overidentification, GaussianProjection, EfficientCriterion;
GEL/Stationarity. Spatial/ErrorMoments and GMM-001/002 were read as algebra;
their full spatial implementation was not reviewed in this lane.

Read Hansen's relevant consistency/limit/efficient-weight statements and appendix,
Newey–Smith's saddle-point/CUE/normalization/identification passages, and Arraiz
et al.'s assumptions and equations 10–13. No direct Lean contradiction was
established. GEL-001 proves conditional positive balancing, not profile
identification or chi-square calibration. The new nonspatial IV symbols have no
direct claim mapping; broader GMM family membership supplies no automatic proof.
All human fidelity reviews remain pending.

## JAX and simplification decisions

J002: private GEL envelope AD is first order at a converged optimum. The central
closed-form CUE profile has Hessian one at zero; nested AD returns zero after
stopping its multiplier. Keep that explicit limitation; do not advertise general
profile Hessians or fitted-solution derivatives. Finite outer/profile convergence
needs its own stationarity and retained-coordinate identification evidence.

J003: qualify 2SLS outcome JVP against its independent linear map. Dynamic IV
while-loop reverse derivatives need an explicit strategy; JIT/vmap support alone
is insufficient. Measure equal-shape compile reuse before asserting a cache issue.

S009: remove the IV producer token if it remains write-only; otherwise make the
intended producer check concrete. S010: consolidate duplicate IV cluster-code/
meat construction under one scientific owner with explicit normalization, keeping
estimation independent of inference. S011: move repeated moment/Jacobian
validators to the existing shared problem owner. S012: defer unused Bartlett
weight helper APIs after caller checks, preserving their formula assertions.
S013: share the duplicate rank-deficient GEL fixture without merging independent
primal and numerical solver tests.

Do not delete preliminary/final variability, actual W or native/host boundary
tests as duplication. Finite unconverged affine GEL point-report rows need an
explicit diagnostic-iterate policy; metadata currently reports failed convergence
while each finite parameter row says available. Keep this open as a reporting
decision rather than misclassify it as an introduced regression.

The central named run passed 45 tests with one slow case deselected. Those tests
did not catch F018–F021. No statistical or performance campaign ran.
