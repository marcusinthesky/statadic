# Measures, numerical contracts and reference-pilot evidence

Authored on 2026-11-09. Agent C's corrective work starts from `850797b`, following
the review of `17fdc54`. This record grants no source-fidelity, statistical,
performance, publication or alpha approval. The coordinator owns integration,
serialized tests, final source identity and release disposition. Pricing
Perspective production sources, dependency pins and retained outputs are unchanged.

## Corrective implementation

| Work | Change and bounded controls |
| --- | --- |
| R02/R03 | Raw PackedClouds masses must be immutable JAX leaves. Checked concatenation retains original atom probabilities. Weighted MMD/energy preserve unavailable lanes and invalid diagonals without poisoning valid neighboring measures. Tests include independent weighted point distances and a nonuniform pooled mean. |
| R06 | Actual empirical V-energy producers retain static ground/exponent/axiom construction evidence. Metrization no longer accepts unrelated ground or population-moment witnesses. Euclidean point controls, angular nonseparation rejection and JIT carrier reuse delimit this empirical contract. |
| R11 | Zero-width quantile intervals mask sites before potentially overflowing powers. PSD projection reuses its spectral JVP; repeated mixed spectra have a first-derivative contract, while Hessian/HVP controls cover definite interiors. Free-support weighted-mean Wasserstein updates reject unsupported non-Euclidean grounds. |
| R12 | Nonnegative QP coordinate updates compute one row derivative. Exact simplex barycentres use the existing checked QP owner with the original objective and independent energy/FW diagnostics; no ridge, clipping or renormalization changes the problem. |
| S019–S023/S031/S033 | Removed the duplicate barycentre QP, fake direct solver selectors, unused poor-hedge quantile wrapper, duplicate mixture reduction, unused transport closure and tautological alias test. Public packed utilities and real precomputed consumers remain. Kernel-balance PGD remains distinct. The retired notebook contained no unique independent numerical oracle. |

Associated source, typing and adverse tests are retained beside their owners.
Independent read-only challenge found no additional confirmed arithmetic defect
in these repairs within the inspected diffs and critical callers. It did not
certify mixed-spectrum higher derivatives or every scientific configuration.

## Executed optimized-work probe

The bounded probe at `build/alpha-readiness/qp-optimized-work/probe.py` executed
successfully through the locked performance uv project using the materialized
devenv PATH and library paths. It retained its source/lock hashes, exact fixtures,
six `compiled.as_text()` artifacts, compiler allocation estimates and
`summary.json`. The initial read-only-cache and missing-library attempts remain
in separate logs; neither reached numerical execution.

CPU float64, Python 3.13.15, JAX/jaxlib 0.11.0 and NumPy 2.4.6 were recorded.
For K=8 and K=32, the actual native one-sweep result and an isolated direct-row
loop agreed exactly with independent host scalar cyclic minimizers. A matched
isolated loop using the historical full-gradient expression differed by at most
`1.11e-16`. Native objective and finite/convex/bounded diagnostics also passed.
This is one-sweep correctness, not convergence to the QP optimum.

| K | Direct-loop compiler temporary bytes | Historical-loop compiler temporary bytes | Complete native compiler temporary bytes |
| --- | ---: | ---: | ---: |
| 8 | 800 | 840 | 1208 |
| 32 | 8672 | 8904 | 10040 |

The optimized direct/native coordinate body selects a 1-by-K row and fuses its
product/reduction. The historical coordinate body retains two full
matrix-vector dots on the K-by-K matrix before selecting the scalar derivative.
Thus the source simplification survives this compiler's optimization for these
two shapes. Native PSD preparation and out-of-loop diagnostic work remain.
The complete native and isolated-loop allocation totals have different output
contracts and are not a like-for-like memory comparison.

No timing benchmark or whole-process RSS measurement ran in this probe. Compiler
temporary bytes are not RSS, and these observations establish neither a 10%
resource win nor universal solver performance superiority. No production change
was needed after the probe.

## Strict reference pilot and release boundaries

The performance app now has six bounded pilot adapters: OLS/HC3 and SLX with
classical covariance against statsmodels; weighted RBF V-MMD with an analytic
coordinate gradient against scikit-learn kernels; capped QP against SciPy SLSQP;
fixed-graph Gaussian SAR fit/covariance against spreg; and ordinary censored
lognormal AFT fit/information against an independent SciPy objective and analytic
derivatives. The AFT peer's derivatives have a separate distribution/finite-
difference test. Initial pilot attempts are retained under
`build/alpha-readiness/reference-pilot*`; the weighted RBF tracing failure led to
moving checked empirical preparation outside JIT. Coordinates and masses stay
dynamic in the measured carrier. These attempts do not qualify performance.

Fresh isolated workers preserve venv interpreter symlinks, verify actual imported
wheel payload/metadata, keep peers JAX-free and retain raw outputs, timing samples,
whole-process RSS, process identities and scope/lock/harness/fixture identities.
The verifier derives agreement and session ratios from those records. Pilot
observations mint no release contract evidence. `QUALIFICATION_CASES` remains
empty: all 15 required workload definitions and reviewed subcases must exist
before qualification can run. Both warm-time and total-RSS ratios must be at most
0.90 in each of three independent sessions; averaging cannot hide a failed cell.

The scoped scientific gate additionally rejects complete-but-unusable point
studies and requires the frozen failure budget using a one-sided exact 95%
failure bound. Finite descriptive bias/MSE remains descriptive. Release inspection
selects CPU/float64 before comparing campaign policy. Manuscript generation now
rejects output symlinks before resolution and freezes the consumed source,
registry, lock and formal inputs, including added source files.

## Boundary QP and caller-precision follow-up

The coordinator's broad selected tests exposed a weak-complementarity boundary
case after simplex-QP consolidation. With target Dirac(3,3,3,3) and candidates at
3 and 9, the original objective is `24*w[1]^2`, with exact optimum `(1,0)` and
zero active multiplier. Retained diagnostics at
`build/alpha-readiness/qp-degenerate/` confirm the actual full and split cloud
components, and that both maintained QPAX backends exhausted their iteration
budgets. The implicit backend's actual stationarity residual `3.416e-9` missed
the requested `1e-9`; its very small primal gap did not justify accepting it.

The shared owner now permits a proposed analytic vertex only after original
primal, dual and complementarity checks at the unchanged requested tolerance.
The outer finite, convexity and cap gates remain. This directly certifies the
exact boundary solution without modifying a failed backend iterate. Other
problems continue to use the existing backend, including interior and capped
nonvertex optima. Associated controls cover duplicates, interior/cap rejection,
invalid lanes, and the actual full/split cloud sizes.

The studies failures also exposed an artificial internal tolerance factor:
public `tol=1e-5` had become a `1e-7` canonical-QP request on caller float32.
The barycentre adapter now passes the caller's tolerance through and still
requires its independent original-energy Frank-Wolfe certificate. It does not
change global precision, floor the public tolerance or admit invalid draws.
An additional exact seed-7 permutation passed canonical KKT at `1e-7` while its
original-energy FW gap `1.77384e-7` exceeded the unchanged `1.11080e-7` bound.
The retained independent NumPy interval optimum confirmed an interior solution.
The adapter now attempts a tighter canonical solve only after an accepted first
point misses its original-energy certificate, using static iteration allowance
remaining within the caller's total cap. It retains the first unmet point if
refinement cannot certify both conditions. Float64 refinement reached FW
`2.35772e-8` with 19 selected-path iterations; float32 at caller `1e-5` retained
its accepted first point. Exact pools, component witnesses, script source and
both diagnostics remain under `qp-degenerate/` in separate records.

Under `vmap`, JAX can turn the per-lane conditional into selection and execute
both solver branches. Selected-path iteration fields do not measure all physical
work in that transformed program. The static combined allowances remain bounded,
but no conditional batching cost reduction or performance win is claimed. A
batching-wide scheduling change is deferred pending measurement.

Focused tests and final-wheel qualification remain the coordinator's work.

## Independent free-signed energy counterexample

Actual frozen Pricing Perspective source
`pipeline/_kernels/typed_barycentre/energy.py` solves `G w = c` for its free-signed
arm, then reports convergence and zero gap. For target Dirac mass at zero and
candidates at minus/plus one, Euclidean exponent-one energy has
`G=[[0,2],[2,0]]`, `c=[1,1]` and zero target self-term. The returned weights
`(1/2,1/2)` give energy 1. Along allowed free-signed weights `(t,t)`, the objective
is `4t-4t²`, which tends to minus infinity. The stationary point is a saddle;
there is no finite global minimizer. This is an exact algebraic counterexample,
confirmed against source without executing or changing PP.

The affine unit-mass constraint instead gives weights `(t,1-t)` and objective
`4(t-1/2)²+1`. Affine-signed and simplex procedures must retain their distinct
contracts. Retiring free-signed energy does not justify removing either one.

Agent C ran the bounded compiler/cyclic-control probe and isolated retained
permutation/refinement diagnoses described above, plus read-only discovery.
Named tests, app checks, reference pilots, scientific
campaigns, final `just check` and exact-wheel release verification remain under
the coordinator's serialized schedule. Passing counts from earlier central
lanes are not automatically evidence for later edits.

## Final-sweep simplification disposition audit

The 2026-11-09 source audit checked all S001–S033 against current owners and
callers. These are design/implementation dispositions, not measured resource
gains. The final-sweep validation record owns subsequent executed results.

| IDs | Disposition and preserved boundary |
| --- | --- |
| S001 | Generic parameter covariance delegates to existing composed-PSD validation; native QR and panel contribution counts retained. |
| S002 | Legacy DMat aliases removed; active `core.matrices.DMat` retained. |
| S003 | LazyFrame removed from accepted union; eager rejection retained. |
| S004 | Deferred by accepted scope; no derivative target retired implicitly. |
| S005 | Duplicate linear test plumbing consolidated; independent numerical oracles retained. |
| S006 | One current conditional guide; migration history remains dated. |
| S007 | IV numerical work moved below reporting. |
| S008 | Retired affine-J wrapper/carriers removed after actual-weight migration. |
| S009 | Unread IV token producer removed. |
| S010 | Common cluster algebra consolidated without merging normalization conventions. |
| S011 | Identical eager finite-moment validation moved to `estimation.moments`; traced GMM preparation and GEL-only Jacobian validation remain separate because their contracts differ. |
| S012 | Retain documented public Bartlett weight constructors in `operators.longrun`. External callers can construct their own lag kernels; absence of internal consumers alone does not establish dead code. Do not claim these helpers were deleted. |
| S013 | Duplicate rank-deficient GEL fixture moved to existing `tests.inference._moments`; both test bodies remain. |
| S014 | Graph-mean and covariance numerics have native owners. |
| S015 | Unused private refined-estimation path removed; actual public refinement retained. |
| S016 | Redundant dependent self-parity work replaced by independent witnesses and contract checks; supplied-parameter profile checks do not validate optimization. |
| S017 | Rectangular observed-panel assembly simplified with row identity retained. |
| S018 | Actual-row HAC simplification preserves quadratic-spectral support. |
| S019 | Barycentre QP uses the checked shared solver and original-objective diagnostics. |
| S020 | False backend choices removed; distinct kernel-balance PGD retained. |
| S021 | Unused poor-hedge quantile wrapper removed. |
| S022 | Mixture-score duplication removed; real precomputed consumers retained. |
| S023 | Dead private transport closure removed; documented packed utilities retained. |
| S024 | Bootstrap fixed-seed containment names/prose explicitly describe regression witnesses. Their arithmetic assertions remain; calibration belongs to repeated-sampling studies. |
| S025 | PSD implementation diary moved to dated `psd-history.md`; current mathematical/derivative contracts remain at the code owner. |
| S026 | Prediction assembly helpers privatized and callers migrated; public option types retained. |
| S027 | Live/frozen predictions share pure graph-mean algebra; schema-specific codecs remain distinct. |
| S028 | Duplicate Tach API policy consolidated with diagnostics/graph dependencies retained. |
| S029 | Docs report and PGF figure compilation both use the shared vendored `src/latex/bundle/tlextras-2022.0r0-statadic.zip`; publishing retains its hash. |
| S030 | Unloaded Fell assets removed; live Latin Modern and applicable notices retained. |
| S031 | Untested mixing notebook retired; current studies and independent oracles retained. |
| S032 | Optional Lean research-helper trim remains deferred; active foundation retained. |
| S033 | Tautological legacy-alias test removed; meaningful optimization controls retained. |

The additional reviewed pre-alpha cut removes 21 duplicate exports from the
`statadic.optimize` navigation namespace. Canonical algorithms remain in child
modules; callers and inventory links migrate explicitly, without aliases.
