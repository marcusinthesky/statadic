# Known limitations

Ordination retains signed inertia and explicit query capabilities. Negative
inertia requires an explicit descriptive positive-subspace choice; neither a
finite PSD matrix nor a fitted Isomap graph supplies a population geometry law.
Independent numerical references do not establish manifold recovery,
out-of-sample predictive calibration or a retained Monte Carlo campaign. See
the [ordination guide](../reference/ordination.md) for exact scope.

Selected-mixture equivalence uses an approximate percentile bootstrap with no
proved Type I error bound. Holm (default) and Bonferroni control familywise error
under arbitrary dependence only when each true-null input p-value is valid.
The former adaptive Bonferroni and `k_minus_1` rules have been removed.

A non-finite observed statistic or scheduled null draw makes the affected
p-value unavailable (NaN in numerical and float-returning functions; existing
host APIs retain None). Failed draws are not discarded to recover significance.
Dyadic reports may retain successful-draw standard-error and interval diagnostics
while withholding the p-value.

Selected added likelihood, IV and survival-mean configurations have bounded
descriptive adapters and numerical tests. The catalogue records the remaining
adapter gaps; none has retained installed-wheel qualification from this work.
Log-logistic RMST uses bounded quadrature and can remain unavailable at extreme
horizon/shape combinations even when the mathematical integral is finite.
The existing Weibull RMST formula can also become unavailable when its
incomplete-gamma factor underflows, including a very large scale with a moderate
finite horizon. Neither case should be interpreted as a nonexistent integral.

OLS residual diagnostics use asymptotic chi-square or F references. Their size
is qualified only in the documented 200-row Gaussian study; power, other sample
sizes and non-Gaussian or heteroskedastic nulls are not qualified. They are
defined for unweighted OLS only, and Durbin-Watson has no p-value. See
[residual diagnostics](../linear-diagnostics.md).

These configuration-specific dispositions are recorded in the [validation guide](../validation.md). They do not describe entire method families.

| Affected procedure or regime | Recorded limitation | Documentation consequence |
|---|---|---|
| Ordinary lognormal AFT, n=128, zero entry | Qualification inconclusive; recorded coverage lower bound 0.937081 | Do not imply all ordinary AFT cells qualified |
| Ordinary Weibull AFT, n=128, positive entry | Qualification inconclusive; lower bound 0.939556; one unavailable fit among 10,000 requested | Preserve failure-inclusive accounting |
| Repeated-spell fixed-effect and dependent spatial survival Wald inference | Not qualified by the iid ordinary-AFT study | No transfer of an iid claim to dependent/repeated settings |
| High-censoring spatial log-time likelihood integration | Recorded 50%/80% censoring stress cases miss the stated numerical target; selected high-budget cases still fail | Convergence/refinement stability must not be called an integration-error certificate |
| Preliminary spatial-survival stress records | Pre-run source manifest absent | Do not describe them as final-source qualification |
| Gaussian SAR Wald inference | 48-unit cell exceeds size budget; 96-unit cell inconclusive | Put finite-sample cautions beside SAR inference examples |
| Pooled SAC panel optimization, seed-0 fixture with T=30 and N=9 | Independent searches on fixed and repeated graph storage can stop at different points; in float64 the repeated-graph candidate fails the average profile-score tolerance and is now unavailable | Retain stalled attempts in failure-inclusive accounting; the acceptance guard does not recover an optimum |
| Random-effects SEM panel, CPU float64 seed-26310 fixture with T=6, N=8 and tolerance 1e-8 | Converges at the spatial boundary -0.999; the boundary guard correctly withholds the fit | Count the unavailable procedure in qualification; valid inputs need not yield an interior fit |
| Stationary-bootstrap mean interval, T=256 | IID normal cell inconclusive; AR(1) coefficient 0.3 cell undercovers | A supported executor is not a coverage guarantee |
| Public Sharpe CI, T=128 | Gaussian, skewed and dependent-return cells report excess error under the prespecified coverage criterion | Avoid “validated performance inference” as a family-wide label |
| Canonical dyadic node-bootstrap qualification | Manual-only study remains unexecuted in the source record | Display “not run”, not “passed” or “failed” |
| Correlated GMS QLR and Wolak projections in float32 | Tested two-/three-moment fixtures contain unconverged numerical lanes; float64 reference tolerances also exceed float32 accuracy | Inspect native convergence evidence and retain unavailable trials; changing precision is an explicit caller choice |

The float32 projection observation comes from the native GMS/Wolak regression
fixtures and study smoke cases, not a completed operating-characteristic campaign.
An eager rejected projection now carries typed numerical availability diagnostics;
unexpected backend/runtime exceptions still stop a study. The standard package
test configuration defaults to float64; passing it does not establish float32
accuracy or convergence for these fixtures.

The SAC observation is a numerical counterexample, not a retained statistical
campaign. Profile calculations agree at identical supplied parameters;
independent searches need not return the same fit. Pooled SAC now checks the
gradient of average negative profile likelihood with respect to spatial
parameters normalized by their search limit, independently of BFGS termination.
Its solver residual records that score norm. The float64 fixed-graph fixture
passes with residual about `4.34e-9`; the repeated graph is stalled with residual
about `0.00474` at tolerance `1e-7`. Both float32 searches are unavailable under
that same tolerance. This guard changes acceptance, not the search budget or
precision. It establishes neither global optimality nor statistical qualification.

## Keep the criterion fixed

The ordinary qualification protocol uses 10,000 independent trials per scenario, with method-specific inner resampling as documented. Its nominal 95% interval criterion requires a one-sided coverage lower bound of at least 0.94, counting unavailable intervals as uncovered. Type-I and failure bounds are separate. These are prespecified study criteria, not targets to relax after inspecting results, and are not simultaneous guarantees across every method.

The survival table contains six other ordinary-AFT cells marked qualified. That is evidence for those cells, not permission to hide the two inconclusive cells or generalize to every sample size and censoring pattern.

## What users should do with a limitation

Read the exact configuration and output affected. Preserve the warnings and numerical evidence in the analysis record. A different precision, asymptotic reference or integration budget is a scientific/computational choice that needs its own justification; the documentation should not silently change it to make an example look successful.
