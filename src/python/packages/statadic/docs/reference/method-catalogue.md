# statadic — by the question you are asking

The package tree is organized by scientific role and cohesive family. The
[module guide](../module-organization.md) explains canonical ownership and imports;
`ARCHITECTURE.md` at the package root explains the enforced dependency DAG. This
index connects the same capabilities by the inferential question.

The [composition reference](composition.md) connects these owners through aligned
observations and retained representations, including weighted empirical measures
used in both group analysis and supervised prediction.

Every path below is live. If a link here does not resolve, the index is stale,
not the code.

Numerical JAX pytrees versus eager Python/report objects are inventoried in
[`numerical-host-boundaries.md`](../numerical-host-boundaries.md); use the former
inside transforms and call a named host adapter only after computation completes.
The normative carrier, equality, transformation, law, evidence, and authority
vocabulary is in [`mathematical-contracts.md`](../mathematical-contracts.md).
Migration examples for typed ground, energy, MMD, and DISCO composition are in
[`mathematical-contract-migration.md`](../mathematical-contract-migration.md).
The [capability guide](../capabilities.md) distinguishes checked JIT, batching,
objective/fitted-solution derivatives and reproducible CPU measurements.

---

## Are these two things associated?

*Independence / correlation between two objects measured on the same units.*

| you want | call | notes |
|---|---|---|
| correlation between two **distance matrices** | `statadic.api.association.Mantel` | Pearson or Spearman on the off-diagonal; native-stream permutation null (`statadic.inference.mantel.mantel_permutation_kernel`) plus caller-scheduled `mantel_bootstrap_ci` / `paired_mantel_bootstrap` node intervals |
| variance **decomposition by group** over a typed distance matrix | `statadic.association.decomposition.disco_components` | Pure DISCO components; use `Regression(GeometricResponse(..., distance_power=1), terms=...)` with an explicit `ResponsePermutation` inference request, or native `statadic.inference.disco.disco_permutation_null` for group-only Monte Carlo calibration |
| whether groups differ in **dispersion** rather than location | `statadic.association.dispersion.dispersion_f_ratio` | Pure PERMDISP statistic on an `(n, k)` coordinate cloud; `statadic.inference.dispersion.dispersion_permutation_null` supplies the full-model residual-permutation null |

## Are these two samples from the same distribution?

*Homogeneity / two-sample testing.*

| you want | call | notes |
|---|---|---|
| an energy discrepancy | `statadic.discrepancy.mmd.energy.energy_distance` | **`metric` is required; `metric="angular"` cannot see scale** — pass `metric="euclidean"` when identification matters. See *Axioms* below |
| raw empirical pairwise ground means | `statadic.discrepancy.mmd.energy.empirical_mean_ground_distances` | accepts unequal cloud sizes; defaults to Euclidean ground; returns a `PairwiseEstimateMatrix` with within-cloud diagonal means, rather than an energy-distance matrix |
| a raw V- or signed U-statistic MMD² estimate | `statadic.discrepancy.mmd.mmd_squared` | only the V-statistic has a distance-law declaration; the U-statistic may be negative |
| a branded matrix of V-statistic MMD distances | `statadic.discrepancy.mmd.mmd_matrix` | RBF bandwidth must be finite/positive; distance-induced `alpha` lies strictly in `(0, 2)` |
| a point-level Mercer Gram | `statadic.geometry.ground.similarities.gram` | linear, cosine, RBF, or distance-induced; PSD conditions differ by entry |
| a full permutation test | `statadic.inference.resampling.permutation.permutation_test_result` | generic: give it any statistic function and a PRNG key |
| a typed, distance-hoisted energy permutation test | `statadic.inference.resampling.permutation.energy_permutation_test` | accepts one compatible `EnergyGeometry`; result preserves ground law, exponent, energy law, statistic origin, homogeneity null, and resampled calibration |
| a transformable full permutation test | `statadic.inference.resampling.permutation.permutation_test_result` | accepts a PRNG key and returns `TestResult`; nested `jit` ✓ leading-axis `vmap` ✓ |
| the **traceable** half, for a `vmap`-ed Monte-Carlo grid | `statadic.inference.resampling.permutation.permutation_test_kernel` | pure JAX, returns `TestResult[S]`; `jit` ✓ `vmap` ✓ |
| a studentized / re-optimized mixture test | `statadic.inference.studentized`, `statadic.inference.mixture` | `corrected_mixture_test` is the selection-corrected form |
| a 1-D optimal-transport distance | `statadic.discrepancy.transport.sliced` | `w1_1d` and `w2_1d` are metrics; a finite-design multivariate `sliced_wasserstein` is a pseudometric |
| an **exact** balanced Wasserstein-1 | `statadic.discrepancy.transport.exact` | Hungarian assignment on a chord ground cost; a *pseudo*metric, see *Axioms* |

## Are these two things close *enough*?

*Equivalence, not the absence of evidence for difference.*

| you want | call |
|---|---|
| a TOST / relevance-margin test | `statadic.inference.equivalence.equivalence_test` |
| an equivalence p-value from a bootstrap draw | `statadic.inference.equivalence.equivalence_pvalue_from_bootstrap` |
| Holm (default) or Bonferroni across claims, conditional on valid true-null p-values | `statadic.inference.multiplicity.equivalence_multiplicity_control` |

The selected-mixture percentile-bootstrap equivalence procedure is approximate:
its input p-values have no proved size guarantee. Applying Holm or Bonferroni
does not supply that missing guarantee.

## What is the structure in this distance matrix?

*Estimation of geometry.*

| you want | call | notes |
|---|---|---|
| coordinates from features, kernels or distances | `statadic.api.ordination.Ordination` | Compose linear, kernel, distance or shortest-path geometry with `PrincipalCoordinates`; inspect signed inertia and retained query capabilities in the [ordination guide](ordination.md). `statadic.geometry.embedding.pcoa_kernel` remains the pure fixed-width distance kernel |
| a geometric median per group | `statadic.geometry.median.spatial_medians` | converged, with a per-group flag; `weiszfeld_medians` is the fixed-budget approximation |
| barycentre weights over candidate clouds | `statadic.geometry.barycentre.mmd.energy.energy_barycentre_weights` | Exact simplex QP with convergence diagnostics; `_batched` / `_pairwise` variants |
| barycentre weights from a declared PSD kernel | `statadic.geometry.barycentre.mmd.kernel_barycentre_weights` | Target-directed MMD objective; energy is a specialized reduction over the same solver |
| measure or free-support transport barycentres | `statadic.geometry.barycentre.wasserstein` | Objective-specific assignment and convergence evidence |
| neighbour-set stability across encoders | `statadic.geometry.neighbour` | |
| a metrizing square root of an empirical energy matrix | `statadic.core.matrices.metrize_energy` | Roots the actual retained empirical V-energy construction when its ground/exponent branch separates measures; persisted values alone carry no such evidence |

## How do I fit a linear model?

| you want | call | notes |
|---|---|---|
| a complete-array OLS or relative-precision WLS fit | `statadic.api.regression.Regression`; brick `statadic.estimation.linear.LeastSquares` over `LinearProblem` | [Linear workflow](../linear-models.md); explicit intercept terms, optional `precision=RelativePrecision(...)` and retained fitted state |
| predictions with checked column order | `statadic.estimation.linear.LinearFit.predict` | named fits require the same ordered column schema; unnamed fits use positional columns |
| affine moment fitting independently of J inference | `statadic.estimation.gmm.IteratedGMM` / `FixedWeightGMM` | preview [GMM workflow](../gmm-fitting.md); exactly identified fits and actual last-solve weighting are retained |
| binomial-logit or Poisson-log fitting | `statadic.api.regression.Regression` with `law=Binomial(...)` / `Poisson(...)`; bricks `statadic.estimation.glm.BinomialLogitMLE` / `PoissonMLE` | [GLM workflow](../glm-fitting.md); top-level predictor terms, explicit trials, offsets/exposure and finite-optimum evidence |
| GLM link, probability or expected-count prediction | `statadic.estimation.glm.GLMFit.predict` | prediction scale is explicit; named columns must match |

## How do I fit panel or spatial models?

The [spatial/panel workflow](../spatial-models.md) describes preparation, fitting,
uncertainty and matched optional reference tests, including time-varying W.

| you want | call | notes |
|---|---|---|
| prepare unbalanced entity/time rows or spatial graphs | `statadic.operators.panel`, `statadic.operators.spatial` | Explicit keys, normalization, missing/singleton policy and retained order |
| pooled, fixed/random effects, between, first differences or Fama–MacBeth | `Regression(..., structure=PanelLayout(...), estimator=...)`; bricks `statadic.estimation.panel.Pooled` / `Within` / `Between` / `FirstDifference` / `RandomEffects` / `FamaMacBeth` over `PanelProblem` | General static panels; `statadic.inference.panel_covariance` owns uncertainty |
| SLX, SAR/SEM/SAC or selected Durbin terms | `statadic.estimation.spatial.likelihood` | Fits retain the graph, parameter schema and solver evidence |
| spatial IV or quadratic-moment GMM | `statadic.estimation.spatial.iv`, `statadic.estimation.spatial.gmm` | Explicit instruments, moments and actual last-solve state |
| balanced static spatial panels with exogenous W_t | `statadic.estimation.spatial.panel` | Actual period filters and Jacobians; KKP's distinct constant-W model lives in `statadic.estimation.spatial.kkp` |
| spatial covariance, impacts or specification diagnostics | `statadic.inference.spatial` | Uses actual fits; reference/null and numerical availability remain separate |

## How much sampling error is in this estimate?

*Calibration of uncertainty under dependence.*

| you want | call | notes |
|---|---|---|
| fit-derived classical, HC, HAC or one-way cluster covariance | `statadic.inference.linear_covariance` | preview; actual fit owns row/weight/parameter alignment; correction and reference choices remain separate |
| actual-weight GMM sampling covariance | `statadic.inference.gmm.covariance.gmm_covariance` | full sandwich from retained weight and final moment variability, not an efficient-weight shortcut |
| model-information or HC covariance for a GLM fit | `statadic.inference.glm_covariance.glm_covariance` | shared mean-equation composer; fixed dispersion one and explicit correction |
| checked linear restrictions and coefficient intervals | `statadic.inference.parameters` | preview; explicit chi-square/F or normal/Student reference, separate degrees of freedom and availability status |
| a HAC / long-run covariance | `statadic.operators.longrun` | `newey_west`, `hall_centered_hac`, `bartlett_weights`, `optimal_bandwidth` |
| limiting covariance from a mean estimating equation | `statadic.core.equations.sandwich_covariance` | accepts a derivative panel or `MeanEquationJacobian`; no division by observation count |
| observation-level parameter sampling covariance | `statadic.core.equations.parameter_covariance` | explicit retained observation count and separate correction; inspect identification/numerical flags; component alignment remains the caller's obligation |
| a covariance operator from distances and marginal variances | `statadic.operators.covariance` | `sigma_dist` and heterogeneous-κ `sigma_dist_hetero_kappa` |
| sample covariance or linear shrinkage toward a target | `statadic.estimation.covariance` | fixed/independent targets use guarded `optimal_lambda_dist`; same-sample targets use the explicitly heuristic `plug_in_lambda_dist` |
| a diagonal-preserving PSD repair | `statadic.optimize.psd` | Qi–Sun semismooth Newton in correlation space; ridge, clipping, diagonal-shrink and array-valued distortion diagnostics are also available |
| a rank-clean undirected dyadic design | `statadic.operators.design` | symmetric endpoint effects (`node_effect` by default, caller-configurable label prefix) and Frisch--Waugh--Lovell residualization |
| both FWL residual series and their slope | `statadic.operators.design.fwl_residualize` | array-valued `FWLResiduals` pytree; fixed-shape JIT/batching with a static focal index; caller precision and rank-clean design; `fwl_coefficient` returns an eager Python float |
| multinomial node-bootstrap inference for a dyadic regression | `statadic.inference.dyadic` | Consumes model specifications from `statadic.model.dyadic`; point fits live in `statadic.estimation.dyadic`; dataframe schemas, labels and publication ladders stay with callers |
| a block length for a dependent series | `statadic.inference.resampling.block_length.optimal_block_length` | Politis–White + Patton. **`heuristic_block_length` in the same module is a different estimator** — `T^{1/3}(1+2\|ρ₁\|)`, disagreeing by 3.4 %–710 % on this repo's fixtures |
| a stationary-bootstrap CI | `statadic.inference.resampling.block_length.stationary_bootstrap_ci` | |
| a transformable bootstrap estimate | `statadic.inference.resampling.bootstrap.bootstrap_estimate` | accepts a PRNG key and returns `EstimateResult`; nested `jit` ✓ leading-axis `vmap` ✓ |
| an eager bootstrap CI for a scalar statistic | `statadic.inference.resampling.bootstrap.bootstrap_confidence_interval` | compatibility adapter returning two Python floats |
| paired vector/pytree statistics or fit refits under supplied row draws | `statadic.inference.resampling` | [schedules and strict summaries](../resampling.md); iid/stationary generators, one chunk executor, retained IDs/status/counts; no silent failed-draw deletion |
| circular block bootstrap indices | `statadic.operators.longrun.circular_block_bootstrap_indices` | |

## Does this moment model satisfy its restrictions?

*Equality, over-identification, inequality, and boundary inference.*

| you want | call | notes |
|---|---|---|
| a fixed-moment HAC Wald test | `statadic.inference.gmm.gmm_wald` | fails closed if the HAC moment covariance loses rank |
| conventional J inference from a retained affine fit | `statadic.inference.gmm.hansen_j_from_fit` | requires explicit correct-specification acknowledgement and efficient-weight evidence; unavailable for exactly identified or arbitrary fixed-weight fits |
| named host restrictions or retained-fit diagnostics | `statadic.api.hypotheses.Hypothesis`, `statadic.api.diagnostics.HansenJ` | Hypotheses state claims; procedures retain their own reference, assumptions and numerical availability. See [hypotheses](hypotheses.md). |
| fixed or affine generalized empirical likelihood fitting | `statadic.estimation.gel.FixedGEL` / `AffineGEL` | [GEL workflow](../gel-fitting.md); retains multipliers, structural parameters and separate solver/rank evidence |
| GEL test calibration from the actual fit | `statadic.inference.gel.gel_test` | Fixed fits support LR/LM/Wald; affine fits support LR; chi-square inference requires explicit iid/serially-uncorrelated moments |
| moment inequalities | `statadic.inference.wolak.wolak_test`, `statadic.inference.gms.indicator_gms` | Wolak uses a χ-bar-square reference; indicator GMS is explicitly uncorrected unless a caller supplies a correction, which is not certified as Table-I calibration |
| reduce a large pair-moment panel | `statadic.inference.clustering.cluster_pairs` | drops empty clusters so they cannot fabricate rank or degrees of freedom |
| κ uncertainty and a lower-boundary test | `statadic.inference.kappa` | delta method, Jacobian-concentration diagnostic, and the `½χ²₀+½χ²₁` boundary tail |

## Is this track record real?

*Performance inference.*

| you want | call |
|---|---|
| a Sharpe ratio | `statadic.estimation.sharpe.sharpe_ratio` |
| probabilistic / deflated Sharpe | `statadic.inference.sharpe.probabilistic_sharpe_ratio`, `deflated_sharpe_ratio` |
| a paired Sharpe-difference test | `statadic.inference.sharpe.sharpe_difference_test` |
| Romano–Wolf / Reality Check across strategies | `statadic.inference.sharpe.multiple_sharpe_test` |

---

## Axioms — which of these is actually a metric?

`geometry/ground/` and `discrepancy/` are **axiom-neutral by construction**: a family name
says what is being compared (points vs measures), never which axioms hold.
Preconditions declare primitive capabilities, not contested names. The complete
contract also records the carrier space, equality, transformation, evidence,
origin, and authority; see [`mathematical-contracts.md`](../mathematical-contracts.md).
Compatibility tables remain executable audit registries while typed strategies
and typed statistical results are the propagation surface. Hypothesis-generated
draws and case-owned adversarial witnesses check the declarations independently.

**The trap worth reading before you pick a default.** Anything that normalises
before comparing cannot see scale, so `d(x, 2x) = 0` and IDENTITY fails:

| dissimilarity | brand |
|---|---|
| `euclidean_distance` | `NegativeType` |
| `angular_distance`, `cosine_distance`, `normalized_euclidean_distance` | **`Premetric`** |
| `energy_distance[euclidean]` | `Semimetric` |
| `energy_distance[angular]` | **`Premetric`** |
| `mmd[rbf]`, `mmd[distance_induced[euclidean]]` at the declared V-statistic domain | `NegativeType` |
| `mmd[linear]`, `mmd[cosine]`, `mmd[distance_induced[angular]]` | **`Premetric`** |
| unbiased `mmd_squared` | **unbranded signed estimate** |
| `w1_1d`, `w2_1d` | `Metric` |
| finite-design `sliced_wasserstein` | `Premetric` compatibility brand; typed law is pseudometric |
| `metrize_energy` | `NegativeType` |
| `exact_balanced_wasserstein` for finite `p >= 1` (including W1/W2 wrappers) | **`Premetric`** on row arrays; a metric on equal-cardinality empirical measures over the unit sphere |

Random property tests do not find this — random draws essentially never produce
parallel vectors. Fixed adversarial witnesses do, and the suite carries them.

## How do I get my data into the right shape?

*S1 — everything that happens before the first pairwise distance.* No question on
this page is answered by `statadic.sample`; every one of them starts there.

| you want | call | notes |
|---|---|---|
| ragged clouds as one **rectangular batch** | `statadic.sample.packed.pack` | `PackedClouds` + `masked_sum`/`masked_mean`/`masked_row_mask`; `pair_chunks` bounds peak memory in the O(n²) kernels |
| **reduce dimension without leaking** the test | `statadic.sample.reduction.fit_projection` | fit on split A, apply to both. This is correction #5 — fitting on the pooled sample invalidates the two-sample test that follows |
| **reference alternatives** to size a test | `statadic.sample.alternatives.draw` | `AltSpec` covers normal, Student-t, mean-shift, scale and mixture departures — power curves without writing a DGP |

Sample preparation depends on shared contracts without selecting a discrepancy,
model or inferential procedure. Its outputs can feed several statistical families;
that direction is part of the explicit dependency DAG.

## Substrate

| package | contents |
|---|---|
| `statadic.core` | matrix/equation contracts, jaxtyping aliases, `TestResult`/`EstimateResult`, `SolverResult`/`SolverStatus`, `cell_key` |
| `statadic.optimize` | Navigation namespace: import `backends`, `simplex`, `psd`, `glm` or `integration` explicitly for numerical algorithms and their evidence |
| `statadic.testing` | the axiom battery, shipped in the wheel so dependents can assert the same properties |

## Import conventions

- Import from the owning public family or leaf, for example
  `from statadic.discrepancy.mmd.energy import energy_distance`. Root and compound
  namespaces provide navigation; they do not re-export the complete package.
- Family initializers explain purpose, inputs, outputs and children, then curate
  family exports. Private helpers stay within the family; reusable component
  reductions have narrow public interfaces.
- `*_distance` / `*_statistic` return numerical values; `*_test` returns a
  procedure-specific test result; `*_fit` retains actual fitted state.
  Uncertainty, reference distributions and optimization status remain separate.
  Kernel strategies and compiled numerical kernels are distinguished by their
  owning module and documented contract.
- The [module guide](../module-organization.md) records current clean import breaks,
  canonical replacements and persisted qualified-name considerations. Older
  release migrations remain in the changelog.
