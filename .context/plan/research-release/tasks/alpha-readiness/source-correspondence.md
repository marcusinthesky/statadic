---
type: document
title: Required measures and permutation source correspondences
description: Exact primary-passage correspondences and bounded coordinator formula-source dispositions; human fidelity remains pending.
links:
  - .context/plan/research-release/tasks/alpha-readiness/README.md
  - .context/plan/research-release/tasks/alpha-readiness/contracts.md
  - .context/plan/research-experience/tasks/studies/tasks/sources/decisions.md
  - .context/plan/research-release/tasks/alpha-review/formal-review.md
---

# Required measures and permutation source correspondences

Reviewed 2026-11-09 against the corrective working source. The original recommendations
below preceded canonical adjudication. The final section records the coordinator's
subsequent exact29 formula-source dispositions; neither stage grants human fidelity approval.
The table covers 36 named variants: three permutation, sixteen MMD, one prototype,
six energy, eight transport and two spatial-survival variants. Numerical tests,
sampling-law qualification, Lean body checking and human review remain separate.
This supporting record creates no new task or reference authority.

## Primary passages actually read

| Source | Complete relevant passage read | Scope used here |
| --- | --- | --- |
| [Hemerik–Goeman 2018](https://link.springer.com/content/pdf/10.1007/s11749-017-0571-1.pdf) | Printed pp.819–822, Definition 2, Theorem 2 and both proofs, Proposition 3, §3.4 | Random group transformations, identity inclusion and tail ranks. Assumes invariance and appropriate uniform transformation sampling. |
| [Gretton et al. 2012](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) | Printed pp.726–729: Lemmas 3–6 and proofs, equations 3–5 | Kernel-mean functional, unequal-size U estimator, empirical rooted V estimator. Kernel and moment assumptions remain explicit. |
| [Sejdinovic et al. 2013](https://arxiv.org/pdf/1207.6076) | Reprint pp.4–5,8–11,15–17,20–21,28–29: definitions 2,5,10, Proposition 3, Example 15, Theorem 22 and proof, Proposition 29, equation 7.1, Appendix B | Finite probability-measure substitution and distance-kernel normalization; negative type and separation differ. |
| [Lyons, spheres, arXiv v3](https://arxiv.org/pdf/1905.02863) | All six pages, including §2 proofs; especially proof of Theorem 1.2 on p.4 | Hemispherical representation of angular distance; negative type on the sphere. Antipodal restrictions matter for stronger separation. This is a proposed additional reference, not an existing governed key. |
| [Peyré–Cuturi, author manuscript v2](https://arxiv.org/html/1803.00567v2) | §2.2 equations 2.2–2.4; §2.6 remarks 2.28–2.30, equations 2.33–2.39; §10.4 equations 10.13–10.17 | Assignment, real-line quantiles and sliced objectives. This review uses an explicit L2 argument only for the p=2 Hilbert claim. |
| [Botev, author manuscript](https://arxiv.org/pdf/1603.04166) | §2–3 through Algorithm 3.1, including equations 5–10; §4 theorem hypotheses inspected | Sequential importance law and saddle equations. Current numerical proposal fitting establishes no minimax or error-bound theorem. Proposed reference: 2017 JRSS B 79(1),125–148, DOI 10.1111/rssb.12162. |

These passages were read in their primary full texts, not inferred from abstracts.
The complete Gretton, Sejdinovic, Peyré–Cuturi and Botev works were not read
end-to-end in this closure pass. No bibliography, source dossier, registry or human
review flag was changed. Wiley's PERMDISP full-text URL still redirected to an
abstract; it supplies no new exact residual-permutation approval.

## Exact code owners

The links name current implementation lines; line numbers may move on formatting.
All abbreviated producer names below mean these exact modules.

- **PT**: [permutation_test_result](../../../../../src/python/packages/statadic/src/statadic/inference/resampling/permutation/test.py#L209), with [permutation_test_kernel](../../../../../src/python/packages/statadic/src/statadic/inference/resampling/permutation/kernel.py#L44).
- **MS**: [scalar mmd_squared and mmd](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/scalar.py#L126); **MM**: [mmd_squared_matrix and mmd_matrix_values](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/matrix.py#L30).
- **MW**: [weighted_mmd_squared_cross, weighted_mmd_cross and weighted_mmd_matrix](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/weighted.py#L73).
- **KS**: [linear/cosine/RBF Gram definitions and the unhalved distance kernel](../../../../../src/python/packages/statadic/src/statadic/geometry/ground/similarities.py#L128); **DK**: [typed distance_induced_kernel](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/energy/kernel.py#L117).
- **ES**: [energy_distance and energy_distance_with_geometry](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/energy/scalar.py#L172); **EM**: [energy_distance_matrix](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/energy/matrix.py#L73).
- **EF**: [energy_functional_matrix](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/energy/matrix.py#L237) and [metrize_energy](../../../../../src/python/packages/statadic/src/statadic/core/matrices.py#L793).
- **EW**: [weighted_energy_squared_cross, weighted_energy_cross and weighted_energy_matrix](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/energy/weighted.py#L83); **EI**: [mixture_energy_distance and mixture_energy_distance_from_blocks](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/energy/mixture.py#L74).
- **TE**: [exact_balanced_wasserstein and order-specific eager/JAX wrappers](../../../../../src/python/packages/statadic/src/statadic/discrepancy/transport/exact.py#L364), backed by [balanced_wasserstein_from_cost_result](../../../../../src/python/packages/statadic/src/statadic/discrepancy/transport/exact.py#L293).
- **TU**: [wasserstein_1d_power and wasserstein_1d](../../../../../src/python/packages/statadic/src/statadic/discrepancy/transport/univariate.py#L125); **TS**: [w1_1d, w2_1d and sliced functions](../../../../../src/python/packages/statadic/src/statadic/discrepancy/transport/sliced.py#L199); **TM**: [quantile_w2_matrix and finite_sliced_w2_matrix](../../../../../src/python/packages/statadic/src/statadic/discrepancy/transport/matrix.py#L70).
- **GP**: [gaussian_tilt_proposal](../../../../../src/python/packages/statadic/src/statadic/optimize/integration/gaussian.py#L183) and [gaussian_log_survival](../../../../../src/python/packages/statadic/src/statadic/optimize/integration/gaussian.py#L424); **SP**: [prepare_spatial_survival_proposals](../../../../../src/python/packages/statadic/src/statadic/estimation/survival/spatial.py#L265), [conditional Gaussian construction](../../../../../src/python/packages/statadic/src/statadic/estimation/survival/spatial.py#L215), and [refine_spatial_survival](../../../../../src/python/packages/statadic/src/statadic/estimation/survival/refinement.py#L144).

**Formula candidate** below means the stated representative mathematical formula
correspondence is supported for coordinator review. It excludes floating-point
refinement, automatic tuning, universal inferential validity and completed human
review. **Keep pending** means the entire named variant exceeds the passage/code
bridge established here. Existing four linear/RBF unrooted approvals stay bounded.

## Per-variant recommendations

| Exact variant ID | Narrow producer/configuration | Recommendation |
| --- | --- | --- |
| `permutation.two-sample.greater` | PT, `alternative="greater"` | Formula candidate: upper rank with add-one; fixed finite statistic and exchangeable pooled rows. |
| `permutation.two-sample.less` | PT, `alternative="less"` | Formula candidate: apply the upper-rank argument to the negated statistic. |
| `permutation.two-sample.two-sided` | PT, `alternative="two-sided"` | Formula candidate: absolute-statistic upper rank, not doubled one-sided tails. |
| `mmd.estimation.linear-u-squared` | MS.mmd_squared, MM.mmd_squared_matrix; linear, `unbiased=True` | Retain bounded approval; Gretton equation 3. |
| `mmd.estimation.linear-v-squared` | Same two producers; linear, `unbiased=False` | Retain bounded approval; squared empirical kernel-mean norm. |
| `mmd.estimation.linear-v-rooted` | MS.mmd, MM.mmd_matrix_values; linear | Formula candidate; source equation 5. |
| `mmd.estimation.linear-weighted-v-rooted` | MW.weighted_mmd_cross, MW.weighted_mmd_matrix; linear | Formula candidate; finite probability substitution. |
| `mmd.estimation.rbf-u-squared` | MS.mmd_squared, MM.mmd_squared_matrix; positive fixed RBF bandwidth, U | Retain bounded approval; no bandwidth selection. |
| `mmd.estimation.rbf-v-squared` | Same two producers; positive fixed RBF bandwidth, V | Retain bounded approval. |
| `mmd.estimation.rbf-v-rooted` | MS.mmd, MM.mmd_matrix_values; positive fixed RBF bandwidth | Formula candidate. |
| `mmd.estimation.rbf-weighted-v-rooted` | MW.weighted_mmd_cross, MW.weighted_mmd_matrix; positive fixed RBF bandwidth | Formula candidate. |
| `mmd.estimation.cosine-u-squared` | MS.mmd_squared, MM.mmd_squared_matrix; cosine, U | Formula candidate with KS row-normalized feature map; active rows nonzero. |
| `mmd.estimation.cosine-v-squared` | Same two producers; cosine, V | Formula candidate with the same domain. |
| `mmd.estimation.cosine-v-rooted` | MS.mmd, MM.mmd_matrix_values; cosine | Formula candidate; no distribution separation from directional means alone. |
| `mmd.estimation.cosine-weighted-v-rooted` | MW.weighted_mmd_cross, MW.weighted_mmd_matrix; cosine | Formula candidate on nonzero positive-mass rows. |
| `mmd.estimation.distance-induced-u-squared` | MS.mmd_squared, MM.mmd_squared_matrix; DK Euclidean unit-power kernel, U | Formula candidate with common reference; kernel is unhalved. |
| `mmd.estimation.distance-induced-v-squared` | Same two producers and kernel, V | Formula candidate under the same fixed geometry. |
| `mmd.estimation.distance-induced-v-rooted` | MS.mmd, MM.mmd_matrix_values; same DK | Formula candidate; retain the implemented normalization. |
| `mmd.estimation.distance-induced-weighted-v-rooted` | MW.weighted_mmd_cross, MW.weighted_mmd_matrix; same DK | Formula candidate; probability masses and common reference. |
| `mmd.cosine-prototypes.normalized-sample-mean-prototype-distance` | [MS.cosine_mean_mmd](../../../../../src/python/packages/statadic/src/statadic/discrepancy/mmd/scalar.py#L220) only | Formula candidate as supplied-prototype Euclidean chord; no source approval for application pooling/normalization. |
| `energy.estimation.raw-euclidean-exponent1-v-energy` | ES and EM; Euclidean, exponent 1 | Formula candidate; raw functional, no root. |
| `energy.estimation.raw-angular-exponent1-v-energy` | ES and EM; angular, exponent 1 | Formula candidate combining Sejdinovic with the Lyons spherical representation and normalization pullback. No separation claim. |
| `energy.estimation.rooted-euclidean-energy-matrix` | EF only; Euclidean, exponent 1 | Formula candidate; construction-bound empirical V root. |
| `energy.estimation.weighted-empirical-euclidean-energy` | EW.weighted_energy_cross, EW.weighted_energy_matrix; Euclidean, exponent 1 | Formula candidate for the rooted quantity; record rootedness explicitly in authoritative scope. |
| `energy.estimation.weighted-candidate-mixture-energy` | EI only; supplied simplex mixture weights | Formula candidate for raw fixed-mixture objective; excludes fitted-weight selection and unrestricted signed weights. |
| `energy.estimation.fractional-superunit-exponent-regimes` | ES and EM; Euclidean, `0<exponent<2` | Formula candidate for this Euclidean range; retain invalid ground/exponent rejection separately. Do not grant every registered ground the whole range. |
| `transport.empirical.exact-balanced-uniform-w1` | TE exact_balanced_wasserstein1, exact_balanced_wasserstein1_jax; generic TE wrappers only with p=1 | Formula candidate: equal active counts, uniform masses, unit-sphere chord ground. |
| `transport.empirical.exact-balanced-uniform-w2` | Corresponding TE order-two wrappers; generic TE wrappers only with p=2 | Formula candidate; assignment must minimize squared ground costs. |
| `transport.empirical.empirical-one-dimensional-w1` | TU.wasserstein_1d, TU.wasserstein_1d_power with p=1; TS.w1_1d | Formula candidate; real-line probability measures. |
| `transport.empirical.empirical-one-dimensional-w2` | TU.wasserstein_1d with p=2; TS.w2_1d; TM.quantile_w2_matrix | Formula candidate; root retained. |
| `transport.empirical.empirical-one-dimensional-powered-w2` | TU.wasserstein_1d_power with p=2 | Formula candidate; explicitly unrooted. |
| `transport.empirical.fixed-design-sliced-wasserstein` | TS.sliced_wasserstein_under_design, TS.sliced_wasserstein_power_under_design; TM.finite_sliced_w2_matrix only when p=2 | Formula candidate for finite shared directions; powered and rooted outputs need explicit scope. |
| `transport.empirical.random-projection-sliced-wasserstein` | TS.sliced_wasserstein, with fixed explicit key/count/order | Formula candidate for finite Monte Carlo directional functional; exclude exact population-sphere integration. |
| `transport.empirical.weighted-unequal-support-quantile-w2` | TU.wasserstein_1d with p=2; TM.quantile_w2_matrix | Formula candidate; nonnegative normalized masses, possibly unequal supports; no assignment-only restriction. |
| `survival.spatial.fixed-draw-joint-gaussian-spatial-log-time-ml` | SpatialSurvivalML/estimate; optional SP preparation and GP fixed proposal | Keep whole variant pending. Proposal correspondence is bounded; complete likelihood/information, fit existence and high-censor accuracy require their own evidence. Remove refinement from this variant's symbol set. |
| `survival.spatial.adaptive-integration-refined-spatial-ml` | SP.refine_spatial_survival plus actual SpatialSurvivalML/estimate and explicit frozen preparation | Keep pending. Budget/validation agreement is a numerical diagnostic, not an error bound, MLE existence proof or calibration. |

## Implementation bridges and exclusions

1. PT draws uniform pooled-row permutations with split keys and keeps group sizes
   fixed. The assembly counts `>=`, `<=` or absolute exceedances and computes
   `(1+b)/(1+B)`. Supplied-null assembly alone cannot establish the resampling
   law. Label-trained transformations must be recomputed or otherwise shown
   invariant; freezing arbitrary supervised tuning is outside this recommendation.
   Energy-specific permutation wrappers can be associated separately after their
   fixed-statistic geometry contract is recorded.

2. MS's U branch removes within-sample diagonal terms and uses true counts
   `n(n-1)`, `m(m-1)`, with all cross pairs, matching Gretton equation 3 rather
   than the paired equal-size alternative equation 4. It requires uniform atom
   masses and at least two true atoms, and keeps negative U estimates. MW is a
   weighted V functional, never a weighted U estimator. RBF bandwidth and all
   three distance-kernel blocks are fixed consistently. The adapter's NumPy
   finite-support truths are independent arithmetic, not new primary proofs.

3. For the STATADIC distance kernel `d(x,z)+d(y,z)-d(x,y)`, code expansion yields
   MMD-squared equal to raw energy. The primary paper uses half this kernel and
   consequently its factor is two. Source approval must retain that distinction.
   Finite empirical probabilities supply finite moments; they do not establish
   moment conditions for an arbitrary population. Angular normalization loses
   scale, and antipodal obstructions prohibit a blanket distribution metric.
   The current `weighted-empirical-euclidean-energy` study mapping also names
   angular methods. A Euclidean-only source disposition cannot cover those
   rows: split the angular configuration or record its distinct spherical
   bridge and domain explicitly before resolving the whole variant.

4. TE consumes already normalized rows and uniform masses with equal true atom
   counts. It powers costs before Hungarian assignment and roots the selected
   mean afterwards. A common positive scale preserves the mathematical target;
   underflow flags and absence of a flag do not establish solver accuracy.
   TU integrates merged probability-CDF intervals; zero-width intervals carry no
   cost. TM grants a Hilbertian producer only for real-line W2 and finite shared
   sliced W2. The source's informal wording following equation 2.36 must not
   broaden that L2 argument to every p.

5. GP samples truncated shifted standard-normal coordinates, accumulates the
   exact importance ratio and analytically integrates the terminal coordinate.
   The selected permutation and tilt remain constant during each fitted
   objective; Gaussian parameters and conditional censor bounds still change.
   Its recorded numerical saddle residual is distinct from L-BFGS termination.
   SP preserves full-graph missing-outcome marginalization, exact-event density
   and time Jacobian. Draw doubling retains proposals and requires independent
   validation keys. No refitted-proposal derivative or minimax guarantee follows.

## Adjudication and remaining gates

The coordinator can apply the exact producer narrowing independently of a source
status decision. The earlier 32 approved-variant audit found 25 genuinely broad
symbol sets; the DSR set legitimately includes its nested PSR/expected-maximum
dependencies. Root applied those corrections through the canonical configuration.
The unresolved variants above still need the same narrowing; family symbol unions
are not representative configurations.

If accepting a formula candidate, record its configuration, domains, primary
locator and exclusions together. Additional governed references are needed for
the spherical angular and proposal passages before citing new keys in shipped
documentation. Keep all 153 human claim fidelity reviews pending. No proposal,
formula correspondence or Lean fact licenses passing a required campaign with
missing or adverse numerical/statistical evidence.

The spatial-survival contract remains 500 units by 20 periods, both graph
semantics, 10/50/80 percent censoring, independent per-period error at most 1e-3,
and identified refit shifts at most 0.02 observed-information standard errors.
The separate executable build pilot is not a release campaign and does not
close the general-graph, full-fit, information or calibration gates. Ordinary AFT,
PERMDISP's exact residual convention, Mantel/bootstrap conventions, dyadic
bootstrap provenance and newly resumed effects/Durbin cases are outside this
record's closure. They retain visible source gaps.

## Coordinator formula-source adjudication

On 2026-11-09 the coordinator independently challenged the primary Gretton
Lemma3–6/equations3,5; Sejdinovic Theorem22/proof/Example15; Peyre–Cuturi
quantile Remark2.30/sliced equation10.13; and Hemerik–Goeman Theorem2/proof/section3.4
against the narrowed current producers and selectors. The following exact29
variants now carry `verified-method-source` in the canonical studies catalogue,
with passage locators and exclusions in each `source_evidence`:

- `permutation.two-sample.greater`
- `permutation.two-sample.less`
- `permutation.two-sample.two-sided`
- `mmd.estimation.linear-v-rooted`
- `mmd.estimation.linear-weighted-v-rooted`
- `mmd.estimation.rbf-v-rooted`
- `mmd.estimation.rbf-weighted-v-rooted`
- `mmd.estimation.cosine-u-squared`
- `mmd.estimation.cosine-v-squared`
- `mmd.estimation.cosine-v-rooted`
- `mmd.estimation.cosine-weighted-v-rooted`
- `mmd.estimation.distance-induced-u-squared`
- `mmd.estimation.distance-induced-v-squared`
- `mmd.estimation.distance-induced-v-rooted`
- `mmd.estimation.distance-induced-weighted-v-rooted`
- `mmd.cosine-prototypes.normalized-sample-mean-prototype-distance`
- `energy.estimation.raw-euclidean-exponent1-v-energy`
- `energy.estimation.rooted-euclidean-energy-matrix`
- `energy.estimation.weighted-empirical-euclidean-energy`
- `energy.estimation.weighted-candidate-mixture-energy`
- `energy.estimation.fractional-superunit-exponent-regimes`
- `transport.empirical.exact-balanced-uniform-w1`
- `transport.empirical.exact-balanced-uniform-w2`
- `transport.empirical.empirical-one-dimensional-w1`
- `transport.empirical.empirical-one-dimensional-w2`
- `transport.empirical.empirical-one-dimensional-powered-w2`
- `transport.empirical.fixed-design-sliced-wasserstein`
- `transport.empirical.random-projection-sliced-wasserstein`
- `transport.empirical.weighted-unequal-support-quantile-w2`

The four previously approved linear/RBF unrooted U/V dispositions remain.
All MMD U selectors explicitly require uniform atom masses and at least two
active atoms per measure. Weighted Euclidean energy's study mapping now excludes
angular methods. Prototype approval concerns supplied normalized vectors, not
the pipeline's pooling/normalization procedure. Sliced approval is the recorded
finite p=2 design, not an exact full-sphere or selected-geometry procedure.

Raw angular energy and both whole joint spatial-survival variants remain
pending. No human claim-review state, Monte Carlo status or implementation-proof
claim changed. The 176-variant selection remains unchanged: zero missing study
mappings after the new exact adapters were registered, and 124 required formula-source
reviews remain pending. Registration does not establish usable fits, correct
coverage, integration accuracy or passed qualification. Retained proposal/application
scripts and exact proposal JSON are supporting evidence under `build/final-sweep/`;
the canonical catalogue owns the applied dispositions.

## Three further bounded formula dispositions

The coordinator independently read the original primary PDF passages for these
three additional approvals on 2026-11-09. Required pending formula-source reviews
therefore decrease from 124 to 121; the selection remains 176 variants.

| Exact variant | Primary passage and approved boundary |
| --- | --- |
| `linear.ols-wls.analytic-precision-wls` | [Stepniak, Theorem 3.2(b), equation 3.3](https://link.springer.com/content/pdf/10.1155/2010/259672.pdf). Explicit whitening specialization with fixed positive relative-precision weights and full column rank; point-estimation formula only. |
| `glm.covariance.model` | [Nelder–Wedderburn, equations 8–11, pp.373–374](https://jhanley.biostat.mcgill.ca/bios601/Likelihood/NelderWedderburn1972.pdf). Canonical binomial/Poisson information, independent rows and unit dispersion; no quasi-scale or coverage claim. |
| `linear.residual-diagnostics.white` | [White, Corollary 1/equation 3 and redundancy discussion, pp.824–825](https://crooker.faculty.unlv.edu/econ441/econ_papers/White-Heteroskedasticity-Correction-1980.pdf). Auxiliary squared-residual regression on second products with a constant, rank-adjusted degrees, homokurtic null and source regularity. |

The other five new proposals remain unapproved pending adequate primary-passage
challenge: raw Bartlett HAC, GLM HC0, conditional normal/t intervals and
intercept-restricted Jarque–Bera. All 18 likelihood-regression variants remain
unresolved: distribution definitions do not establish full joint ancillary or
robust-inference procedures. A nongoverned White 1982 source cannot silently
replace the existing linear-regression source. The proposal's exact selectors,
locators and access limitations are retained in
`build/final-sweep/a-source-correspondence-proposals.json`.

No human fidelity, implementation verification or campaign disposition changed.
