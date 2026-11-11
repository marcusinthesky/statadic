# Reference-gap verification

Executed through the repository's external lane (`just statadic::test-external` recipe:
`-m external_reference --require-external-references --no-testmon`, `JAX_ENABLE_X64=true`,
the `reference` uv group in `.cache/statadic-reference-venv`, run under `devenv shell`).
Peers: statsmodels 0.14.6, glum 3.4.1, linearmodels 7.1, spreg 1.9.0, lifelines 0.30.0.
Result for the nodes below: 35 passed, 0 skipped, 0 failed (the survival file's six
unrelated panel-design nodes fail; see "Existing failure").

Nothing in the catalogue, dispositions, deferred list or production source was edited.
The mapping below is for the coordinator to apply.

## Comparisons added and the catalogue variant each maps to

### Log-logistic AFT vs lifelines (`survival.independent.loglogistic`)

Added by an earlier agent and reviewed here. File `tests/testing/test_survival_references.py`.

- `tests/testing/test_survival_references.py::test_aft_fit_information_and_predictions_match_lifelines[loglogistic-ordinary]`
- `tests/testing/test_survival_references.py::test_aft_fit_information_and_predictions_match_lifelines[loglogistic-delayed-entry]`

Convention: lifelines S(t) = 1 / (1 + (t / alpha)^beta) with log alpha linear in the
covariates and log beta constant, the same as `LogLogisticAFT`. The test compares
parameters, delayed-entry log likelihood, free-parameter covariance, survival, cumulative
hazard, quantiles and restricted mean. The peer is given tighter L-BFGS-B stopping options
(`ftol=1e-15, gtol=1e-10, maxiter=2000`) for the log-logistic case only. Checked: at
default options lifelines' log likelihood is below the tight-option value and its
coefficients move by about 3e-5 to 2e-4 (delayed-entry case: -214.92674 vs -214.92674,
coefficients 0.35931 vs 0.35928, -0.23623 vs -0.23645), so the default stop is short of
the maximiser and the option change only makes the peer converge. The compared tolerances
are unchanged. Maps to `survival.independent.loglogistic` (its `loglogistic` and
`loglogistic-delayed` selectors); `survival.mean.loglogistic` is also exercised by the
restricted-mean check in the same nodes.

### 2SLS robust covariance vs linearmodels

File `tests/integration/api/test_structural_references.py`; the new test is
`test_public_two_stage_least_squares_robust_covariance_matches_linearmodels`, run for
float32 and float64 inputs (the existing test's pattern, with x64 solved internally).

| Node (suffix of the test id) | Catalogue variant |
| --- | --- |
| `[HC0-float32]`, `[HC0-float64]` | `iv.regression.2SLS.HC0` |
| `[HC1-float32]`, `[HC1-float64]` | `iv.regression.2SLS.HC1` |
| `[HAC-float32]`, `[HAC-float64]` | `iv.regression.2SLS.HAC` (bandwidth 4) |
| `[clustered-float32]`, `[clustered-float64]` | `iv.regression.2SLS.clustered` |

Tolerance 4e-4 (float32 inputs) and 2e-8 (float64), the existing 2SLS test's. Measured
float64 agreement is about 1e-15 relative. The structural error is heteroskedastic so the
robust meats differ from the model-based one. Conventions, written beside the test:

- HC0 is linearmodels `cov_type="robust", debiased=False`.
- HC1 is `debiased=True`, which scales by n / (n - k) with k = 3 coefficients; the same
  factor as statadic's `n / (n - p)`.
- HAC is Bartlett with weights 1 - j / (b + 1), lags j <= b, uncentred scores, no
  small-sample factor; statadic's `bandwidth=4` equals linearmodels `bandwidth=4`.
  (`bandwidth=5` does not match: it is a different kernel width.)
- Clustered is one-way with no G / (G - 1) factor and `debiased=False`. With
  `debiased=True` linearmodels differs by about 11 percent, so that option is not the
  same convention.

`iv.regression.2SLS.model` was already mapped to `test_public_two_stage_least_squares_matches_linearmodels`.

### Gamma, NB2 and probit likelihood regression vs statsmodels

File `tests/estimation/test_likelihood.py`. Tolerances are written in the tests: coefficients
rtol 1e-6 (fixed) and 5e-6 (estimated NB2, because statsmodels' own optimiser stops near
1e-7), log likelihood rtol 1e-10 (gamma, probit), covariance rtol 2e-6 (5e-6 estimated NB2),
all scaled by the largest entry.

| Node | Catalogue variants |
| --- | --- |
| `tests/estimation/test_likelihood.py::test_fixed_gamma_matches_statsmodels_glm` | `likelihood.regression.gamma.fixed.model`, `.fixed.HC0`, `.fixed.HC1` |
| `tests/estimation/test_likelihood.py::test_fixed_nb2_matches_statsmodels_glm` | `likelihood.regression.nb2.fixed.model`, `.fixed.HC0`, `.fixed.HC1` |
| `tests/estimation/test_likelihood.py::test_estimated_nb2_matches_statsmodels_negative_binomial` | `likelihood.regression.nb2.estimated.model`, `.estimated.HC0`, `.estimated.HC1` |
| `tests/estimation/test_likelihood.py::test_probit_matches_statsmodels[binary]` | `likelihood.regression.probit.binary.model`, `.binary.HC0`, `.binary.HC1` |
| `tests/estimation/test_likelihood.py::test_probit_matches_statsmodels[grouped]` | `likelihood.regression.probit.grouped.model`, `.grouped.HC0`, `.grouped.HC1` |
| `tests/estimation/test_likelihood.py::test_estimated_gamma_matches_statsmodels_likelihood` | none recommended (see below) |

Conventions, written beside the tests:

- Gamma dispersion phi has variance phi * mu^2, which is statsmodels' fixed `scale`. NB2
  dispersion is statsmodels' `alpha` (variance mu + alpha mu^2). Coefficients are in the
  log link.
- Model covariance: statadic uses the inverse observed information. statsmodels' default GLM
  covariance is expected information, which differs for non-canonical links (log-link
  gamma and NB2, probit). The statadic-vs-default gap is 9 percent (gamma, NB2) and 0.4 percent
  (grouped probit). The peer used is the inverse of statsmodels' own
  `hessian(params, observed=True)`. For binary `Probit` and for `NegativeBinomial` the
  peer's default already is observed information.
- HC0 is statsmodels `cov_type="HC0"` unchanged (agrees to 1e-9).
- HC1: statsmodels applies no degrees-of-freedom factor in GLM or discrete models (its HC1
  equals its HC0, verified). statadic scales HC0 by n / (n - p) with p counting the dispersion
  when estimated. The test therefore compares statadic HC1 to statsmodels HC0 times that
  factor. This confirms the factor is applied consistently but is not an independent peer
  value for the factor itself.
- Estimated NB2: statsmodels' last parameter is alpha, not its log, matching statadic's
  physical dispersion, so the joint (beta, alpha) covariance compares directly.
- Grouped probit peer is `GLM(Binomial(Probit))` with (successes, failures) endog, whose
  log likelihood carries the same binomial coefficient.

Not mapped: the six `likelihood.regression.gamma.estimated.*` variants have no peer for
what they are about. statsmodels (and glum) estimate the gamma scale by a Pearson/deviance
moment rule, not by maximum likelihood, and offer no joint (beta, dispersion) covariance.
`test_estimated_gamma_matches_statsmodels_likelihood` matches the coefficients and, through
the peer's `loglike(params, scale)` maximised over scale, statadic's dispersion (rtol 1e-5) and
log likelihood (rtol 1e-9). That is a likelihood witness, not a comparison of the estimator
or covariance; I recommend leaving those six variants unmapped.

glum was not used for the new comparisons: its gamma/NB2 dispersion is not estimated by
maximum likelihood either, and the existing glum Poisson comparison already covers its
unpenalised log-link path. This was inferred from glum's documented behaviour and
statsmodels' equivalent, not run.

### Hurdle Poisson vs statsmodels blocks

File `tests/api/regression/test__conditional.py`.

- `tests/api/regression/test__conditional.py::test_hurdle_poisson_matches_statsmodels_blocks[logit]` maps to `likelihood.hurdle-poisson.logit`
- `tests/api/regression/test__conditional.py::test_hurdle_poisson_matches_statsmodels_blocks[probit]` maps to `likelihood.hurdle-poisson.probit`

statsmodels' `HurdleCountModel` is not a peer: its zero hurdle is a right-censored Poisson
(P(0) = exp(-lambda), a complementary log-log gate), it has no logit or probit gate, it
forbids offsets and exposure, and it rejects robust covariance. The hurdle likelihood does
factorise, so the same model is assembled from statsmodels' own blocks: `Logit`/`Probit` for
the indicator y == 0 (statadic's gate models the zero probability) and
`TruncatedLFPoisson(truncation=0)` on positive rows. Coefficients (rtol 1e-6, agreement
about 1e-9), the summed log likelihood (rtol 1e-10), the model covariance of each block
(rtol 1e-6; the count block differs by 1.2e-7 from the optimisers' stopping rules) and
the four predictions (rtol 1e-6) agree. The predictions of `positive_mean` and `mean` are
composed by the model's definition from the peer parameters, lambda / (1 - exp(-lambda)) and
(1 - P0) times that; that algebra is not computed by the peer library. This is a peer
for the blocks, not for a joint hurdle fit; the mapping is justified only on that basis and
the owner may prefer to treat it as partial.

### Spatial panel standard errors vs spreg

File `tests/integration/api/test_spatial_reference_walkthroughs.py`; the existing test
`test_panel_ml_public_routes_match_spreg` now also compares the (beta, rho) covariance for
the `complete` route.

- `tests/integration/api/test_spatial_reference_walkthroughs.py::test_panel_ml_public_routes_match_spreg[complete]`

Convention: spreg `Panel_FE_Lag.vm` and statadic's `PanelML` with `SpatialInformation()` are both
the inverse expected (Anselin) Gaussian information on (beta, rho), scale eliminated after
inversion, with no degrees-of-freedom option in spreg. Agreement is 4e-10 relative; the
tolerance is rtol 1e-6, atol 1e-9. It maps to whichever `spatial.panel.*` variant records
`PanelML` constant-weight unit effects. I inferred, but did not verify, that this is
`spatial.panel.balanced-period-varying-panelml`'s sibling for constant W; the 39 other
variants are not claimed.

Not comparable: `[observed]` and `[graph]`. Those routes report observed information, not
expected, so their standard errors differ from spreg's by 2e-3 to 7e-2 relative (observed
route: 0.04747, 0.04856, 0.08576 vs spreg 0.04759, 0.04923, 0.08060). That is a difference of
estimator convention, not a bug, and spreg has no observed-information option. The graph
route also carries unit-effect parameters (15 vs 3). They stay without an SE comparison.
`[observed]` and `[graph]` keep their existing mapping for coefficients, likelihood and
predictions only.

## Peers that did not match, and what was not forced

- gamma estimated dispersion and covariance (above).
- hurdle Poisson as a whole (above): `HurdleCountModel` is a different model.
- Observed-information spatial panel routes (above).
- The other 39 `spatial.panel` variants were not attempted: they cover random effects,
  changing rosters, multi-operator and learned graphs, which spreg's `Panel_FE_Lag` does not
  implement.
- Pyfixest: not added, no dependency or lock edit; owner decision stays open.

## Real disagreement found: NB2 log likelihood

statadic's NB2 row log likelihood is not accurate when compared with an independent evaluation.
`statadic.model.likelihood.log_likelihood_rows(..., "nb2")` uses `jax.scipy.special.betaln(size, y)`.
In float64, `betaln` itself differs from `scipy.special.betaln` by 4.1e-7 at (10, 1), 6.1e-7 at
(10, 2) and 8e-9 at (1.36, 30). Per-row statadic minus a direct `gammaln` evaluation of the NB2
density: about -4e-7 per row at alpha = 0.1 for y = 1, 2, 7; none to 1e-15 for y = 0.

Consequence in the compared fits: for the estimated NB2 case (n = 200, alpha near 0.74)
`fit.log_likelihood` = -306.0214497, statsmodels' `llf` and its `loglike` evaluated at
statadic's own parameters both give -306.0214467 (relative 9.7e-9). The fixed NB2 case
(alpha = 0.7) differs by 9e-7 absolute. At alpha = 0.1 the total over the same rows is off by
6e-5. Coefficients and covariance still agree with statsmodels, so only the reported
likelihood (and anything built on it, including a likelihood-based comparison and the
estimated-alpha stationary point at the 1e-6 level) is affected. I did not change production
code or widen a tolerance. The two NB2 tests deliberately do not assert the NB2 log
likelihood, with a comment pointing here. The owner should decide on replacing `betaln` by a
`gammaln` combination (the gamma branch is already stable and matches to 1e-10), after which
a `rtol=1e-10` log-likelihood assertion can be added to both NB2 tests.

## Existing failure (not mine, not touched)

In the external lane the six nodes
`tests/testing/test_survival_references.py::test_unbalanced_repeated_spells_match_lifelines_design[...]`
(`unit-family0`, `unit-family1`, `time-family0`, `time-family1`, `two-way-family0`,
`two-way-family1`) fail with `ValueError: Fixed effects require a design without an explicit
intercept.` raised at `statadic/api/survival/_model.py:505`. The test builds
`Intercept() + Linear("x")` with absorbed effects; production rejects that. Neither the test
region nor `_model.py` is in my diff (`_model.py` last changed in b945c85), so this predates my
work. These nodes are excluded from the list above and should not be mapped until fixed.

## Acceptance state

- Listed variants: comparisons added for log-logistic, 2SLS (HC0, HC1, HAC, clustered), NB2,
  probit, fixed gamma and hurdle Poisson (blocks); reasons recorded for estimated gamma
  and the non-matching spatial routes.
- Spatial panel SEs: compared for `PanelML`; reason recorded for the observed and graph routes.
- Pyfixest decision: untouched.
- Reference lane: my nodes ran with external references required and no skips; the full
  `validate-references` lane was not run (coordinator).

## Coordinator follow-up

- Catalogue: 20 variants mapped to the executed nodes above (log-logistic; 2SLS HC0,
  HC1, HAC, clustered; fixed gamma; fixed and estimated NB2; binary and grouped probit).
  The catalogue check reports 116 of 309 variants mapped, up from 96. Hurdle Poisson
  stays unmapped as a partial peer, and the spatial panel standard-error comparison is
  not mapped because its variant identity was not verified.
- Existing failure repaired: the six `test_unbalanced_repeated_spells_match_lifelines_design`
  fixed-effect nodes used an explicit intercept, which host fixed effects now reject.
  The test now omits it, fits the peer with its own intercept and one dropped indicator,
  and restores the host parameters and covariance by an exact linear map. The file passes
  14 of 14 in the external lane. These nodes are not mapped.
- NB2 log likelihood: confirmed and recorded as its own task; no production change.
