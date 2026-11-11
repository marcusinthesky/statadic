# Gamma, NB2 and probit conditional laws

These independent-row likelihood workflows use the same observations, fitted
design, estimator substitution and explicit inference choices as existing
regression workflows. They add no implicit intercept or row deletion.

| Workflow | Conditional mean | Variance | Additional roles |
| --- | --- | --- | --- |
| `Regression(Observed(...), terms, law=Gamma(...))` | `exp(X beta + offset)` | `phi * mean**2` | Positive continuous outcomes |
| `Regression(Observed(...), terms, law=NegativeBinomial2(...))` | `exp(X beta + offset) * exposure` | `mean + alpha * mean**2` | Nonnegative integer outcomes; positive exposure |
| `Regression(Observed(...), terms, law=Binomial(link="probit"))` | `trials * Phi(X beta + offset)` | `trials * p * (1-p)` | Integer successes and positive integer trials |

## Explicit dispersion and reusable estimators

This complete example selects fixed dispersion and checks fitted availability:

:::{literalinclude} ../../examples/gamma_regression.py
:language: python
:::

The [conditional-law reference](conditional-laws.md#counts-and-positive-outcomes) owns
all specification examples, including fixed and estimated dispersion. Use the
same canonical form when substituting a numerical estimator.

The same bricks fit `LikelihoodProblem` arrays independently of the host workflow.
Estimated dispersion participates in the joint score and Hessian. Fixed dispersion
is retained configuration, not an estimated coefficient. The numerical solver
uses log dispersion; the fit's named parameter vector and covariance describe
the physical dispersion. Inspect parameter names before selecting restrictions.

## Prediction and inference

Gamma and NB2 default to expected outcomes. Probit defaults to probability;
`kind="mean"` returns expected successes using the new rows' trials. `kind="link"`
returns the linear predictor. NB2 also supports rate before exposure. Training
offsets, exposure or trials are never recycled as prediction-row values.

`infer(covariance=ModelBased())` uses joint observed information. `HC("HC0")` and
`HC("HC1")` use full joint score contributions; a robust covariance request does
not establish an exact finite-sample reference distribution. Intervals and tests
retain the shared explicit reference-law interface.

Numerical convergence, identification and existence of a finite global likelihood
optimum are distinct. Inspect availability and diagnostics; separation, collapse
and an unresolved estimated NB2 Poisson boundary do not authorize ordinary
interior inference. An estimated NB2 boundary does not silently select Poisson.
Caller precision remains unchanged. Distribution-object and new saved-predictor
codecs are outside this extension.
