# Models, quantities, hypotheses and retained results

STATADIC separates the scientific question, its numerical solution and the evidence
needed to interpret it.

| Object | Responsibility | What it does not establish |
| --- | --- | --- |
| **Ordination** | Construct coordinates from declared feature, kernel or distance geometry; retain the training query map | A population geometry law, manifold recovery or a sampling reference |
| **Regression** | State the response, predictor terms and supported law/structure; estimate with a compatible numerical estimator | Exogeneity, causal identification or valid uncertainty |
| **Estimand** | Specify covariance or supplied moment conditions; estimate with an explicit numerical method | A response model, predictions or a generic inferential reference |
| **Hypothesis** | State a null proposition; an inference or diagnostic procedure produces a `TestOutcome` | A testing procedure or proof that the null holds |
| **Report** | Retain and render computed quantities, assumptions, availability and provenance | A fitted model, new inference or statistical qualification |

An **estimator** computes fitted state. The result keeps the actual numerical
fit and training preparation. Prediction reuses that preparation; inference
adds covariance and reference assumptions. These are separate choices even
when a short constructor such as `Regression.ols(...)` selects familiar defaults.

Ordination can itself be a regression term: learn coordinates from training
predictors, then reuse their map for prediction. Refit the complete composition
inside each training fold. Changing geometry or selecting it using outcomes
requires its own evaluation and inferential argument.

Use the [composition reference](../reference/composition.md) for concrete family
contracts. The five API entry points are [Regression](../reference/regression.md),
[Ordination](../reference/ordination.md), [Estimand](../reference/estimand.md),
[Hypothesis](../reference/hypotheses.md) and [Report](../reference/reporting.md).

:::{toctree}
:maxdepth: 1
:caption: Interpret the ingredients

observations-weights
geometry-contracts
result-validity
Resampling and failure policies <../guides/resampling>
Choosing a survival model <../guides/survival>
:::

[Evidence](../validation/index.md) distinguishes numerical reference agreement,
scoped mathematical results and complete-procedure statistical studies.
