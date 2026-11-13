---
type: explanation
description: Why a successful fit, an available interval and a justified analysis answer different questions.
---

# Understanding result validity

A model can produce finite coefficients even when the relationship between
the observations and its assumptions makes those coefficients difficult to
interpret. STATADIC keeps numerical availability separate from the scientific
judgment needed to connect an estimate to the question behind an analysis.

## Read the status for the output you need

| Question | What to inspect |
| --- | --- |
| Were the inputs supported? | Domain, shape and identity checks |
| Did the solver meet its criterion? | Convergence status and numerical residuals |
| Were the parameters identified? | Rank and identification diagnostics |
| Is uncertainty available? | Covariance and reference-distribution status |
| What supports the procedure? | Matched comparisons and statistical studies |

These checks concern different parts of an analysis, so success at one stage
leaves the assumptions for later stages to be assessed separately. A solver's
convergence criterion describes its numerical calculation for the fitted model, while exogeneity
concerns the relationship between predictors and errors in the population.

## A fit can succeed before inference fails

A saturated linear model can identify its coefficients while leaving zero
residual degrees of freedom for a common variance estimate. Likewise, a finite
tail probability can fall outside the range checked for its reference
calculation. Inspect the inference result separately from the fitted result, because the
requested covariance and reference calculation each have their own conditions.

Reports keep unavailable values visible alongside the estimates that were
computed for the analysis. When a value is unavailable, reconsider the design or requested
inference and keep failed resamples in the accounting for the method.

## Connect a result to its assumptions

Exchangeability, independent censoring and a valid dependence model are
assumptions about the observations and their collection. Numerical checks help you examine an analysis, but those assumptions require
substantive reasoning about how the observations were collected and how the
model relates to the question.

Before interpreting an output, read the method's reference page alongside the
assumptions that apply to your observations. The [evidence guide](what-evidence-means.md)
explains how tests, comparisons, mathematical statements and studies contribute
different kinds of support.
