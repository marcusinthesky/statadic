# What does “valid” actually mean?

STATADIC returns evidence at more than one layer. Read the flag or status associated with the output you plan to use, and do not promote it into a stronger scientific statement.

| Question | Relevant evidence | Not established by that evidence |
|---|---|---|
| Were inputs and shapes supported? | Input/domain/schema checks | The observations are correctly sampled or scientifically comparable |
| Did the numerical solver meet its criterion? | Solver status, residuals, feasibility or optimality diagnostics | Identification, a unique global solution, or accurate inference |
| Are these parameters identified in the retained design? | Rank and identification diagnostics | Exogeneity or causal identification |
| Is a requested numerical result available? | Fit, covariance, interval or test validity/status | Statistical coverage or Type-I error control |
| Does it agree with an independent implementation? | Matched reference test with aligned conventions | Every configuration or the full family is qualified |
| Has the complete inferential procedure met a study criterion? | Prespecified, failure-inclusive qualification result | Universal or simultaneous validity beyond the studied cells |

## Fitting and uncertainty can diverge

A saturated linear fit can have identifiable coefficients but no residual degrees of freedom for a common variance estimate. An interval can be unavailable even when the point fit succeeded. A tiny exact-fit residual can make a covariance construction numerically inappropriate. A finite computed tail can lie outside the validated reference range.

## Failure is an output, not an inconvenience

Unavailable results should remain visible in examples and reports. A tutorial should show what to inspect and which scientific choice needs reconsideration. It should not suggest pseudoinverse rescue for unidentified parameters, silently delete failed resamples, or replace an unqualified model with a green badge.

## Write down the assumptions separately

Numerical evidence cannot establish exchangeability, exogeneity, independent censoring, a valid dependence model or the quality of a research design. State these assumptions in the method page and in the analysis. Link to the relevant [validation evidence](../validation/index.md) rather than claiming that a shared API supplies them.
