<!-- rumdl-disable MD041 -->

(statadic-by-the-question-you-are-asking)=
(are-these-two-things-associated)=
(are-these-two-samples-from-the-same-distribution)=
(are-these-two-things-close-enough)=
(what-is-the-structure-in-this-distance-matrix)=
(how-do-i-fit-a-linear-model)=
(how-do-i-fit-panel-or-spatial-models)=
(how-much-sampling-error-is-in-this-estimate)=
(does-this-moment-model-satisfy-its-restrictions)=
(is-this-track-record-real)=
(axioms-which-of-these-is-actually-a-metric)=
(how-do-i-get-my-data-into-the-right-shape)=
(substrate)=
(import-conventions)=
(statadic--by-the-question-you-are-asking)=
(axioms--which-of-these-is-actually-a-metric)=

# Methods

<!-- rumdl-enable MD041 -->

| Research question | Start with |
| --- | --- |
| Response and predictors | [Regression](tutorials/first-regression.md) |
| Counts or binary responses | [Regression with a response law](reference/conditional-laws.md) |
| Coordinates from features, kernels or distances | [Ordination](reference/ordination.md) |
| Covariance or supplied moment conditions | [Estimand](reference/estimand.md) |
| Instruments or moment restrictions | [IV](iv-fitting.md) / [GMM](gmm-fitting.md) |
| Checking OLS assumptions | [Residual diagnostics](linear-diagnostics.md) |
| Repeated or spatial observations | [Panel/spatial models](spatial-models.md) |
| Censored durations | [Survival](survival-models.md) |
| Distribution differences | [Energy and scale](tutorials/energy-and-scale.md) |
| Distribution-valued rows | [Distributional workflow](tutorials/distribution-workflow.md) |
| Further numerical methods | [Detailed catalogue](reference/method-catalogue.md) |

## Standalone procedures

Some inferential procedures are numerical functions outside the five APIs. They
return their own records, not `TestOutcome`: `equivalence_test` and
`corrected_mixture_test` (mixture energy), `energy_permutation_test`, the Sharpe
tests and intervals, `wolak_test`, `indicator_gms`, `kappa_delta_method` and
`bootstrap_confidence_interval`. Find each in the
[detailed catalogue](reference/method-catalogue.md); they keep their own laws.

`energy_distance()` and `Energy()` differ in ground and rooting; see
[energy and scale](tutorials/energy-and-scale.md). Check [capabilities](capabilities.md)
and [limitations](validation/known-limitations.md).
