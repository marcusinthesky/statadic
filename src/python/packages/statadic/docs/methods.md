---
type: explanation
description: Choose a method using the question you want to answer.
---

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

## Choose by research question

| Research question | Start with |
| --- | --- |
| Response and predictors | [Regression](start/first-regression.py) |
| Counts or binary responses | [Regression with a response law](reference/conditional-laws.md) |
| Coordinates from features, kernels or distances | [Ordination](reference/ordination.md) |
| Covariance or supplied moment conditions | [Estimand](reference/estimand.md) |
| Instruments or moment restrictions | [IV](reference/moment-estimators.md) / [GMM](reference/moment-estimators.md) |
| Checking OLS assumptions | [Residual diagnostics](reference/linear-diagnostics.md) |
| Repeated or spatial observations | [Panel/spatial models](reference/spatial-panels.md) |
| Censored durations | [Survival](reference/survival.md) |
| Distribution differences | [Energy and scale](tutorials/energy-and-scale.py) |
| Distribution-valued rows | [Distributional workflow](tutorials/distributions.py) |

For example, a researcher comparing censored durations can start with survival,
then follow its reference to the assumptions and evidence for the particular
family and prediction target.
