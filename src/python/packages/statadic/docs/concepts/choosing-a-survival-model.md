---
type: explanation
description: How ordinary, repeated-spell and spatial survival models answer different questions about durations.
---

# Choosing a survival model

For a patient still under observation when follow-up ends, the recorded
duration gives a lower bound on the time to an event rather than its exact time.
A survival model accounts for that distinction by using both the duration and
an indicator of whether the event was observed.

## Ordinary durations

STATADIC supports Weibull, lognormal and log-logistic accelerated failure time
models for positive elapsed durations. A row describes one spell, with
`event=1` for an event and `event=0` for right censoring.

Delayed entry conditions on survival until an entry time satisfying
`0 <= entry < duration`. This input describes elapsed durations rather than a
start–stop counting-process model. Choose the family using the scientific
question and the shape of the duration distribution.

## Repeated spells

A panel layout identifies entities and periods, with at most one spell in each
entity-period. Unit or time effects describe differences in the conditional
location of log durations.

The choice of fixed effects describes variation in the conditional location,
while the assumptions for uncertainty must also address dependence between
spells from the same entity. Their coding matters for prediction too, because
new fixed-effect levels are rejected rather than assigned an arbitrary effect.

## Spatial log times

Spatial survival uses lognormal errors and a graph-dependent model of log
durations. Covariates cover active graph nodes, while observed outcomes can
cover only some of them. Missing outcomes are integrated out of the likelihood.

The spatial likelihood needs explicit integration designs, which can make the
calculation expensive when outcomes are censored or missing. Agreement across
independent designs provides evidence about numerical stability rather than a
bound on the error in the integral. The current host rejects positive delayed entry for spatial survival, so a
delayed-entry analysis needs a supported ordinary-duration model instead. Check
the input contract.

## Choose a prediction target

The default prediction is a marginal median, while a mean, survival probability,
quantile or restricted mean answers a different question and must be requested
explicitly. Choose the target first. Some distributions have infinite means even when their medians
and survival probabilities are well defined.

Read [survival reference](../reference/survival.md) for prediction conventions
and [known limitations](../validation/known-limitations.md) for configurations
with unresolved numerical or coverage concerns.
