---
type: explanation
description: Why the resampling unit, preprocessing and failure policy determine the meaning of an interval or p-value.
---

# Resampling and failed draws

A bootstrap interval depends on what was sampled and which parts of the
analysis were recomputed for each draw of that sample. Sampling rows, clusters
or graph nodes describes different experiments, even when the implementation
eventually represents each of them using an array of indices.

## Choose the experiment first

A schedule identifies the observations entering each draw, after which an
evaluator computes the requested statistic or refits the model on those inputs.
The summary has a further responsibility: to apply the stated rules for
uncertainty and for draws whose calculations failed.

A shared schedule makes two calculations comparable draw by draw, because
each calculation receives the same sampled observations. That alignment
alone cannot establish exchangeability or justify a model of dependence, so
state the sampling unit before choosing an executor.

## State what is repeated

For a linear pairs bootstrap, the prepared design, response and relative
precisions are gathered together. Selection, imputation and tuning remain
fixed in the analysis unless the procedure explicitly repeats them.

This distinction changes the uncertainty being measured by the resulting
interval. An interval conditional on a prepared design addresses a narrower
question than an interval that repeats an entire model-selection procedure.

## Preserve the failure policy

The scheduled percentile summary requires at least two draws and makes the
summary unavailable if any draw fails, so an incomplete calculation remains
visible in the result. The add-one p-value calculation also
requires finite observed and null statistics. Some Mantel adapters instead
summarize successful draws, so their results are conditional on those draws.

Read the contract of the entry point you use, and report the requested, valid
and failed counts alongside its interval or p-value. A difficult draw is part
of the experiment. Removing it can change the distribution that the resampling
procedure was intended to approximate.

## Interpret the result at its stated scope

The randomized add-one rule is `(1+b)/(1+B)`, and passing an exhaustively enumerated
null to the calculation leaves that rule unchanged, so the reported p-value
continues to use the add-one convention. State the rule for the report. Percentile intervals for several coefficients
are marginal intervals rather than simultaneous coverage statements.

The [resampling reference](../reference/resampling.md) gives the exact policies.
Consult [known limitations](../validation/known-limitations.md) for the studied
sample sizes and dependence patterns.
