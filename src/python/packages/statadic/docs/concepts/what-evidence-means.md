---
type: explanation
description: What implementation tests, reference comparisons, mathematical proofs and statistical studies can tell you.
---

# What evidence means

A regression can agree with another package on one dataset and still produce
poor interval coverage when used with a different sampling design or reference
distribution. Each kind of evidence answers a particular question about the
method, its implementation or the complete procedure used in an analysis.

## Tests and independent comparisons

An implementation test checks behavior for specified inputs. Agreement between
eager and compiled execution checks consistency across execution modes.
Comparing with another implementation adds an independent numerical control
when the data, options and conventions are aligned.

That agreement concerns the outputs actually compared. It cannot establish
accuracy for every combination of estimators, input distributions and
reference choices exposed by the same interface.

## Mathematical statements

A proof establishes the conclusion in its statement under the hypotheses that
the statement requires. Read those hypotheses first. The objects being
compared, their notion of equality and the sampling assumptions determine
whether that result supports the conclusion you want to draw.

A theorem about exact arithmetic and a numerical implementation have separate
obligations. The [formal pages](../formal/index.md) link individual statements
to their assumptions and the calculations they help explain.

## Statistical studies

A repeated-experiment study examines properties such as interval coverage or
the rate of false rejections. Its result belongs to the complete procedure
that was run, including its sample size, reference choices and failure policy.

Monte Carlo uncertainty matters because a finite number of repeated trials
only estimates the frequency of an outcome, so a small difference between
procedures can reflect sampling variation. Report that uncertainty. An inconclusive study remains
inconclusive, and a procedure that was never run provides a different kind of
record from a completed study. Read the [study catalogue](../validation/research.md) and
[known limitations](../validation/known-limitations.md) together.

## Apply evidence to your analysis

Start with the assumptions for your method, then find evidence for a matching
configuration and output. Preserve the numerical warnings and choices in your
report so that another reader can assess the same analysis.
