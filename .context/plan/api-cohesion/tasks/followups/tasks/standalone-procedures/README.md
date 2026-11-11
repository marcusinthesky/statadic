---
type: task
title: Account for standalone inferential procedures
description: Record a host decision for every inferential procedure outside the five workflow APIs and state that set in the orientation.
links:
  - .context/plan/api-cohesion/tasks/followups/tasks/standalone-procedures/recommendation.md
status: backlog
---

# Account for standalone inferential procedures

## Goal

Read at `00e7d8f`: `statadic.api` imports none of the `statadic.inference` modules
`equivalence`, `mixture`, `sharpe`, `gms`, `wolak`, `kappa`, `multiplicity`,
`studentized`, `clustering` or `resampling`. Their public functions return nine
separate records: `MixtureTestResult`, `HypothesisTest`, `EnergyPermutationResult`,
`SharpeDifferenceTest`, `MultipleSharpeResult`, `DeflatedSharpeResult`, `GmsResult`,
`WolakResult` and `KappaDeltaResult`. `equivalence_test` takes 17 parameters and
`corrected_mixture_test` 15, mixing quantity, estimator and calibration. The
methods page routes all of them through "Further numerical methods".

The 16:51 transcript (lines 751–778 and 884–905) proposed hosting these tests
under Hypothesis and these quantities under Estimand. No ledger row decided it.
Rows U14 and U15 in the [dispositions](../../../../proposal-dispositions.md) now
own that decision. Each procedure keeps its own law (U04, U05, U09): an adapter
retains the native result as evidence and adds no calibration.

A [recommended decision per procedure](recommendation.md) is awaiting the owner.

Owner: Backlog. Trigger: the owner decides whether alpha ships five APIs plus a
named standalone set, or hosts more. See [deferred work](../../deferred.md).

## Acceptance

- [ ] Record one decision per procedure: Hypothesis adapter, Estimand quantity, or standalone with a reason.
- [ ] Name the standalone set in the orientation and methods pages, so the five API claim matches the package.
- [ ] For each accepted adapter, retain the native result as typed evidence and verify parity with the native function.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
