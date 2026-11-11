---
type: task
title: Put the survival family on the model, then add prediction-target tables
description: "Make the implied statistical model inspectable: the AFT family moves from SurvivalML to the model, then results describe their prediction targets and required inputs."
links: []
status: backlog
---

# Put the survival family on the model, then add prediction-target tables

## Goal

A researcher should see what a model assumes and what a fit can predict
before calling it.

## Survival family

`SurvivalML(family=..., max_steps, tolerance)` mixes a modelling choice with
solver settings (`estimation/survival/fit.py`). There are about 20
`SurvivalML(` sites in the package, plus examples, scripts and the studies and
LaTeX apps. The precedent is `GammaRegression` and
`NegativeBinomialRegression`, which hold the family on the model and check
the injected estimator with `_check_fit` (`api/regression/_likelihood.py`).
Spatial survival is lognormal only and must keep saying so.

Open decision: a family-agnostic estimator (family on the problem), or the
model holding the family and checking the estimator agrees (the GLM pattern).

## Prediction-target tables

About 12 `predict` methods validate their `kind` inline. Panel conditional
prediction needs effect keys seen in training; Fama–MacBeth rejects
prediction; survival targets need `value`, accept a time grid and allow entry
conditioning only for survival and cumulative hazard; spatial survival returns
period- and ID-keyed output. Nothing describes this today; the closest is the
artifact manifest's `prediction_kinds`.

Open decisions: per instance (depends on the fitted estimator) rather than a
class constant; required inputs (value, grid, entry, graph IDs) rather than
kind names only; whether the table drives validation or only describes it.

## Acceptance

- [ ] The AFT family is declared on `SurvivalRegression` and reported by it.
- [ ] Each result exposes its targets and their required inputs, and its
  `predict` errors cite the table.

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
