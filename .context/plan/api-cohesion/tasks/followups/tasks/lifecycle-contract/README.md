---
type: task
title: Enumerate lifecycle verbs in one contract test
description: Check every host result's lifecycle verbs against the declared capabilities in one enumerating test.
links: []
status: done
---

# Enumerate lifecycle verbs in one contract test

## Goal

`statadic.api.capabilities` declares structural Protocols. `tests/api/test_capabilities.py`
types four result classes against them by hand. Read at `00e7d8f`, the host layer
defines 23 `infer`, 16 `predict`, 21 `transform`, 7 `test` and 5 `diagnose`
methods. A result whose signature drifts fails nothing: a Protocol mismatch is
silent unless a fixture names that class, and `final` binds only the checker.

The current shapes are consistent, from a syntax read and not a test:

- `infer` takes `covariance=`, `resampling=` or `calibration=`.
- `transform(data)` is uniform.
- `predict` has seven shapes around `(data, *, kind)`.
- No `predict_*` alias and no `estimate_refined` remain.

This task keeps that state. Row U16 in the
[dispositions](../../../../proposal-dispositions.md) owns it.

Owner: Backlog. No scientific trigger. See [deferred work](../../deferred.md).

## Acceptance

- [x] One test walks every concrete result and inference class in `statadic.api` and matches each public lifecycle verb to a declared capability shape or a recorded exception.
- [x] The same test rejects unlisted public `predict_*`, `estimate_*` and `infer_*` verbs.
- [x] A deliberately divergent fixture class fails the test.

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
