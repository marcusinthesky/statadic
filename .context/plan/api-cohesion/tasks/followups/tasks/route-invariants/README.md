---
type: task
title: Test invariants across Regression routes
description: Test constructor compatibility, prepared-state ownership, retained schema, result type and no silent model substitution across every supported Regression route.
links:
  - .context/plan/api-cohesion/tasks/final-review/README.md
status: backlog
---

# Test invariants across Regression routes

## Goal

The final alpha review of `4cc3077` judged the `Regression` dispatcher's
complexity to be maintenance debt, not a public-abstraction defect. Its
recommended protection is a small set of tested invariants across every supported
route, not a registry or plugin framework:

- constructor compatibility;
- prepared-state ownership;
- the retained schema;
- the result type;
- no silent model substitution.

This is not an alpha prerequisite.

Owner: Backlog. See [deferred work](../../deferred.md).

## Acceptance

- [ ] Enumerate the supported constructor, structure, dependence and estimator routes from `api/regression/_model.py`.
- [ ] One parametrised test per invariant covers every route, with a rejected control for each incompatible combination.

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
