---
type: task
title: Separate D and squared-D warrants, then publish the extension guide
description: "Let a geometry warrant negative type of D without a Hilbertian (squared-D) warrant, so DISCO accepts L1-like geometries; then write the extension guide around _L1OnMeans."
links: []
status: backlog
---

# Separate D and squared-D warrants, then publish the extension guide

## Goal

DISCO needs negative type of D. PERMANOVA and PCoA need negative type of D²
(a Hilbertian embedding). `PairwiseGeometry.negative_type` and
`squared_negative_type` (`api/representation.py`) both return
`_witness is not None`, because `HilbertianMeasureMatrix` is the only witness.
This is sound (Hilbertian implies negative type of D) but over-restrictive: an
L1-like geometry, negative type for D but not D², must return no warrant, so
DISCO rejects it. On the corners of a square, Manhattan distance supports the
negative-type decomposition while classical scaling of its square has an
eigenvalue of −1.

The core layer already separates the laws: `NegativeTypeLaw`
(`core/axioms.py`), `SquaredDistanceNegativeTypeLaw` and
`HilbertianPseudometricLaw` (`core/matrices.py`), and DISCO's kernel requires
`DispersionDecomposableLaw` (`association/decomposition.py`).

## Open decision

Two witness classes (a D-only witness beside `HilbertianMeasureMatrix`), or one
witness carrying a law descriptor. Either way `GeometrySpec.matrix` widens,
`pairwise()` checks the union, and DISCO's `matrix` attribute needs a DMat
typed with a negative-type law.

## Extension guide

After the split, promote `_L1OnMeans` (`tests/api/test_representation.py`) to
a guide beside `docs/compose/change-component.md`. Its executable examples
already exist: plug-in, PCoA rejection, raw DMat warrant rejection, the
`cross()` shape check, delegated warrants and `replace` re-authorisation.
Record two gaps: custom geometries cannot be persisted (`encode_geometry`
accepts only built-ins) and `_prepare` is private.

## Acceptance

- [ ] A geometry can warrant negative type of D only; DISCO accepts it;
  PERMANOVA and PCoA reject it.
- [ ] The Manhattan square test: DISCO decomposes, PCoA reports −1.
- [ ] The extension guide runs as a doc test and states the two gaps.

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
