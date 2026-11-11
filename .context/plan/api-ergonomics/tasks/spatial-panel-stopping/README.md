---
type: task
title: Spatial panel stopping diagnostics
description: Reject nonstationary balanced pooled SAC candidates before acceptance.
links: []
status: done
---

# Spatial panel stopping diagnostics

## Goal

Investigate numerical termination and fixed-versus-stacked graph fitting for the
pooled SAC likelihood before making a retained qualification claim. Phase 2 API
changes do not change this optimizer or its acceptance criteria.

## Acceptance

- [x] Reproduce seed 0 with the retained fixed and repeated period graphs.
- [x] Explain the large gradient residual despite backend convergence.
- [x] Establish and test any numerical correction independently of API naming.
- [x] Reassess affected evidence before qualification.

## Counterexample

The 2026-11-05 CPU float64 review used the pooled fixture in
`tests/estimation/spatial/test_panel.py`: `T=30`, `N=9`, seed 0,
`W=.7*roll(I,1,axis=1)+.3*roll(I,-2,axis=1)`, an intercept and normal predictor,
and a normal outcome filtered by `(I-.4W)^{-1}`. Both `PanelML(SAC(), effects="pooled")` paths
reported valid and converged with the default 256 steps and tolerance `1e-7`.

| Graph storage | Log likelihood | Retained gradient infinity norm | Iterations |
| --- | --- | --- | --- |
| One fixed graph | -388.29478170299876 | 0.0000011404402284330957 | 49 |
| The same graph repeated over periods | -388.299251629608 | 1.244084054030824 | 68 |

Direct spatial-coordinate gradient norms were approximately `1.58e-6` and
`1.735`, respectively. Profile evaluations at identical supplied parameters
agreed at the test's `rtol=1e-10`, `atol=1e-12`. Backend convergence is not a
stationarity or global-optimality certificate; changing the graph storage can
change the numerical search path. This is a source-level counterexample, not a
completed statistical campaign. The implementation was unchanged by the typing
and API work that exposed it.

## Acceptance correction

Balanced pooled SAC now independently checks the average profile score in physical
spatial coordinates normalized by the admissible limit. Backend convergence alone
cannot admit a candidate: the score must be finite and at most the explicit solver
tolerance. Screening precedes objective selection. With no eligible candidate, the
best finite diagnostic candidate remains unavailable and reports `STALLED`.
No optimizer recovery, iteration-budget, global BFGS or other-effects change was made.

The same CPU float64 seed-0 fixture now accepts the fixed graph with score
`4.335209935894466e-09` (49 iterations) and rejects the stacked graph with score
`0.0047402631471180935` (68 iterations), at tolerance `1e-7`. Both float32 paths
remain unavailable at that unchanged tolerance. Focused tests cover independent
scores, false backend success, candidate selection and additive objective shifts;
the final focused run passed six tests. Strict typing and lint passed.

Known limitations now record the unavailable paths. This correction does not
establish a global optimum or statistical qualification; retained campaigns must
still account for these failures against the final installed wheel.

## Boundary availability witness

The first Phase 2 full check also exposed an invalid universal-validity assertion
in the newly generalized random-effects optimizer reference test. Its CPU
float64 SEM fixture with `T=6`, `N=8`, seed `26310`, tolerance `1e-8` and 256
steps converged after 106 iterations at spatial value `-0.999`. The solver
residual was approximately `1.58e-8` and the variance ratio `0.826185117194905`.
Identification was available, but the spatial-boundary flag correctly made the
fit unavailable: parameters, innovation variance and log likelihood were NaN.

This is a boundary rejection, not evidence of optimizer nonconvergence. Retain
it in failure-inclusive qualification accounting. The original seed-38 fitted
reference retains its assertions and tolerances; pointwise profile properties
remain separate from requiring every random sample to have an interior fit.

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
