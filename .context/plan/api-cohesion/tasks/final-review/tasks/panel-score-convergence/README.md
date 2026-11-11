---
type: task
title: Require a score check before random-effects panels report convergence
description: Stop complete-panel spatial ML from reporting convergence while its gradient is far from zero, and assert the random-effects SEM optimum against spreg.
links:
  - .context/plan/api-cohesion/tasks/final-review/tasks/spatial-likelihood-references/README.md
status: done
---

# Require a score check before random-effects panels report convergence

## Goal

`spatial-likelihood-references` compared a random-effects SEM panel with spreg
`Panel_RE_Error` and found a false convergence (row F15). The case is
`_panel("error", 2, 8, 20)` in
`tests/integration/api/test_spatial_likelihood_references.py`.

| Quantity | statadic `PanelML` | Maximum of statadic's own profile (Nelder–Mead), matching spreg to about 1e-7 |
| --- | --- | --- |
| λ | 0.30654 | 0.3089901 |
| Variance ratio | 1.9977 | 2.027192 |
| Log likelihood | −153.53067 | −153.5294356 |
| Status | `converged=True`, `residual_norm=0.337` | — |

The likelihood function itself agrees with spreg at both points. The defect is the
stopping rule.

`_multistart_panel_search` in `src/statadic/estimation/spatial/panel.py` gates
convergence on a score (gradient) residual only when `score_sample_size` is
given. The complete-panel random-effects path does not give it, so `converged` is
whatever Optimistix BFGS reports from its step and objective-change test. The
returned `residual_norm`, which is the gradient infinity norm, is not consulted.

## Acceptance

- [x] A complete-panel spatial ML fit reports convergence only when its returned score residual meets a stated tolerance. This covers random effects, which include the variance-ratio coordinate. A false stop is recovered by a continued or restarted search, or is reported as not converged.
- [x] On the recorded case, statadic's estimate, log likelihood, AIC, BIC and HQIC match spreg `Panel_RE_Error` through the offsets the test already states, and the test asserts the optimum.
- [x] Existing spatial panel tests and references are unchanged. That covers the complete, observed and graph routes, the SAR random-effects comparison, and the studies spatial panel variants. Any value that moves is explained in the hand-back.
- [x] The named tests and the external lane pass; lint and strict typing pass for the touched files.

## Specification

- Files: `src/statadic/estimation/spatial/panel.py`,
  `tests/estimation/spatial/test_panel.py`, and the optimum assertion in
  `tests/integration/api/test_spatial_likelihood_references.py`.
- Reuse the existing mechanism, not a new optimiser:
  - **Score gate:** the `score_sample_size` branch, which takes the residual of
    the per-observation average objective and keeps eligible starts only.
  - **Stalled-search recovery:** `_recover_stalled_panel_search`.
- Extend both to the random-effects coordinate:
  - Evaluate the score in the unconstrained coordinates the search optimises,
    with the ratio on the log scale.
  - Scale the score by the sample size, exactly as the observed route does.
- Thread `score_sample_size` from the complete-panel entry point to
  `_panel_search`.
- Keep JAX-traceable behaviour, and keep the existing tolerances unless the
  hand-back justifies a change.
- Do not change the likelihood, its constants or `k`.

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
