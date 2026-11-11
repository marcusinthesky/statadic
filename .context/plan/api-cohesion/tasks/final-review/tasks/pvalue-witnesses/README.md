---
type: task
title: Witness permutation and GMS p-values independently
description: Add independent numpy witnesses for the permutation-kernel p-value definition and the indicator GMS statistics.
links:
  - .context/plan/api-cohesion/tasks/followups/tasks/reference-gaps/README.md
status: done
---

# Witness permutation and GMS p-values independently

## Goal

The generic permutation kernel and the indicator GMS diagnostic report p-values,
but no test compares them with an independent calculation (row F07). Add tests
only; no source changes.

## Acceptance

- [x] An independent numpy implementation of the documented permutation p-value definition matches the kernel on the kernel's own null draws, for every supported alternative.
- [x] Exact enumeration of all relabelings of a 4+4 two-sample case matches an `itertools` witness.
- [x] `scipy.stats.permutation_test` is compared only where its definition coincides; otherwise the docstring states why not.
- [x] A numpy witness reproduces the GMS MMM and QLR statistics and the indicator selection on a fixed panel. It also reproduces the critical value or p-value, if the Monte Carlo draws can be supplied or reproduced.
- [x] New tests carry one scope marker plus `reference`, with oracle marking copied from `tests/inference/test_mantel.py`; `just python::statadic::test-layout` accepts their placement.
- [x] The named test files pass; lint passes for them.

## Specification

- Owners: the existing test file for
  `src/statadic/inference/resampling/permutation/kernel.py` (find it with `git grep -l
  permutation_test_kernel tests/`), and `tests/inference/test_gms.py`. Add a new
  mirrored file only if no owner exists and `test-layout` accepts it.
- Read the docstrings of `permutation_test_kernel` and `indicator_gms` first, and
  implement exactly what they state. For the kernel, that covers the count rule,
  the +1 correction and the alternatives. The two-sided rule counts
  `abs(null) >= abs(observed)`, so a peer that doubles the smaller tail is a
  different definition.
- GMS: implement MMM and QLR from their definitions in numpy, with QLR via
  `scipy.optimize` on a nonnegative QP or a closed form in two moments. Use x64.
  Compare statistics to `rtol=1e-10`. If the calibration draws are internal and
  cannot be reproduced, witness only the statistic and selection, and say so in
  the docstring.
- Source files may not be edited. Report any defect found instead of fixing it.

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
