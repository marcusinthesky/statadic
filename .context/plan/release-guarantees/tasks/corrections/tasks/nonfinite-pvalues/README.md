---
type: task
title: Make non-finite statistics give unavailable p-values
description: "A NaN or infinite observed statistic or null draw makes the add-one p-value NaN, under eager, jit and vmap, in every producer that lacks a guard."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md"]
status: done
---

# Make non-finite statistics give unavailable p-values

## Goal

An undefined statistic must never become evidence. Today a NaN observed statistic gives p = 0.001 with 999 draws, and NaN null draws make p-values smaller. The [evidence](../../../../evidence.md#non-finite-p-values) has the measurements.

## Owns

- `statadic/inference/resampling/permutation/kernel.py`, and `permutation/test.py` only if its callers need it.
- `statadic/inference/mixture/_split_runner.py`, around line 348.
- `statadic/inference/gms.py`: `_finite_mc_calibration` and how its result reports unavailability.
- `statadic/inference/dyadic/bootstrap.py`, around lines 320-345.
- `statadic/inference/calibration.py`: `add_one_p_kernel` and its mirrored tests,
  added by the coordinator after the audit found that it drops non-finite draws.
- `statadic/inference/sharpe/multiple.py` extreme-count guards and mirrored tests,
  added after the same audit identified unguarded joint and step-down tails.
- `statadic/inference/sharpe/difference.py` paired extreme-count guards and mirrored
  tests; cross-review reproduced `p=0` from a NaN return hidden by a ratio fallback.
- `statadic/inference/dyadic/report.py` and mirrored tests; cross-review found the
  public report discarded failed draws before calling the guarded kernel.
- `statadic/inference/resampling/results.py` and mirrored tests: preserve unavailable
  NaN p-values through the existing float-valued legacy host adapter, with no
  new fields or result types; continue rejecting infinities and finite values
  outside `[0, 1]`.
- The `TestResult` docstring in `statadic/core/results.py`. Change no fields.
- The mirrored tests for each of these.

Not owned: `statadic/inference/equivalence.py`, which the equivalence lane fixes, and `statadic/api/*`, which is already guarded.

## Specification

**1. The contract.** The p-value is unavailable when the observed statistic or any null draw is NaN or infinite. Then:

- **Inside JAX:** use `jnp.where(valid, pvalue, jnp.nan)`. Keep it traceable under `jit` and `vmap`, with no host concretization.
- **Functions returning a host float:** return `math.nan`.
- **Results with a validity or status convention:** if a producer's result already has one (check GMS and dyadic first), use that instead and record which.

Add no result fields. Follow `statadic/api/association.py:217-226`, which treats any non-finite draw as invalid.

**2. Documentation.**

- In the kernel docstring, state the finite-input condition.
- In the `TestResult` docstring, say that `pvalue` is in [0, 1], or NaN when unavailable.

**3. Tests, for each producer:**

- a NaN statistic, an infinite statistic and one NaN draw;
- all three alternatives, for the kernel;
- under eager, `jax.jit` and `jax.vmap`, a batch with one invalid lane and one valid lane. The valid lane must be bit-for-bit unchanged, and the invalid lane must be NaN.

**4. Audit.**

- Search `statadic/inference` and `statadic/api` for every other add-one or extreme-count p-value, for example with `rg -n '\(1(\.0)? \+|extreme' src/python/packages/statadic/src/statadic`.
- List each site in `verification.md` as fixed here, guarded already, or owned by another lane.
- Check downstream consumers of these p-values. A NaN compared with α is not significant, which is the conservative result. Record any consumer that ranks p-values, sums them or subtracts them from one.

## Acceptance

- [x] In the kernel, a non-finite statistic or any non-finite draw gives a NaN p-value for every alternative, under eager, `jit` and `vmap`. Valid lanes are unchanged.
- [x] The split runner, GMS and dyadic bootstrap follow the same rule, or a recorded existing status convention.
- [x] Shared calibration applies the same all-draw finite rule and does not drop
  invalid draws; focused transformation tests cover this audited producer.
- [x] Multiple-Sharpe joint/step-down tails are unavailable when any required
  statistic or replicate is non-finite, and the legacy permutation adapter
  preserves an unavailable NaN instead of raising solely because of that NaN.
- [x] `verification.md` holds the add-one producer audit and the downstream-consumer check.
- [x] Focused tests, lint, typecheck and architecture pass, or each failure is shown to be only the expected inventory staleness.

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
