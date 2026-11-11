---
type: task
title: Prove the linear MMD reduction as an algorithm refinement
description: "Prove the exact-real identity behind STATADIC's linear-kernel MMD shortcut and register it as an algorithm-refinement claim."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md"]
status: done
---

# Prove the linear MMD reduction as an algorithm refinement

## Goal

STATADIC computes linear-kernel V-MMD through weighted means rather than Gram matrices. Prove the exact-real identity and register it through the existing claim system, keeping it distinct from implementation verification.

## Acceptance

- [x] A new `StatadicFormal/Discrepancy/MMD/LinearReduction.lean`, imported by the library root and annotated `@[blueprint "STATADIC-MMD-002"]`, proves the statement below using only permitted axioms.
- [x] `src/lean/registry/claims.json` adds `STATADIC-MMD-002` with kind `algorithm-refinement`. Its `review` says human review is pending, and its limitations exclude IEEE rounding and XLA.
- [x] `src/lean/tools/correspondence.py` adds `_linear_mmd_cross` to `SUBJECTS`, and the pins are updated after review. `python_formally_verified` stays false.
- [x] A test compares the Gram and mean routes in float32 and float64, with large coordinates, nonuniform masses, zero-mass padding and integer inputs. It is described as a witness, not a proof.
- [x] No method obligation is closed without explicit scope review.
- [x] `just lean::certify`, `just lean::statement-pin` after review, `just lean::registry-test` and `just check` pass.

## Statement

For finite index types, real weights `w` and `v` with `∑ w = ∑ v`, and any reference `r`:

```text
∑∑ w w' ⟪x, x'⟫ + ∑∑ v v' ⟪y, y'⟫ − 2 ∑∑ w v ⟪x, y⟫ = ‖∑ w (x − r) − ∑ v (y − r)‖²
```

## Why equal mass is a hypothesis

- `linear_gram` is uncentred (`geometry/ground/similarities.py:124`).
- The mean route subtracts a reference (`discrepancy/mmd/scalar.py`, `_linear_mmd_cross` in `discrepancy/mmd/weighted.py`).
- The reference cancels only when total masses agree. Also cover zero-mass atoms, which `_linear_mmd_cross` masks to zero.
- This is the V-statistic. The U-statistic path has no such shortcut.

`docs/formal/manuscripts.md` says a refinement claim must identify its algorithm and arithmetic model. Here that is the weighted-mean route over exact reals. `MaskedReduction.lean` shows the local module style, and `src/lean/tools/tests/test_bridge.py:66` shows how refinement claims validate.

## Non-goals

No floating-point bounds, XLA semantics, new registry format, new claim kinds or claims that an implementation is verified.

Follow the [handoff guide](../../handoff.md). Measurements are in [evidence](../../evidence.md).

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
