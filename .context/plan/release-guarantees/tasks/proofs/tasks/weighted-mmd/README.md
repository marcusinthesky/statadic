---
type: task
title: Prove weighted V-MMD nonnegativity and the U contrast
description: "Prove the weighted V-MMD nonnegative for linear and PSD kernels on a finite pooled support, show the U estimator can be negative, and show zero-weight padding leaves V unchanged."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md", ".context/reference/gretton-etal-2012/README.md"]
status: in-progress
---

# Prove weighted V-MMD nonnegativity and the U contrast

## Goal

Give users the first visible difference that a scientific choice makes. The weighted V-statistic is provably nonnegative, so its square root is defined. The U-statistic can be negative, so rooting it is invalid.

## Owns

- `StatadicFormal/Discrepancy/MMD/*`, including a new module such as `Nonnegativity.lean`.
- The claim drafts, recorded in `verification.md`.

Reuse `quadratic` and `KernelPSD` from `StatadicFormal/Discrepancy/Energy/Finite.lean`. Do not redefine them. The negative-type lane owns that file.

## Specification

**1. Linear kernel.** Prove the weighted linear V-statistic is nonnegative under equal mass, as a corollary of STATADIC-MMD-002's norm-square identity (`LinearReduction.lean:24`).

**2. PSD kernels.**

- On a finite pooled index ι ⊕ κ with signed weights c = (w, −v), prove that the weighted V-statistic equals `quadratic k c`, and is therefore nonnegative whenever `KernelPSD k` holds.
- Prove that a Gram kernel k(x, y) = ⟪φ x, φ y⟫ satisfies `KernelPSD`.

**3. The U contrast.**

- Prove an explicit witness: a concrete sample where `mmdU` (`FiniteBias.lean:104`) is negative. For example, take Ω = `Unit`, the linear kernel on ℝ and x = y = (1, −1), which gives −2.
- State it as an existential, or as a closed instance proved by `norm_num`.

**4. Masking.**

- Prove that adding zero-weight points to either sample leaves the weighted V-statistic unchanged.
- Relate this to `maskedReduction_eq_leading` (`MaskedReduction.lean:23`), and record the existing limitation that the Python shortcut masks nonpositive masses.

**5. Python links.**

- `statadic.discrepancy.mmd.weighted.weighted_mmd_squared_cross` for V.
- Record that the U route (`statadic/discrepancy/mmd/contracts.py:59`, which refuses to root it) has no nonnegativity claim.
- Separation by characteristic kernels stays open.

## Acceptance

- [ ] Weighted V-MMD ≥ 0 is proved for the linear kernel and for any PSD kernel on a finite pooled support, using the existing `KernelPSD`.
- [ ] A Lean witness shows that `mmdU` can be negative.
- [ ] Zero-weight padding is proved to leave the weighted V-statistic unchanged.
- [ ] Builds under the lock. Only the three standard axioms appear, and nothing states or assumes `True`.
- [ ] `verification.md` holds the draft claims with Gretton et al. locators, limitations and Python links, with LaTeX matching each Lean statement.

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
