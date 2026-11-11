---
type: task
title: Prove WLS whitening and the HC covariance algebra
description: "Close the STATADIC-LINEAR-005 statement gap, prove WLS as whitened OLS with Aitken optimality from the existing claims, and prove the sandwich and leverage facts that HC0-HC3 rely on."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md", ".context/reference/stepniak-gauss-markov-2010/README.md", ".context/reference/white-1980/README.md", ".context/reference/mackinnon-white-1985/README.md", ".context/reference/long-ervin-2000/README.md"]
status: in-progress
---

# Prove WLS whitening and the HC covariance algebra

## Goal

When a researcher switches from OLS to WLS or HC3, the evidence should show which assumptions changed.

- WLS is proved to be OLS on whitened data, with Gauss–Markov optimality carried across.
- HC0–HC3 are proved as exact algebra, together with the leverage facts their corrections need.

## Owns

- `StatadicFormal/Estimation/Linear/*`, including new modules such as `WLS.lean` and `Sandwich.lean`.
- The claim drafts, recorded in `verification.md`.

## Specification

**1. Close the LINEAR-005 gap.**

- Its LaTeX speaks of tᵀβ̂, but its Lean conclusion is about bᵀY (`OLS.lean:44-69`).
- Restate the registered conclusion in terms of `t ⬝ᵥ olsCoefficients X Y`, using `ols_contrast_eq` (`OLS.lean:84`), so the LaTeX and Lean agree.
- Request the changed claim draft for the coordinator.

**2. WLS.** With weights wᵢ > 0 and W = diag(w):

- β̂_W = (XᵀWX)⁻¹XᵀWy equals OLS on W^½X and W^½y;
- β̂_W solves the weighted normal equations;
- β̂_W is the unique minimizer of Σ wᵢ(yᵢ − xᵢᵀβ)² when XᵀWX is invertible.

Under E ε = 0 and E εεᵀ = σ²W⁻¹, derive unbiasedness and the BLUE ordering for contrasts from STATADIC-LINEAR-004 and LINEAR-005 by whitening. Do not prove Gauss–Markov again.

Locate the source that states WLS optimality for a known diagonal covariance, such as Aitken (1935) or a textbook already recorded like `searle-1971`. Record its locator, and create a reference record if needed.

**3. Sandwich and leverage,** on a finite law with E ε = 0 and E εεᵀ = diag(σᵢ²):

- Var(tᵀβ̂) = tᵀ(XᵀX)⁻¹Xᵀ diag(σ²) X (XᵀX)⁻¹t.
- Leverage hᵢ = xᵢᵀ(XᵀX)⁻¹xᵢ lies in [0, 1], and Σ hᵢ equals the number of columns.
- Under homoskedasticity, E êᵢ² = σ²(1 − hᵢ). That is why HC2 divides by (1 − hᵢ). HC3 divides by (1 − hᵢ)² and needs hᵢ < 1. HC1 multiplies HC0 by n/(n − p).
- Sources: White (1980), MacKinnon and White (1985) and Long and Ervin (2000). All have reference records.

**4. Compare with the implementation.**

- Check each formula against `statadic/inference/linear_covariance.py:188-215`: the scaling, the leverage, and the guard when hᵢ = 1.
- Record mismatches as findings for the coordinator. Do not change Python in this lane.

**5. Out of scope:** interval coverage, consistency, and HAC or clustered covariance (`linear-covariance-002`).

## Acceptance

- [ ] STATADIC-LINEAR-005's Lean conclusion states tᵀβ̂ through `olsCoefficients`, and its LaTeX matches.
- [ ] WLS is proved: the whitening identity, the weighted normal equations and the unique minimizer, plus Aitken unbiasedness and the BLUE ordering, derived from the existing claims.
- [ ] The sandwich variance identity, the leverage bounds and trace, and E êᵢ² = σ²(1 − hᵢ) under homoskedasticity are proved.
- [ ] `verification.md` compares each formula with `linear_covariance.py` and lists any mismatches.
- [ ] Builds under the lock. Only the three standard axioms appear, nothing states or assumes `True`, and the claim drafts match LaTeX to Lean.

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
