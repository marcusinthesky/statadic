---
type: task
title: Prove sampled-permutation add-one validity
description: "Prove that the add-one p-value from the identity plus B independent uniform group draws has P(p <= alpha) <= alpha under an invariant finite law, with inclusive ties and all three alternatives."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md", ".context/reference/hemerik-goeman-2018/README.md", ".context/reference/phipson-smyth-2010/README.md"]
status: in-progress
---

# Prove sampled-permutation add-one validity

## Goal

STATADIC-RANDOMIZATION-001 covers enumeration of the full group. Users run sampled permutations through `permutation_test_kernel`, which computes (1 + b) / (1 + B). Prove that sampled scheme valid.

## Owns

- A new module, `StatadicFormal/Inference/SampledRandomization.lean`.
- Edits to `StatadicFormal/Inference/Randomization.lean`, only to generalize a reusable lemma.
- The claim draft, recorded in `verification.md`.

## Target statement

**Setting:**

- G is a finite group acting on a finite Ω.
- w is a `FiniteLaw`-style weight on Ω with w(g • ω) = w(ω), as in RANDOMIZATION-001.
- T : Ω → ℝ is the statistic.
- The draws g : Fin B → G are independent and uniform, giving the product weight |G|^(−B).

**Definitions:**

- b(ω, g) = #{k : T(ω) ≤ T(g_k • ω)}, which counts ties as extreme.
- p = (1 + b) / (B + 1).

**Claim:** for every α ≥ 0, the total weight of {p ≤ α} under the product law is at most α.

**Corollaries:**

- `less`, by applying the claim to −T;
- `two-sided`, by applying the claim to |T|, which matches `statadic/inference/resampling/permutation/kernel.py:97-102`.

**A suggested route,** not a requirement:

1. Extend the draws to h : Fin (B + 1) → G with h₀ = 1.
2. For each j, the map (ω, h) ↦ (h_j • ω, (h_k h_j⁻¹)_k) preserves the product law: invariance handles ω, and the map is a bijection on the free draws. It permutes the B + 1 statistic values and puts position j at the identity.
3. Averaging over j and applying `upper_rank_count` (`Randomization.lean:58`) on Fin (B + 1) gives the bound.

## Specification

- Locate the exact theorem in Hemerik and Goeman (2018) on random transformations with the identity added, and the add-one discussion in Phipson and Smyth (2010). Record the theorem, page and equation locators, and every way the Lean specialization differs from the source.
- **Limitations to record:**
  - caller-supplied schedules;
  - sampling without replacement;
  - refitting or selection inside permutations, such as `reoptimize_in_permutation`;
  - PRNG quality;
  - non-finite statistics, which the nonfinite-pvalues lane's contract covers instead.
- **Python link:** `statadic.inference.resampling.permutation.permutation_test_kernel`. The relationship text must say that the kernel computes the formula from draws supplied by the caller, and that how those draws are generated is not verified.

## Acceptance

- [ ] A Lean theorem bounds P(p ≤ α) by α for the identity plus B uniform draws, with the greater tail and inclusive ties. The `less` and two-sided corollaries are proved.
- [ ] The module builds under the lock. `#print axioms` shows only the three standard axioms, and nothing states or assumes `True`.
- [ ] `verification.md` holds a complete draft claim object with source locators, assumptions, limitations and the Python link. Its LaTeX statement matches the Lean statement term for term.
- [ ] No definition duplicates one in StatadicFormal or mathlib.

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
