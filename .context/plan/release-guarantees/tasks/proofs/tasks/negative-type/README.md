---
type: task
title: Port Euclidean negative type and energy nonnegativity
description: "Port pricing-perspective's sorry-free Euclidean negative-type proof without its duplicate definitions or placeholders, derive finite Euclidean energy nonnegativity, and replace the dangling docstring reference with the claim."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md", ".context/reference/sejdinovic-equivalence-2013/README.md", ".context/reference/szekely-rizzo-2013/README.md"]
status: in-progress
---

# Port Euclidean negative type and energy nonnegativity

## Goal

STATADIC-ENERGY-002 proves energy nonnegativity only for a PSD generating kernel, so there is no Euclidean instance. `statadic/geometry/ground/metrics.py:149-150` cites a proof in another repository. Bring that proof into STATADIC, use it, and cite it properly.

## Owns

- `StatadicFormal/Discrepancy/Energy/*`, including a new `NegativeType.lean`.
- The docstring sentence at `statadic/geometry/ground/metrics.py:149-150`.
- The reference records this lane creates.
- The claim draft, recorded in `verification.md`.

## Specification

**1. Port only what is needed.**

- From `~/Git/pricing-perspective/src/lean/packages/EnergyStatistics/EnergyStatistics/`, take `DistNegativeType` (`Defs.lean:141-143`) and what `distNegativeType_of_innerProductSpace` (`NegativeType.lean:340`) needs.
- Do not port `EnergyStatistics.ProbabilityMeasure` (`Defs.lean:31`), `PermutationTest.lean`, `PaperReconstructions` or anything that depends on them.
- Move the code into the `StatadicFormal.Discrepancy.Energy` namespace.
- Record the source repository, commit `66169ad5f`, files and line ranges. pricing-perspective has no licence file, so flag that the owner must confirm the reuse.

**2. Check the definition.** Confirm `DistNegativeType` against the source definition: Σᵢⱼ wᵢwⱼ d(xᵢ, xⱼ) ≤ 0 for weights summing to zero (Lyons 2013, Definition 3.1, or Schoenberg). Create and check reference records for the sources cited.

**3. Energy nonnegativity.**

- Factor the identity rawEnergy d p q = −quadratic d (p − q), now inside the proof of `finite_energy_nonneg` (`Energy/Finite.lean:118-132`), into a lemma that needs only a symmetric d and equal mass.
- Then prove that for points in a real inner-product space and probability weights on a finite index, `rawEnergy` with d = ‖xᵢ − xⱼ‖ is ≥ 0.

**4. Exponent.** STATADIC's energy distance accepts exponents in (0, 2). This proof covers exponent 1 only. Record that as a limitation, and do not attempt fractional powers.

**5. The docstring.** Replace the `metrics.py` sentence with the new claim id. Keep the caveat about strong negative type.

**6. Python link.** `statadic.geometry.ground.metrics.EuclideanLaw` and `statadic.discrepancy.mmd.energy.scalar.energy_distance` with the Euclidean metric. Check that these are the right public symbols.

## Acceptance

- [ ] The ported module builds under the lock, with only the three standard axioms, nothing that states or assumes `True`, and no duplicate of a StatadicFormal or mathlib concept.
- [ ] Finite Euclidean energy nonnegativity is proved and drafted as a claim, with the exponent limitation recorded.
- [ ] `metrics.py` cites the STATADIC claim, not a file in another repository.
- [ ] Provenance and the licence flag are in `verification.md`.

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
