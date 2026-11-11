# Verification: Euclidean negative type

## Changes and source review

- `Energy/Finite.lean`: extracted `rawEnergy_eq_neg_quadratic`; symmetry alone suffices, so the factored identity is stronger than the requested equal-mass version. Existing theorem signatures remain unchanged.
- `Energy/NegativeType.lean`: ports the one-dimensional indicator integral, Gaussian projection and finite-span reduction, with a finite-probability energy result. No duplicate probability measure or squared-distance auxiliary results were ported. The definition allows the empty family as well as the nonempty families in Lyons.
- `geometry/ground/metrics.py`: replace the external Lean path with the claim reference; retain the strong-negative-type caveat.
- `lyons-distance-covariance-2013` and `schoenberg-positive-definite-1938`: authored bibliography records, metadata checked against published papers. Source: Lyons, equation (3.1), p. 3291 (zero sum and inequality), pp. 3292–3293 (Euclidean negative type), equation (3.3), p. 3293 (measure inequality). It is equation (3.1), not “Definition 3.1.” The Lean Gaussian projection is a different proof from Lyons's Riesz embedding. Schoenberg 1938 is cited by Lyons p. 3292 for the embedding characterization, not as the exact proof ported here.
- Port provenance: `/home/marcussky/Git/pricing-perspective`, commit `66169ad5ff07f7779fe2543e24f215454bb24704`; `src/lean/packages/EnergyStatistics/EnergyStatistics/Defs.lean:141-143` and `NegativeType.lean:129-356`. The latter's comments and proof were adapted; unused `NegativeType.lean:60-127` and `Defs.lean:31` were not copied.
- `EuclideanLaw` is exported in `statadic.geometry.ground.metrics.__all__`; `statadic.discrepancy.mmd.energy.scalar.energy_distance` is publicly defined and exported by its family. Its `metric="euclidean"`, exponent 1 route computes unrooted V-energy on weighted `PackedClouds`; the Lean theorem uses two weights on a common finite support, achievable by union and zero padding, not formally linked to the implementation.

## Owner flag

The source repository has no root or Lean-package licence file; the only licence found is under the unrelated `src/python/apps/manimize` subtree. The owner must confirm permission to reuse and redistribute this port before publication. Publication remains on hold. The claim and source correspondence still require human review.

## Coordinator requests

Add the root import `StatadicFormal.Discrepancy.Energy.NegativeType`; integrate and review this complete claim draft, and review the partial `energy-002` obligation mapping. Refresh implementation and statement pins, inventory, bibliography and generated views through the established coordinator workflow. Do not claim the full exponent range or floating-point implementation correctness.

```json
{
  "id": "STATADIC-ENERGY-003",
  "title": "Finite Euclidean energy nonnegativity at exponent one",
  "source": {
    "key": "lyons-distance-covariance-2013",
    "locator": "Section 3, equation (3.1), p. 3291; Euclidean negative type, pp. 3292–3293; equation (3.3), p. 3293",
    "proof_locator": "Section 3, pp. 3292–3293; Euclidean Riesz embedding (the Lean proof uses Gaussian projection instead)",
    "url": "https://rdlyons.pages.iu.edu/pdf/dcov-published.pdf",
    "review": "source comparison by implementing agent; human review pending"
  },
  "scope": {
    "assumptions": [
      "A finite family of points in a real inner product space",
      "Two finite probability laws over the same finite index family"
    ],
    "conclusion": "The unrooted V-energy using ordinary norm distances is nonnegative.",
    "relationship": "Finite-support specialization of the negative-type inequality for Euclidean points; a finite-span reduction extends it to real inner product spaces. The proof uses Gaussian projection rather than the source's embedding proof.",
    "limitations": [
      "Only distance exponent 1 is proved; other exponents in (0, 2) are not covered.",
      "Strong negative type, equality implying equality of laws, general measures and separation are not proved.",
      "The signed U-statistic and floating-point execution are not certified nonnegative.",
      "The Python connection, including union and zero padding for different supports, is conceptual rather than a formal implementation refinement."
    ],
    "proof_steps": [
      "Represent one-dimensional absolute differences by an integral of interval-indicator squares.",
      "Integrate one-dimensional inequalities over a standard Gaussian projection and reduce arbitrary finite point families to their finite-dimensional span.",
      "Use equal total probability mass and symmetry to identify raw energy with the negative zero-sum distance quadratic."
    ]
  },
  "lean": {
    "module": "StatadicFormal.Discrepancy.Energy.NegativeType",
    "declaration": "StatadicFormal.Discrepancy.Energy.finite_euclidean_energy_nonneg"
  },
  "python": {
    "symbols": ["statadic.geometry.ground.metrics.EuclideanLaw", "statadic.discrepancy.mmd.energy.scalar.energy_distance"],
    "relationship": "Exact-real conceptual correspondence for metric=euclidean and exponent=1; implementation refinement unverified."
  },
  "methods": ["energy"],
  "obligations": [],
  "kind": "published-specialization"
}
```

## Commands and results

- Initial `devenv shell -- ... lake build` failed because the sandbox could not connect to the Nix daemon; retried with escalated daemon access.
- `devenv shell -- just reference-new` for each of two new reference records: 2 successful.
- Focused `devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake build StatadicFormal.Discrepancy.Energy.NegativeType'`: 3 attempts with daemon access. First built dependencies and `Finite.lean` but failed on import visibility, four lint diagnostics and one deprecated lemma; second failed on the replacement lemma namespace; third passed, 3236 jobs. Each exited and released the lock.
- Scratch `devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake env lean /tmp/statadic-negative-type-axioms.lean'`: passed for six public declarations, each depending only on `propext`, `Classical.choice` and `Quot.sound`. This includes the transitive private proof helpers through the main theorem.
- Forbidden-token scan of both Energy modules: zero matches for admissions, custom axioms, `True`, custom syntax, debug options and `native_decide`. `git diff --check` on the two tracked edits passed. Bibliographic metadata and source locators were manually checked against the linked published papers; generated reference validation awaits coordinator integration.
- Full `just check`, `just sync`, certification, statement pins and inventory refresh are coordinator-only and were not run here.
