# Sampled permutation verification

## Changes

- `src/lean/packages/StatadicFormal/StatadicFormal/Inference/SampledRandomization.lean`: defines the inclusive add-one p-value and proves its finite-law level bound with independent uniform group draws, then proves the less and two-sided variants. The proof reuses `upper_rank_count` and `finite_randomization_level`. Private recentering and rank helpers make the product-law argument explicit. No change to `Randomization.lean` was needed.
- This record supplies the source decisions and the complete claim draft for coordinator integration. No shared registry, root import, status, generated region or bibliography file was changed by this task.

## Source decisions

- Hemerik and Goeman (2018), [Definition 2 and Theorem 2, p. 819, and its proofs, pp. 819–820](https://link.springer.com/content/pdf/10.1007/s11749-017-0571-1.pdf), treat a group with the identity included and random transformations; the theorem uses an order-statistic rejection rule at levels in [0, 1). Its with-replacement conditioning proof is on p. 820. The discussion immediately after the proof asserts weak p-value validity for the inclusive add-one expression. Section 3.4, equation (6), pp. 822–823, defines the inclusive upper-tail count. The Lean theorem specializes to a finite data space, finite group, data-law invariance, real-valued statistics and independent uniform group draws with replacement. Its nonnegative alpha domain includes the trivial range alpha ≥ 1. It uses an inclusive rank rule and does not formalize equivalence to the paper's order-statistic formulation. The paper also treats other sampling schemes, which are not covered here.
- Phipson and Smyth (2010), [corrected preprint, §4, p. 5, and §§5–6, pp. 5–7](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf), discuss the add-one formula. Section 4's equality is a Monte Carlo calculation under its stated assumptions; §6 distinguishes the with-replacement exact p-value from its add-one upper bound. The present result claims validity of the bound, not equality to the exact randomized p-value or exact size.

## Coordinator requests

Add the module to the library root and review the following complete `claims.json` object. The principal blueprint statement describes the exact Lean conclusion; the less and two-sided theorems are consequences obtained by applying it to `-T` and `|T|`. The `permutation-002` mapping should be reviewed as the stated finite, with-replacement specialization; it does not close unmodeled sampling schemes or implementation refinement.

```json
{
  "id": "STATADIC-RANDOMIZATION-002",
  "title": "Finite sampled-permutation add-one p-value level",
  "source": {
    "key": "hemerik-goeman-2018",
    "locator": "Section 3.3, Definition 2 and Theorem 2, p. 819; Section 3.4, equation (6), pp. 822–823",
    "proof_locator": "Section 3.3, Theorem 2, first proof, pp. 819–820",
    "url": "https://link.springer.com/article/10.1007/s11749-017-0571-1",
    "review": "source comparison by authoring agent; human review pending"
  },
  "scope": {
    "assumptions": [
      "Finite data space with nonnegative weights summing to one",
      "Finite group action preserving each data-law weight",
      "Independent uniformly distributed group draws with replacement, independent of the data",
      "Real-valued statistic, inclusive ties and nonnegative alpha"
    ],
    "conclusion": "The product-law probability that the inclusive upper-tail add-one p-value is at most alpha is at most alpha; applying the result to negative and absolute statistics gives the less and two-sided bounds.",
    "relationship": "Finite invariant-data-law, with-replacement inclusive-rank specialization of the sampled randomization argument. Alpha at least one is also included. Equivalence to the source order-statistic rule is not formalized.",
    "limitations": [
      "Caller-supplied schedules are not assumed to satisfy the independent uniform sampling model and are not verified.",
      "Sampling without replacement is not covered.",
      "Refitting or data-dependent selection inside permutations, including reoptimize_in_permutation, is not verified.",
      "PRNG quality and independence of generated draws are not verified.",
      "Real-valued statistics exclude non-finite values; behavior on NaN or infinity and floating-point arithmetic are not modeled.",
      "The bound is a valid p-value guarantee, not a claim of exact size or equality to the exact randomized p-value."
    ],
    "proof_steps": [
      "Include the identity at index zero and recenter at each index using a swap and group multiplication; the resulting map on data and tuples is an involution preserving the product weights.",
      "Reindex the sampled statistics and apply the inclusive upper-rank counting bound.",
      "Average the pointwise rank bound over indices and invariant finite weights."
    ]
  },
  "lean": {
    "module": "StatadicFormal.Inference.SampledRandomization",
    "declaration": "StatadicFormal.Inference.sampleP_level"
  },
  "python": {
    "symbols": ["statadic.inference.resampling.permutation.permutation_test_kernel"],
    "relationship": "The kernel computes the add-one formula from draws supplied by the caller; how those draws are generated is not verified, nor is floating-point implementation refinement."
  },
  "methods": ["permutation"],
  "obligations": ["permutation-002"],
  "kind": "published-specialization"
}
```

Owner flag: independently review source correspondence and obligation scope before treating the registry mapping as reviewed. Existing statement pins bind the whole Lean source graph and require coordinator re-pinning after review.

## Verification

- Locked module build: `devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake build StatadicFormal.Inference.SampledRandomization'` passed, 1,079 jobs.
- Locked `lake env lean /tmp/StatadicSampleAxioms.lean`: all four public declarations (`sampleP`, `sampleP_level`, `sampleP_level_less`, `sampleP_level_two_sided`) list only `propext`, `Classical.choice`, and `Quot.sound`. These closures transitively cover the private helpers used by the public theorem. Attempts to directly name private declarations from another module failed; independent private-declaration census remains for coordinator certification.
- Scratch checks: one recentering-only version passed, one complete proof version passed; six intermediate scratch checks failed and were corrected. The first sandboxed devenv attempt could not connect to the Nix daemon; subsequent pinned Lean commands required approved escalation. Three unsuccessful diagnostics attempted an unsupported `#print prefix` command and two forms of inaccessible private internal names; the corrected public axiom check passed.
- One scratch text-edit command invoked system `python3` instead of a locked uv project. It edited only `/tmp/StatadicSampleScratch.lean`, not project data; subsequent edits used shell tools.
- `just check`, synchronization, certification, inventory refresh and statement pinning are coordinator-owned and were not run.
