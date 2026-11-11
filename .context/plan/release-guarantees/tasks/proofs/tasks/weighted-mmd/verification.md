# Weighted MMD verification

## Changes and evidence

- `src/lean/packages/StatadicFormal/StatadicFormal/Discrepancy/MMD/Nonnegativity.lean` adds six declarations. `linear_mmd_nonneg` applies the existing STATADIC-MMD-002 norm-square identity. `gram_kernel_psd` proves Gram PSD from a squared norm. `pooled_v_eq_quadratic` expands four blocks and requires symmetry; `pooled_v_nonneg` applies the existing `KernelPSD`. Neither equal mass nor nonnegative weights is needed for the pooled result. `mmdU_negative_witness` computes −2 for two identical (1,−1) samples under the real linear kernel. `weighted_v_zero_padding` splits both padded index sums, including arbitrary, potentially asymmetric kernels; taking either padding type empty covers padding one side. This matches the finite-sum exclusion idea in `MaskedReduction.maskedReduction_eq_leading`, which separately models a leading rational block, rather than claiming a formal theorem connecting the two declarations.
- Gretton et al., Lemma 6, equation (3), p. 728 defines the U expression; its proof p. 729 explicitly notes that U can be negative. Equation (5), p. 729 gives the empirical V expression rooted after Gram expansion. The arbitrary real weighted and padding statements are exact finite algebraic extensions, not verbatim claims of that paper. Source checked against <https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf>, pp. 728–729.
- Python `weighted.py:68-125` provides the V path and `contracts.py:44-65` refuses rooting U. The shortcut and general route mask nonpositive masses (`weighted.py:58-62,106-108`), so correspondence to the unrestricted signed-weight statements requires nonnegative masses. The Python function uses floating arithmetic and validity checks; no implementation refinement follows from the exact-real proof. Characteristic-kernel separation remains open.

## Coordinator requests

Add `import StatadicFormal.Discrepancy.MMD.Nonnegativity` to the library root. Add the following complete claim objects to `registry/claims.json` after independent source and statement review. Do not mark `mmd-001` complete: it also requires characteristic-kernel separation. Review whether partial-obligation tracking is possible before mapping any of these claims; draft `obligations` are empty. No implementation pin is requested without implementation review. The declarations' Blueprint statements carry the LaTeX, matching the exact conclusions.

```json
[
  {
    "id": "STATADIC-MMD-003",
    "title": "Weighted linear V-MMD nonnegativity",
    "kind": "algorithm-refinement",
    "review": "Proof worker comparison; independent and human semantic review pending",
    "scope": {
      "assumptions": ["Finite index types and real weights with equal total mass", "Vectors in a real inner product space"],
      "conclusion": "The linear-kernel weighted V expression is nonnegative.",
      "relationship": "Immediate exact-real corollary of STATADIC-MMD-002's squared-norm identity.",
      "limitations": ["The Python shortcut masks nonpositive masses; correspondence requires nonnegative masses.", "No floating-point execution, kernel separation, sampling or input normalization is verified."],
      "proof_steps": ["Apply linear_mmd_reduction.", "A real square is nonnegative."]
    },
    "lean": {"module": "StatadicFormal.Discrepancy.MMD.Nonnegativity", "declaration": "StatadicFormal.Discrepancy.MMD.linear_mmd_nonneg"},
    "python": {"symbols": ["statadic.discrepancy.mmd.weighted.weighted_mmd_squared_cross"], "relationship": "Exact-real property of the linear weighted-mean route under equal mass; Python refinement is unverified."},
    "methods": ["mmd"], "obligations": []
  },
  {
    "id": "STATADIC-MMD-004",
    "title": "Finite pooled PSD weighted V-MMD nonnegativity",
    "kind": "published-proof-step",
    "source": {"key": "gretton-etal-2012", "locator": "Equation (5), p. 729", "proof_locator": "Lemma 6 proof, empirical feature-space mean expansion, p. 729", "url": "https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf", "review": "Proof worker comparison; independent and human semantic review pending"},
    "scope": {
      "assumptions": ["Finite disjoint union of sample indices", "A symmetric real kernel that satisfies KernelPSD on the pooled support", "Arbitrary real weights"],
      "conclusion": "The weighted V expression is nonnegative.",
      "relationship": "Finite exact-real weighted extension of the empirical Gram expansion behind equation (5); pooled_v_eq_quadratic supplies the signed quadratic equality and gram_kernel_psd supplies a Gram instance.",
      "limitations": ["The paper states empirical equal-weight feature means; arbitrary real weights are an algebraic extension.", "Symmetry is explicit because KernelPSD is defined only by quadratic nonnegativity.", "No characteristic-kernel separation, probability normalization, RKHS construction, or floating-point implementation refinement is proved.", "Python masks nonpositive masses; correspondence requires nonnegative masses."],
      "proof_steps": ["Split the pooled signed quadratic form into four blocks.", "Use symmetry and exchange finite sums for the reverse cross block.", "Apply KernelPSD; Gram PSD follows from a squared norm."]
    },
    "lean": {"module": "StatadicFormal.Discrepancy.MMD.Nonnegativity", "declaration": "StatadicFormal.Discrepancy.MMD.pooled_v_nonneg"},
    "python": {"symbols": ["statadic.discrepancy.mmd.weighted.weighted_mmd_squared_cross"], "relationship": "Exact-real V-expression sign under the stated finite PSD and symmetry assumptions; floating-point behavior and kernel-law implementation are unverified."},
    "methods": ["mmd"], "obligations": []
  },
  {
    "id": "STATADIC-MMD-005",
    "title": "A negative linear-kernel MMD U statistic",
    "kind": "published-proof-step",
    "source": {"key": "gretton-etal-2012", "locator": "Lemma 6, equation (3), p. 728", "proof_locator": "Lemma 6 proof, observation that the U estimate may be negative, p. 729", "url": "https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf", "review": "Proof worker comparison; independent and human semantic review pending"},
    "scope": {
      "assumptions": ["Two identical samples (1, −1), real linear kernel, and the unique Unit outcome"],
      "conclusion": "The defined unequal-sample MMD U statistic equals −2.",
      "relationship": "Concrete finite instance illustrating the source's observation that its U estimator can be negative; the specific example is derived here.",
      "limitations": ["This instance does not establish sampling properties, U estimator nonnegativity, or floating-point refinement.", "Python's rooted route rejects U estimates; the U route has no nonnegativity guarantee."],
      "proof_steps": ["Expand the two within-sample off-diagonal sums and the cross sum.", "Evaluate the finite arithmetic."]
    },
    "lean": {"module": "StatadicFormal.Discrepancy.MMD.Nonnegativity", "declaration": "StatadicFormal.Discrepancy.MMD.mmdU_negative_witness"},
    "python": {"symbols": ["statadic.discrepancy.mmd.mmd_squared", "statadic.discrepancy.mmd.mmd_squared_matrix"], "relationship": "Illustrates why the U route is signed and contracts._require_v_distance refuses rooting; no implementation refinement is proved."},
    "methods": ["mmd"], "obligations": []
  },
  {
    "id": "STATADIC-MMD-006",
    "title": "Zero-weight padding preserves weighted V-MMD",
    "kind": "algorithm-refinement",
    "review": "Proof worker comparison; independent and human semantic review pending",
    "scope": {
      "assumptions": ["Finite original and padding index types", "An arbitrary real kernel on the padded pooled support", "Original real weights and exactly zero weights on both padding blocks"],
      "conclusion": "The padded weighted V expression equals the corresponding expression restricted to original indices.",
      "relationship": "Finite exact-real zero-padding identity; parallels maskedReduction_eq_leading's exclusion of padded entries.",
      "limitations": ["The proof does not formalize a connection to maskedReduction_eq_leading, whose separate model uses rational leading blocks.", "Python masks nonpositive masses and may encounter undefined kernel values; floating-point execution and implementation refinement are not verified.", "No PSD or separation claim is made."],
      "proof_steps": ["Split each finite sum over original and padding components.", "Terms involving zero weights vanish."]
    },
    "lean": {"module": "StatadicFormal.Discrepancy.MMD.Nonnegativity", "declaration": "StatadicFormal.Discrepancy.MMD.weighted_v_zero_padding"},
    "python": {"symbols": ["statadic.discrepancy.mmd.weighted.weighted_mmd_squared_cross"], "relationship": "Exact-real analogue of excluding zero-mass padding, conditional on finite kernel values; implementation refinement is unverified."},
    "methods": ["mmd"], "obligations": []
  }
]
```

## Checks and owner flags

- Initial pinned build blocked before Lean: sandbox could not connect to Nix daemon (`Operation not permitted`).
- First escalated locked build failed: Gram symmetry rewrite and an unused simp argument (2 errors).
- Second escalated locked build failed: an unbounded commutativity rewrite and incorrect sum-type parentheses (12 type errors plus a consequent unsolved goal).
- Third locked build succeeded: 2,032 jobs. After Blueprint annotations, fourth locked build succeeded: 2,032 jobs; fifth build after statement text correction also succeeded: 2,032 jobs. Command each time: `devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake build StatadicFormal.Discrepancy.MMD.Nonnegativity'`.
- `devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake env lean /tmp/statadic_mmd_axioms.lean'`: succeeded for all 6 new declarations, each using exactly `propext`, `Classical.choice`, `Quot.sound`.
- Full certification, statement pinning, sync, and `just check` were reserved for the coordinator and not run. Baseline was failed/interrupted; GLM timeout remains a separate follow-up. Human source and semantic review remain pending.

## Independent review correction

The STATADIC-MMD-003 Blueprint statement now explicitly quantifies the theorem's reference point `r` in the real inner product space, as requested by independent review. Its conclusion and proof are unchanged. Rebuilt with `devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake build StatadicFormal.Discrepancy.MMD.Nonnegativity'`: passed, 2,032 jobs. Independent re-review requested.

Re-review identified the separate handoff rule that the LaTeX must not mention a quantity absent from the conclusion. Removed the redundant reference parameter from `linear_mmd_nonneg` and its statement; its proof now invokes the existing reduction with reference zero. The draft assumptions above reflect the revised signature. A combined locked build of this module and OLS passed (2,235 jobs); the six-declaration axiom inspection was repeated and each still reports exactly `propext`, `Classical.choice`, and `Quot.sound`.
