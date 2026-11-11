# Linear covariance verification

## Changes and sources

- `OLS.lean`: moved the existing coefficient and contrast identities ahead of
  `finite_fullrank_ols_contrast`, and restated that theorem directly in terms of
  `t ⬝ᵥ olsCoefficients`. Its original blueprint statement now matches its type.
  Stępniak (2010), Theorem 3.2 p. 3 and Theorem 4.1 pp. 4–5, remains the source.
- `WLS.lean`: exact square-root whitening, weighted normal equations and unique
  minimization via the existing OLS minimizer; finite-law unbiasedness and BLUE
  via `finite_fullrank_ols_contrast` for whitened data. The competing coefficient
  vector is in whitened units; positive weights make the mapping to original
  observation weights bijective. Hinkelmann, “Aitken Equations,” pp. 103–104,
  equations (3)–(4) and the following paragraph, states the known positive-definite
  covariance, weighted equations, minimization and BLUE. The task-owned
  `hinkelmann-aitken-equations-2006` record records publisher metadata and text.
- `Sandwich.lean`: diagonal pair moments give a finite-law contrast variance;
  an algebraic identity gives its matrix sandwich and nonnegative diagonal meats
  give nonnegative contrast quadratics. HC row influence identities cover HC0,
  scalar HC1, and HC2/HC3 denominator powers. Symmetry and idempotence of the
  hat matrix give leverage bounds and trace and, with spherical second moments,
  the residual moment and HC2 row adjustment. White (1980), Section 2 p. 819,
  gives the fixed-design diagonal covariance sandwich (normalized by n), and
  p. 820 replaces variance terms with squared residuals. MacKinnon and White
  (1985), Section 2, motivates leverage corrections; Long and Ervin (2000),
  Section 3 manuscript pp. 6–7, supplies conventional uncentered HC3. These
  exact finite algebraic identities do not prove asymptotic results in White.
  This attempt to fetch the MacKinnon–White full text timed out; its locator
  relies on the existing reference record and needs independent review.

## Python comparison

In `linear_covariance.py:188–215`, `coefficient_influence_rows` consumes
`sqrt(w) * residual` and the coefficient map of the whitened design; the
independent covariance sums influence outer products (`operators/parameters.py:28–72`).
Thus HC0 has the same row-sandwich algebra; HC1 applies `n / max(df_resid,1)`
to HC0 and availability requires positive residual degrees of freedom, giving
`n/(n-p)` for full-rank fits. HC2 divides each outer product by `1-h`, and HC3
by `(1-h)^2`. Both require finite leverage in `[0,1)` and produce unavailable
results at `h=1`; the underlying operator produces NaNs for invalid leverage.
The leverage is row-sum-of-squares of the whitened QR Q (`estimation/linear.py:179–185`),
matching the diagonal of its projection in exact full-rank algebra. An optional
`finite_sample_correction` is applied separately. No formula mismatch found in
this scope. Historical MacKinnon–White jackknife HC3 includes centering and a
prefactor; the implemented conventional uncentered HC3 is the Long–Ervin form.
Numerical QR, tolerance guards, row masks and floating-point behavior are not
proved by these Lean statements. HC variance algebra and positive semidefiniteness
do not establish HC unbiasedness under heteroskedasticity, consistency, or
interval coverage; only the individual HC2 residual correction is unbiased
under the stated homoskedastic model.

## Coordinator requests

Add root imports `StatadicFormal.Estimation.Linear.WLS` and
`StatadicFormal.Estimation.Linear.Sandwich`; refresh and review statement pins after
integrating these drafts. `linear-002` is addressed for positive diagonal known
weights and full column rank only. `linear-covariance-001` is partially addressed
by exact HC algebra and leverage facts, not by asymptotic consistency or coverage;
do not mark that broader obligation closed. No Python implementation pin change
is requested. Human source-to-claim and implementation correspondence review
remain pending. Claims below are complete objects in the existing schema.

```json
[
  {
    "id": "STATADIC-LINEAR-005",
    "title": "Full-rank OLS contrast and normal equations",
    "source": {"key": "stepniak-gauss-markov-2010", "locator": "Theorem 3.2, p. 3; Theorem 4.1 and (4.4), pp. 4–5", "proof_locator": "Theorem 4.1 proof, pp. 4–5; Theorem 3.2 proof, p. 3", "url": "https://link.springer.com/content/pdf/10.1155/2010/259672.pdf", "review": "independent-agent-source-comparison; human review pending"},
    "scope": {"assumptions": ["Finite probability law", "Fixed design, zero-mean errors and spherical second moments", "Strictly positive variance", "Competitor reproduces the scalar target for every parameter vector", "The Gram matrix XᵀX has an invertible determinant."], "conclusion": "The inverse-Gram OLS coefficient contrast is unbiased and has no greater variance than a reproduced linear competitor.", "relationship": "Full-column-rank, scalar, finite-law specialization. Supporting declarations establish the normal-equation/minimizer equivalence, unique normal-equation solution, coefficient/contrast identity, and uniqueness of contrast weights at equal minimum variance.", "limitations": ["The general rank-deficient and vector-valued source theorem remains outside this specialization.", "Finite probability laws only; no random-design conditioning or continuous-law extension.", "No WLS or JAX/QR implementation refinement."], "proof_steps": ["Construct weights X(XᵀX)⁻¹t in the design column space.", "Use the inverse-Gram identity to reproduce t and solve the normal equations.", "Identify the coefficient contrast with the weighted observation sum.", "Apply the sourced orthogonality and variance decomposition.", "The published residual Pythagorean identity yields the full-rank normal-equation/minimizer equivalence.", "Invertibility and a zero sum of squared weight differences establish the specialized uniqueness conclusions."]},
    "lean": {"module": "StatadicFormal.Estimation.Linear.OLS", "declaration": "StatadicFormal.Estimation.Linear.finite_fullrank_ols_contrast"}, "python": {"symbols": ["statadic.estimation.linear.LeastSquares"], "relationship": "conceptual correspondence; implementation refinement unverified"}, "methods": ["linear"], "obligations": [], "kind": "published-specialization"
  },
  {
    "id": "STATADIC-LINEAR-006", "title": "Finite full-rank diagonal WLS contrast",
    "source": {"key": "hinkelmann-aitken-equations-2006", "locator": "Aitken Equations, pp. 103–104, equations (3)–(4) and following paragraph", "proof_locator": "Aitken Equations, p. 104, equation (4) and subsequent minimum-variance statement; proof cited there to Hinkelmann and Kempthorne (1994)", "url": "https://catalogimages.wiley.com/images/db/pdf/0471743917.01.pdf", "review": "source comparison completed; human review pending"},
    "scope": {"assumptions": ["Finite probability law and fixed design", "Strictly positive deterministic diagonal weights and positive error scale", "Invertible weighted Gram determinant", "Zero error means and diagonal pair moments σ²/wᵢ", "Competitor reproduces the contrast on the whitened design"], "conclusion": "The inverse-weighted-Gram coefficient contrast is unbiased and no less efficient than a reproduced whitened linear competitor.", "relationship": "Finite-law scalar contrast specialization of the known positive-definite covariance result; diagonal covariance and full rank. Supporting results prove whitening, normal equations and a unique weighted residual minimizer.", "limitations": ["Finite probability laws and known positive diagonal weights only.", "The competitor is represented in whitened coordinates; positive weights make this representation bijective.", "Pair-moment hypotheses carry the full covariance model; no independence is inferred.", "No numerical QR or JAX refinement, adaptive weights, survey weights, coverage, or asymptotic result."], "proof_steps": ["Show that square-root diagonal whitening has the weighted Gram and right-hand side.", "Derive spherical whitened moments from the supplied diagonal moments.", "Apply STATADIC-LINEAR-005 to the whitened model.", "Use the existing OLS residual minimizer and uniqueness for the weighted criterion."]},
    "lean": {"module": "StatadicFormal.Estimation.Linear.WLS", "declaration": "StatadicFormal.Estimation.Linear.finite_wls_contrast"}, "python": {"symbols": ["statadic.estimation.linear.LeastSquares"], "relationship": "conceptual correspondence to positive precision weights; numerical refinement unverified"}, "methods": ["linear"], "obligations": [], "kind": "published-specialization"
  },
  {
    "id": "STATADIC-LINEAR-007", "title": "Finite diagonal-error OLS sandwich variance",
    "source": {"key": "white-1980", "locator": "Section 2, p. 819, fixed-regressor diagonal covariance sandwich", "proof_locator": "Finite-law algebra corresponding to the fixed-regressor sandwich immediately after Lemma 2, p. 819", "url": "https://crooker.faculty.unlv.edu/econ441/econ_papers/White-Heteroskedasticity-Correction-1980.pdf", "review": "source comparison completed; human review pending"},
    "scope": {"assumptions": ["Finite probability law", "Fixed real design and deterministic signal", "Zero error means and supplied diagonal pair moments"], "conclusion": "The variance of an inverse-Gram coefficient contrast equals the diagonal-error matrix sandwich.", "relationship": "Exact finite-law algebra corresponding to White's fixed-regressor covariance expression; the source uses normalized matrices in an asymptotic analysis.", "limitations": ["The pair-moment hypothesis explicitly supplies the diagonal second moments.", "Lean's nonsingular inverse is total; invertible Gram is needed for the conventional full-rank OLS interpretation.", "No HC estimator unbiasedness, consistency, coverage, random-design conditioning, or floating-point implementation theorem."], "proof_steps": ["Expand the finite-law variance of the contrast as a double sum.", "Apply the diagonal pair moments.", "Rewrite the weighted sum as the matrix sandwich."]},
    "lean": {"module": "StatadicFormal.Estimation.Linear.Sandwich", "declaration": "StatadicFormal.Estimation.Linear.finite_ols_sandwich"}, "python": {"symbols": ["statadic.inference.linear_covariance.hc_covariance"], "relationship": "the exact variance is the population analogue of the row-influence sandwich; estimator and numerical refinement unverified"}, "methods": ["linear-covariance"], "obligations": [], "kind": "published-proof-step"
  },
  {
    "id": "STATADIC-LINEAR-008", "title": "Finite homoskedastic OLS residual moment",
    "source": {"key": "mackinnon-white-1985", "locator": "Section 2, leverage corrections and residual bias", "proof_locator": "Section 2, residual leverage adjustment", "url": "http://qed.econ.queensu.ca/working_papers/papers/qed_wp_537.pdf", "review": "source record comparison; human review pending"},
    "scope": {"assumptions": ["Finite probability law", "Fixed full-column-rank design", "Zero error means and spherical second moments"], "conclusion": "The squared OLS residual has expectation σ²(1−hᵢ).", "relationship": "Finite-law scalar exact algebra underlying the homoskedastic residual leverage correction; supporting declarations prove leverage bounds, trace and the HC2 adjusted row mean when hᵢ<1.", "limitations": ["Spherical pair moments are assumed, not inferred from independence.", "HC2 row unbiasedness under homoskedasticity does not establish unbiasedness under heteroskedasticity.", "HC3 convention, asymptotic consistency, coverage and numerical implementation refinement are separate."], "proof_steps": ["Show the hat matrix and its complement are symmetric idempotent.", "Express the residual as the complementary projection of the error.", "Use the spherical moment identity and the complementary projection diagonal."]},
    "lean": {"module": "StatadicFormal.Estimation.Linear.Sandwich", "declaration": "StatadicFormal.Estimation.Linear.ols_residual_second_moment"}, "python": {"symbols": ["statadic.inference.linear_covariance.hc_covariance"], "relationship": "the HC2 row correction corresponds in exact arithmetic; numerical refinement unverified"}, "methods": ["linear-covariance"], "obligations": [], "kind": "published-proof-step"
  }
]
```

## Checks

- `devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake build StatadicFormal.Estimation.Linear.OLS'`: passed (1,814 jobs).
- Corresponding locked WLS and Sandwich builds passed together (2,256 jobs),
  followed by a Sandwich build after adding HC row identities (1,815 jobs).
- Scratch `#print axioms` under the same lock reported only `propext`,
  `Classical.choice`, and `Quot.sound` for all 21 changed or new theorems,
  including `hc_row_influence`.
- The fenced claim JSON parsed successfully with `jq`, yielding four objects.
- A text scan of the two new modules found no prohibited placeholders or bypasses.
- The sandboxed devenv invocation initially failed to reach the local Nix daemon;
  approved escalated invocations succeeded. Intermediate builds reported Lean
  proof/linter diagnostics and were corrected.
- Full `just check`, certification, synchronization, statement pinning and
  inventory refresh are coordinator-only per the task handoff and were not run.

## Independent review correction

In `OLS.lean`, the STATADIC-LINEAR-005 Blueprint statement now defines only the conclusion's `β̂`; the proof sketch introduces its auxiliary `b = X(XᵀX)⁻¹t` before using it. The theorem body is unchanged. `devenv shell -- bash -c 'cd src/lean && flock /tmp/statadic-lean.lock lake build StatadicFormal.Discrepancy.MMD.Nonnegativity StatadicFormal.Estimation.Linear.OLS'` passed (2,235 jobs). Independent re-review requested.
