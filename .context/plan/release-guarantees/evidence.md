# Release-guarantees evidence

Measured on 2026-11-06 at `fe98cae` on `feat-docs`. Path prefixes follow the [handoff](handoff.md).

## Multiplicity

**The adaptive route.** `statadic/inference/multiplicity.py:72-79` estimates π̂₀ = min(1, 2·mean(p > 0.5)) and m̂₀ = max(1, ⌈π̂₀·m⌉), then rejects when p ≤ α/m̂₀. The real function was evaluated with `jax.vmap` over every outcome:

| Setting | Rejections when every null is true |
| --- | --- |
| Two independent nulls, p uniform on {1/200, …, 1}, α = 0.05 | 2,900 of 40,000 outcomes (7.25%) |
| Continuous uniform, 200,000 draws, m = 2 | 7.28% |
| m = 5 | 7.96% |
| m = 10 | 6.31% |
| m = 20 | 5.60% |

When every p ≤ 0.5, the threshold is α rather than α/m. The docstring of `tests/inference/test_multiplicity.py:152-153` says the route "never exceeds Bonferroni", which is false.

**The k−1 route.** `multiplicity.py:66-71` divides α by k−1 for all pairwise comparisons among k groups.

- This rule is the subject of Röhmel, "On familywise type I error control for multiplicity in equivalence trials with three or more treatments", *Biometrical Journal*, DOI 10.1002/bimj.201100073.
- Its full text was not read.
- A later stepwise paper cites a "Caffo, Lauzon and Röhmel (2013)" Bonferroni-type procedure.
- **Unconfirmed:** the external claim that α/(k−1) is too liberal for k ≥ 4.

**Untraceable citation.** The module docstring cites "interval-null non-uniformity (Statistical Papers 2024)" without an identifiable reference.

**Consumers:**

- `studies/src/statadic_studies/definitions/inference.py:88-129`;
- `studies/src/statadic_studies/adapters/inference.py:281-359`;
- `tests/qualification/test_returns.py:220-261`;
- `tests/inference/test_equivalence.py:268-324` (`TestMultiplicity`);
- `tests/inference/test_multiplicity.py`;
- docs: `reference/package-development.md:295` and `reference/method-catalogue.md:65`;
- catalogue variants `equivalence.familywise.k-minus-1` and `equivalence.familywise.adaptive-bonferroni`;
- `statadic/inference/mixture/contracts.py:36` and `statadic/inference/equivalence.py:115,129`.

## Non-finite p-values

**The permutation kernel** (`statadic/inference/resampling/permutation/kernel.py:97-111`):

- A NaN observed statistic gives p = 1/(B+1) for all three alternatives: 0.001 at B = 999.
- With 500 of 999 null draws set to NaN and a statistic of 0, `greater` gives 0.233 rather than about 0.5. NaN draws count as not extreme.
- Its callers are `statadic/inference/resampling/permutation/test.py:267,347`.

**Other add-one producers with no finiteness guard:**

- `statadic/inference/mixture/_split_runner.py:348`;
- `statadic/inference/gms.py:331-345` (`_finite_mc_calibration`);
- `statadic/inference/dyadic/bootstrap.py:336-337`;
- `statadic/inference/equivalence.py:53-57`, which the equivalence lane owns.

**Already guarded,** and the pattern to follow: `statadic/api/association.py:217-226` and `statadic/api/comparison/_results.py:131-142`. Each computes `valid` from `isfinite` on the observed value and every draw, and returns `pvalue=None` otherwise.

**The result type.** `statadic/core/results.py:58-72` documents `TestResult.pvalue` as lying in [0, 1] and "not validated". `statadic/core/results.py:127` defines `InferenceStatus`.

## Equivalence

**What STATADIC claims.** `statadic/inference/equivalence.py:29-57` computes p = (1 + #{Ê* ≥ δ}) / (1 + B) from a nonparametric percentile bootstrap of the energy distance. The mixture weights are selected from the data first. Lines 103-105 claim "a genuine Type-I guarantee", and line 132 grounds it on "Liu & Gandy (2026)".

**What the cited theorem covers.** Liu and Gandy, "Kernel Tests of Equivalence", arXiv 2603.10886v2 (15 March 2026), Theorem 8 (E-MMD-Boot):

- the null is MMD(Q, P) ≥ θ;
- the statistic is θ − MMD of a two-sample V-statistic;
- the critical value comes from a multiplier bootstrap with weights (W − 1);
- Lemma 7 assumes a bounded kernel, 0 ≤ k ≤ K;
- the distributions are fixed, with no selection step.

STATADIC differs in the bootstrap, in the kernel (energy distance is unbounded) and in the selection step. The theorem does not cover STATADIC's procedure.

**Sources with no reference record:** Liu & Gandy 2026, Berger & Hsu 1996, Wellek 2010 and Schuirmann 1987. The catalogue's `equivalence.test.*` variants cite only `szekely-rizzo-2013`, and are source-unresolved.

## Bibliography export

**DOIs are dropped.** The reference records hold DOIs, for example `white-1980` (10.2307/1912934), `peyre-cuturi-2019`, `morris-etal-2019`, `phipson-smyth-2010` and `wilson-1927`. `src/latex/refs.bib` has no `doi` field.

- `src/python/scripts/workflow/script.py:344-351` runs `pandoc --from csljson --to bibtex`.
- Pandoc's `bibtex` writer drops DOI; its `biblatex` writer keeps it. Checked on one record with devenv's Pandoc.
- The script comes from the Copier template `exordio` (`.copier-answers.yml`, `v0.1.0-alpha`).

**Smaller metadata notes:**

- `wilson-1927` records the publisher DOI 10.1080/01621459.1927.10502953. An external audit suggested JSTOR's 10.2307/2276774.
- `bhat-etal-2024` writes "Alex Keizer", where the author's own record uses "Alex C. Keizer". This is cosmetic.
- `ledoit-wolf-2008` writes "Oliver", which matches the publisher's record. Keep it.

## Lean

**Historical size and coverage at the 2026-11-06 measurement:**

- StatadicFormal has 12 files and 1,368 lines.
- The then-registered claims cover method ids `energy`, `linear`, `mmd` and `permutation`, each with `obligations: []`.
- `registry/methods.json` lists 46 methods with 53 required obligations.
- Every claim at this checkpoint records "human review pending". Current scope
  and review state come from the live registry and subsequent review evidence.

**Integrity.** A text scan finds no `sorry`, `admit`, `axiom`, `native_decide`, `macro`, `elab`, `syntax`, `set_option` or `True`. `src/lean/tools/Audit.lean` and `tools/integrity.py` check the compiled declarations:

- only the three standard axioms may appear;
- statements, premises and definitions that reduce to `True` are rejected;
- debug options are rejected;
- the build treats warnings as errors.

**Per-claim findings:**

- **STATADIC-LINEAR-005.** The LaTeX is about tᵀβ̂, but the Lean conclusion (`StatadicFormal/Estimation/Linear/OLS.lean:44-69`) is about bᵀY, with b = X(XᵀX)⁻¹t. `ols_contrast_eq` (`OLS.lean:84`) proves the link, but it is not part of the registered statement.
- **STATADIC-RANDOMIZATION-001** uses the full group and an invariant law. Its limitations already exclude sampled transformations and add-one p-values. `finite_randomization_level` (`StatadicFormal/Inference/Randomization.lean:23`), `upper_rank_count` (`:58`) and `count_rank` (`:97`) can be reused.
- **STATADIC-MMD-002** (`StatadicFormal/Discrepancy/MMD/LinearReduction.lean:24`) gives V = ‖Σwᵢ(xᵢ−r) − Σvⱼ(yⱼ−r)‖² under equal mass, but nonnegativity is not registered as a claim.
  - `MaskedReduction.lean` models the leading-block mask over ℚ.
  - `FiniteBias.lean` defines `mmdU` and `mmdV` with pair-moment hypotheses.
- **STATADIC-ENERGY-002** (`StatadicFormal/Discrepancy/Energy/Finite.lean:164`) assumes a PSD generating kernel. There is no Euclidean instance.

**Obligations to target:** `permutation-002`, `linear-002`, the nonnegativity part of `mmd-001`, the HC part of `linear-covariance-001`, and the Euclidean part of `energy-002`.

**Dangling reference.** `statadic/geometry/ground/metrics.py:149-150` cites `EnergyStatistics/NegativeType.lean`, a file in another repository.

**pricing-perspective** (read-only, at `66169ad5f`) has the same Lean 4.35.0 and mathlib `56a00585`. Its first-party Lean is 195 files and 49k lines.

- **Port:** `src/lean/packages/EnergyStatistics/EnergyStatistics/NegativeType.lean` (358 lines, no `sorry`). It proves `distNegativeType_of_innerProductSpace` (`:340`) from `DistNegativeType` (`Defs.lean:141`). The proof goes through a Gaussian integral.
- **Do not port `Defs.lean:31`.** It defines its own `ProbabilityMeasure`, which duplicates mathlib's.
- **Do not port these placeholders:**
  - `PermutationTest.lean:36,53`: `permutation_test_distribution` and `dcor_asymptotic_null`, each `True := by sorry`;
  - `PaperReconstructions/Shanken1992.lean:137`: `(h_full_rank : True) : True := by trivial`.
- **No licence file** at the repository root.

**LeanArchitect** at the pinned revision:

- It has `#show_blueprint` (`Architect/Output.lean:237`).
- It infers `uses` and `proofUses` by default.
- It has no `autoDeps` option.

## Reproduction

Save this outside the repository and run `devenv shell -- bash -c 'cd src/python && uv run --package statadic python -I FILE'`:

```python
import jax
import jax.numpy as jnp
from statadic.inference.multiplicity import equivalence_multiplicity_control as control
from statadic.inference.resampling.permutation import permutation_test_kernel

grid = jnp.arange(1, 201) / 200.0
pairs = jnp.stack(jnp.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
rejected = jax.vmap(lambda p: control(p, 0.05, "adaptive_bonferroni").any())(pairs)
print(int(rejected.sum()), pairs.shape[0])  # 2900 40000
null = jax.random.normal(jax.random.key(1), (999,))
print(float(permutation_test_kernel(jnp.nan, null, 999).pvalue))  # 0.001
```
