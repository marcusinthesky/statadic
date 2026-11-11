---
type: task
title: Replace the familywise corrections with valid ones
description: "Remove adaptive_bonferroni, keep k_minus_1 only with a proving source, and offer Bonferroni and Holm with their exact validity condition."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md"]
status: done
---

# Replace the familywise corrections with valid ones

## Goal

`equivalence_multiplicity_control` should offer only corrections whose familywise error control is proved for valid input p-values, and its docstring should state that condition.

The [evidence](../../../../evidence.md#multiplicity) shows that `adaptive_bonferroni` reaches 7.25% at a nominal 5%, and that the `k_minus_1` rule is disputed in the literature.

## Owns

- `statadic/inference/multiplicity.py`.
- `tests/inference/test_multiplicity.py`.
- The `TestMultiplicity` class in `tests/inference/test_equivalence.py:268-324`. Move anything worth keeping into `test_multiplicity.py`, then delete the class. The equivalence lane owns the rest of that file, so re-read it before each edit.
- Its now-unused multiplicity import in `test_equivalence.py`, authorized by the
  coordinator as part of that test-class migration before the equivalence lane starts.
- The multiplicity parametrization in `tests/qualification/test_returns.py:220-261`.
- The multiplicity entries in `studies/src/statadic_studies/definitions/inference.py:88-129`, `studies/src/statadic_studies/adapters/inference.py:281-359` and their studies tests.
- The reference records this lane creates.

## Specification

**1. Read the sources first, and record each theorem locator.**

- Bonferroni's inequality, and Holm (1979), "A simple sequentially rejective multiple test procedure", in the Scandinavian Journal of Statistics.
- For `k_minus_1`, find Lauzon and Caffo's α/(k−1) proposal for pairwise TOST and Röhmel's Biometrical Journal paper (DOI 10.1002/bimj.201100073). Search for any later correction, such as Caffo, Lauzon and Röhmel (2013).
- Verify every bibliographic detail against the publisher. Do not rely on the titles given here.

**2. The API.**

- `method` accepts `"bonferroni"` and `"holm"`. The default is `"holm"`, because it is never less powerful under the same assumptions.
- Remove `"adaptive_bonferroni"`.
- Keep `"k_minus_1"` only if a recorded published theorem proves its control for all-pairwise equivalence families at every `n_groups` the code accepts. Restrict `n_groups` to that range. Otherwise remove the method and the `n_groups` parameter.
- Keep the refusal of `"bh"`.
- Holm must stay traceable under `jit` and `vmap`: sort, compare with α/(m − i), take a cumulative product, and scatter back.

**3. The docstring.**

- State the guarantee exactly: familywise error at most α under any dependence, provided each true-null p-value satisfies P(p ≤ u) ≤ u.
- Say that `equivalence_test`'s bootstrap p-value is not shown to satisfy this.
- Remove "Statistical Papers 2024" unless you identify the paper and record it.
- A NaN p-value is never rejected. Document that.

**4. Tests.**

- Enumerate the 200-point grid exhaustively for two true nulls. Each remaining method must reject at most 2,000 of the 40,000 outcomes.
- Test that `"adaptive_bonferroni"` raises `ValueError`.
- Compare Holm against a plain-Python step-down oracle using Hypothesis.
- Check that eager, `jax.jit` and `jax.vmap` agree.
- Remove the false "never exceeds Bonferroni" docstring test at `test_multiplicity.py:152`.

**5. Studies.**

- Update the configurations, adapter loops and the source note at `definitions/inference.py:100`.
- In `verification.md`, request these catalogue changes: remove the `k-minus-1` and `adaptive-bonferroni` variants of `equivalence.familywise`, and add the new ones with citation keys.

**6. Requests for the coordinator.**

- CHANGELOG lines covering removed methods, the default change and any parameter removal.
- Replacement text for `package-development.md:295` and `method-catalogue.md:65`.
- A `known-limitations.md` entry saying the input p-values must be valid.
- Wording for the multiplicity references in `equivalence.py` and `mixture/contracts.py`, which the equivalence lane applies.

## Acceptance

- [x] Every remaining method cites a recorded source theorem, and its docstring states that theorem's assumptions.
- [x] `adaptive_bonferroni` is gone. `k_minus_1` is gone unless a recorded source proves it for every accepted `n_groups`.
- [x] The exhaustive two-null test shows familywise error ≤ α for every remaining method. The Holm oracle and the `jit`/`vmap` parity tests pass.
- [x] Studies definitions, adapters and tests use the new methods, and the coordinator requests are recorded.
- [x] Focused tests, lint, typecheck and architecture pass, or each failure is shown to be only the expected inventory or catalogue staleness.
- [x] The owner is flagged on the default change and on the `k_minus_1` decision.

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
