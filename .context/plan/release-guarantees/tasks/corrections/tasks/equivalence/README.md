---
type: task
title: Remove the unsupported equivalence guarantee
description: "Stop claiming a Type I guarantee for the energy-distance equivalence test, describe what Liu and Gandy's theorem does and does not cover, and make non-finite bootstrap replicates give a NaN equivalence p-value."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md"]
status: done
---

# Remove the unsupported equivalence guarantee

## Goal

The equivalence documentation should claim only what is supported, and an undefined bootstrap replicate must not count as evidence of equivalence.

The procedure, a percentile bootstrap of energy distance after the mixture weights are selected from the data, is not covered by the theorem it cites. The [evidence](../../../../evidence.md#equivalence) explains why.

## Owns

- `statadic/inference/equivalence.py`.
- The docstrings in `statadic/inference/mixture/contracts.py`.
- `tests/inference/test_equivalence.py`, except the `TestMultiplicity` class, which the multiplicity lane removes. Re-read the file before each edit.
- The reference records this lane creates.

## Specification

**1. Remove the claim.** Remove "a genuine Type-I guarantee" (`equivalence.py:103-105`) and any other size, level or familywise claim in these two files. Say plainly that:

- the equivalence p-value and the confidence-interval verdict are bootstrap approximations;
- no size result is proved for energy distance with selected weights.

**2. Liu and Gandy.** Add one sentence:

- their Theorem 8 covers a multiplier bootstrap for bounded-kernel MMD with fixed distributions;
- STATADIC's procedure differs in the bootstrap, the kernel and the selection step.

Cite it as related work. Do not implement it.

**3. Citations.**

- For each of Berger & Hsu 1996, Wellek 2010 and Schuirmann 1987, keep it only if the docstring relies on that source's content, for example the intersection-union principle or TOST.
- Each kept source, and Liu & Gandy 2026, needs a reference record checked against the source, with the section or theorem cited.
- Remove any citation you do not record.

**4. Non-finite input.**

- `equivalence_pvalue_from_bootstrap` returns `math.nan` when δ or any replicate is non-finite.
- Check how `equivalence_test`'s `equivalence` flag behaves with non-finite interval endpoints. It must not declare equivalence. Record the behaviour.

**5. Multiplicity references.**

- Reword `equivalence.py:111-116,129` and `mixture/contracts.py:7,36` to match the multiplicity lane's final API. The coordinator relays that lane's wording.
- Name only methods that still exist.

**6. Tests.** Cover a NaN replicate, a non-finite δ and non-finite interval endpoints.

## Acceptance

- [x] No docstring in `equivalence.py` or `mixture/contracts.py` claims a Type I, size, level or familywise guarantee.
- [x] Every remaining citation has a checked reference record, and the docstring says that Liu and Gandy's theorem does not cover this procedure.
- [x] Non-finite replicates or δ give a NaN equivalence p-value, non-finite intervals never declare equivalence, and tests cover both.
- [x] Focused tests, lint, typecheck and architecture pass, or each failure is shown to be only the expected inventory staleness.

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
