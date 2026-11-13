---
type: evidence
description: Mathematical statements, hypotheses and comparisons for STATADIC methods.
---

# Formal mathematics

STATADIC connects mathematical claims with their assumptions, source literature and Lean
proofs in a claim registry. Select a claim to see its precise statement, assumptions and available evidence, while keeping in mind that the programme to cover supported methods is ongoing.

Lean checks stated mathematical propositions under their hypotheses, and links to Python identify the calculations related to those statements. Verification of
JAX code, floating-point rounding and statistical calibration requires separate
evidence about the implementation and complete procedure.

<!-- statadic:formal-registry -->

## How to read this

- [Using a proof to justify a method](assumptions-and-diagnostics.md) works through
  an embedding that fails its spectral check and an optimization fit with a measured gap.
- [Sources by method family](family-sources.md) lists representative methods, literature
  and reference calculations across the full inventory.
- [Method crosswalk](method-crosswalk.md) pairs the current estimators and inference rules
  with their proof limits and independent implementations.
- [Reading the foundations](literature.md) explains how the claims connect.
- [Bibliography](../reference/bibliography.md) lists the cited sources.
- [Mathematical contracts](../concepts/geometry.md) explains laws,
  transformations and the evidence carried by STATADIC objects.

:::{toctree}
:hidden:
:maxdepth: 1
:glob:

literature
assumptions-and-diagnostics
family-sources
method-crosswalk
manuscripts
claims/*
:::
