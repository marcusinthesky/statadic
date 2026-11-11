# Formal mathematics

STATADIC connects mathematical claims, their assumptions, source literature and Lean
proofs in a claim registry. Select a claim to see its exact scope and current
evidence. The research programme to cover all supported methods is ongoing.

Lean checks stated mathematical propositions. A proof about real-valued objects
does not by itself verify JAX code, floating-point rounding, a numerical solver,
or a statistical calibration. Links to Python identify a mathematical
relationship; they do not certify an implementation.

<!-- statadic:formal-registry -->

## Explore

- [Using a proof to justify a method](assumptions-and-diagnostics.md) works through
  an embedding that fails its spectral check and an optimization fit with a measured gap.
- [Sources by method family](family-sources.md) lists representative methods, literature
  and reference calculations across the full inventory.
- [Method crosswalk](method-crosswalk.md) pairs the current estimators and inference rules
  with their proof limits and independent implementations.
- [Reading the foundations](literature.md) explains how the claims connect.
- [Bibliography](../reference/bibliography.md) lists the cited sources.
- [Mathematical contracts](../mathematical-contracts.md) explains laws,
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
