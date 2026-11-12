---
type: explanation
description: How representations, distances and mathematical laws affect what an analysis can distinguish.
---

# Choosing a geometry

Suppose one observation is a cloud of points and another contains the same
points at twice their original scale. A comparison based on directions can
treat these clouds as identical, while one based on Euclidean distances can
distinguish the change in their magnitudes. The representation determines which differences remain visible to the analysis,
including whether a change in the scale of the cloud contributes to the result.
Choose it deliberately.

## Start with the objects being compared

Points, directions and probability distributions have different notions of
equality. Normalizing each nonzero vector preserves its direction and discards
its magnitude, so comparing the resulting distributions answers a question
about normalized data. The original scale is lost.

Angular comparisons require nonzero inputs because a zero vector has no
direction to compare. If magnitude matters for the question, choose a
representation that preserves it and read the [energy reference](../reference/energy.md)
before selecting a ground distance.

## Keep the output convention visible

Raw energy and its square root have different scales, as do MMD squared and
rooted MMD, so the output convention matters when comparing results. An
unbiased U-statistic estimate can be negative. Clipping it before taking a
square root changes the quantity being reported and its statistical interpretation.

The high-level `Energy` and `MMD` representations use rooted V-statistics.
The scalar functions expose their own conventions, so check the function's
reference before comparing numbers from two interfaces.

## Match a method to the laws it needs

A distance can be a metric without satisfying every requirement of an embedding
or kernel method. Principal-coordinate analysis needs the appropriate structure
of squared distances, and a kernel method needs a positive semidefinite kernel.
These requirements depend on the construction of the input.

A finite spectrum describes one matrix, whereas a universal law concerns the
whole class of inputs that a method is intended to accept. Similarly, a finite
set of projections compares objects through that particular design, so two
objects can agree on those projections while differing elsewhere.

## Distinguish a checked value from a general claim

Checking that a vector has unit length establishes a property of that value.
A theorem about distances between unit vectors states a general result under
its hypotheses. The source and hypotheses of that result remain relevant
even when a particular input passes its numerical checks.

Use the [distance catalogue](../reference/catalogue-metrics.md) to compare
constructions and the [formal mathematics](../formal/index.md) to inspect the
scope of individual statements.
