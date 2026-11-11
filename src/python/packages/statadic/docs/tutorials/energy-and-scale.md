# Energy distance: what exactly are you comparing?

**Goal:** distinguish a raw energy functional, a rooted distance and a scale-insensitive ground comparison.
**Prerequisites:** arrays with rows as observations; no hypothesis-testing background required.
**Data:** tiny deterministic point clouds with an analytic check.

## Begin with an explicit estimand

For samples of sizes `n` and `m`, the scalar V-energy implementation computes

```text
ÊV = 2 × mean(cross distances^α)
      − mean(within-X distances^α)
      − mean(within-Y distances^α).
```

Within-sample means include self-pairs and use divisors `n²` and `m²`. This is not a signed U-statistic with diagonal terms removed. The scalar function returns the raw functional, not its square root.

With Euclidean ground, `X = {0,1}`, `Y = {2,3}` and `α=1`, the cross-distance mean is 2 and both within-distance means are 0.5. The answer is exactly `2×2−0.5−0.5=3` in real arithmetic. A rooted representation would use the corresponding square-root convention instead.

## Run the example

From the repository root, with the [installed environment](../start/installation.md) active:

```bash
python src/python/packages/statadic/examples/energy_and_scale.py
```

```{literalinclude} ../../examples/energy_and_scale.py
:language: python
```

{download}`Download the complete script <../../examples/energy_and_scale.py>`

## Why explicit ground selection matters

The low-level `energy_distance` requires `metric`. Angular normalization makes it insensitive to positive scale changes on its supported nonzero domain. Multiplying every point by three can therefore give zero angular discrepancy even though the Euclidean comparison detects a change.

Never use a zero vector in the angular illustration: this ground requires nonzero rows, and its documented zero-row behavior is NaN. Explicit Euclidean ground permits the simple zero-containing analytic example above.

The host `Energy()` representation has a **different** default: Euclidean ground and **rooted** V-energy. See the [side-by-side contract](../reference/energy.md). Do not change one default during a documentation-only refactor, and do not imply that `Energy()` is merely an object-oriented spelling of the scalar default.

## What this result does not tell you

A discrepancy is not a p-value. A homogeneity-test p-value is not an equivalence certificate. To claim distributions are sufficiently close, specify an equivalence margin and a justified inferential design. A large p-value from a difference test does not establish closeness.

Supported exponents and ground laws constrain the mathematical interpretation. “Metric”, “negative type”, “PSD kernel” and “distribution separating” are different claims. An observed finite matrix does not prove a population law.

## Exercise

Explain the result of multiplying both nonzero clouds by a positive constant under angular and Euclidean ground. Then identify which reported number changes when switching between raw and rooted energy. Answers should name the ground, exponent and root convention—not merely the function name.
