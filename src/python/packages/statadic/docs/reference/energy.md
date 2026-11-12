---
type: reference
description: Compare empirical distributions with energy distance, and choose between the raw scalar function and the rooted Energy representation.
---

# Energy

## Usage

```python
import jax.numpy as jnp

from statadic.api.representation import Energy
from statadic.discrepancy.mmd.energy import energy_distance
from statadic.sample.packed import PackedClouds

x = PackedClouds.from_points(jnp.array([[0.0], [1.0]]))
y = PackedClouds.from_points(jnp.array([[2.0], [3.0]]))
raw = energy_distance(x, y, exponent=1.0, metric="euclidean")

clouds = PackedClouds.from_clouds(
    [jnp.array([[0.0], [1.0]]), jnp.array([[2.0], [3.0]]), jnp.array([[0.0], [4.0]])]
)
rooted = Energy(ground="euclidean", exponent=1.0).matrix(clouds)
print(float(raw), float(jnp.sqrt(raw)), rooted.matrix.values[0, 1])
```

## Description

Energy distance compares two probability distributions through expected
distances between their draws. For distributions $X$ and $Y$ with independent
copies $X'$ and $Y'$, it equals
$2\,\mathbb E\lVert X-Y\rVert^\alpha - \mathbb E\lVert X-X'\rVert^\alpha - \mathbb E\lVert Y-Y'\rVert^\alpha$.
STATADIC evaluates this formula on empirical distributions held in
`PackedClouds`, where each atom carries a probability, and the self-pairs are
included so that uniform masses recover the denominators $n^2$ and $m^2$.
The two interfaces share this formula. They return different quantities for
different uses of the data.

| | `energy_distance(x, y, exponent, *, metric)` | `Energy(ground, exponent)` |
| --- | --- | --- |
| Returns | The raw V-energy, a squared-distance scale quantity | Its square root, a distance between distributions |
| Ground | Required through `metric` | `"euclidean"` unless set |
| Input | Two distributions | A collection of distributions with row identities |
| Use it for | A single comparison or a test statistic | Distance matrices, ordination and graph geometry |

The rooted value is the square root of the raw value for the same ground and
exponent. In the example the raw value is 3, and the rooted entry is
$\sqrt3 \approx 1.732$. Rooting gives a metric between distributions when the
ground separates them. A rooted matrix can feed
{py:class}`~statadic.api.ordination.Ordination` directly. The
`Energy` constructor always returns the rooted value. Use the scalar function
for the raw functional, for example as a statistic whose scale you interpret
yourself.

<!-- statadic:math STATADIC-ENERGY-002 -->

## Options

The ground distance and the exponent select the geometry, and both
interfaces check the compatibility of the two when constructed or called.

| Choice | Values | Effect |
| --- | --- | --- |
| `metric` or `ground` | `"euclidean"` | Retains location and scale differences |
| | `"angular"` | Compares directions only, so positive rescaling is invisible |
| `exponent` | Number in $(0, 2)$, default `1.0` | Power $\alpha$ applied to ground distances |

For example, two clouds with the same directions but different radii have a
positive Euclidean energy distance and an angular distance of exactly zero.
The exponent 2 regime is a different quantity and raises a `ValueError`, and
leaving `metric` out of `energy_distance` raises a `TypeError` because the
function assumes no default ground.

## Results

For two single distributions, `energy_distance` returns a zero-dimensional JAX
array, and the leading cloud axes of the inputs broadcast against each other.
You build a distribution with `PackedClouds.from_points`, optionally with
`masses`, and a collection with `PackedClouds.from_clouds`. The result is
nonnegative in exact arithmetic, but roundoff can produce a small negative
value in finite precision.

`Energy(...).matrix(clouds)` returns a `HilbertianMeasureMatrix` whose
`matrix.values` holds the rooted pairwise distances, and
`Energy(...).cross(left, right)` returns the rooted values between two
collections. The method `prepare(measures, row_ids=...)` freezes the geometry
and the order of the measures for the pairwise and cross evaluation of new
rows, while atom counts and masses stay inside each observation.

For advanced numerical work, `energy_distance_with_geometry(x, y, geometry)`
takes a prevalidated `EnergyGeometry` from `statadic.discrepancy.mmd.energy`.
The related `MMD` and `Wasserstein1D` representations follow their own
contracts, and `Wasserstein1D` currently supports `p=2` on the real line.

## Assumptions and limits

- Nonnegativity and the energy-kernel identity are proved for a compatible
  symmetric positive-semidefinite generating kernel. The proof does not cover
  floating-point evaluation or every ground an API can name.
- The angular ground cannot detect positive changes of scale and is defined
  for nonzero vectors, returning NaN where a zero vector appears.
- Only the V-statistic is supported, with no option for an unbiased
  U-statistic.
- A rooted result separates distributions only when the ground and the
  exponent give a geometry of negative type with enough moments.

## Evidence

- Mathematical results: the {ref}`energy and kernel identity <formal-STATADIC-ENERGY-001>`,
  {ref}`nonnegativity with a PSD kernel <formal-STATADIC-ENERGY-002>`,
  {ref}`Euclidean nonnegativity <formal-STATADIC-ENERGY-003>` and
  {ref}`rooting to a separating metric <formal-STATADIC-ROOTENERGY-001>`.
- Reference comparisons: weighted Euclidean energy distance is matched against
  SciPy.
- Studies: the {ref}`energy estimation <research-procedure-energy.estimation>`
  procedure in the study catalogue.

## See also

- [Energy and scale tutorial](../tutorials/energy-and-scale.py)
- [Geometry](../concepts/geometry.md)
- [Ordination](ordination.md), which consumes rooted energy matrices
