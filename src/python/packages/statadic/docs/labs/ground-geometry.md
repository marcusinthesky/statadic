# What does distance notice?

**Learning objective:** separate direction from magnitude before choosing a
statistical distance. You need only a basic understanding of vectors.

## Predict first

Vector A is `(1, 0)`. Vector B is `(3, 0)`. Their directions match, but their
lengths do not. Which distances should be zero?

## Try it

Set the angle to zero, then vary B's length. Next, keep its length fixed and
rotate it. Finally, set its length to zero: the lab stops the angular comparison
and explains the invalid input instead of presenting NaN as a scientific result.

Euclidean distance records magnitude and direction. Angular distance records the
angle in radians and discards positive radial scale. On raw vectors, zero angular
distance does not establish equal vectors. Whether that is useful depends on
which differences your scientific question treats as meaningful.

**Analytic check:** A=`(1,0)`, B=`(3,0)` gives Euclidean distance 2 and angular
distance 0. The native lab tests check that witness.

## What this does not establish

A plotted point pair cannot prove the mathematical laws of a distance. A metric
on a direction-only space and a metric on raw vectors are different claims.
Angular input vectors must be finite and nonzero. The lab guards the zero-vector
case; it does not redefine the package's mathematical domain.

[Next: compare whole distributions](energy-distance.md)

```{raw} html
<a href="snapshots/ground-geometry.html">View the executed static snapshot</a> · <a href="downloads/ground-geometry.py">Download the notebook</a> · <a href="snapshots/ground-geometry.provenance.json">Snapshot provenance</a>
```

[Run this lab locally](index.md)
