# Is the observed difference surprising?

**Learning objective:** connect a discrepancy to a specified null and a valid
randomization scheme. Prerequisite: the [energy lesson](energy-distance.md).

## Predict first

Would the observed separation still look unusual if the group labels had been
assigned differently? More separation is not automatically stronger evidence:
sample size and the null distribution matter.

## Experiment

Set the shift to zero to simulate the null. Increase the shift to investigate an
alternative. Use 19, 49, 99 or 199 permutations and watch the p-value resolution.
Results describe the settings last submitted, not any later unsubmitted edits.

This lab fixes Euclidean ground and exponent one before seeing the data. It uses
the typed `energy_permutation_test_result` path with `alternative="greater"`.
The statistic is `T = n*m/(n+m) * raw_V_energy`. A histogram of T is not a
histogram of the unscaled discrepancy or its square root.

The reported p-value is `(1 + exceedances) / (B + 1)`, including ties. With 19
random permutations, the smallest possible p-value is 0.05, not zero.

## Interpretation and limits

Under this lab's null, the observations are IID from the same distribution, so
unrestricted label permutations are appropriate. Paired, clustered, time-series
or otherwise dependent observations require a suitable design. Extra shuffles
do not repair a wrong design.

A p-value is not the probability that the null is true. Failure to reject a
difference is not evidence of equivalence. Repeatedly trying seeds or choosing a
metric after looking for a small p-value changes the analysis.

**Challenge:** can B=19 ever produce p=0.001? Explain using the calculation,
not by searching seeds.

[Next: fit and interpret a regression](regression.md)

```{raw} html
<a href="snapshots/permutation-evidence.html">View the executed static snapshot</a> · <a href="downloads/permutation-evidence.py">Download the notebook</a> · <a href="snapshots/permutation-evidence.provenance.json">Snapshot provenance</a>
```

[Run this lab locally](index.md)
