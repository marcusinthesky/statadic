# Does the test behave as advertised?

**Learning objective:** distinguish repeated-sample statistical behavior from
numerical agreement and from a single p-value.

## Predict first

If a small null study records no rejections, is the test perfectly conservative,
or is the estimate imprecise?

## Experiment

The null scenario has location shift zero; the two alternatives have shifts 0.75
and 1.5. Every trial calls the native upper-tail energy permutation test, with
alpha fixed at 0.05. Independent seeds generate the data and the permutation
stream for each trial. Increasing permutations and increasing independent trials
address different sources of Monte Carlo variability.

All work is gated by form submission. The public helper bounds sample size,
permutations and trials before invoking STATADIC. A failure aborts the study; failed
trials are not silently excluded from its denominator.

## Read uncertainty, not a green badge

The error bars are approximate 95% Wilson score intervals for the Bernoulli
rejection probabilities under these specific scenarios. They are not confidence
intervals for a regression coefficient and are not simultaneous bands.

For example, zero rejections in twenty trials still gives an upper Wilson bound
of approximately 0.161. A point estimate of zero is not proof of zero false
positive probability. Save the study's JSON record to retain settings, runtime
versions and all reported counts.

**Teaching study, not qualification evidence.** This lab does not replace `docs/validation.md` or the scheduled qualification suite. Do not choose a favorable seed and present its chart as calibration. Independent reference comparisons, JIT support and numerical convergence establish different claims from repeated-sample statistical behavior.

The former homogeneity notebook is retired. This bounded teaching lab and the
[permutation lab](permutation-evidence.md) cover a smaller scenario set; its full
distribution, dimension and effect-size grid has not been migrated. Historical
benchmark notebooks remain in the performance application's history directory.

[Return to the documentation index](../index.md)

```{raw} html
<a href="snapshots/calibration.html">View the executed static snapshot</a> · <a href="downloads/calibration.py">Download the notebook</a> · <a href="snapshots/calibration.provenance.json">Snapshot provenance</a>
```

[Run this lab locally](index.md)
