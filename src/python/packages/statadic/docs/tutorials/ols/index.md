# An OLS analysis, from fit to report

Ordinary least squares estimates a conditional mean. A useful analysis also asks
what the residuals reveal, which uncertainty calculation is justified, and what
can be saved for another reader.

Work through four lessons in order. They use one executable example and the
same 80-row sample throughout; switching covariance never estimates a new line.

:::{toctree}
:maxdepth: 1

fit
diagnose
infer
report
:::

Download {download}`ols_analysis.py <../../../examples/ols_analysis.py>` and run:

```bash
python ols_analysis.py --output ols-results
```

The example needs installed STATADIC, JAX and NumPy. It writes only to the requested
directory. Without `--output`, it uses a temporary directory. Caller precision
governs the calculation; this lesson does not enable float64 globally.

For reactive controls, open the [native regression lab](../../labs/regression.md).
It calls this same example, with the same default sample, rather than maintaining
another statistical recipe. The [concepts guide](../../concepts/index.md) explains
the relationship between specification, fitted state, inference and reporting.

Evidence links below have different meanings: a Lean result checks a mathematical
statement under assumptions; a reference comparison checks named numerical
quantities; a registered study describes a statistical experiment. None makes
this dataset causal or establishes every reference distribution's validity.
