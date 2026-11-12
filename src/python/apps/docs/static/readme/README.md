# README example artifacts

These files come from the marked OLS Python example in the root README.
`report.txt`, `regression.tex` and `coefficients.tex` are its actual outputs;
`ols-coefficients.png` renders the same retained coefficient plot with Matplotlib.
`provenance.json` records the source hash, package versions and runtime settings.

Regenerate from the pinned devenv shell, with the docs workspace prepared:

```bash
uv run --project src/python --offline --frozen --package statadic-docs python src/python/apps/docs/scripts/render_readme.py
```

The renderer executes the example in a temporary directory with CPU float64,
then copies the report, figure and TeX outputs here. It does not change the
package's default precision policy. The intervals use HC3 covariance and a normal
reference; this six-observation illustration establishes no coverage guarantee.
The marked numerical table in the root README is refreshed from the same report.
