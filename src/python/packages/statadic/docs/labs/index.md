# Interactive labs

These seven lessons use synthetic data and call the installed STATADIC package. Each page
has a static explanation, an executed snapshot, and a downloadable marimo source.
The snapshots show fixed settings; run the notebook locally to change controls.

:::{toctree}
:maxdepth: 1

ground-geometry
energy-distance
permutation-evidence
regression
distributional-workflow
calibration
study-results
:::

From the repository root, run `just setup` once, then open a lab by its slug:

```bash
just python::docs::lab ground-geometry
```

The available slugs are `ground-geometry`, `energy-distance`,
`permutation-evidence`, `regression`, `distributional-workflow`,
`calibration`, and `study-results`.

Alternatively, with Python 3.13 and uv installed, use the locked workspace
without Just:

```bash
uv sync --locked --project src/python --package statadic-docs
uv run --locked --project src/python --package statadic-docs python src/python/apps/docs/scripts/lab.py ground-geometry
```

Replace `ground-geometry` with any slug above.
