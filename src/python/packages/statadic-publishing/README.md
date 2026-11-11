# STATADIC publishing

Local publishing specifications and evidence manifests for researcher-authored
analyses. Run `just python::publishing::check` in the pinned environment.
See `ARCHITECTURE.md` for ownership and the project and manifest modules for
schema definitions. Publication and scientific interpretation remain the
researcher's responsibility.

From the repository root, using the pinned shell:

```sh
devenv shell -- just python::publishing::build examples/ols-hc3 /tmp/ols-bundle
devenv shell -- just python::publishing::verify /tmp/ols-bundle examples/ols-hc3
devenv shell -- just python::publishing::archive /tmp/ols-bundle /tmp/ols-bundle.tar
```

The equivalent CLI is `python -m statadic_publishing build`, `verify`, or `archive`
with the same arguments. `just python::publishing::check` runs the fresh-wheel
reconstruction outside the checkout, including the locked reference comparison
and offline TeX compilation. It requires the pinned Tectonic bundle and engine.
