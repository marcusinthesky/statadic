<!-- rumdl-disable MD041 -->

(install-statadic)=

# Installation

<!-- rumdl-enable MD041 -->

Use Python 3.13; Python 3.14 has targeted compatibility coverage. From the repository root:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python './src/python/packages/statadic[data]'
```

CPU JAX is installed by default. GPU installation and floating precision are
caller choices, with [method-specific support](../capabilities.md).

Tutorials use Polars/Arrow adapters (`[data]`). Run examples with
`.venv/bin/python`; array-only consumers can omit `[data]`.

Contributors: run `just setup` in the pinned devenv shell.
See [isolated development workflows](../reference/package-development.md).
