---
type: how-to
title: Install STATADIC
description: Install STATADIC from a checkout and choose dependencies for your analysis.
---

<!-- rumdl-disable MD041 -->

(install-statadic)=

# Install STATADIC

Install the package in an environment and open a teaching notebook in a separate sandbox.

## Introduction

An isolated environment keeps this checkout's dependencies together, so the
interpreter used for an analysis can find the same packages each time. The
commands below use Python 3.13 from the repository root and include the optional
adapters for Polars and Arrow. STATADIC also has targeted compatibility checks
for Python 3.14, while the notebooks declare Python 3.13 or later.

## Step 1: Create an environment

Create a virtual environment for the Python version used by the notebooks:

```bash
uv venv --python 3.13
```

The environment lives in `.venv` under the checkout, so the next command can
select its interpreter explicitly even if another Python environment is active.
Use that interpreter for the installation.

## Step 2: Install the package

Install STATADIC and its optional data adapters into that interpreter:

```bash
uv pip install --python .venv/bin/python './src/python/packages/statadic[data]'
```

An analysis using arrays alone can omit the `data` extra, while workflows that
read Polars or Arrow data need the corresponding adapters installed with the
package in the environment. Run `.venv/bin/python`.

## Step 3: Open a teaching notebook

Download the notebook source from its page, then open the downloaded file with
a sandboxed marimo environment:

```bash
uvx marimo edit --sandbox first-regression.py
```

The notebook declares its dependencies and the Git source for STATADIC, so
the first installation needs network access. The website shows precomputed outputs from the same notebook, which you can
read without starting a notebook kernel or installing its dependencies.

## Result

Your checkout has a Python environment with STATADIC installed, and a downloaded
notebook can create its own environment from its declared dependencies.

## Assumptions and evidence

The package installs CPU JAX by default, providing a backend for the tutorial
calculations on a CPU. Check the backend for your own analysis. Check the [transformation support
reference](../reference/transformation-support.md) and [known limitations](../validation/known-limitations.md)
when choosing a backend or floating precision.

## See also

- Continue with the [first regression](first-regression.py).
- Repository contributors can use the pinned environment in [CONTRIBUTING](https://github.com/marcusinthesky/statadic/blob/main/CONTRIBUTING.md).
