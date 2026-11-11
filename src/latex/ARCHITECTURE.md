---
description: Why manuscripts are independent Tectonic projects over a vendored bundle, with generated results.
---

# Manuscript build

The layout follows the Pricing Perspective publishing workspace. Dependencies
point one way: analysis code writes `generated/`; chapters read semantic
commands; target adapters own document class and layout; Tectonic renders a
target to PDF.

## One engine, offline builds

Tectonic is the single PDF engine, and devenv pins it with tex-fmt and ChkTeX.
Every `Tectonic.toml` points at `bundle/tlextras-2022.0r0-statadic.zip`, the subset
of the upstream `tlextras-2022.0r0` bundle that the projects and the PGFplots
fixture resolve. Tectonic reads a local zip as a bundle when it carries a
`SHA256SUM` member. Vendoring removes two problems of a URL bundle: builds
depending on a third-party host that answers HTTP 429 under load, and a pin that
holds only by trust on first use. `bundle-refresh` rebuilds the subset in a
scratch cache and writes a deterministic archive.

The effective asset set is TeX Live 2022. An `arxiv` target uses a conservative
`article` class and long-stable packages, because arXiv recompiles uploads under
its own installation. `just latex::bundle` stages only the files a target needs,
then unpacks the archive into an empty directory and recompiles it from the
vendored bundle before reporting success.

## Projects and targets

A project is independently buildable. Named outputs map to thin files under
`src/targets/`, so venue choices do not become conditional preamble branches.
Chapters know manuscript semantics but not layout. A venue adapter, such as a
journal class, is added as another target over the same `body.tex`.

## Results enter through a narrow interface

Numbers, tables and figures are analysis artifacts. A project's
`analysis/generate.py` writes `\jcDeclareValue` bindings and table or figure
fragments to `src/generated/`. Prose reads values through `\jcvalue`, and TeX
never computes or loads data. Printed code listings live in `src/listings/`, and
the analysis executes them verbatim, so a listing cannot drift from the API.

STATADIC's native `PlotSpec.to_pgfplots` output is included unchanged. A target
declares `pgfplots`; the manuscript may adjust axis styles with
`\pgfplotsset` without changing the computed coordinates.

## Bibliography

CSL records in `.context/reference/` are canonical. `just sync` derives
`refs.bib`, and `_prepare` copies it into each project before compiling. One
bibliography therefore serves every manuscript, the documentation site and the
Lean bridge.

## Log policy

`tools/check-log` fails a compile on unresolved or multiply defined citations,
references and labels, on missing glyphs, and on overfull boxes of at least
`STATADIC_OVERFULL_PT` (default 1pt).
