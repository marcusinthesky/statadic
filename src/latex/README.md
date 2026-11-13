---
description: Multi-project LaTeX workspace built offline by one pinned Tectonic engine and a vendored bundle.
---

# LaTeX

Each paper is an independent project under `projects/<slug>/` with its own
`Tectonic.toml` and named targets. Run from the repository root:

```bash
just latex::new 02_example                # scaffold a project from _template
just latex::generate 01_statadic              # rerun a project's analysis script
just latex::build 01_statadic arxiv           # compile one target offline
just latex::watch 01_statadic draft           # rebuild on save
just latex::bundle 01_statadic arxiv          # archive and clean-recompile an upload
just latex::preview figure.tex            # view one PlotSpec PGFplots fragment
just latex::check                         # format, lint and compile every target
just latex::bundle-refresh                # regenerate the vendored TeX bundle
```

`bundle/tlextras-2022.0r0-statadic.zip` is the subset of the pinned upstream bundle
that these projects resolve, tracked in Git LFS. Builds therefore never use the
network. `just latex::setup` fetches the LFS content in a fresh clone. Run
`bundle-refresh` (network required) only after adding a package or font size the
subset lacks, then commit the regenerated zip.

`just latex::generate` runs `analysis/generate.py` in the locked reference
environment prepared by `just setup-references`. It writes `src/generated/`,
which is tracked so a checkout builds without rerunning the analysis.

## Layout

| Path | Owns |
| --- | --- |
| `templates/statadic-manuscript.sty` | Shared semantic commands, generated-value macros and listing style |
| `projects/<paper>/Tectonic.toml` | Pinned bundle and named build targets |
| `projects/<paper>/src/targets/` | Document class, layout, packages and front matter |
| `projects/<paper>/src/body.tex` | Stable composition of content chapters |
| `projects/<paper>/src/chapters/` | Layout-free manuscript sections |
| `projects/<paper>/src/generated/` | Analysis-owned values, tables and PGFplots figures |
| `projects/<paper>/src/listings/` | Code listings the analysis executes verbatim |
| `projects/<paper>/analysis/` | The script that regenerates every reported result |
| `refs.bib` | BibTeX derived by `just sync` from `.context/reference/` records |
| `tools/check-log` | Rejects unresolved references, missing glyphs and overfull boxes |

`_prepare` copies the shared style and bibliography into a project before each
compile; those copies are ignored. Compiled PDFs and upload bundles live under
each project's ignored `build/<target>/` directory.

See [ARCHITECTURE.md](ARCHITECTURE.md) for ownership and pinning decisions.
