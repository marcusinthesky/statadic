# STATADIC documentation application

The canonical site source is `../../packages/statadic/docs/`. This app owns
Sphinx, notebook rendering and the checks that guard the site. The STATADIC
wheel has no dependency on documentation tools.

## Build, serve and check

Run these commands from the repository root in the pinned devenv environment:

```bash
just python::docs::build
just python::docs::build-fast
just python::docs::serve
just python::docs::notebook tutorials/energy-and-scale.py
just python::docs::check
just python::docs::check-size
just python::docs::check-browser
```

`build` certifies the Lean sources, builds the blueprint and writes native HTML
and notebook downloads under ignored `build/`. `build-fast` reuses the existing
Lean certification and blueprint. Both execute canonical notebooks on cache
misses. `serve` previews the result at `http://127.0.0.1:8000`.

`check` validates word budgets, page schemas and prose, then checks app sources,
notebook integrity, app tests and the complete build. Checks never repair source.
Use `just python::docs::fmt` to apply formatting explicitly. Root `just check`
also runs the package's doctest command.

`just python::setup` prepares the locked dependencies and pinned TeX resources.
`just python::docs::setup` prepares the TeX resources explicitly and may use the
network. Report tests compile with the repository Tectonic tool and cached
bundle. External links remain a separate `just links` check.

`check-browser` inspects built HTML using installed Chrome/Chromium, or the
executable supplied with `--browser /path/to/executable`. It checks responsive
layouts, light and dark system preferences, local fonts, navigation, search,
local targets, figures and mathematics. Screenshots and hashes are saved under
`build/ci/browser/`.

## Authoring

[STYLE.md](STYLE.md) defines page types, schemas and voice. Navigation separates
Start, Tutorials, How-to, Concepts, Reference, Evidence and Blog. Contributor
procedures belong in repository Markdown, and each method's assumptions and
evidence have one home in its reference page.

Tutorials and how-to guides are self-contained Marimo notebooks in `start/`,
`tutorials/` and `guides/`. Copy
`../../packages/statadic/docs/_templates/notebook.py`, then check an edited file:

```bash
just python::docs::check-notebooks /absolute/path/to/notebook.py
```

The runner checks strict Marimo rules, unparsable cells, script execution in
both JAX precisions, collected output tests and any doctests. Schema and prose
checks run separately. Output tests and doctests share one pytest invocation;
published notebooks must collect an ordinary output test even when doctests pass.
Notebook tests live in a hidden Appendix. Independent
comparison tests belong in `tests/` and import the notebook's top-level helpers.

The renderer runs Marimo's public `export ipynb --include-outputs --sort top-down`
command with real JAX and adapts that standard export to native Sphinx pages.
Marimo supplies Markdown cells, code visibility, static tables and executed
outputs. The site hides Appendix cells and adds a source download, an Open in
molab link and a local sandbox command. Reading a static page needs no hosted
kernel. Changing notebook, package, renderer or locked dependency contents
invalidates cached executions.

Marimo also supports `marimo export md notebook.py -o notebook.md` and
`marimo convert notebook.md > notebook.py`. Use these native commands for
source conversion. Markdown export omits executed outputs, so the published
site uses the output-inclusive Jupyter export. The remaining adapter handles
Sphinx links, launch controls and static presentation of interactive controls.
Every build checks both source prose and the executed Markdown outputs.

## Evidence and release

The formal extension projects the claim registry into canonical source stubs
at build time. Preserve its markers and use explicit creation/synchronization
commands for authored repository records. Build success remains separate from
scientific qualification.

Release selections and GitHub Pages publication are documented in the
[release guide](../../../../infra/release/README.md). The
[branding notes](static/brand/README.md) record the local serif fonts, Delft-blue
stylesheet, logo and asset provenance. `scripts/render_readme.py` regenerates
the root README's marked OLS outputs using the same locked app runner.

## Blog

Posts live in `../../packages/statadic/docs/blog/`. Preserve the author's
wording. A post has `type: blog`, description frontmatter, a title, and an author
and date line. Add its summary and hidden toctree entry to `blog/index.md`, then
build the site. Blog prose is measured as a voice reference and is exempt from
the documentation prose gate.
