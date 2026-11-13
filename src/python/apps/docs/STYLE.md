# Writing the STATADIC documentation

This guide describes how the user documentation in
`src/python/packages/statadic/docs/` is organised and written. It exists because
documentation drifts. Each new page answers the question in front of its author,
and over time the site fills with pages that half-teach, half-specify and
half-apologise. A fixed structure for each kind of page, checked on every
build, keeps the site readable as the library grows.

## One purpose for each page

Every page serves exactly one of four reader needs, following the
[Diátaxis](https://diataxis.fr) framework. A **tutorial** teaches by guiding a
newcomer through a complete analysis that is certain to work. A **how-to guide**
helps a practitioner reach a goal they already understand. A **reference** page
describes an interface precisely enough to look things up. An **explanation**
discusses why the library works the way it does and what its choices mean for
an analysis.

Three supporting types complete the site. A **landing** page introduces a
section and links into it. An **evidence** page presents the mathematical
results, reference comparisons and statistical studies behind the methods. A
**blog** post is an essay owned by its author.

Each Markdown page declares its type in frontmatter:

```yaml
---
type: reference
description: One sentence that appears in search results and link previews.
---
```

Author-owned blog posts keep their wording. Their prose metrics are reported as
a voice reference, while the hard prose rules apply to documentation pages.

Notebooks take their type from their directory. Those in `start/` and
`tutorials/` are tutorials, and those in `guides/` are how-to guides. The schema
for each type lives in `infra/schemas/documents/docs-<type>.document-schema.yaml`
and fixes the sections a page must contain, their order and the length of each.
Notebook page budgets apply to the published content before `## Appendix`.
The hidden Appendix remains covered by parsing, execution, pytest and doctests.

## Where content belongs

The site is written for people who use STATADIC. Release procedures, module
rules, numerical boundary inventories and notes about how a change was made
belong in the repository's own Markdown: `CONTRIBUTING.md`, the package
`ARCHITECTURE.md`, the application READMEs and `infra/release/README.md`. The
history of a change belongs in `CHANGELOG.md`.

Each fact has one home. The assumptions, limits and evidence for a method live
in its reference page, under `## Assumptions and limits` and `## Evidence`.
Tutorials and how-to guides link to those sections once rather than repeating
their caveats. When the same idea appears on a second page, link to the first.

## Voice

The documentation should read like the project's README and blog: clear,
measured prose that explains how things relate. In practice, three habits
separate that voice from text that merely lists facts.

**Use prepositions to show relationships.** Compressed noun phrases hide how
ideas connect. "The retained training design" asks the reader to unpack four
stacked words, while "the design built from the training rows" says what
happened and to what. Measured on the project's essays, good prose uses about
11 to 15 prepositions for every 100 words. Earlier pages in this site used about
half as many.

**Vary sentence length around twenty words.** Short sentences carry emphasis,
and longer ones carry reasoning. A page made entirely of eight-word sentences
reads as a checklist. Aim for a mean of 15 to 22 words in narrative prose, with
real variation between sentences, and never exceed 40 words.

**State what happens before stating its limits.** A reader first needs to know
what a method does and why it matters. The limit follows in its own sentence,
and on most pages it belongs in the reference page's `Assumptions and limits`
section. A paragraph should contain at most one negative sentence.

Some smaller habits follow from these:

- Write two sentences instead of joining clauses with a semicolon.
- Open a paragraph with a word that frames it, such as "However", "In practice"
  or "Since then", and keep each paragraph to one idea.
- Give a concrete example before the general statement.
- Use the terms defined in the [glossary](../../packages/statadic/docs/start/glossary.md)
  consistently, and avoid internal vocabulary such as ticket numbers, paper
  numbers or words like "witness", "lane" and "campaign".
- Describe the library as it is, not how it changed. Words like "former" and
  "no longer" belong in the changelog.

Two rewrites show the difference:

> Caller precision governs the calculation; this lesson does not enable
> float64 globally.
>
> The calculation runs at the precision you have configured in JAX. This
> lesson leaves that setting unchanged.
>
> Numerical availability does not establish scientific assumptions.
>
> A fit can succeed numerically even when the assumptions behind it fail, so
> check those assumptions against how your data were collected.

## Checks

Two scripts enforce this guide on every build, and both run as part of
`just python::docs::check`.

`scripts/check_schema.py` validates each page against the schema for its type.
For notebooks it first runs `marimo export md`, which turns every pure `mo.md`
cell into Markdown, and validates the result.

`scripts/check_prose.py` applies the hard rules to every page: no semicolons in
prose, no sentence over 40 words, no internal vocabulary and no more than one
negative sentence per paragraph outside `Assumptions and limits`. It also
prints advisory measurements for each page, including mean sentence length,
its variation, prepositions per 100 words and negations per sentence. These
measurements never fail a build, but a page far outside the ranges above
deserves another draft.

Sentence metrics describe narrative prose. Labels and fragments in tables,
and lists of links in navigation sections, remain subject to the hard rules
while being excluded from narrative sentence measurements. Full sentences in
tables contribute to those measurements. Curated evidence tables are checked
at their authored data sources as well as being rendered from those sources.

## Notebooks

Tutorials and how-to guides are [marimo](https://marimo.io) notebooks. Each one
is executed during the documentation build with the real library and rendered as
a static page, with an **Open in molab** button and a command for running it
locally. Readers who want to change the controls can launch the notebook,
while everyone else reads the executed result.

Source conversion uses Marimo's native Markdown export and conversion commands.
The site uses its public Jupyter export with outputs, since Markdown export
contains source without executed results. The adapter consumes standard cells
and output metadata, preserving figures, tables and computed prose without
depending on Marimo's private notebook parser.

### A common structure

Every notebook follows the same sequence of sections, so a reader always knows
where to look. Start from `docs/_templates/notebook.py`, which contains each
section in order and already passes every check.

| Section | Purpose | Tutorial | How-to |
| --- | --- | --- | --- |
| `# Title` | What the reader will do, followed by a one-sentence summary | required | required |
| `## Introduction` | The question, why it matters and what the reader will produce | required | required |
| `## Before you start` | The knowledge the notebook assumes, with links | required | optional |
| `## The data` | Where the observations come from, with a preview and a figure | required | optional |
| `## Step N: …` | One stage of the analysis, ending in a visible result | 2 to 8 | 1 to 8 |
| `## Explore` | Controls that rerun the steps, with questions to predict first | required | optional |
| `## What you learned` or `## Result` | A short summary or the final artifact | required | required |
| `## Assumptions and evidence` | One callout linking to the reference page | required | required |
| `## Next steps` or `## See also` | Two to four links | required | required |
| `## Appendix` | Helper functions and tests, hidden from the page | optional | optional |

Each heading sits in its own cell containing a single `mo.md` call with a plain
string. That cell exports as Markdown, which is how the schema sees it. A cell
whose `mo.md` call uses an f-string is treated as code, so set `hide_code=True`
on it and only its rendered text will appear.

### Writing the steps

Each step opens with one or two sentences explaining why it comes next. A short
code cell follows, ideally under fifteen lines, and ends with something the
reader can see: a report, a table or a figure. The step closes with an
interpretation written as an f-string over the computed values, so the prose
always agrees with the output:

```python
@app.cell(hide_code=True)
def _(fit):
    _slope = float(fit.parameters[1])
    mo.md(f"Each additional unit of `x` raises the fitted outcome by {_slope:.2f}.")
    return
```

Some of marimo's own recommendations matter especially for documentation:

- Keep global names few and descriptive, and prefix names that are local to a
  cell with an underscore.
- Put reusable logic in top-level functions marked with `@app.function`. The
  steps, the Explore section and the tests can then share it.
- Create a new object in each cell rather than mutating one defined elsewhere,
  and avoid `on_change` handlers and `mo.state`.
- Seed every random draw, so that each cell produces the same output on every
  run.
- Gate expensive reruns in the Explore section behind `mo.ui.run_button` with
  `mo.stop`, and cache them with `mo.cache`.
- Show reports with `mo.plain_text(report.to_text())` or `mo.md` rather than
  `print`, which marimo sends to the console instead of the page.

Each notebook is self-contained. molab opens a single file, so a notebook
cannot import from its neighbours, and each one carries its own data generator
in the Appendix.

### Figures and interactive controls

The static page must make sense without a running kernel. Narrative figures
therefore use matplotlib, which marimo renders to SVG under the style set in
the scaffold's setup cell. Give every figure one idea, label its axes with
units and leave out decoration.

Interactive elements belong in the Explore section. Controls from `mo.ui` and
widgets from [wigglystuff](https://github.com/koaning/wigglystuff) work well
there, and reactive plots such as `mo.ui.matplotlib` let a reader select points
and watch the results update. Each control's default must match the values in
the narrative, because the static page shows the defaults. Pair every widget
with a matplotlib output of its current value, so that the page still shows a
meaningful result. When a plot supports a selection, give it an initial value
that illustrates the point being made, such as the observations with the
highest leverage.

### Tests and integrity

Writing a marimo file by hand makes one failure especially easy. If marimo
cannot parse a cell, it stores the code as `app._unparsable_cell(...)`, and the
notebook will open but cannot run. `scripts/check_notebooks.py` guards against
this and related problems by running five checks on every notebook:

1. `marimo check --strict --select MB,MR,MF`, which reports unparsable cells,
   duplicate definitions, dependency cycles and formatting problems.
2. A search for `_unparsable_cell` anywhere in the file.
3. A run of the notebook as a script, once with 32-bit and once with 64-bit
   JAX precision.
4. The in-notebook tests, collected by pytest.
5. The doctests in the notebook's `@app.function` helpers.

Tests live in the Appendix. Following marimo's pytest rules, a test cell may
contain only test functions, test classes or fixtures, and it receives the
values it checks as cell arguments. A cell that mixes a helper with a test is
silently skipped. Comparisons against independent implementations belong in
`src/python/apps/docs/tests/`, which import the notebook's top-level functions.
Each notebook header sets `reactive_tests = false`, so readers in molab do not
see the tests run.

## Reference pages

Reference pages share one structure, adapted from the entries in statistical
software manuals:

1. `# ObjectName` names the class or function being described.
2. `## Usage` shows the shortest complete example.
3. `## Description` explains what the object does and when to use it.
4. `## Options` lists the arguments and choices, usually as a table.
5. `## Results` describes what the call returns and what the result keeps.
6. `## Methods and formulas` gives the estimator and its variance, when useful.
7. `## Assumptions and limits` collects every caveat about the method.
8. `## Evidence` links to the mathematical results, reference comparisons and
   studies that support it.
9. `## See also` points to related pages.

Write reference prose to be scanned. Sentences can be shorter than in a
tutorial, but they should still explain how ideas relate rather than list
nouns.
