# StatLLM AgentBench

A small research-workflow pilot: **Pydantic Evals → Pydantic Coder → devenv-built
OCI workspace → fresh replay → statistical and LaTeX grading**. Two treatments,
STATADIC and statsmodels, share 20 explicitly adapted StatLLM tasks. No bespoke
agent loop, Dockerfile, proxy, web UI, database or LLM judge.

[Environment and commands](../../../../infra/agentbench/README.md) ·
[Rubric](rubric.md) · [Public contract](prompt.md) · [Tasks](tasks.json)

## Boundaries

`core.py` owns preprocessing, schemas, independent NumPy/SciPy witnesses and the
rubric. `run.py` adapts Pydantic's Case/Dataset/Evaluator/CaseLifecycle and sandbox
APIs. `reporting.py` produces CSV, JSON, Markdown and quality–cost scatter plots.
Tests require no credentials; native-framework and OCI tests are separately marked.

The application is deliberately a standalone uv controller project, excluded from
the parent workspace. It is never installed in the STATADIC wheel or agent images.
Treatment builds use the existing scientific lock, not the controller environment.
This is an explicit exception to the usual shared application lock: model-provider
packages should not change the scientific package versions under comparison.

## Tasks and provenance

| IDs | Family | Notes |
| --- | --- | --- |
| PD0003, PD0013, PD0088, PD0096 | ANOVA | Explicit categorical coding, marginal F tests; PD0096 adds LS means |
| PD0007, PD0035, PD0055, PD0068, PD0098, PD0100, PD0107, PD0129, PD0176 | Regression | Ordinary, multiple, quadratic and transformed-response fits |
| PD0119, PD0128, PD0134, PD0135 | GLM | Logistic and Poisson; PD0128 adds adjusted odds-ratio contrasts |
| PD0036, PD0056, PD0142 | Diagnostics | Residuals, leverage, studentization and Cook distance |

Data are fetched only by the explicit `prepare` command from StatLLM commit
`3194677cc1c1590a95e2bf482af57dd7bb37f35a`. Preparation keeps raw CSVs, source URLs,
SHA-256 hashes, normalized complete-case inputs, public schemas and private
witness values. Numeric accounting notation such as `(21.00)` is decoded as
negative, preserving source signs and scales. No upstream datasets or SAS solutions are vendored. Review their
source/licensing terms before downloading or redistributing them. Adaptations,
including corrections of ambiguous source instructions, are in each task's notes.
This is not an official StatLLM score or a reproduction of every SAS output.

Agents receive public inputs, the assigned package and its installed
source/docstrings. `/docs/installed-api.txt` is generated from that package during
image building. Documentation conditions can add the frozen public references
described below. Agents do not receive the checkout's `.agents`, tests or oracles.
The same capability/tool configuration, budgets and preprocessing apply to both
treatments. No multi-agent delegation, repository-context injection or web access.

## Documentation A/B pilot

[The example configuration](../../../../infra/agentbench/config.docs-ab.example.json)
selects the first existing catalog task in each family: PD0003 (ANOVA), PD0007
(OLS), PD0119 (logistic GLM), and PD0036 (diagnostics). They use four distinct
datasets. Two libraries × two documentation conditions × four tasks × three
repetitions produces **48 attempts**, with GPT-6 Luna, medium effort, seed 42,
600 seconds, 50 requests and 200,000 tokens per attempt. The seed shuffles the
complete plan; it does not fix a hosted model's random sampling.

`conditions` replaces the legacy `images` mapping with `minimal` and `reference`,
each containing immutable STATADIC and statsmodels image IDs. Legacy two-image
configurations remain supported. The condition is part of every job and run
identity, and the existing Pydantic Evals runner shuffles and runs the complete
balanced plan. There is no separate agent loop or Skills injection.

All four images contain `/docs/INDEX.md`, root-package `installed-api.txt`, runtime
identity, a license and a hashed corpus manifest. Reference images add grouped
public API documentation generated from the installed package; STATADIC also
includes selected unmodified package reference pages. Source, locks, modules,
objects, files and corpus bytes are frozen with the images. Corpus sizes are
**not matched between libraries**: the experiment compares those documented
bundles, not equal bytes or equal prior model knowledge. It supplies no
benchmark-specific cookbook, private calibration code or expected results.

Both conditions receive the same instruction to read `/docs/INDEX.md` before
implementation. Both use `Coder(unrestricted_filesystem=True)`, so file tools
can read `/input` and `/docs` inside the OCI workspace. The OCI mounts, network restrictions,
packages, public statistical contract, reference calculations and grading remain
the same. Documentation reading consumes the ordinary per-attempt budget.

The previous 20-task, one-repetition campaign is a historical baseline, not the
minimal arm of this new pilot: its documentation/access instructions differ.
All selected tasks have already been observed. This pilot is exploratory;
selection by catalog order does not make it a new confirmatory study. Preserve
all failures and incomplete measurements, and **stop after the pilot**. A
20-task follow-up requires a separate decision and frozen campaign.

## Artifacts and grading

Each attempt must generate one self-contained `solution.py`, full-precision
`results.json` and a short `report.tex` fragment, plus task-specific plots.
Only solution.py is copied into each of two fresh replay containers. The original
answers are not trusted. TeX is compiled in a third clean container with only
report/image artifacts, using untrusted, offline Tectonic.

Pydantic Evals invokes the deterministic rubric. It awards **85 automatic points**;
15 further points require explicit human review for interpretation, report/JSON
agreement, plots and genuine assigned-package use. A final grade remains null
until review. A model that calls another estimator must not win merely by matching
numbers. Inspect retained code and traces; no lexical compliance heuristic is used.

```sh
# From infra/agentbench, after a run:
just bench review campaigns/pilot/runs/run-00000 /absolute/path/review.json
just bench summarize campaigns/pilot
```

Review file format is in rubric.md. Source hashes bind reviews to the submitted
program. Preserve review amendments separately. No agent sees the review files.

## Measurements

Runs retain prompts, framework message traces, model/effort/image identity,
provider usage, elapsed time, replay outputs, compilation logs and grading details.
Credentials stay in the trusted controller. Library/model failures remain records.
Elapsed time covers the agent phase, including its workspace lifecycle, and
excludes independent replay and report compilation.
Native request hooks append preparation, response and step events to each run's
`requests.jsonl`. Usage completeness depends on every prepared attempt returning
measured usage and native response counts agreeing, rather than agent status.
A budget failure after a measured response can have complete usage; an unresolved
or unmeasured request leaves complete totals unavailable. Observed partial counts
and explicit incompleteness reasons remain retained; unknown cost is **not** zero.

Summaries require the full balanced plan. They average repetitions within tasks,
then tasks equally; they do not drop failures or compare different eligible task
sets. Automatic-score plots are labeled /85, reviewed plots /100. Unknown usage
suppresses that cost-frontier point. Grade intervals resample shared-dataset
clusters. Pareto flags are descriptive, not a significance test of dominance.
No benchmark results or superiority claims are supplied by this implementation.

Condition campaigns produce four summary groups (`condition`, `library`,
`effort`). `summary.json`, `.csv` and `.md` retain the ordinary equal-task grades,
strict complete token/cost means, elapsed time and descriptive cluster intervals.
Quality–cost SVGs include condition labels and omit unavailable points. New
`per-task` and `attempts` JSON/CSV/Markdown tables retain every planned repetition,
automatic component points, null or reviewed final grades, completion, replay,
TeX, usage completeness and observed input/cache/output counts. Observed subtotals
can include partial runs; they are not complete token totals or dollar costs.
Subscription runs with null pricing retain unavailable estimated costs.

`condition-differences` and `task-condition-differences` tables report reference
minus minimal within each library/effort, averaging repetitions before tasks.
Unknown full means produce unknown differences. These are descriptive comparisons;
four selected datasets and three repetitions do not establish superiority.
Attempts remain in their assigned condition even when they never read docs.

Failure categories use precedence: token/request/unknown usage limit, timeout,
other agent/tool error, replay/contract, LaTeX, numerical mismatch, then none.
Per-attempt flags and failed quantities preserve additional failures. Trace
counters count `/docs` path mentions, associated returns, explicit root denials,
introspection markers and returns of at least 10,000 characters. Missing traces
remain unavailable. A returned tool call does not prove a document was read,
and no counter or matching number adjudicates assigned-package compliance.
Human-review fields and final /100 scores remain null until actual review.

## Checks

```sh
# From this directory, after explicit setup/lock resolution:
uv run --locked --offline pytest tests/test_core.py
uv run --locked --offline pytest tests/test_integration.py
# OCI smoke, no model calls:
AGENTBENCH_STATADIC_IMAGE=sha256:... AGENTBENCH_STATSMODELS_IMAGE=sha256:... \
  uv run --locked --offline pytest tests/test_integration.py -m integration
```

Before publishing a study, calibrate native solutions for BOTH packages on all
20 prepared datasets, including marginal means, contrasts and small-sample
conventions. Synthetic oracle tests are not a substitute for that calibration.
Do not change tolerances after seeing model scores without a new experiment.

The controller requires Pydantic AI/Evals 2.68.0 and Harness 0.68.0. A Codex-backed
Pydantic Coder is not the Codex CLI; label the evaluated agent accurately. A pinned
runtime cannot make a hosted model or its billing perfectly deterministic.
