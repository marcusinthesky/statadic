# Review brief

## Contract and identity

The owner authorized implementation of the review programme on 2026-11-09.
Candidate: `17fdc5410de31356df72d6fc6459853ca39260f9`.
Merge base: `52268aa5078d94c9bf8b092c7655fbf3ae54031a`.
[The stack](stack.json) contains PRs 4, 5, 6, 7, 8, 9, 11 and 12.
The net tracked diff has 1,433 paths, 143,023 insertions and 19,119 deletions.
Review records added by this programme are outside that frozen candidate.

Scope is the whole shipped stack. Output is evidence, adjudicated dispositions
and an implementation-ready repair backlog. Do not implement production repairs,
delete code, optimize numerical methods, commit, push, post or publish under this
review. Long campaigns follow the scope and corrective-source freeze.

## Governing decisions

Use [the decision ledger](decisions.md). Explicit current owner instructions
control; earlier assistant sketches and implementation summaries are evidence to
inspect, not cumulative requirements. Record contradictions and do not silently
reopen settled choices. Architectural assertions also require implementation
inspection. A finding is not closed by a passing source inventory.

The coordinator owns shared contracts, lifecycle, public-surface cohesion,
dependency direction and final dispositions. Reviewers propose changes. Human
semantic review cannot be completed by an agent or inferred from certification.

## Execution

Use one coordinator and three leaf reviewers on the same bounded slice.
Pass A: code/tests; sources/proofs; independent evidence/workflow/documentation.
Pass B: JAX/performance/memory; aggressive simplification; independent challenge.
Rotate reviewers before sign-off. The review-agent skill applies to defect-first
leaf reviews only; its no-delegation rule does not apply to the coordinator.

Each assignment declares exact paths, operations, revisions, decision IDs,
evidence questions and exclusions. Read the full assigned authored diff and
enough callers/tests. Generated views are checked through their owning sources,
hashes and generation contracts. Report unexamined scope explicitly.

Workers do not execute test suites. They propose exact named paths or bounded
reproducers. The coordinator serializes Python/test/benchmark execution through
the pinned devenv and locked uv projects. Full suites remain CI-only.

## Evidence and findings

One operation may have many substantive variants. Sources, Lean, independent
numerical references, Monte Carlo outcomes, JAX execution and design quality
retain separate statuses. A catalogue mapping is registration, not qualification.
Stale, absent, adverse, inconclusive and not-applicable-with-reason are distinct.
Aliases/classes/host wrappers do not inherit mathematical evidence automatically.

Each finding has an ID, candidate, scope, category/severity, concrete trigger,
expected contract, observed behavior, source location, evidence and limits,
proposed disposition, challenge, verification and repair dependencies.
Introduced defects, pre-existing release risks, design decisions and missing
evidence remain distinguishable.

## Performance and simplification

Primary winning regime: repeated numerical fitting, batching/resampling and
derivative objectives on available CPU hardware. Cold public calls and total
process memory remain separately reported. No unmeasured ARM/GPU claim follows.
Correctness, matched outputs/conventions and independent trial identities precede
timing. Warm superiority requires every one of at least three independent worker
median ratios below one; mixed ordering is inconclusive. Memory comparisons must
use the same measured quantity. Compiler estimates cannot be compared with a
reference process's RSS. Joint superiority requires both gates.

Review grad, JVP/VJP, Hessian/HVP, jit, vmap and lax choices where meaningful,
including objective/finite-iterate/optimal-value/fitted-solution distinctions.
Do not change precision, acceptance, objectives or sampling to produce a win.

Every dead/duplicate/verbose/misowned candidate gets delete, merge, move,
simplify or retain-with-reason. Public export alone does not justify retention.
Inspect registries, protocols, codecs and downstream use before declaring dead
code. Preserve independent oracles and real alternative numerical strategies.
Line-count reduction alone is not acceptance.

## Completion

Every changed path and scientific operation must be accounted for. A slice needs
both passes, an independent challenge and coordinator dispositions. A pending
inspection or unverified hypothesis cannot be relabelled completed coverage.
Review completion may still recommend no alpha release. Required production
repairs, human approvals and frozen-candidate campaigns remain separate gates.
