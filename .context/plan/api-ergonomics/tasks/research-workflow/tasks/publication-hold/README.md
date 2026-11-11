---
type: task
title: Hold the software paper and public release
description: "Publication is on hold by owner decision. Records what must be true before any paper or public release."
links: [".context/plan/api-ergonomics/tasks/research-workflow/handoff.md", ".context/plan/api-ergonomics/tasks/research-workflow/evidence.md", ".context/plan/release-experience/publishing-design.md", ".context/plan/publishing-companion/README.md"]
status: blocked
---

# Hold the software paper and public release

## Goal

The owner is holding off publication of the library. Keep the known prerequisites in one place, and take no public or irreversible step under this work.

## Acceptance

- [ ] The owner lifts the hold and chooses a venue.
- [ ] A human reviews all 8 Lean claims, whose review fields say review is pending. A model cannot complete this.
- [ ] The `publishing-companion` OLS/HC3 slice exists and can serve as replication material.
- [ ] The release is frozen, retained qualification campaigns have run, and the cold hosted release gate passes.
- [ ] The repository is public, with more than six months of public history if targeting JOSS.
- [ ] The owner confirms the AI usage disclosure drafted in `declarations.tex` and adds a human-validation statement.
- [ ] Paper claims are regenerated and checked against the frozen revision.
- [ ] The [walkthrough](../../../../../research-release/tasks/walkthroughs/README.md) lane passes on the frozen wheel.
- [ ] The owner confirms the post-release artifact compatibility promise on the page written by [artifact-contract](../../../../../research-release/tasks/artifact-contract/README.md).
- [ ] The [public-surface](../../../../../public-surface/README.md) plan is complete: every public module declares `__all__`, and the inventory enforces it.

## Do not

Make the repository public, tag a release, mint a DOI, upload a package or submit anywhere.

## Manuscript draft

At the owner's request, an arXiv-style draft exists in `src/latex/projects/01_statadic`. Its analysis script executes the printed listings and regenerates every number, so it records the working tree it ran on rather than a frozen release. `just latex::bundle 01_statadic arxiv` builds a source archive, but nothing has been submitted. Before submission the owner must confirm the affiliation, the AI disclosure and the missing funding and competing-interest statements. JOSS needs a short `paper.md` instead of this article.

## Notes for when the hold lifts

- **Positioning:** "publication-native" describes a companion that is still in backlog.
- **Lean:** do not argue that Lean proofs will become mandatory. No statistics or economics journal requirement was found. Lean proves stated mathematics, not JAX execution.
- **Accelerators:** the docs state there are no GPU measurements, so measure before claiming support.
- **Repairs:** the repair record says five further review findings were never supplied. Resolve this before citing that review.

The companion's first slice is now planned under the [research-release plan](../../../../../research-release/README.md). Until its replication task passes, do not present a publication-native workflow as shipped. If the first public release is only the numerical `statadic` package, the companion does not block it.

Follow the [handoff guide](../../handoff.md). Measurements are in [evidence](../../evidence.md).

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Decide repository hygiene before publication](../../../../../api-cohesion/tasks/followups/tasks/publication-hygiene/README.md)
- [Tighten the public import surface before release](../../../../../public-surface/README.md)
- [Prove one coherent research workflow before release](../../../../../research-release/README.md)

<!-- insitu:end -->
