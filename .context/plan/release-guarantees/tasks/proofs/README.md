---
type: task
title: Prove the first release guarantees in Lean
description: "Four finite exact-real proof lanes (sampled permutations, weighted V-MMD, WLS and HC algebra, Euclidean negative type), then an independent review before any statement is re-pinned."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md", ".context/plan/formal-literature/tasks/literature-proofs/README.md"]
status: in-progress
---

# Prove the first release guarantees in Lean

## Goal

Prove the first headline guarantees that users rely on as finite, exact-real Lean theorems. Record each one's scope honestly, and close or narrow named obligations in `registry/methods.json`.

Each lane follows the Lean rules in the [handoff](../../handoff.md#settled-decisions). The integrity gate cannot see two failures: hypotheses that assume the main step, and LaTeX that differs from the Lean statement. [claim-review](tasks/claim-review/README.md) is the check for both.

## Acceptance

- [ ] [sampled-permutation](tasks/sampled-permutation/README.md), [weighted-mmd](tasks/weighted-mmd/README.md), [linear-covariance](tasks/linear-covariance/README.md) and [negative-type](tasks/negative-type/README.md) are complete.
- [ ] [claim-review](tasks/claim-review/README.md) signed off every new or changed claim.
- [ ] The coordinator:
  - merged the claim drafts;
  - added root imports;
  - mapped each claim to an obligation only where the claim's scope covers that obligation's title, with every other relationship recorded as partial;
  - re-pinned statements only after sign-off.
- [ ] `just lean::certify`, `just lean::registry-test` and `just lean::check` pass.

## Order and ownership

| Wave | Work | Owns | Target obligation |
| --- | --- | --- | --- |
| 1 | [sampled-permutation](tasks/sampled-permutation/README.md) | `StatadicFormal/Inference/*` | `permutation-002` |
| 1 | [weighted-mmd](tasks/weighted-mmd/README.md) | `StatadicFormal/Discrepancy/MMD/*` | `mmd-001`, nonnegativity part |
| 1 | [linear-covariance](tasks/linear-covariance/README.md) | `StatadicFormal/Estimation/Linear/*` | `linear-002`; `linear-covariance-001`, HC part |
| 1 | [negative-type](tasks/negative-type/README.md) | `StatadicFormal/Discrepancy/Energy/*` | `energy-002`, Euclidean part |
| 2 | [claim-review](tasks/claim-review/README.md) | Its own `verification.md` only | None |

`StatadicFormal/Probability/*` is shared and read-only. A lane that needs a new `FiniteLaw` lemma adds it in its own module.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

| Status | Item | Description |
| --- | --- | --- |
| in-progress | [Independently review every new or changed claim](tasks/claim-review/README.md) | A reviewer who wrote none of the proofs checks each new or changed claim for statement fidelity, satisfiable and honest hypotheses, definition shadowing, integrity, source fidelity and Python scope, before any re-pin. |
| in-progress | [Prove WLS whitening and the HC covariance algebra](tasks/linear-covariance/README.md) | Close the STATADIC-LINEAR-005 statement gap, prove WLS as whitened OLS with Aitken optimality from the existing claims, and prove the sandwich and leverage facts that HC0-HC3 rely on. |
| in-progress | [Port Euclidean negative type and energy nonnegativity](tasks/negative-type/README.md) | Port pricing-perspective's sorry-free Euclidean negative-type proof without its duplicate definitions or placeholders, derive finite Euclidean energy nonnegativity, and replace the dangling docstring reference with the claim. |
| in-progress | [Prove sampled-permutation add-one validity](tasks/sampled-permutation/README.md) | Prove that the add-one p-value from the identity plus B independent uniform group draws has P(p <= alpha) <= alpha under an invariant finite law, with inclusive ties and all three alternatives. |
| in-progress | [Prove weighted V-MMD nonnegativity and the U contrast](tasks/weighted-mmd/README.md) | Prove the weighted V-MMD nonnegative for linear and PSD kernels on a finite pooled support, show the U estimator can be negative, and show zero-weight padding leaves V unchanged. |

`5 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
