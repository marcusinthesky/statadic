---
type: task
title: Independently review every new or changed claim
description: "A reviewer who wrote none of the proofs checks each new or changed claim for statement fidelity, satisfiable and honest hypotheses, definition shadowing, integrity, source fidelity and Python scope, before any re-pin."
links: [".context/plan/release-guarantees/handoff.md", ".context/plan/release-guarantees/evidence.md"]
status: in-progress
---

# Independently review every new or changed claim

## Goal

The certify gate rejects admissions, custom axioms and `True` placeholders. It cannot tell whether:

- the LaTeX says what the Lean proves;
- the hypotheses can hold, and are honest about what they assume;
- a statement uses a home-made version of a standard concept.

This review is that check.

## Depends on

The four proof lanes must be frozen, and the coordinator must have merged their claim drafts. The reviewer must not have written any of the proofs reviewed.

## Owns

Only this task's `verification.md`. Everything else is read-only. Findings go to the coordinator, who returns them to the owning lane. Each fixed claim is reviewed again.

## Checklist for every new or changed claim

1. **Statement.** Every quantity, quantifier, hypothesis and conclusion in the LaTeX appears in the Lean statement, and the reverse. STATADIC-LINEAR-005 is the known past failure.
2. **Satisfiable.** Give a concrete instance in which all hypotheses hold. Use prose, or a Lean `example` if one is short.
3. **No hidden main step.** No hypothesis assumes the theorem's main step. Each hypothesis with real content is listed in `scope.limitations`.
4. **Definitions.** Print every constant in the statement with full names, for example `set_option pp.fullNames true` and `#check`. Each must be from StatadicFormal or mathlib, and none may duplicate an existing concept.
5. **Integrity.**
   - `#print axioms` on each claimed declaration shows only the three standard axioms.
   - No statement or premise is `True`.
   - The new files contain no `macro`, `elab`, `syntax` or `set_option` lines.
6. **Proof sketch.** The LaTeX proof sketch follows the Lean proof.
7. **Source.** Open the cited source at the recorded locator. Confirm the claim is a faithful specialization, and record every difference.
8. **Python scope.**
   - The relationship text is accurate.
   - The limitations name what is not verified: floating point, PRNG, selection, non-finite input and exponent.
9. **Port.** The pricing-perspective port contains none of the placeholders or duplicate definitions listed in the [evidence](../../../../evidence.md#lean).

## Output

Write one table per claim, giving pass or fail and a note for each item. List the open findings separately, with the owning lane for each.

## Acceptance

- [ ] Every new or changed claim, including STATADIC-LINEAR-005, passes all nine items, or has an exception approved by the owner and recorded here.
- [ ] Every claim fixed after a finding was reviewed again.
- [ ] The reviewer wrote none of the reviewed proofs.

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

_No declared backlinks._

<!-- insitu:end -->
