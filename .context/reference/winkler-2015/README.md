---
type: article-journal
title: Multi-level block permutation
description: Exchangeability blocks for dependent permutation designs.
author:
  - family: Winkler
    given: Anderson M.
  - family: Webster
    given: Matthew A.
  - family: Vidaurre
    given: Diego
  - family: Nichols
    given: Thomas E.
  - family: Smith
    given: Stephen M.
issued:
  date-parts: [[2015]]
container-title: NeuroImage
volume: '123'
page: 253-268
DOI: 10.1016/j.neuroimage.2015.05.092
URL: https://doi.org/10.1016/j.neuroimage.2015.05.092
links: []
---

# Multi-level block permutation

Defines exchangeability blocks and the permutations they admit: shuffling
within blocks, or moving whole blocks intact, nested over several levels.
STATADIC's `PermutationPlan(blocks=...)` follows the within-block case;
`unit=...` reassigns labels among whole assignment units, which need not be
equal in size because labels, not rows, move. Whole-block shuffles of
equal-size blocks with preserved internal order are not implemented.

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
