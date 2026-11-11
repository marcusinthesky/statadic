---
type: "article-journal"
title: "A Kernel Two-Sample Test"
description: "Source for the finite-moment MMD U reduction; iid sampling and RKHS foundations remain open."
author: [{"family": "Gretton", "given": "Arthur"}, {"family": "Borgwardt", "given": "Karsten M."}, {"family": "Rasch", "given": "Malte J."}, {"family": "Schölkopf", "given": "Bernhard"}, {"family": "Smola", "given": "Alexander"}]
issued: {"date-parts": [[2012]]}
container-title: "Journal of Machine Learning Research"
volume: "13"
issue: "25"
page: "723-773"
URL: "https://www.jmlr.org/papers/v13/gretton12a.html"
links: []
---

# A Kernel Two-Sample Test

Publication metadata was checked against <https://www.jmlr.org/papers/v13/gretton12a.html>.
The integrating agent compared Lemma 6, equation (3), p. 728, and its
empirical-estimator proof step, p. 729, with the finite-law reduction in
`STATADIC-MMD-001`. Human semantic review remains pending; current
compilation and axiom status come from native certification.

The proof assumes common pair expectations instead of deriving them from iid
sampling. It does not construct the RKHS embeddings or prove separation or
calibration. The accompanying V-bias identity is a supporting diagonal-counting
consequence of equation (5), p. 729, rather than a separate theorem attributed to
the paper. See [source mapping](source.json) for exact locations.

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Integrate finite-moment MMD reference proofs](../../plan/formal-literature/tasks/literature-proofs/tasks/mmd-finite-moments/README.md)
- [Prove weighted V-MMD nonnegativity and the U contrast](../../plan/release-guarantees/tasks/proofs/tasks/weighted-mmd/README.md)

<!-- insitu:end -->
