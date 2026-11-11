# Public diagnostic reference journeys

Two new live comparisons are authored in
`tests/integration/api/test_spatial_reference_walkthroughs.py`:

- `test_ols_moran_lm_public_journey_matches_spreg`: canonical Regression.ols,
  named observations and explicit graph requests compared with independently
  constructed spreg.OLS/MoranRes inputs. Checks all five LM statistics/tails and
  residual Moran I, its adjusted expectation/variance and normal tail. The retained
  core fit identity is unchanged.
- `test_spatial_iv_ak_public_journey_matches_spreg`: canonical spatial Regression,
  named SpatialData and the actual SpatialIV fit compared with independently
  constructed spreg.GM_Lag/AKtest(case="gen"). Checks statistic/tail and retained
  fit identity.

Canonical mappings extend existing spatial diagnostic variant records;
they add no scientific variants or source approvals. The first node supports
`moran-residual-normal`, `lag-lm`, `error-lm`, `robust-lag-lm`,
`robust-error-lm`, and `joint-sarma-lm` under `spatial.diagnostics`. The second
supports `anselin-kelejian`. Their lane is external-float64. This does not extend
to learned graphs, panel diagnostics, arbitrary heteroskedasticity or weak-IV
robust inference.

The initial static pass exposed a missing SpatialIV constructor overload; its
canonical owner repaired the constructor rather than hiding the mismatch with
a cast. The coordinator subsequently reported the public journeys passing.

The second wave retains exact float32/float64 array cells for public unit fixed
effects, named 2SLS and training-only PCA composed with OLS. Their independent
peers are linearmodels, scikit-learn and statsmodels. These cells run inside an
x64-enabled process and therefore belong to the external-float64 lane; they are
not declarations for the separate precision-only test command. The PCoA signed
spectrum/table control is an independent NumPy formula mapping with no external
execution lane and does not retire the legacy embedding carrier.

Two bounded missing comparisons were added after inspecting installed peer source:

- `tests/discrepancy/mmd/test_energy_independent_reference.py::test_euclidean_v_test_statistic_matches_dcor_scaling`
  has exact exponent cells 0.5, 1.0 and 1.5. It compares the scaled two-sample
  statistic on unequal counts, complementing the existing raw V-energy comparison.
- `tests/discrepancy/transport/test_sinkhorn.py::test_product_mass_kl_cost_and_debiased_geometry_match_pot`
  has exact float32/float64 input cells. POT supplies a checked transport plan;
  explicit product-mass KL terms match the native regularized objective. The
  transport-only `sinkhorn2` scalar would compare a different quantity. Both
  debiased squared and rooted public geometry outputs are checked at fixed
  epsilon and iteration budget.

The coordinator reported 38 passing tests in the bounded R07/reference wave,
including the new controls and eight structural fixtures. This arm ran no
numerical tests, campaigns or qualification.

The public inventory was explicitly regenerated with `just lean::inventory-refresh`
after the source freeze. `statadic-studies check --repository .` then passed with
2486 public symbols, 216 method records, 51 scientific procedures and 309 variants.
The index now maps 96 variants and 18 method records. Seventeen new bounded
declarations preserve exact configurations and quantities. Existing source-review,
statistical evaluation and resource statuses remain unchanged; execution is never
inferred from mappings. A repeated-JSON patch-placement error was caught by the
catalogue validator and corrected using complete uniquely identified records
before the passing reconciliation.
