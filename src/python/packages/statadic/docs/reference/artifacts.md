# Saved artifacts and reproducibility

STATADIC separates prediction, rendering and reproduction into three tiers:

| Tier | Written by | Restores | Does not restore |
| --- | --- | --- | --- |
| Predictor | Fitted result's `save_predictor`; `save_predictor` and `save_ordination_predictor` helpers | Prediction or transformation with the declared roles | Inference, covariance or refitting |
| Analysis snapshot | `save_analysis` | Report values, warnings, notes and plots for rendering | A fit or sampling uncertainty |
| Reproduction | `statadic-publishing` companion | Source, lockfile, data identity and rebuild instructions | A restored runtime fit |

Analysis snapshots containing TeX markup require explicit `trust_tex=True` when
calling `load_analysis`.

Current artifacts use schema 6. Prediction kinds are derived from the approved
family rather than repeated in the saved state. Fixed wire records validate
required fields, discriminants and values before numerical reconstruction.
Kernel bandwidths retain their checked scalar payload, dtype and JAX weak-scalar
status, so loading preserves promotion against feature and measure arrays.
Malformed reconstructed predictor state raises `ArtifactError` with its cause.

Before a public release, a reader accepts only its own schema version; any change
to persisted content bumps that version. There is no migration: re-save from the
original analysis. Each manifest records the writer version for diagnosis only;
readers do not compare it. The post-release compatibility promise requires the
owner's confirmation in publication hold.
