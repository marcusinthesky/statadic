# Observations, identities and four different kinds of weight

The first question is “What is one statistical observation?” In a scalar regression, it may be one row. In a distribution-valued regression, it is still one outer row, even though that row contains many atoms. In a dyadic problem, shared endpoints introduce a different sampling structure.

## Identity before shape

Equal array lengths do not prove that values belong to the same rows. `Observations` keeps numeric, categorical and measure roles aligned by observation identity. A `MeasureColumn` must carry exactly the outer row identities in order. Selection applies across roles together; it is not independent column-wise deletion of missing values.

Prediction retains learned design state and column schema. Row identities identify observations; parameter names identify design columns. Neither recovers an ordering that the caller has already discarded.

## Weight meanings are not interchangeable

| Weight | What it weights | What it does not mean |
|---|---|---|
| Empirical-measure atom mass | Probability attached to an atom inside a whole observation | Extra regression sample size or precision |
| Relative-precision WLS weight | The conditional error variance is proportional to its inverse | Frequency, survey weight or known absolute variance |
| Graph weight | The declared spatial operator after its chosen normalization | Atom probability or a regression precision |
| Weight over candidate measures/mixtures | Contribution of a whole measure to a composed objective | Automatically the number of independent observations |

Omitted empirical atom masses are uniform. Supplied masses must obey the declared normalization contract; requesting normalization is explicit. Padding is storage, not observed data or probability mass.

WLS's common scale is estimated using its retained sample and precision-weighted residuals. Rescaling every relative precision by the same positive factor does not create additional information; coefficients and estimated-scale covariance retain their documented invariance.

## A practical check

Before fitting, write down the unit of observation, row identifiers, feature order, missingness policy, weight interpretation and independent sampling unit. The library can reject several incompatible shapes or schemas. It cannot establish that your economic or scientific identification assumptions are true.
