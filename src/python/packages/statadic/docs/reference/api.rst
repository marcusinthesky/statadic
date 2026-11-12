Selected Python API
===================

These public entry points support the introductory tutorials. Follow the linked family guides for assumptions, shapes, limits and result diagnostics.

Data and design
---------------

.. autoclass:: statadic.api.data.Observations
   :members: take, numeric, categorical, resolve_grouping

.. autoclass:: statadic.api.data.ColumnSource

.. py:type:: statadic.api.roles.InputRole

   One of ``numeric``, ``categorical`` or ``measure``.

.. autoclass:: statadic.api.roles.InputRoleProvider

.. autofunction:: statadic.api.roles.merge_input_roles

.. autoclass:: statadic.api.data.MeasureColumn

.. autoclass:: statadic.api.data.RelativePrecision

.. autoclass:: statadic.api.design.Intercept

.. autoclass:: statadic.api.design.Linear

.. autoclass:: statadic.api.design.DesignContrast

.. autoclass:: statadic.api.design.LinearPredictorContrast

.. autoclass:: statadic.api.design.DistanceToLandmarks

Regression and uncertainty
--------------------------

.. autoclass:: statadic.api.regression.Observed

.. autoclass:: statadic.api.regression.Gaussian

.. autoclass:: statadic.api.survival.RightCensored

.. autoclass:: statadic.api.data.PanelLayout

.. autoclass:: statadic.api.dyadic.Dyads

.. autoclass:: statadic.api.dyadic.EndpointEffects

.. autoclass:: statadic.api.regression.Regression
   :members: ols, wls, logit, probit, poisson, aft, estimate

.. py:type:: statadic.api.regression.RegressionEstimator

   A compatible native estimator for the selected scientific family.

.. py:type:: statadic.api.regression.RegressionFit

   The supported family-specific result union; capabilities depend on the actual fit.

.. py:type:: statadic.model.survival.AFTFamily

   A declared Weibull, lognormal or log-logistic accelerated failure-time law.

.. py:class:: statadic.optimize.integration.IntegrationDesign
   :canonical: statadic.optimize.integration.gaussian.IntegrationDesign

   A replayable key or supplied integration points with fixed draw and replicate
   counts. See the :doc:`spatial survival guide <survival>` for scope.

.. autoclass:: statadic.api.regression.LinearResult
   :members: diagnose, predict, infer

.. py:class:: statadic.api.regression.LogisticResult
   :canonical: statadic.api.regression._glm.LogisticResult

   Binomial-logit fit with retained design, prediction and explicit covariance choices.

.. py:class:: statadic.api.regression.PoissonResult
   :canonical: statadic.api.regression._glm.PoissonResult

   Poisson-log fit with retained exposure, offset and prediction targets.

.. py:class:: statadic.api.regression.LikelihoodResult
   :canonical: statadic.api.regression._likelihood.LikelihoodResult

   Probit, Gamma or NB2 fit; numerical validity and model-based inference remain separate.

.. py:class:: statadic.api.survival.SurvivalResult
   :canonical: statadic.api.survival._model.SurvivalResult

   Independent AFT fit with the declared censoring, family and supported predictions.

.. py:class:: statadic.api.fit_statistics.FitStatistics

   Fit-quality metrics and the notes that state their conventions, from ``fit.fit_statistics()``.

.. py:class:: statadic.api.fit_statistics.LikelihoodSummary

   Maximised log likelihood, parameter count and observation count that a family supplies for its criteria.

.. py:class:: statadic.api.capabilities.ReportsFitStatistics

   Protocol for results that return ``FitStatistics`` from ``fit_statistics()``.

.. py:class:: statadic.api.capabilities.SummarizesLikelihood

   Protocol for results that return a ``LikelihoodSummary`` from ``likelihood_summary()``.

.. py:function:: statadic.api.fit_statistics.information_metrics

   Build ``n_parameters``, ``log_likelihood``, AIC, BIC and HQIC from one log likelihood.

.. py:class:: statadic.api.base.PreparedRegression

   Frozen row-wise design and observations, bound to the preparing model instance.

.. py:class:: statadic.api.survival.PreparedSurvival
   :canonical: statadic.api.survival._model.PreparedSurvival

   Frozen AFT inputs, including the observation mechanism and fitted predictor design.

.. automodule:: statadic.api.diagnostics
   :members: JarqueBera, BreuschPagan, White, DurbinWatson, BreuschGodfrey, RESET, HansenJ, Influence, Diagnostics, InfluenceDiagnostic, LinearDiagnostic, IVDiagnostic, MomentsDiagnostic, RetainedLinearFit, RetainedIVFit, RetainedMomentsFit

.. automodule:: statadic.api.hypotheses
   :members: Hypothesis, Proposition, LinearRestriction, ConstantVariance, GaussianMoments, NoSerialCorrelation, NoAddedFittedPowerEffect, OveridentifyingRestrictions, ChiSquare, FisherF, TestOutcome

.. py:currentmodule:: None

.. py:type:: statadic.api.hypotheses.Reference

   A ``ChiSquare`` or ``FisherF`` reference attached to an actual calibrated tail.

.. py:type:: statadic.api.reporting.MetadataValue

   Explicit JSON-compatible report metadata, including nested mappings and sequences.

.. py:type:: statadic.api.reporting.FrozenMetadataValue

   The recursively immutable representation retained by a report snapshot.

.. py:type:: statadic.api.reporting.ReportNumber

   An integer or floating-point report value.

.. autoclass:: statadic.api.inference.InferenceContext

.. py:class:: statadic.estimation.iv.IVFit

   Native structural IV fit retaining the actual last weighting solve, original
   aligned rows and weighting/convergence provenance.

.. py:class:: statadic.estimation.gmm.AffineGMMFit

   Native affine-GMM fit retaining the generating weight, final contributions
   and variability scheme. Point-fit validity and J calibration remain separate.

.. py:class:: statadic.core.equations.VariabilityScheme

   Nominal axis distinguishing variability constructions; a scheme name alone
   establishes neither source correspondence nor statistical validity.

.. py:type:: statadic.api.diagnostics.Diagnostic

   One diagnostic outcome: a ``TestOutcome`` or an ``InfluenceDiagnostic``.

.. autoclass:: statadic.estimation.linear.LeastSquares

.. autoclass:: statadic.estimation.glm.BinomialLogitMLE

.. autoclass:: statadic.estimation.glm.PoissonMLE

.. autoclass:: statadic.estimation.likelihood.ProbitMLE

.. autoclass:: statadic.estimation.survival.SurvivalML

.. autoclass:: statadic.api.inference.HC

.. autoclass:: statadic.api.inference.HAC

.. py:function:: statadic.inference.linear_covariance.hc_covariance(fit, *, kind="HC0", finite_sample_correction=1.0)

   Estimate HC0–HC3 covariance from the fit's whitened influences. HC1 applies
   the count correction; HC2 and HC3 require leverage below one. The optional
   positive correction is separate. Choosing a covariance does not choose a
   reference distribution or establish interval coverage.

A ``LinearResult`` retains its numerical fit as ``core``. The tutorials inspect
``core.valid`` before interpreting ``core.parameters``, then inspect the interval's
own validity after requesting inference. See the :doc:`regression guide <regression>`.

Reports and retained plots
--------------------------

.. autoclass:: statadic.api.reporting.Report
   :members: to_text, to_html, to_latex, save_latex, to_values, save_values, coefficient_plot, require_available

.. autoclass:: statadic.api.reporting.ParameterRow

.. autoclass:: statadic.api.reporting.TestRow

.. autoclass:: statadic.api.reporting.Metric

.. autoclass:: statadic.api.reporting.PlotSpec
   :members: to_pgfplots, save_pgfplots

.. autoclass:: statadic.api.reporting.ReferenceLine

These objects retain already-computed values. Analysis snapshots restore
rendering, while predictor artifacts restore only their declared prediction
capabilities. See :doc:`reporting <reporting>` and :doc:`artifacts <artifacts>`.

.. py:class:: statadic.api.artifacts.PredictionArtifact

   A prediction-only artifact restored by ``load_predictor(path)``. It keeps the
   fitted design and coefficients needed for compatible new observations.

   .. py:method:: predict(data, *, kind=None)

      Apply the saved design and prediction convention to new observations.
      The artifact does not restore live fitted inference.

Quantities and supplied moments
-------------------------------

.. py:class:: statadic.api.estimand.Estimand(quantity, *, estimator)

   Prepare and estimate a declared covariance or supplied-moment quantity.
   The actual covariance, GMM or GEL result retains its family capabilities.
   See :doc:`estimand <estimand>` for inputs and inference requests.

.. py:class:: statadic.api.estimand.Covariance(columns=None)

   Covariance of an ordered set of observed variables; no inference is implied.

.. py:class:: statadic.api.estimand.MomentConditions()

   Supplied fixed or affine moments; sample arrays belong to ``MomentData``.

.. py:class:: statadic.estimation.covariance.CovarianceProblem(values)

   Native rectangular numerical input to a covariance estimator.

.. py:class:: statadic.estimation.covariance.SampleCovariance(ddof=1)

   Numerical sample covariance with divisor ``n - ddof``.

.. py:class:: statadic.estimation.covariance.LedoitWolf()

   Constant-correlation Ledoit--Wolf covariance brick.

.. py:class:: statadic.api.moments.GMMSandwich(finite_sample_correction=1.0)

   Explicit actual-fit GMM sampling covariance request.

.. py:class:: statadic.api.moments.GELCalibration(kind="lr", assume_iid=False)

   Bound GEL statistic/reference request; the hypothesis is evaluated at ``test``.

Distributions, embedding and comparison
---------------------------------------

.. autofunction:: statadic.discrepancy.mmd.energy.energy_distance

.. autoclass:: statadic.geometry.ground.metrics.EuclideanLaw

.. autoclass:: statadic.api.representation.Energy

.. autoclass:: statadic.api.representation.MMD

.. autoclass:: statadic.api.ordination.Ordination
   :members: pca, pcoa, estimate

.. autoclass:: statadic.api.ordination.PrincipalCoordinates

.. py:class:: statadic.api.ordination_terms.OrdinationEvidence

   Immutable actual native fit, training context and retained coordinate map.

.. py:class:: statadic.api.ordination_terms.FittedOrdination

   A fitted predictor term with named inspection through
   ``fit.design.ordination(name)``. Prediction-only artifacts omit inspection.

.. autoclass:: statadic.api.representation.ObservedDissimilarity

.. autoclass:: statadic.api.regression.GeometricResponse

.. autoclass:: statadic.api.regression.Reduced

.. autoclass:: statadic.api.regression.FreedmanLane

.. autoclass:: statadic.api.comparison.PermutationPlan

.. py:type:: statadic.api.comparison.ComparisonInput

   Aligned prepared geometry or a supported supplied matrix for group comparison.

.. py:type:: statadic.api.comparison.DesignLabels

   A grouping column name or explicitly aligned integer/string labels.

.. autofunction:: statadic.discrepancy.mmd.mmd_squared

.. autofunction:: statadic.discrepancy.mmd.mmd_squared_matrix

.. autofunction:: statadic.discrepancy.mmd.weighted.weighted_mmd_squared_cross

.. autofunction:: statadic.inference.dispersion.dispersion_permutation_null

.. py:function:: statadic.inference.resampling.permutation.permutation_test_kernel(statistic, null_distribution, num_permutations, alternative="two-sided")

   Assemble a numerical test result from an observed statistic and supplied
   null draws. Alternatives are greater, less and absolute-value two-sided,
   with inclusive ties and add-one calibration. Any non-finite statistic or
   draw makes the p-value unavailable. The caller supplies the sampling scheme;
   this kernel does not establish its exchangeability.

A fitted Ordination result retains coordinates, signed inertia and its supported
frozen query map. Geometric Regression requests inference separately through
``ResponsePermutation`` or ``FreedmanLane`` with an explicit ``PermutationPlan``.
Inspect validity, calibration status and failed draws before interpreting a
report. See :doc:`ordination <ordination>`, :doc:`energy <energy>`,
:doc:`resampling <resampling>` and the
:doc:`composition guide <composition>`.
