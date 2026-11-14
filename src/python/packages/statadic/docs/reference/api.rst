Selected Python API
===================

These public entry points support the introductory tutorials. Follow the linked family guides for assumptions, shapes, limits and result diagnostics.

Data and design
---------------

.. autoclass:: statadic.observation.data.Observations
   :members: take, numeric, categorical, resolve_grouping

.. autoclass:: statadic.observation.data.ColumnSource

.. py:type:: statadic.observation.roles.InputRole

   One of ``numeric``, ``categorical`` or ``measure``.

.. autoclass:: statadic.observation.roles.InputRoleProvider

.. autofunction:: statadic.observation.roles.merge_input_roles

.. autoclass:: statadic.observation.data.MeasureColumn

.. autoclass:: statadic.observation.data.RelativePrecision

.. autoclass:: statadic.observation.design.Intercept

.. autoclass:: statadic.observation.design.Linear

.. autoclass:: statadic.observation.design.DesignContrast

.. autoclass:: statadic.observation.design.LinearPredictorContrast

.. autoclass:: statadic.observation.design.DistanceToLandmarks

Regression and uncertainty
--------------------------

.. autoclass:: statadic.regression.Observed

.. autoclass:: statadic.regression.Gaussian

.. autoclass:: statadic.structure.survival.RightCensored

.. autoclass:: statadic.observation.data.PanelLayout

.. autoclass:: statadic.structure.dyadic.Dyads

.. autoclass:: statadic.structure.dyadic.EndpointEffects

.. autoclass:: statadic.regression.Regression
   :members: ols, wls, logit, probit, poisson, aft, estimate

.. py:type:: statadic.regression.RegressionEstimator

   A compatible native estimator for the selected scientific family.

.. py:type:: statadic.regression.RegressionFit

   The supported family-specific result union; capabilities depend on the actual fit.

.. py:type:: statadic.law.survival.AFTFamily

   A declared Weibull, lognormal or log-logistic accelerated failure-time law.

.. py:class:: statadic.numerics.quadrature.IntegrationDesign
   :canonical: statadic.numerics.quadrature.IntegrationDesign

   A replayable key or supplied integration points with fixed draw and replicate
   counts. See the :doc:`spatial survival guide <survival>` for scope.

.. autoclass:: statadic.regression.LinearResult
   :members: diagnose, predict, infer

.. py:class:: statadic.regression.LogisticResult
   :canonical: statadic.regression._glm.LogisticResult

   Binomial-logit fit with retained design, prediction and explicit covariance choices.

.. py:class:: statadic.regression.PoissonResult
   :canonical: statadic.regression._glm.PoissonResult

   Poisson-log fit with retained exposure, offset and prediction targets.

.. py:class:: statadic.regression.LikelihoodResult
   :canonical: statadic.regression._likelihood.LikelihoodResult

   Probit, Gamma or NB2 fit; numerical validity and model-based inference remain separate.

.. py:class:: statadic.structure.survival.SurvivalResult
   :canonical: statadic.structure.survival._model.SurvivalResult

   Independent AFT fit with the declared censoring, family and supported predictions.

.. py:class:: statadic.model.fit_statistics.FitStatistics

   Fit-quality metrics and the notes that state their conventions, from ``fit.fit_statistics()``.

.. py:class:: statadic.model.fit_statistics.LikelihoodSummary

   Maximised log likelihood, parameter count and observation count that a family supplies for its criteria.

.. py:class:: statadic.model.capabilities.ReportsFitStatistics

   Protocol for results that return ``FitStatistics`` from ``fit_statistics()``.

.. py:class:: statadic.model.capabilities.SummarizesLikelihood

   Protocol for results that return a ``LikelihoodSummary`` from ``likelihood_summary()``.

.. py:function:: statadic.model.fit_statistics.information_metrics

   Build ``n_parameters``, ``log_likelihood``, AIC, BIC and HQIC from one log likelihood.

.. py:class:: statadic.model.lifecycle.PreparedRegression

   Frozen row-wise design and observations, bound to the preparing model instance.

.. py:class:: statadic.structure.survival.PreparedSurvival
   :canonical: statadic.structure.survival._model.PreparedSurvival

   Frozen AFT inputs, including the observation mechanism and fitted predictor design.

.. automodule:: statadic.hypothesis.specification
   :members: JarqueBera, BreuschPagan, White, DurbinWatson, BreuschGodfrey, RESET, HansenJ, Influence, Diagnostics, InfluenceDiagnostic, LinearDiagnostic, IVDiagnostic, MomentsDiagnostic, RetainedLinearFit, RetainedIVFit, RetainedMomentsFit

.. automodule:: statadic.hypothesis.core
   :members: Hypothesis, Proposition, LinearRestriction, ConstantVariance, GaussianMoments, NoSerialCorrelation, NoAddedFittedPowerEffect, OveridentifyingRestrictions, ChiSquare, FisherF, TestOutcome

.. py:currentmodule:: None

.. py:type:: statadic.hypothesis.core.Reference

   A ``ChiSquare`` or ``FisherF`` reference attached to an actual calibrated tail.

.. py:type:: statadic.report.core.MetadataValue

   Explicit JSON-compatible report metadata, including nested mappings and sequences.

.. py:type:: statadic.report.core.FrozenMetadataValue

   The recursively immutable representation retained by a report snapshot.

.. py:type:: statadic.report.core.ReportNumber

   An integer or floating-point report value.

.. autoclass:: statadic.hypothesis.uncertainty.InferenceContext

.. py:class:: statadic.estimation.moments.instrumental.IVFit

   Native structural IV fit retaining the actual last weighting solve, original
   aligned rows and weighting/convergence provenance.

.. py:class:: statadic.estimation.moments.gmm.AffineGMMFit

   Native affine-GMM fit retaining the generating weight, final contributions
   and variability scheme. Point-fit validity and J calibration remain separate.

.. py:class:: statadic.core.equations.VariabilityScheme

   Nominal axis distinguishing variability constructions; a scheme name alone
   establishes neither source correspondence nor statistical validity.

.. py:type:: statadic.hypothesis.specification.Diagnostic

   One diagnostic outcome: a ``TestOutcome`` or an ``InfluenceDiagnostic``.

.. autoclass:: statadic.estimation.least_squares.linear.LeastSquares

.. autoclass:: statadic.estimation.likelihood.glm.BinomialLogitMLE

.. autoclass:: statadic.estimation.likelihood.glm.PoissonMLE

.. autoclass:: statadic.estimation.likelihood.conditional.ProbitMLE

.. autoclass:: statadic.estimation.likelihood.survival.SurvivalML

.. autoclass:: statadic.hypothesis.uncertainty.HC

.. autoclass:: statadic.hypothesis.uncertainty.HAC

.. py:function:: statadic.inference.covariance.linear.hc_covariance(fit, *, kind="HC0", finite_sample_correction=1.0)

   Estimate HC0–HC3 covariance from the fit's whitened influences. HC1 applies
   the count correction; HC2 and HC3 require leverage below one. The optional
   positive correction is separate. Choosing a covariance does not choose a
   reference distribution or establish interval coverage.

A ``LinearResult`` retains its numerical fit as ``core``. The tutorials inspect
``core.valid`` before interpreting ``core.parameters``, then inspect the interval's
own validity after requesting inference. See the :doc:`regression guide <regression>`.

Reports and retained plots
--------------------------

.. autoclass:: statadic.report.core.Report
   :members: to_text, to_html, to_latex, save_latex, to_values, save_values, coefficient_plot, require_available

.. autoclass:: statadic.report.core.ParameterRow

.. autoclass:: statadic.report.core.TestRow

.. autoclass:: statadic.report.core.Metric

.. autoclass:: statadic.report.core.PlotSpec
   :members: to_pgfplots, save_pgfplots

.. autoclass:: statadic.report.core.ReferenceLine

These objects retain already-computed values. Analysis snapshots restore
rendering, while predictor artifacts restore only their declared prediction
capabilities. See :doc:`reporting <report>` and :doc:`artifacts <prediction>`.

.. py:class:: statadic.prediction.artifacts.PredictionArtifact

   A prediction-only artifact restored by ``load_predictor(path)``. It keeps the
   fitted design and coefficients needed for compatible new observations.

   .. py:method:: predict(data, *, kind=None)

      Apply the saved design and prediction convention to new observations.
      The artifact does not restore live fitted inference.

Quantities and supplied moments
-------------------------------

.. py:class:: statadic.estimand.core.Estimand(quantity, *, estimator)

   Prepare and estimate a declared covariance or supplied-moment quantity.
   The actual covariance, GMM or GEL result retains its family capabilities.
   See :doc:`estimand <estimand>` for inputs and inference requests.

.. py:class:: statadic.estimand.core.Covariance(columns=None)

   Covariance of an ordered set of observed variables; no inference is implied.

.. py:class:: statadic.estimand.core.MomentConditions()

   Supplied fixed or affine moments; sample arrays belong to ``MomentData``.

.. py:class:: statadic.estimation.covariance.CovarianceProblem(values)

   Native rectangular numerical input to a covariance estimator.

.. py:class:: statadic.estimation.covariance.SampleCovariance(ddof=1)

   Numerical sample covariance with divisor ``n - ddof``.

.. py:class:: statadic.estimation.covariance.LedoitWolf()

   Constant-correlation Ledoit--Wolf covariance brick.

.. py:class:: statadic.estimand.moments.GMMSandwich(finite_sample_correction=1.0)

   Explicit actual-fit GMM sampling covariance request.

.. py:class:: statadic.estimand.moments.GELCalibration(kind="lr", assume_iid=False)

   Bound GEL statistic/reference request; the hypothesis is evaluated at ``test``.

Distributions, embedding and comparison
---------------------------------------

.. autofunction:: statadic.geometry.discrepancy.mmd.energy.energy_distance

.. autoclass:: statadic.geometry.ground.metrics.EuclideanLaw

.. autoclass:: statadic.observation.geometry.Energy

.. autoclass:: statadic.observation.geometry.MMD

.. autoclass:: statadic.ordination.fit.Ordination
   :members: pca, pcoa, estimate

.. autoclass:: statadic.ordination.fit.PrincipalCoordinates

.. py:class:: statadic.observation.coordinates.OrdinationEvidence

   Immutable actual native fit, training context and retained coordinate map.

.. py:class:: statadic.observation.coordinates.FittedOrdination

   A fitted predictor term with named inspection through
   ``fit.design.ordination(name)``. Prediction-only artifacts omit inspection.

.. autoclass:: statadic.observation.geometry.ObservedDissimilarity

.. autoclass:: statadic.regression.GeometricResponse

.. autoclass:: statadic.regression.Reduced

.. autoclass:: statadic.regression.FreedmanLane

.. autoclass:: statadic.hypothesis.homogeneity.PermutationPlan

.. py:type:: statadic.hypothesis.homogeneity.ComparisonInput

   Aligned prepared geometry or a supported supplied matrix for group comparison.

.. py:type:: statadic.hypothesis.homogeneity.DesignLabels

   A grouping column name or explicitly aligned integer/string labels.

.. autofunction:: statadic.geometry.discrepancy.mmd.mmd_squared

.. autofunction:: statadic.geometry.discrepancy.mmd.mmd_squared_matrix

.. autofunction:: statadic.geometry.discrepancy.mmd.weighted.weighted_mmd_squared_cross

.. autofunction:: statadic.inference.homogeneity.dispersion.dispersion_permutation_null

.. py:function:: statadic.inference.resampling.permutation_kernel.permutation_test_kernel(statistic, null_distribution, num_permutations, alternative="two-sided")

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
:doc:`composition guide <model>`.
