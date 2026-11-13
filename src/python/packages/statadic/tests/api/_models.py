"""Canonical conditional specifications shared by host contract fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from statadic.api.regression import Binomial, Observed, Poisson, Regression

if TYPE_CHECKING:
    from statadic.api.base import Estimator, PreparedRegression
    from statadic.api.design import Term
    from statadic.api.regression import LogisticResult, PoissonResult
    from statadic.estimation.glm import BinomialProblem, GLMFit, PoissonProblem
    from statadic.model.glm import BinomialLogitEquation, PoissonLogEquation


def poisson_model(
    response: str,
    terms: Term,
    *,
    exposure: str | None = None,
    offset: str | None = None,
    estimator: Estimator[PoissonProblem, GLMFit[PoissonLogEquation]] | None = None,
) -> Regression[PoissonResult, PreparedRegression]:
    """Declare a Poisson law with the roles named by the test."""
    return Regression(
        Observed(response),
        terms,
        estimator=estimator,
        exposure=exposure,
        offset=offset,
        law=Poisson(),
    )


def logit_model(
    response: str,
    terms: Term,
    *,
    trials: str | None = None,
    offset: str | None = None,
    estimator: Estimator[BinomialProblem, GLMFit[BinomialLogitEquation]] | None = None,
) -> Regression[LogisticResult, PreparedRegression]:
    """Declare a grouped-binomial logit law without changing the estimator."""
    return Regression(
        Observed(response, trials=trials),
        terms,
        estimator=estimator,
        offset=offset,
        law=Binomial(link="logit"),
    )
