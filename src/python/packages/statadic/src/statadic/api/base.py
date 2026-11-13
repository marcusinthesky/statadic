"""Shared model, result and estimator interfaces for the host workflows.

``Model`` is the reusable specification lifecycle: ``prepare`` retains learned
state with a model token, ``estimate_prepared`` checks that token and estimates,
and ``estimate`` composes the two. It is the only estimation entry point.
Private table preparation retains a response and predictor terms.
``Result`` is the facade
over one native core fit; ``RegressionResult`` adds the retained design. An
``Estimator`` brick maps a typed numeric problem to its native fit and lives
beside its mathematics in ``statadic.estimation``.
Nothing here implements a numerical estimator or chooses a reference law.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol, override, runtime_checkable

import jax.numpy as jnp

from statadic.api.data import Observations
from statadic.api.design import FittedDesign, Term  # noqa: TC001
from statadic.api.fit_statistics import (
    FitStatistics,
    LikelihoodSummary,  # noqa: TC001 -- runtime annotation
)
from statadic.api.inputs import RegressionInput, as_observations
from statadic.api.reporting import Metric, ParameterRow, Report
from statadic.api.roles import (  # noqa: TC001 -- runtime annotations
    InputRole,
    merge_input_roles,
)
from statadic.core.typing import Array  # noqa: TC001
from statadic.inference.contracts import ParameterFit

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "Estimator",
    "FitContext",
    "InvalidFitError",
    "Model",
    "ParameterResult",
    "PreparedRegression",
    "PreparedState",
    "RegressionFit",
    "RegressionResult",
    "Result",
]


class InvalidFitError(ValueError):
    """A regression fit cannot supply valid point estimates."""


@runtime_checkable
class Estimator[ProblemT, FitT](Protocol):
    """A static estimator brick mapping a typed numeric problem to a native fit."""

    def estimate(self, problem: ProblemT) -> FitT:
        """Estimate ``problem`` with the single existing numerical implementation."""
        ...


class PreparedState(Protocol):
    """Prepared state carrying the token of the model that produced it."""

    @property
    def model_token(self) -> object:
        """Return the identity token of the producing model instance."""
        ...


@dataclass(frozen=True)
class FitContext:
    """The exact aligned host sample and model identity surrounding a core fit."""

    observations: Observations = field(repr=False)
    model_name: str
    response: str

    @property
    def row_ids(self) -> tuple[int | str, ...]:
        """Return the exact retained outer observation identities."""
        return self.observations.row_ids

    @property
    def available_columns(self) -> tuple[str, ...]:
        """Return the columns a later inference step can name, such as clusters."""
        return self.observations.available_columns


@dataclass(frozen=True)
class PreparedRegression:
    """Retained design and sample belonging to one reusable model instance."""

    values: Array
    design: FittedDesign
    context: FitContext
    model_token: object = field(repr=False)


@dataclass(frozen=True)
class Model[InputT, PreparedT: PreparedState, ResultT](ABC):
    """Reusable analysis specification with a prepare/estimate lifecycle.

    A subclass implements ``prepare`` and ``_estimate_prepared``. The shared
    entry points retain one-data estimation and producing-model ownership.
    """

    _token: object = field(
        default_factory=object, init=False, repr=False, compare=False
    )

    @abstractmethod
    def prepare(self, data: InputT, /) -> PreparedT:
        """Validate ``data`` and retain learned preparation for this model."""
        ...

    def _check(self, prepared: PreparedT) -> None:
        if prepared.model_token is not self._token:
            message = (
                "Prepared state belongs to another model instance, even if the "
                "options match, and dataclasses.replace creates a new instance. "
                "Call prepare on this model, or estimate_prepared on the instance "
                "that prepared this state."
            )
            raise ValueError(message)

    def estimate_prepared(self, prepared: PreparedT) -> ResultT:
        """Check preparation ownership once, then estimate its numerical problem."""
        self._check(prepared)
        return self._estimate_prepared(prepared)

    @abstractmethod
    def _estimate_prepared(self, prepared: PreparedT) -> ResultT:
        """Estimate this model's checked prepared state."""
        ...

    def estimate(self, data: InputT, /) -> ResultT:
        """Prepare ``data`` and estimate the matching prepared state."""
        return self.estimate_prepared(self.prepare(data))


@dataclass(frozen=True)
class _RegressionPreparation[ResultT](
    Model[RegressionInput, PreparedRegression, ResultT]
):
    """Regression specification: a numeric ``response`` and predictor ``terms``."""

    response: str
    terms: Term
    _accepts_regression_inputs: ClassVar[bool] = False

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return predictor and response roles required for fitting."""
        return merge_input_roles(self.terms.input_columns, {self.response: "numeric"})

    @override
    def prepare(self, data: RegressionInput) -> PreparedRegression:
        """Retain the training transformation and its explicit column schema."""
        if self._accepts_regression_inputs:
            data = as_observations(data, self.input_columns, broadcast_scalars=False)
        elif not isinstance(data, Observations):
            message = "This workflow requires Observations."
            raise TypeError(message)
        design = FittedDesign(self.terms._prepare_term(data))
        response = data.numeric(self.response)
        values = design.transform(data)
        if not jnp.issubdtype(values.dtype, jnp.floating):
            values = values.astype(jnp.result_type(response, 1.0))
        return PreparedRegression(
            values,
            design,
            FitContext(data, "Regression", self.response),
            self._token,
        )


@dataclass(frozen=True)
class Result[FitT, ContextT](ABC):
    """Fitted-result facade: the native core fit and its family-specific context.

    A subclass names its native fit type, implements ``report`` and adds its
    own prediction or covariance methods. Nothing here chooses a reference
    distribution.
    """

    core: FitT = field(repr=False)
    context: ContextT

    @abstractmethod
    def report(self) -> Report:
        """Describe fitting evidence without manufacturing inferential quantities."""
        ...

    def likelihood_summary(self) -> LikelihoodSummary | None:  # noqa: PLR6301
        """Return the maximised likelihood summary, or ``None`` without one."""
        return None

    def fit_statistics(self) -> FitStatistics:
        """Return fit-quality metrics derived from ``likelihood_summary``.

        Empty unless the result has a likelihood summary or a family overrides.
        """
        summary = self.likelihood_summary()
        if summary is None:
            return FitStatistics()
        return FitStatistics.from_likelihood(summary)

    def _repr_html_(self) -> str:
        """Render the report for notebook display."""
        return self.report()._repr_html_()


class RegressionFit(ParameterFit, Protocol):
    """A parameter fit with validity evidence for regression results."""

    @property
    def valid(self) -> Array:
        """Return the fit's validity evidence."""
        ...


@dataclass(frozen=True)
class ParameterResult[FitT: RegressionFit](Result[FitT, FitContext]):
    """Parameter estimates and an explicit numerical-validity check."""

    @property
    def valid(self) -> bool:
        """Return whether the native fit is valid."""
        return bool(self.core.valid)

    def require_valid(self) -> Self:
        """Return this result or explain why the point fit is unavailable."""
        if not self.valid:
            report = self.report()
            status = ", ".join(
                f"{name}={report.metadata[name]}"
                for name in (
                    "solver_status",
                    "identified",
                    "finite_optimum",
                    "existence_status",
                )
                if name in report.metadata
            )
            message = " ".join(
                (
                    f"{self.context.model_name}:",
                    *report.warnings,
                    *((f"({status})",) if status else ()),
                )
            )
            raise InvalidFitError(message)
        return self

    @property
    def parameters(self) -> Array:
        """Return the native coefficients; invalid fits hold NaN."""
        return self.core.parameters


@dataclass(frozen=True)
class RegressionResult[FitT: RegressionFit](ParameterResult[FitT]):
    """Regression result retaining the fitted design used for prediction."""

    design: FittedDesign = field(repr=False)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the coefficient names used by ``report``."""
        return self.core.parameter_names or self.design.column_names

    @override
    def report(self) -> Report:
        """Describe fitting evidence without manufacturing inferential quantities."""
        valid = bool(self.core.valid)
        rows = tuple(
            ParameterRow(
                name,
                float(self.core.parameters[index]) if valid else None,
                valid=valid,
                status="available" if valid else "unavailable",
                unavailable_reason=None if valid else "invalid-fit",
            )
            for index, name in enumerate(
                self.core.parameter_names or self.design.column_names
            )
        )
        return Report(
            title=f"{self.context.model_name} fit",
            parameters=rows,
            metrics=(Metric("n_observations", int(self.context.observations.n_rows)),),
            metadata={
                "response": self.context.response,
                "fit_valid": valid,
                "solver_converged": bool(self.core.solver.converged),
                "solver_status": int(self.core.solver.status),
            },
            warnings=()
            if valid
            else ("Point fitting is unavailable; inspect retained core diagnostics.",),
        )
