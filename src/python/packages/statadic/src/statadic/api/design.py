"""Named predictor terms with retained training transformations.

Host preparation learns category levels and selects landmark observations once.
Numerical feature arithmetic remains in JAX and existing geometry producers.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping  # noqa: TC003 -- runtime annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, override, runtime_checkable

import jax.numpy as jnp

from statadic.api.data import Observations
from statadic.api.inputs import as_observations
from statadic.api.ordination_terms import FittedOrdination
from statadic.api.representation import GeometrySpec, PreparedGeometry  # noqa: TC001
from statadic.api.roles import (  # noqa: TC001
    InputRole,
    InputRoleProvider,
    merge_input_roles,
)
from statadic.core.statistics import PopulationEstimand
from statadic.core.typing import Array  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Categorical",
    "CyclicSpline",
    "DesignContrast",
    "DistanceToLandmarks",
    "FittedCategorical",
    "FittedDesign",
    "FittedDistanceToLandmarks",
    "FittedInteraction",
    "FittedIntercept",
    "FittedLinear",
    "FittedSpline",
    "FittedSumTerm",
    "FittedTerm",
    "Fourier",
    "Interaction",
    "Intercept",
    "Linear",
    "LinearPredictorContrast",
    "RowDifference",
    "Spline",
    "SumTerm",
    "Term",
    "formula_to_terms",
    "level_codes",
]


_CUBIC_DEGREE = 3


def _label(value: int | str) -> str:
    """Keep unlike host labels distinct in the static coefficient schema."""
    return f"int:{int(value)!r}" if isinstance(value, int) else f"str:{str(value)!r}"


def _level_key(label: int | str) -> tuple[bool, int | str]:
    return (isinstance(label, str), label)


def level_codes(
    labels: Sequence[int | str],
) -> tuple[tuple[int | str, ...], Array]:
    """Return sorted unique levels and int32 codes, linear in the label count.

    Integers sort before strings, each by value, so int ``1`` and str ``"1"``
    stay distinct levels in a reproducible order.
    """
    levels = tuple(sorted(set(labels), key=_level_key))
    index = {level: position for position, level in enumerate(levels)}
    codes = jnp.asarray([index[label] for label in labels], dtype=jnp.int32)
    return levels, codes


def _formula_atom(value: str) -> Term:
    """Resolve one identifier or the sole allowlisted categorical constructor."""
    value = value.strip()
    if value.isidentifier():
        return Linear(value)
    category = re.fullmatch(r"C\(\s*([^()]+?)\s*\)", value)
    if category is not None and category.group(1).isidentifier():
        return Categorical(category.group(1))
    message = f"Unsupported formula term: {value!r}."
    raise ValueError(message)


def formula_to_terms(formula: str) -> tuple[str, Term]:
    """Compile a bounded formula into the canonical explicit term expression.

    Supported syntax is ``response ~ x + C(group) + x:z``. An intercept is
    included unless a standalone ``0`` occurs; ``1`` explicitly includes it.
    Column names must be identifiers. No evaluation, subtraction, arithmetic,
    arbitrary calls, ``*`` expansion, quoted names or transformations occur.
    """
    if formula.count("~") != 1:
        message = "A formula requires exactly one response ~ predictors separator."
        raise ValueError(message)
    response, expression = (part.strip() for part in formula.split("~"))
    if not response.isidentifier() or not expression:
        message = "Formula response must be an identifier with a predictor expression."
        raise ValueError(message)
    pieces = [piece.strip() for piece in expression.split("+")]
    if any(not piece for piece in pieces) or ("0" in pieces and "1" in pieces):
        message = (
            "Formula terms must be nonempty with one unambiguous intercept policy."
        )
        raise ValueError(message)

    terms: list[Term] = [] if "0" in pieces else [Intercept()]
    for piece in pieces:
        if piece in {"0", "1"}:
            continue
        factors = piece.split(":")
        term = _formula_atom(factors[0])
        for factor in factors[1:]:
            term = Interaction(term, _formula_atom(factor))
        terms.append(term)
    if not terms:
        message = "A formula must retain at least one predictor column."
        raise ValueError(message)
    return response, SumTerm(tuple(terms))


@runtime_checkable
class FittedTerm(InputRoleProvider, Protocol):
    """A training-fitted transformation, without prediction-time learning."""

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the retained coefficient-column order."""
        ...

    def transform(self, data: Observations) -> Array:
        """Apply retained training state without learning from prediction rows."""
        ...


class Term(ABC):
    """A declared predictor component; addition concatenates explicit terms.

    Addition combines columns, while multiplication creates an interaction.
    Include an intercept and any main effects explicitly.

    Examples:
        >>> terms = Intercept() + Linear("x")
        >>> terms.input_columns
        {'x': 'numeric'}

    """

    @property
    @abstractmethod
    def input_columns(self) -> Mapping[str, InputRole]:
        """Return the input roles needed for this term."""
        ...

    @abstractmethod
    def _prepare_term(self, training: Observations) -> FittedTerm:
        """Retain the training transformation and its explicit column schema."""
        ...

    def __add__(self, other: Term) -> SumTerm:
        """Concatenate explicit terms without adding an intercept."""
        left = self.terms if isinstance(self, SumTerm) else (self,)
        right = other.terms if isinstance(other, SumTerm) else (other,)
        return SumTerm((*left, *right))

    def __mul__(self, other: Term) -> Interaction:
        """Return only the interaction; main effects remain explicit terms."""
        return Interaction(self, other)


@dataclass(frozen=True)
class Intercept(Term):
    """Declare an explicit unit column.

    Examples:
        >>> Intercept().input_columns
        {}

    """

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return {}

    @override
    def _prepare_term(self, training: Observations) -> FittedIntercept:
        """Retain the training transformation and its explicit column schema."""
        del training
        return FittedIntercept()


@dataclass(frozen=True)
class FittedIntercept:
    """An unchanged unit column for any prediction row count."""

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return {}

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the retained coefficient-column order."""
        return ("intercept",)

    def transform(self, data: Observations) -> Array:
        """Apply retained training state without learning from prediction rows."""
        return jnp.ones((data.n_rows, len(self.column_names)), dtype=bool)


@dataclass(frozen=True)
class Linear(Term):
    """Select one numeric column without implicit scaling or centering.

    Args:
        column: Name of the numeric predictor.

    Examples:
        >>> Linear("age").input_columns
        {'age': 'numeric'}

    """

    column: str

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return {self.column: "numeric"}

    @override
    def _prepare_term(self, training: Observations) -> FittedLinear:
        """Retain the training transformation and its explicit column schema."""
        training.numeric(self.column)
        return FittedLinear(self.column)


@dataclass(frozen=True)
class FittedLinear:
    """An explicit numeric column selection."""

    column: str

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return {self.column: "numeric"}

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the retained coefficient-column order."""
        return (self.column,)

    def transform(self, data: Observations) -> Array:
        """Apply retained training state without learning from prediction rows."""
        return data.numeric(self.column)[:, None]


def _count(value: int, minimum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        message = f"{name} must be an integer >= {minimum}."
        raise ValueError(message)


def _period(period: float | int) -> None:  # noqa: PYI041 -- runtime annotation
    if isinstance(period, bool) or not math.isfinite(period) or period <= 0:
        message = "The period must be finite and positive."
        raise ValueError(message)


@dataclass(frozen=True)
class CyclicSpline(Term):
    """Uniform cyclic cubic spline with exactly ``df`` nonconstant columns.

    The grid starts at zero and has ``df + 1`` cells over one period. The
    coefficient of cardinal basis function zero is fixed at zero. Add an
    intercept to span the unrestricted space, including its constant mode.

    Args:
        column: Numeric phase column.
        df: Number of nonconstant basis columns, at least three.
        period: Positive finite period in the units of the phase.

    Raises:
        ValueError: If the basis dimension or period is invalid.

    Examples:
        >>> CyclicSpline("phase", df=3, period=1.0).column_names
        ('phase:cyclic[1]', 'phase:cyclic[2]', 'phase:cyclic[3]')

    """

    column: str
    df: int
    period: float | int

    def __post_init__(self) -> None:
        """Require four distinct wrapped neighbours and a physical period."""
        _count(self.df, _CUBIC_DEGREE, "df")
        _period(self.period)

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the numeric phase input."""
        return {self.column: "numeric"}

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the nonreference knot indices in cyclic order."""
        return tuple(f"{self.column}:cyclic[{k}]" for k in range(1, self.df + 1))

    @override
    def _prepare_term(self, training: Observations) -> CyclicSpline:
        """Validate the column; the declared period fixes the entire grid."""
        training.numeric(self.column)
        return self

    def transform(self, data: Observations) -> Array:
        """Evaluate the four local cubic weights, including across the seam."""
        count = self.df + 1
        phase = jnp.remainder(data.numeric(self.column), self.period) / self.period
        grid = phase * count
        cell = jnp.floor(grid)
        u = grid - cell
        weights = (
            jnp.stack(
                (
                    (1 - u) ** 3,
                    3 * u**3 - 6 * u**2 + 4,
                    -3 * u**3 + 3 * u**2 + 3 * u + 1,
                    u**3,
                ),
                axis=1,
            )
            / 6
        )
        indices = (cell.astype(jnp.int32)[:, None] + jnp.arange(-1, 3)) % count
        basis = jnp.zeros((data.n_rows, count), dtype=weights.dtype)
        return basis.at[jnp.arange(data.n_rows)[:, None], indices].set(weights)[:, 1:]


@dataclass(frozen=True)
class Fourier(Term):
    """Paired sine and cosine columns with zero phase at the origin.

    Args:
        column: Numeric phase column.
        harmonics: Number of sine and cosine pairs.
        period: Positive finite period in the units of the phase.

    Raises:
        ValueError: If the harmonic count or period is invalid.

    Examples:
        >>> Fourier("phase", harmonics=1, period=1.0).column_names
        ('phase:period[1.0]:sin[1]', 'phase:period[1.0]:cos[1]')

    """

    column: str
    harmonics: int
    period: float | int

    def __post_init__(self) -> None:
        """Require at least one harmonic and a finite positive period."""
        _count(self.harmonics, 1, "harmonics")
        _period(self.period)

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the numeric phase input."""
        return {self.column: "numeric"}

    @property
    def column_names(self) -> tuple[str, ...]:
        """Name each sine/cosine pair in increasing harmonic order."""
        return tuple(
            f"{self.column}:period[{float(self.period)!r}]:{kind}[{k}]"
            for k in range(1, self.harmonics + 1)
            for kind in ("sin", "cos")
        )

    @override
    def _prepare_term(self, training: Observations) -> Fourier:
        """Validate the phase input and retain the declared frequencies."""
        training.numeric(self.column)
        return self

    def transform(self, data: Observations) -> Array:
        """Apply the same origin, period, and frequencies to every row."""
        phase = jnp.remainder(data.numeric(self.column), self.period) / self.period
        angle = (2 * jnp.pi * phase[:, None]) * jnp.arange(1, self.harmonics + 1)
        return jnp.stack((jnp.sin(angle), jnp.cos(angle)), axis=2).reshape(
            data.n_rows, 2 * self.harmonics
        )


@dataclass(frozen=True)
class Spline(Term):
    """Unpenalized regression spline with equally spaced interior knots.

    Training fixes the range once. The truncated-power basis contains
    powers one through ``degree`` and ``df - degree`` hinge functions.
    Prediction continues the end polynomials beyond the training range.

    Args:
        column: Numeric predictor column.
        df: Number of nonconstant basis columns.
        degree: Polynomial degree.

    Raises:
        ValueError: If the degree or basis dimension is invalid.

    Examples:
        >>> Spline("age", df=4, degree=3).input_columns
        {'age': 'numeric'}

    """

    column: str
    df: int
    degree: int = _CUBIC_DEGREE

    def __post_init__(self) -> None:
        """Validate the polynomial degree and nonconstant basis dimension."""
        _count(self.degree, 1, "degree")
        _count(self.df, self.degree, "df")

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the numeric predictor input."""
        return {self.column: "numeric"}

    @override
    def _prepare_term(self, training: Observations) -> FittedSpline:
        """Retain the observed endpoints for all subsequent evaluations."""
        values = training.numeric(self.column)
        if not values.size:
            message = "Spline preparation requires a nonempty training range."
            raise ValueError(message)
        return FittedSpline(self, float(jnp.min(values)), float(jnp.max(values)))


@dataclass(frozen=True)
class FittedSpline:
    """A fixed grid in the original training coordinate system."""

    spec: Spline
    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Require an ordered finite training range."""
        if not (
            math.isfinite(self.lower)
            and math.isfinite(self.upper)
            and self.lower < self.upper
        ):
            message = "A spline needs distinct finite training endpoints."
            raise ValueError(message)

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the retained predictor role."""
        return self.spec.input_columns

    @property
    def column_names(self) -> tuple[str, ...]:
        """Name polynomial powers first, then interior hinges."""
        return tuple(
            f"{self.spec.column}:spline[{k}]" for k in range(1, self.spec.df + 1)
        )

    def transform(self, data: Observations) -> Array:
        """Evaluate on the retained grid, including polynomial extrapolation."""
        degree = self.spec.degree
        count = self.spec.df - degree
        u = (data.numeric(self.spec.column) - self.lower) / (self.upper - self.lower)
        powers = u[:, None] ** jnp.arange(1, degree + 1)
        knots = jnp.arange(1, count + 1, dtype=u.dtype) / (count + 1)
        hinges = jnp.maximum(u[:, None] - knots, 0) ** degree
        return jnp.concatenate((powers, hinges), axis=1)


@dataclass(frozen=True)
class Categorical(Term):
    """Sorted-level reference coding, rejecting unseen prediction levels.

    The smallest observed level is the reference unless explicitly selected.
    This term does not add an intercept or infer whether one is appropriate.

    Args:
        column: Categorical predictor column.
        reference: Training level to omit, or ``None`` for the smallest level.

    Raises:
        TypeError: If the reference is a boolean.

    Examples:
        >>> Categorical("group", reference="control").input_columns
        {'group': 'categorical'}

    """

    column: str
    reference: int | str | None = None

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return {self.column: "categorical"}

    def __post_init__(self) -> None:
        """Reject boolean reference labels rather than aliasing an integer level."""
        if isinstance(self.reference, bool):
            message = "Categorical reference labels exclude bool."
            raise TypeError(message)

    @override
    def _prepare_term(self, training: Observations) -> FittedCategorical:
        """Retain the training transformation and its explicit column schema."""
        labels = training.categorical(self.column)
        if not labels:
            message = "Categorical preparation requires observed levels."
            raise ValueError(message)
        levels, _ = level_codes(labels)
        reference = levels[0] if self.reference is None else self.reference
        if reference not in levels:
            message = "The categorical reference must occur in training."
            raise ValueError(message)
        return FittedCategorical(self.column, levels, reference)


@dataclass(frozen=True)
class FittedCategorical:
    """Retained training levels and their omitted reference."""

    column: str
    levels: tuple[int | str, ...]
    reference: int | str

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return {self.column: "categorical"}

    def __post_init__(self) -> None:
        """Own and validate the retained training coding."""
        levels = tuple(self.levels)
        if (
            not levels
            or len(set(levels)) != len(levels)
            or self.reference not in levels
            or any(
                isinstance(level, bool) or not isinstance(level, (int, str))
                for level in (*levels, self.reference)
            )
        ):
            message = (
                "Retained categorical levels must be unique with an observed reference."
            )
            raise ValueError(message)
        object.__setattr__(self, "levels", levels)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the retained coefficient-column order."""
        return tuple(
            f"{self.column}[{_label(level)}]"
            for level in self.levels
            if level != self.reference
        )

    def transform(self, data: Observations) -> Array:
        """Apply retained training state without learning from prediction rows."""
        labels = data.categorical(self.column)
        position = {level: index for index, level in enumerate(self.levels)}
        if any(label not in position for label in labels):
            message = f"Unseen categorical level in {self.column!r}."
            raise ValueError(message)
        retained = tuple(level for level in self.levels if level != self.reference)
        if not retained:
            return jnp.empty((data.n_rows, 0), dtype=bool)
        codes = jnp.asarray([position[label] for label in labels], dtype=jnp.int32)
        columns = jnp.asarray([position[level] for level in retained], dtype=jnp.int32)
        return codes[:, None] == columns[None, :]


@dataclass(frozen=True)
class SumTerm(Term):
    """Concatenate declared terms in their retained order."""

    terms: tuple[Term, ...]

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return merge_input_roles(*(term.input_columns for term in self.terms))

    def __post_init__(self) -> None:
        """Own the declared order independently of the caller's container."""
        object.__setattr__(self, "terms", tuple(self.terms))

    @override
    def _prepare_term(self, training: Observations) -> FittedSumTerm:
        """Retain the training transformation and its explicit column schema."""
        return FittedSumTerm(tuple(term._prepare_term(training) for term in self.terms))


@dataclass(frozen=True)
class FittedSumTerm:
    """Concatenation of independently retained term transformations."""

    terms: tuple[FittedTerm, ...]

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return merge_input_roles(*(term.input_columns for term in self.terms))

    def __post_init__(self) -> None:
        """Own the retained term sequence."""
        object.__setattr__(self, "terms", tuple(self.terms))

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the retained coefficient-column order."""
        return tuple(name for term in self.terms for name in term.column_names)

    def transform(self, data: Observations) -> Array:
        """Apply retained training state without learning from prediction rows."""
        if not self.terms:
            return jnp.empty((data.n_rows, 0))
        return jnp.concatenate([term.transform(data) for term in self.terms], axis=1)


@dataclass(frozen=True)
class Interaction(Term):
    """Cartesian products of two explicit terms, without implicit main effects.

    Args:
        left: First term in the product.
        right: Second term in the product.

    Examples:
        >>> (Linear("x") * Linear("z")).input_columns
        {'x': 'numeric', 'z': 'numeric'}

    """

    left: Term
    right: Term

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return merge_input_roles(self.left.input_columns, self.right.input_columns)

    @override
    def _prepare_term(self, training: Observations) -> FittedInteraction:
        """Retain the training transformation and its explicit column schema."""
        return FittedInteraction(
            self.left._prepare_term(training), self.right._prepare_term(training)
        )


@dataclass(frozen=True)
class FittedInteraction:
    """Training-schema products in left-major coefficient order."""

    left: FittedTerm
    right: FittedTerm

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return merge_input_roles(self.left.input_columns, self.right.input_columns)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the retained coefficient-column order."""
        return tuple(
            f"{left}:{right}"
            for left in self.left.column_names
            for right in self.right.column_names
        )

    def transform(self, data: Observations) -> Array:
        """Apply retained training state without learning from prediction rows."""
        left, right = self.left.transform(data), self.right.transform(data)
        return (left[:, :, None] * right[:, None, :]).reshape(
            data.n_rows, left.shape[1] * right.shape[1]
        )


@dataclass(frozen=True)
class DistanceToLandmarks(Term):
    """Distribution features using explicitly selected training-row landmarks."""

    column: str
    geometry: GeometrySpec
    landmarks: tuple[int | str, ...]

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return {self.column: "measure"}

    def __post_init__(self) -> None:
        """Own explicit landmark identities without accepting boolean aliases."""
        if any(
            isinstance(label, bool) or not isinstance(label, (int, str))
            for label in self.landmarks
        ):
            message = (
                "Landmark identities require integer/string labels excluding bool."
            )
            raise ValueError(message)
        object.__setattr__(self, "landmarks", tuple(self.landmarks))

    @override
    def _prepare_term(self, training: Observations) -> FittedDistanceToLandmarks:
        """Retain the training transformation and its explicit column schema."""
        if not self.landmarks or len(set(self.landmarks)) != len(self.landmarks):
            message = "Landmarks must be a nonempty tuple of unique row IDs."
            raise ValueError(message)
        positions = {label: index for index, label in enumerate(training.row_ids)}
        if any(label not in positions for label in self.landmarks):
            message = "Every landmark must identify a training observation."
            raise ValueError(message)
        selected = training.take([positions[label] for label in self.landmarks])
        prepared = self.geometry.prepare(
            selected.measure(self.column), row_ids=selected.row_ids
        )
        return FittedDistanceToLandmarks(self.column, prepared)


@dataclass(frozen=True)
class FittedDistanceToLandmarks:
    """Frozen landmark measures, geometry parameters and reference order."""

    column: str
    geometry: PreparedGeometry

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the declared input roles."""
        return {self.column: "measure"}

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the retained coefficient-column order."""
        return tuple(
            f"{self.column}:distance[{_label(label)}]"
            for label in self.geometry.row_ids
        )

    def transform(self, data: Observations) -> Array:
        """Apply retained training state without learning from prediction rows."""
        return self.geometry.cross(data.measure(self.column))


@dataclass(frozen=True)
class LinearPredictorContrast(PopulationEstimand):
    """The linear-predictor difference between two predictor settings."""

    at: Mapping[str, float | int | str]
    versus: Mapping[str, float | int | str]
    scope: Literal["rows", "term"]

    def __post_init__(self) -> None:
        """Retain the targeted settings independently of caller mutation."""
        object.__setattr__(self, "at", MappingProxyType(dict(self.at)))
        object.__setattr__(self, "versus", MappingProxyType(dict(self.versus)))

    def describe(self) -> str:
        """Name the targeted contrast in words."""
        if self.scope == "rows":
            return f"linear predictor at {dict(self.at)} versus {dict(self.versus)}"
        ((column, at),) = self.at.items()
        ((_, versus),) = self.versus.items()
        return f"structural linear-predictor contrast: {column}={at} versus {versus}"


@dataclass(frozen=True)
class RowDifference:
    """Two complete predictor rows compared on the linear-predictor scale."""

    at: Mapping[str, float | int | str]
    versus: Mapping[str, float | int | str]

    def __post_init__(self) -> None:
        """Retain the research question independently of caller mutation."""
        object.__setattr__(self, "at", MappingProxyType(dict(self.at)))
        object.__setattr__(self, "versus", MappingProxyType(dict(self.versus)))

    def compile(self, design: FittedDesign) -> DesignContrast:
        """Evaluate both covariate settings in this design's learned basis."""
        columns = design.input_columns
        if self.at.keys() != columns.keys() or self.versus.keys() != columns.keys():
            message = f"Each contrast row requires exactly {sorted(columns)}."
            raise ValueError(message)
        rows = (
            as_observations(
                {name: [self.at[name], self.versus[name]] for name in columns},
                columns,
                broadcast_scalars=False,
            )
            if columns
            else Observations({}, row_ids=(0, 1))
        )
        values = design.transform(rows)
        at, versus = values.astype(jnp.result_type(values, 1.0))
        return DesignContrast(
            at - versus,
            design.column_names,
            LinearPredictorContrast(self.at, self.versus, "rows"),
            design,
        )


@dataclass(frozen=True)
class DesignContrast:
    """A structural linear-predictor contrast in a fitted coefficient schema."""

    coefficients: Array
    parameter_names: tuple[str, ...]
    estimand: LinearPredictorContrast
    design: FittedDesign = field(repr=False, compare=False)


def _fitted_ordinations(term: FittedTerm) -> tuple[FittedOrdination, ...]:
    if isinstance(term, FittedOrdination):
        return (term,)
    if isinstance(term, FittedSumTerm):
        return tuple(
            found for child in term.terms for found in _fitted_ordinations(child)
        )
    if isinstance(term, FittedInteraction):
        return (*_fitted_ordinations(term.left), *_fitted_ordinations(term.right))
    return ()


@dataclass(frozen=True)
class FittedDesign:
    """A complete retained term expression and unique coefficient schema."""

    terms: FittedTerm

    @property
    def input_columns(self) -> Mapping[str, InputRole]:
        """Return the declared input roles."""
        return self.terms.input_columns

    def __post_init__(self) -> None:
        """Validate the retained coefficient schema."""
        names = self.column_names
        if not names or len(set(names)) != len(names):
            message = (
                "A design requires nonempty, unique coefficient names; "
                "use distinct name= values for multiple ordination terms."
            )
            raise ValueError(message)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the retained coefficient-column order."""
        return self.terms.column_names

    def ordination(self, name: str) -> FittedOrdination:
        """Find one named live ordination within built-in sums and interactions.

        Extension terms remain opaque. Predictive-only artifacts retain maps,
        not the live decomposition evidence required by this inspection door.
        """
        fitted = _fitted_ordinations(self.terms)
        matches = [
            term
            for term in fitted
            if term.evidence is not None and term.evidence.name == name
        ]
        if len(matches) > 1:
            message = "Use distinct name= values for independently fitted ordinations."
            raise ValueError(message)
        if not matches:
            if any(term.evidence is None for term in fitted):
                message = (
                    "Predictive-only artifacts do not retain ordination inspection."
                )
                raise ValueError(message)
            message = f"No fitted ordination has name {name!r}."
            raise KeyError(message)
        return matches[0]

    @property
    def intercept_columns(self) -> tuple[int, ...]:
        """Find literal intercept terms instead of columns with matching names."""

        def origins(term: FittedTerm) -> tuple[bool, ...]:
            if isinstance(term, FittedSumTerm):
                return tuple(flag for child in term.terms for flag in origins(child))
            retained = getattr(term, "intercept_columns", ())
            if retained:
                return tuple(i in retained for i in range(len(term.column_names)))
            return (isinstance(term, FittedIntercept),) * len(term.column_names)

        return tuple(i for i, flag in enumerate(origins(self.terms)) if flag)

    @property
    def has_intercept(self) -> bool:
        """Whether a top-level Intercept term contributes to the design."""
        return bool(self.intercept_columns)

    def design_contrast(
        self, term: Term, *, at: float, versus: float
    ) -> DesignContrast:
        """Evaluate one matched, already-fitted additive basis at two values."""
        parts = (
            self.terms.terms if isinstance(self.terms, FittedSumTerm) else (self.terms,)
        )
        matches = []
        start = 0
        for part in parts:
            match = (
                part == term
                or (
                    isinstance(term, Spline)
                    and isinstance(part, FittedSpline)
                    and part.spec == term
                )
                or (
                    isinstance(term, Linear)
                    and isinstance(part, FittedLinear)
                    and part.column == term.column
                )
            )
            if match:
                matches.append((start, part))
            start += len(part.column_names)
        if len(matches) != 1:
            message = "Contrast requires exactly one matching additive fitted term."
            raise ValueError(message)
        if not isinstance(term, (Linear, Spline, CyclicSpline, Fourier)):
            message = "Contrast requires a scalar numeric predictor term."
            raise TypeError(message)
        start, fitted = matches[0]
        rows = Observations({term.column: jnp.asarray([at, versus])})
        values = fitted.transform(rows)
        delta = values[0] - values[1]
        coefficients = jnp.zeros(len(self.column_names), dtype=delta.dtype)
        coefficients = coefficients.at[start : start + len(fitted.column_names)].set(
            delta
        )
        return DesignContrast(
            coefficients,
            self.column_names,
            LinearPredictorContrast({term.column: at}, {term.column: versus}, "term"),
            self,
        )

    def transform(self, data: Observations) -> Array:
        """Apply retained training state without learning from prediction rows."""
        values = self.terms.transform(data)
        if values.shape != (data.n_rows, len(self.column_names)):
            message = "Term output does not match its declared row/column schema."
            raise ValueError(message)
        return values
