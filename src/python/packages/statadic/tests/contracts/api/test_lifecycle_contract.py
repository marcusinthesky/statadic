"""Lifecycle verbs of every host result match a capability or recorded exception."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING, Final

import pytest
import statadic.api
from statadic.api import capabilities
from statadic.api.base import Model

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.component, pytest.mark.contract]

VERBS: Final = ("infer", "predict", "transform", "diagnose", "test")
FORBIDDEN_PREFIXES: Final = ("predict_", "estimate_", "infer_")

# Capability shapes, read from the declared structural Protocols.
CAPABILITIES: Final[dict[str, tuple[type, ...]]] = {
    "infer": (capabilities.CovarianceInference, capabilities.ResamplingInference),
    "predict": (capabilities.Predicts,),
    "transform": (capabilities.Transforms,),
    "diagnose": (capabilities.Diagnoses,),
    "test": (capabilities.Tests,),
}

# The model lifecycle hook is the only public ``estimate_*`` verb, on ``Model``.
LISTED_VERBS: Final = frozenset({"estimate_prepared"})

# Recorded signature exceptions: (class qualified name, verb) -> reason.
SIGNATURE_EXCEPTIONS: Final[dict[tuple[str, str], str]] = {
    ("InferenceResult", "test"): (
        "the reference law is a required keyword; chi-square and F are "
        "alternative references, so the shape is not Tests[Hypothesis, ...]"
    ),
    ("AffineGELResult", "infer"): (
        "binds a GEL calibration by keyword; no declared capability names "
        "calibration, so it is recorded rather than widened"
    ),
    ("FixedGELResult", "infer"): "as AffineGELResult: ``infer(*, calibration)``",
    ("TimeBlockPairs", "infer"): (
        "a resampling plan applying itself to a result, ``infer(result, /)``; "
        "it is the argument of ResamplingInference, not an implementer"
    ),
}


def _protocol_method(protocol: type) -> tuple[str, inspect.Signature]:
    names = [n for n in vars(protocol) if n in VERBS]
    assert len(names) == 1, protocol
    return names[0], inspect.signature(getattr(protocol, names[0]))


def _call_shape(
    protocol: type,
) -> tuple[list[object], dict[str, object]]:
    """Return placeholder arguments that exercise the protocol's call shape."""
    _, signature = _protocol_method(protocol)
    args: list[object] = []
    kwargs: dict[str, object] = {}
    for name, parameter in list(signature.parameters.items())[1:]:
        if parameter.kind is parameter.VAR_POSITIONAL:
            args.extend((object(), object()))
        elif parameter.kind is parameter.KEYWORD_ONLY:
            kwargs[name] = object()
        else:
            args.append(object())
    return args, kwargs


def accepts(candidate: inspect.Signature, protocol: type) -> bool:
    """Whether a method signature accepts the protocol's call and demands no more."""
    args, kwargs = _call_shape(protocol)
    try:
        bound = candidate.bind(object(), *args, **kwargs)
    except TypeError:
        return False
    bound.apply_defaults()
    return True


def matching_capabilities(verb: str, candidate: inspect.Signature) -> list[type]:
    """Return declared capabilities whose call shape this signature accepts."""
    return [p for p in CAPABILITIES[verb] if accepts(candidate, p)]


def public_verbs(cls: type) -> list[str]:
    """Public lifecycle and lifecycle-like verbs of a class, including inherited."""
    return [
        name
        for name in dir(cls)
        if not name.startswith("_")
        and (name in VERBS or name.startswith(FORBIDDEN_PREFIXES))
        and callable(getattr(cls, name))
    ]


def audit(cls: type) -> list[str]:
    """Return every lifecycle-contract violation of one class."""
    problems: list[str] = []
    qualname = cls.__qualname__
    for verb in public_verbs(cls):
        if verb in LISTED_VERBS:
            if not issubclass(cls, Model):
                problems.append(f"{qualname}.{verb} listed only for Model subclasses")
            continue
        if verb not in VERBS:
            problems.append(f"{qualname}.{verb} is an unlisted lifecycle verb")
            continue
        if (qualname, verb) in SIGNATURE_EXCEPTIONS:
            continue
        signature = inspect.signature(getattr(cls, verb))
        if not matching_capabilities(verb, signature):
            problems.append(f"{qualname}.{verb}{signature} matches no capability")
    return problems


def concrete_api_classes() -> list[type]:
    """Every concrete class defined in ``statadic.api``, private modules included."""
    found: dict[type, None] = {}
    for info in pkgutil.walk_packages(statadic.api.__path__, "statadic.api."):
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if (
                inspect.isclass(value)
                and value.__module__.startswith("statadic.api")
                and not getattr(value, "_is_protocol", False)
                and value.__module__ != capabilities.__name__
            ):
                found[value] = None
    return sorted(found, key=lambda c: (c.__module__, c.__qualname__))


def test_every_api_lifecycle_verb_matches_a_capability_or_recorded_exception() -> None:
    """No result or inference class drifts from the declared capability shapes."""
    classes = concrete_api_classes()
    problems = [p for cls in classes for p in audit(cls)]
    assert problems == []


def test_walk_reaches_the_known_result_and_inference_families() -> None:
    """The enumeration is not vacuous: representative classes are audited."""
    audited = {c.__qualname__ for c in concrete_api_classes() if public_verbs(c)}
    for name in (
        "LinearResult",
        "IVResult",
        "SurvivalResult",
        "SpatialPanelResult",
        "GraphRegressionResult",
        "DyadicResult",
        "NodeBootstrapResult",
        "MantelInference",
        "GELInference",
        "OrdinationResult",
        "FittedDesign",
    ):
        assert name in audited


def test_recorded_exceptions_name_real_methods() -> None:
    """A stale exception fails rather than silently widening the contract."""
    by_name = {c.__qualname__: c for c in concrete_api_classes()}
    for qualname, verb in SIGNATURE_EXCEPTIONS:
        signature = inspect.signature(getattr(by_name[qualname], verb))
        assert not matching_capabilities(verb, signature), (qualname, verb)


class _DivergentPredict:
    def predict(self, data: object, mode: str) -> object:
        return data, mode


class _DivergentInfer:
    def infer(self, *, covar: object) -> object:
        return covar


class _DivergentTransform:
    def transform(self) -> object:
        return None


class _DivergentDiagnose:
    def diagnose(self, request: object) -> object:
        return request


class _AliasVerb:
    def predict_mean(self, data: object) -> object:
        return data


class _RefinedEstimate:
    def estimate_refined(self, data: object) -> object:
        return data


class _Conforming:
    def predict(self, data: object, *, kind: str = "mean") -> object:
        return data, kind

    def infer(self, *, covariance: object) -> object:
        return covariance

    def transform(self, data: object) -> object:
        return data

    def diagnose(self, *requests: object) -> object:
        return requests

    def test(self, hypothesis: object) -> object:
        return hypothesis


@pytest.mark.parametrize(
    "fixture",
    [
        _DivergentPredict,
        _DivergentInfer,
        _DivergentTransform,
        _DivergentDiagnose,
        _AliasVerb,
        _RefinedEstimate,
    ],
)
def test_divergent_fixture_fails_the_audit(fixture: Callable[[], object]) -> None:
    """Drifted signatures and unlisted verbs are reported."""
    assert audit(fixture) != []  # type: ignore[arg-type]


def test_conforming_fixture_passes_the_audit() -> None:
    """The audit accepts a class whose verbs match declared capabilities."""
    assert audit(_Conforming) == []
