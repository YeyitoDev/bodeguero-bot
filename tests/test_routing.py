import pytest

from bodeguero.application.routing import FallbackRouter, KeywordRouter
from bodeguero.domain.models import Domain, Intent, ModelTier, Route


@pytest.mark.parametrize(
    ("question", "domain"),
    [
        ("¿Cuánto cuesta el aceite Primor?", Domain.INVENTORY),
        ("¿Qué fue lo que más vendí la semana pasada?", Domain.SALES),
        ("Dame un consejo para mejorar mis ventas de verano", Domain.ADVICE),
        ("hola", Domain.UNKNOWN),
    ],
)
def test_keyword_router_domain(question, domain):
    assert KeywordRouter().route(question).domain is domain


def test_reasoning_questions_go_to_large_model():
    assert KeywordRouter().route("¿Por qué bajaron mis ventas?").tier is ModelTier.LARGE
    assert KeywordRouter().route("¿Cuánto cuesta el arroz?").tier is ModelTier.SMALL


class _Fixed:
    def __init__(self, route=None, error=None):
        self._route, self._error = route, error

    def route(self, question):
        if self._error:
            raise self._error
        return self._route


def _route(intent, confidence, **spans):
    return Route(domain=Domain.UNKNOWN, tier=ModelTier.SMALL, intent=intent, confidence=confidence, spans=spans)


def test_fallback_is_not_asked_when_the_primary_is_sure():
    sure = _route(Intent.PRICE, 0.95)
    assert FallbackRouter(_Fixed(sure), _Fixed(error=AssertionError("must not run")), 0.7).route("q") is sure


def test_unsure_primary_takes_the_fallback_intent_but_keeps_its_spans():
    unsure = _route(Intent.SALES, 0.4, PRODUCTO="aceite")
    routed = FallbackRouter(_Fixed(unsure), _Fixed(_route(Intent.PRICE, 0.9)), 0.7).route("q")
    assert routed.intent is Intent.PRICE
    assert routed.spans == {"PRODUCTO": "aceite"}


def test_failing_fallback_keeps_the_primary():
    unsure = _route(Intent.SALES, 0.4)
    assert FallbackRouter(_Fixed(unsure), _Fixed(error=TimeoutError()), 0.7).route("q") is unsure
