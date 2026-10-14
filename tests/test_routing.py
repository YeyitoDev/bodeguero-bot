import pytest

from bodeguero.application.routing import KeywordRouter
from bodeguero.domain.models import Domain, ModelTier


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
