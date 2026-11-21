from pathlib import Path

import pytest

from bodeguero.domain.models import Domain, Intent

MODEL_DIR = Path(__file__).resolve().parents[1] / "modelos" / "lector"
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "lector.onnx").exists(), reason="lector model not built")


@pytest.fixture(scope="module")
def lector():
    from bodeguero.adapters.outbound.classifier.lector import LectorRouter

    return LectorRouter(MODEL_DIR)


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("¿a cuánto vendo el aceite Primor?", Intent.PRICE),  # "vendo" but it's a price question
        ("¿cuánto vendí de aceite la semana pasada?", Intent.SALES),
        ("¿cuál es el límite de fiado?", Intent.ADVICE),
        ("¿me quedan huevos?", Intent.STOCK),
        ("hola causa", Intent.GREETING),
        ("¿cuánto está el dólar?", Intent.OTHER),
    ],
)
def test_lector_reads_the_intent(lector, question, intent):
    assert lector.route(question).intent is intent


def test_lector_extracts_product_brand_and_size(lector):
    route = lector.route("¿cuánto cuesta la gaseosa Inca Kola de litro y medio?")
    assert route.domain is Domain.INVENTORY
    assert route.spans == [{"PRODUCTO": "gaseosa", "MARCA": "Inca Kola", "MEDIDA": "litro y medio"}]


def test_lector_splits_several_products(lector):
    route = lector.route("frituras, panetón, turrón?")
    assert [s.get("PRODUCTO") for s in route.spans] == ["frituras", "panetón", "turrón"]
