from bodeguero.application.catalog import find_product, reply
from bodeguero.domain.models import Chunk, Domain, Intent, RetrievedChunk

BEBIDAS = """# Bebidas

## Gaseosas
Inca Kola 500 ml: precio de venta S/ 2.50. Stock actual: 36 unidades.

Inca Kola 1.5 litros: precio de venta S/ 6.50. Stock actual: 12 unidades.

## Cervezas
Cerveza Pilsen Callao 630 ml: precio de venta S/ 6.00. Stock actual: 30 botellas. Se cobra S/ 1.00 sin envase."""


def _chunks(text: str = BEBIDAS) -> list[RetrievedChunk]:
    return [RetrievedChunk(Chunk("catalogo/bebidas.md", Domain.INVENTORY, 0, text), 0.9)]


def test_spoken_size_and_section_heading_find_the_right_line():
    product = find_product("gaseosa litro y medio", _chunks())
    assert product.name == "Inca Kola 1.5 litros"
    assert reply(Intent.PRICE, product) == "Inca Kola 1.5 litros: S/ 6.50. [1]"


def test_stock_reply_keeps_the_note_after_the_stock():
    product = find_product("cerveza Pilsen", _chunks())
    assert reply(Intent.STOCK, product) == (
        "Cerveza Pilsen Callao 630 ml: te quedan 30 botellas. Se cobra S/ 1.00 sin envase. [1]"
    )


def test_unknown_brand_is_not_answered_with_another_brand():
    # "Cristal" is a beer we don't sell: answering Pilsen's price would be wrong.
    assert find_product("cerveza Cristal", _chunks(), brand="Cristal") is None


def test_price_per_unit_is_kept():
    chunks = _chunks("Huevos a granel: precio de venta S/ 0.60 la unidad. Stock actual: 180 huevos.")
    assert reply(Intent.PRICE, find_product("huevo", chunks)) == "Huevos a granel: S/ 0.60 la unidad. [1]"


def test_unknown_size_is_not_answered_with_another_size():
    assert find_product("Inca Kola 3 litros", _chunks(), brand="Inca Kola", size="3 litros") is None
    assert find_product("Inca Kola litro y medio", _chunks(), size="litro y medio").name == "Inca Kola 1.5 litros"
