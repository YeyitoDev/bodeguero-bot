from bodeguero.application.catalog import find_products, products, reply
from bodeguero.domain.models import Chunk, Domain, Intent, RetrievedChunk

BEBIDAS = """# Bebidas

## Gaseosas
Inca Kola 500 ml: precio de venta S/ 2.50. Stock actual: 36 unidades.

Inca Kola 1.5 litros: precio de venta S/ 6.50. Stock actual: 12 unidades.

## Cervezas
Cerveza Pilsen Callao 630 ml: precio de venta S/ 6.00. Stock actual: 30 botellas. Se cobra S/ 1.00 sin envase."""


def _chunks(text: str = BEBIDAS) -> list[RetrievedChunk]:
    return [RetrievedChunk(Chunk("catalogo/bebidas.md", Domain.INVENTORY, 0, text), 0.9)]


def _one(query, chunks=None, **kw):
    [product] = find_products(query, chunks or _chunks(), **kw)
    return product


def test_spoken_size_and_section_heading_find_the_right_line():
    product = _one("gaseosa litro y medio")
    assert product.name == "Inca Kola 1.5 litros"
    assert reply(Intent.PRICE, product) == "Inca Kola 1.5 litros: S/ 6.50."


def test_stock_reply_keeps_the_note_after_the_stock():
    assert reply(Intent.STOCK, _one("cerveza Pilsen")) == (
        "Cerveza Pilsen Callao 630 ml: te quedan 30 botellas. Se cobra S/ 1.00 sin envase."
    )


def test_unknown_brand_is_not_answered_with_another_brand():
    # "Cristal" is a beer we don't sell: answering Pilsen's price would be wrong.
    assert find_products("cerveza Cristal", _chunks(), brand="Cristal") == []


def test_price_per_unit_is_kept():
    chunks = _chunks("Huevos a granel: precio de venta S/ 0.60 la unidad. Stock actual: 180 huevos.")
    assert reply(Intent.PRICE, _one("huevo", chunks)) == "Huevos a granel: S/ 0.60 la unidad."


def test_unknown_size_is_not_answered_with_another_size():
    assert find_products("Inca Kola 3 litros", _chunks(), brand="Inca Kola", size="3 litros") == []
    assert _one("Inca Kola litro y medio", size="litro y medio").name == "Inca Kola 1.5 litros"


def test_a_category_or_a_brand_without_size_gives_every_matching_line():
    assert [p.name for p in find_products("gaseosas", _chunks())] == ["Inca Kola 500 ml", "Inca Kola 1.5 litros"]
    assert len(find_products("Inca Kola", _chunks(), brand="Inca Kola")) == 2


def test_inventory_line_has_price_and_stock():
    assert [reply(Intent.INVENTORY, p) for p in products(_chunks())] == [
        "Inca Kola 500 ml: S/ 2.50, quedan 36 unidades.",
        "Inca Kola 1.5 litros: S/ 6.50, quedan 12 unidades.",
        "Cerveza Pilsen Callao 630 ml: S/ 6.00, quedan 30 botellas.",
    ]
