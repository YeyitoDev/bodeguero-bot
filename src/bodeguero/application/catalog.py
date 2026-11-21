"""Catalogue tool: find the product line inside retrieved chunks and answer price or stock.

The catalogue lines look like
    "Aceite Primor 900 ml: precio de venta S/ 9.90. Stock actual: 10 botellas. Quedan pocas, reponer esta semana."
Only the product name (before ":") plus its section heading ("## Gaseosas") is
matched against the question, because "precio" and "stock" appear on every line.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from bodeguero.domain.models import Intent, RetrievedChunk

# Ways people say a size that the catalogue writes differently.
_SIZES = {"litro y medio": "1.5 litros", "medio litro": "500 ml", "medio kilo": "500 g", "un kilo": "1 kg"}
_STOPWORDS = {
    "a", "al", "de", "del", "el", "la", "las", "lo", "los", "un", "una", "y", "en", "me", "mi", "que", "cuanto",
    "cuanta", "cuantos", "cuantas", "cuesta", "vale", "precio", "esta", "tengo", "hay", "queda", "quedan", "stock",
    "vendo", "dejo", "cobro", "es", "por", "para", "todavia",
}
_PRICE_RE = re.compile(r"S/\s*(\d+(?:[.,]\d+)?)(\s+la unidad)?")
_STOCK_RE = re.compile(r"Stock actual:\s*([^.]+)\.\s*(.*)")


def _words(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    for said, written in _SIZES.items():
        text = text.replace(said, written)
    # Plural to singular, enough for "huevos"/"huevo" and "litros"/"litro".
    words = re.findall(r"\d+(?:\.\d+)?|[a-z]+", text)
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words} - _STOPWORDS


@dataclass(frozen=True)
class ProductLine:
    name: str
    line: str
    chunk: RetrievedChunk
    keys: frozenset[str]  # words of the section heading + product name


def products(chunks: list[RetrievedChunk]) -> list[ProductLine]:
    """Every catalogue line in the chunks, in document order, once."""
    found, seen = [], set()
    for rc in sorted(chunks, key=lambda c: (c.chunk.source, c.chunk.position)):
        heading = ""
        for line in (p.strip() for p in rc.chunk.text.splitlines()):
            if line.startswith("#"):
                heading = line.lstrip("# ")
                continue
            name, sep, _ = line.partition(":")
            if sep and _PRICE_RE.search(line) and line not in seen:
                seen.add(line)
                found.append(ProductLine(name.strip(), line, rc, frozenset(_words(f"{heading} {name}"))))
    # A chunk starts with the cut-off tail of the previous one (chunking overlap): drop those halves.
    return [p for p in found if not any(p.line != q.line and p.line in q.line for q in found)]


def find_products(query: str, chunks: list[RetrievedChunk], brand: str = "", size: str = "") -> list[ProductLine]:
    """The catalogue lines whose names share the most words with the query.

    Ties are all returned: "gaseosas" matches every line under "## Gaseosas", and
    "Inca Kola" without a size matches both sizes. A brand or size the bodeguero
    named must be in the line: a price for another brand or size would be wrong money.
    """
    wanted, required = _words(query), _words(brand) | _words(size)
    scored = [(len(wanted & p.keys), p) for p in products(chunks) if required <= p.keys]
    best = max((s for s, _ in scored), default=0)
    return [p for s, p in scored if s == best] if best else []


def reply(intent: Intent, product: ProductLine) -> str:
    """Template answer line for a price, stock or inventory question."""
    price = _PRICE_RE.search(product.line)
    cost = f"S/ {price.group(1)}{price.group(2) or ''}"
    stock = _STOCK_RE.search(product.line)
    if intent is Intent.PRICE or not stock:
        return f"{product.name}: {cost}."
    left = stock.group(1).strip()
    if intent is Intent.INVENTORY:
        return f"{product.name}: {cost}, quedan {left}."
    note = f" {stock.group(2).strip()}" if stock.group(2).strip() else ""
    return f"{product.name}: te quedan {left}.{note}"
