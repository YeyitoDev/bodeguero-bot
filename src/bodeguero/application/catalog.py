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


def find_product(query: str, chunks: list[RetrievedChunk], brand: str = "", size: str = "") -> ProductLine | None:
    """The catalogue line whose name shares the most words with the query.

    A brand or size the bodeguero named must be in that line: a price for another
    brand or another size would be a wrong answer about money.
    """
    wanted, required = _words(query), _words(brand) | _words(size)
    best: tuple[int, ProductLine] | None = None
    for rc in chunks:
        heading = ""
        for line in (p.strip() for p in rc.chunk.text.splitlines()):
            if line.startswith("#"):
                heading = line.lstrip("# ")
                continue
            name, sep, _ = line.partition(":")
            if not sep or not _PRICE_RE.search(line):
                continue
            have = _words(f"{heading} {name}")
            if not required <= have:
                continue
            score = len(wanted & have)
            if score and (best is None or score > best[0]):
                best = (score, ProductLine(name=name.strip(), line=line, chunk=rc))
    return best[1] if best else None


def reply(intent: Intent, product: ProductLine) -> str:
    """Template answer for a price or stock question, citing the chunk as [1]."""
    if intent is Intent.STOCK:
        m = _STOCK_RE.search(product.line)
        if m:
            note = f" {m.group(2).strip()}" if m.group(2).strip() else ""
            return f"{product.name}: te quedan {m.group(1).strip()}.{note} [1]"
    price = _PRICE_RE.search(product.line)
    return f"{product.name}: S/ {price.group(1)}{price.group(2) or ''}. [1]"
