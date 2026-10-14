"""Loads .md / .txt files from a folder. The sub-folder name becomes the domain:

    data/catalogo/bebidas.md  -> Domain.INVENTORY
    data/ventas/octubre.md    -> Domain.SALES
    data/consejos/fiado.md    -> Domain.ADVICE
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from bodeguero.domain.models import Document, Domain

FOLDER_TO_DOMAIN = {
    "catalogo": Domain.INVENTORY,
    "inventario": Domain.INVENTORY,
    "ventas": Domain.SALES,
    "consejos": Domain.ADVICE,
}


class FileSystemSource:
    def __init__(self, root: str | Path, extensions: tuple[str, ...] = (".md", ".txt")) -> None:
        self._root = Path(root)
        self._extensions = extensions

    def load(self) -> Iterator[Document]:
        # TODO(yee): add loaders for PDF and CSV (sales exports). A CSV row is
        # better turned into a sentence ("El 12/10 se vendieron 30 Inca Kola
        # 500ml") than embedded raw.
        for path in sorted(self._root.rglob("*")):
            if path.suffix.lower() not in self._extensions or not path.is_file():
                continue
            rel = path.relative_to(self._root)
            domain = FOLDER_TO_DOMAIN.get(rel.parts[0], Domain.UNKNOWN) if len(rel.parts) > 1 else Domain.UNKNOWN
            yield Document(source=rel.as_posix(), text=path.read_text(encoding="utf-8"), domain=domain)
