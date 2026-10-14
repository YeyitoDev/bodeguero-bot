"""Document source port: where raw documents come from (folder, S3, a database...)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from bodeguero.domain.models import Document


class DocumentSourcePort(Protocol):
    def load(self) -> Iterable[Document]:
        ...
