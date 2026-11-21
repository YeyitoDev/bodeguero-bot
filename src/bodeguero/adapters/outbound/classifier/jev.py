"""Jev (TypeSafe): a decision model that answers CLOSED questions with probabilities.

Same service and key as restaurant-system's llm-service/jev.py. It never writes
text: here it only picks the intent when the lector isn't sure. Any failure
raises, and FallbackRouter keeps the lector's answer.
"""

from __future__ import annotations

import httpx

from bodeguero.application.routing import route_for
from bodeguero.domain.models import Intent, Route

URL = "https://api.typesafe.ai/v1/systemone"

_CRITERIA = {
    Intent.PRICE.value: "Pregunta cuánto cuesta o a cuánto vende un producto.",
    Intent.STOCK.value: "Pregunta si tiene un producto o cuántas unidades le quedan.",
    Intent.SALES.value: "Pregunta por lo que vendió, ganó, gastó o le deben (fiado) en un periodo.",
    Intent.ADVICE.value: "Pide un consejo o pregunta una regla del negocio (fiado, ofertas, orden, pagos).",
    Intent.GREETING.value: "Saluda o pide ayuda sin preguntar nada todavía.",
    Intent.THANKS.value: "Agradece, se despide o cierra la conversación.",
    Intent.OTHER.value: "Habla de algo que no es su bodega.",
}


class JevRouter:
    def __init__(self, api_key: str, model: str = "jev-latest", timeout: float = 3.0) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def route(self, question: str) -> Route:
        body = {
            "model": self._model,
            "state": {"mensaje": question},
            "questions": {
                "intencion": {
                    "type": "choice",
                    "instructions": "¿Qué quiere el bodeguero con su `mensaje`, escrito al asistente de su bodega?",
                    "criteria": _CRITERIA,
                }
            },
        }
        r = httpx.post(URL, json=body, timeout=self._timeout, headers={"Authorization": f"Bearer {self._api_key}"})
        r.raise_for_status()
        answer = (r.json().get("answers") or {}).get("intencion") or {}
        intent = Intent(answer["choice"])
        confidence = float((answer.get("probabilities") or {}).get(answer["choice"], 0.0))
        return route_for(intent, confidence, f"jev p={confidence:.2f}")
