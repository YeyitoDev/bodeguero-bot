"""The lector: a small DistilBERT classifier (int8 ONNX) trained on bodega questions.

It reads, it doesn't write: one pass gives the intent and the spans
(PRODUCTO, MARCA, MEDIDA, CANTIDAD, PERIODO). Training lives in lector/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from bodeguero.adapters.outbound.classifier.tokenizador import cargar, tramos_de_bio
from bodeguero.application.routing import route_for
from bodeguero.domain.models import Intent, Route


class LectorRouter:
    def __init__(self, model_dir: str | Path) -> None:
        d = Path(model_dir)
        self._tok = cargar(str(d / "tokenizador.json"), str(d / "vocab.txt"))
        labels = json.loads((d / "etiquetas.json").read_text(encoding="utf-8"))
        self._intents = [Intent(n) for n in labels["intenciones"]]
        self._bio = labels["bio"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        # Same int8 kernels on the Mac and on x86 servers (see lector/entrenar/exportar.py CAMINOS).
        so.add_session_config_entry("mlas.disable_kleidiai", "1")
        self._session = ort.InferenceSession(str(d / "lector.onnx"), so, providers=["CPUExecutionProvider"])
        self._needs_types = any(i.name == "token_type_ids" for i in self._session.get_inputs())

    def route(self, question: str) -> Route:
        enc = self._tok.codificar(question)
        ids = np.array([enc.input_ids], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        if self._needs_types:
            feed["token_type_ids"] = np.zeros_like(ids)
        p_intent, _flags, p_bio = self._session.run(["intencion", "banderas", "tramos"], feed)

        best = int(p_intent[0].argmax())
        intent = self._intents[best]
        bio_ids = p_bio[0].argmax(-1)
        word_tags = [self._bio[bio_ids[p]] if p >= 0 else "O" for p in enc.primera_pieza]
        spans: dict[str, str] = {}
        for t in tramos_de_bio(word_tags, [(w.inicio, w.fin) for w in enc.palabras]):
            spans.setdefault(t["etiqueta"], question[t["inicio"] : t["fin"]])

        confidence = float(p_intent[0][best])
        return route_for(intent, confidence, f"lector p={confidence:.2f}", spans)
