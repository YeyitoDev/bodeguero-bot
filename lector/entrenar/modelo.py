"""
El lector: un codificador preentrenado con tres cabezas encima.

  - intencion: 7 clases sobre el [CLS];
  - banderas: 3 sigmoides sobre el [CLS] (apuro, abierto, usted), cada una
    por su cuenta, porque el apuro no implica abierto ni al revés;
  - tramos: el BIO de cada pieza (sólo cuenta la primera pieza de cada
    palabra).

¿Por qué un solo codificador y no tres modelos? Porque las tres preguntas
miran las mismas palabras: compartirlo cuesta una pasada en el Express en
vez de tres, y cada cabeza ayuda a las otras a entrenar (saber que «Tanta»
es un LOCAL ayuda a decir que la intención es «local»).
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class Lector(nn.Module):
    def __init__(self, base: str, n_intenciones: int, n_banderas: int, n_bio: int,
                 abandono: float = 0.1, atencion: str | None = None, pesos_base: bool = True):
        super().__init__()
        kw = {'attn_implementation': atencion} if atencion else {}
        if pesos_base:
            self.codificador = AutoModel.from_pretrained(base, **kw)
        else:
            self.codificador = AutoModel.from_config(AutoConfig.from_pretrained(base), **kw)
        cfg = self.codificador.config
        # DistilBERT no tiene token_type_ids: sólo BERT los recibe.
        self.usa_tipos = cfg.model_type == 'bert'
        h = cfg.hidden_size
        self.abandono = nn.Dropout(abandono)
        self.intencion = nn.Linear(h, n_intenciones)
        self.banderas = nn.Linear(h, n_banderas)
        self.tramos = nn.Linear(h, n_bio)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {}
        if self.usa_tipos:
            kw['token_type_ids'] = token_type_ids if token_type_ids is not None else torch.zeros_like(input_ids)
        h = self.codificador(input_ids=input_ids, attention_mask=attention_mask, **kw).last_hidden_state
        cls = self.abandono(h[:, 0])
        return self.intencion(cls), self.banderas(cls), self.tramos(self.abandono(h))


class LectorParaExportar(nn.Module):
    """Lo que va al .onnx: las mismas cabezas, ya pasadas a probabilidades.

    El Express recibe probabilidades y no logits para poder poner umbrales de
    confianza sin reimplementar el softmax: intencion [lote, 7] (softmax),
    banderas [lote, 3] (sigmoide), tramos [lote, largo, 23] (softmax por pieza).
    """

    def __init__(self, lector: Lector):
        super().__init__()
        self.lector = lector

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        i, b, t = self.lector(input_ids, attention_mask, token_type_ids)
        return torch.softmax(i, -1), torch.sigmoid(b), torch.softmax(t, -1)
