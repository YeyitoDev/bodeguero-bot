"""
Los conjuntos del lector pasados a lo que come el modelo, y las métricas.

Las etiquetas salen de `lector/generar/comun.py`, que es la definición única
de lo que predice el lector: si allí cambia una etiqueta, aquí cambia sola.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

AQUI = os.path.dirname(os.path.abspath(__file__))
LECTOR = os.path.dirname(AQUI)
DATOS = os.path.join(LECTOR, 'datos')
sys.path.insert(0, os.path.join(LECTOR, 'generar'))
# El tokenizador vive en el bot, que lo usa para leer en producción: una sola copia.
sys.path.insert(0, os.path.join(os.path.dirname(LECTOR), 'src', 'bodeguero', 'adapters', 'outbound', 'classifier'))

from comun import BANDERAS, ETIQUETAS, INTENCIONES  # noqa: E402

from tokenizador import Tokenizador, tramos_de_bio  # noqa: E402,F401

# El BIO por pieza: O primero (id 0), luego B- e I- de cada etiqueta en el
# orden de comun.ETIQUETAS. El orden queda escrito en etiquetas.json.
BIO = ['O'] + [f'{p}-{e}' for e in ETIQUETAS for p in ('B', 'I')]
BIO_ID = {b: i for i, b in enumerate(BIO)}
INTENCION_ID = {n: i for i, n in enumerate(INTENCIONES)}
IGNORAR = -100  # piezas sin etiqueta: especiales, símbolos y piezas que no son la primera


def leer(nombre: str, carpeta: str | None = None) -> list[dict]:
    """Un conjunto de lector/datos, o de otra carpeta (los pliegues de la validación cruzada)."""
    with open(os.path.join(carpeta or DATOS, f'{nombre}.jsonl'), encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


@dataclass
class Ejemplo:
    fila: dict
    input_ids: list[int]
    intencion: int
    banderas: list[float]
    tramos: list[int]           # una por pieza, IGNORAR donde no va etiqueta
    palabras: list[tuple[int, int]]
    primera_pieza: list[int]


def bio_de_palabras(palabras, tramos) -> list[str]:
    """La etiqueta BIO de cada palabra según los tramos de la fila.

    La primera palabra dentro de un tramo es B-, las siguientes I-. Se mira
    la contención y no la igualdad de inicio, para que una palabra que
    empiece con una marca suelta (descartada por el tokenizador) no quede
    fuera de su tramo.
    """
    res = []
    vistos = set()
    for (a, b) in palabras:
        e = 'O'
        for k, t in enumerate(tramos):
            if t['inicio'] <= a and b <= t['fin']:
                e = ('I-' if k in vistos else 'B-') + t['etiqueta']
                vistos.add(k)
                break
        res.append(e)
    return res


def preparar(filas: list[dict], tok: Tokenizador) -> list[Ejemplo]:
    ejemplos = []
    for f in filas:
        c = tok.codificar(f['texto'])
        palabras = [(u.inicio, u.fin) for u in c.palabras]
        bio = bio_de_palabras(palabras, f['tramos'])
        tramos = [IGNORAR] * len(c.input_ids)
        for p, e in zip(c.primera_pieza, bio):
            if p >= 0:
                tramos[p] = BIO_ID[e]
        ejemplos.append(Ejemplo(
            fila=f, input_ids=c.input_ids, intencion=INTENCION_ID[f['intencion']],
            banderas=[1.0 if f['banderas'][b] else 0.0 for b in BANDERAS],
            tramos=tramos, palabras=palabras, primera_pieza=c.primera_pieza))
    return ejemplos


# ── De la salida del modelo a la lectura ────────────────────────────────────

def lectura(ej: Ejemplo, p_intencion, p_banderas, bio_por_pieza) -> dict:
    """La lectura completa de una fila a partir de las salidas del modelo.

    p_intencion: lista de 7 probabilidades; p_banderas: 3 probabilidades;
    bio_por_pieza: el id BIO elegido en cada pieza.
    """
    etiquetas = [BIO[bio_por_pieza[p]] if p >= 0 else 'O' for p in ej.primera_pieza]
    return {
        'intencion': INTENCIONES[max(range(len(p_intencion)), key=lambda i: p_intencion[i])],
        'banderas': {b: bool(p_banderas[i] >= 0.5) for i, b in enumerate(BANDERAS)},
        'tramos': tramos_de_bio(etiquetas, ej.palabras),
    }


# ── Métricas ────────────────────────────────────────────────────────────────

def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def metricas(filas: list[dict], lecturas: list[dict]) -> dict:
    n = len(filas)
    acierto_int = sum(f['intencion'] == l['intencion'] for f, l in zip(filas, lecturas))
    banderas = {}
    btp = bfp = bfn = 0
    for b in BANDERAS:
        tp = sum(f['banderas'][b] and l['banderas'][b] for f, l in zip(filas, lecturas))
        fp = sum((not f['banderas'][b]) and l['banderas'][b] for f, l in zip(filas, lecturas))
        fn = sum(f['banderas'][b] and not l['banderas'][b] for f, l in zip(filas, lecturas))
        p, r, f1 = _f1(tp, fp, fn)
        banderas[b] = {'precision': p, 'recall': r, 'f1': f1, 'apoyo': tp + fn}
        btp, bfp, bfn = btp + tp, bfp + fp, bfn + fn
    por_etiqueta = {}
    TP = FP = FN = 0
    for e in ETIQUETAS:
        tp = fp = fn = 0
        for f, l in zip(filas, lecturas):
            oro = {(t['inicio'], t['fin']) for t in f['tramos'] if t['etiqueta'] == e}
            dicho = {(t['inicio'], t['fin']) for t in l['tramos'] if t['etiqueta'] == e}
            tp += len(oro & dicho)
            fp += len(dicho - oro)
            fn += len(oro - dicho)
        p, r, f1 = _f1(tp, fp, fn)
        por_etiqueta[e] = {'precision': p, 'recall': r, 'f1': f1, 'apoyo': tp + fn}
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    p, r, f1 = _f1(TP, FP, FN)
    macro = sum(v['f1'] for v in por_etiqueta.values() if v['apoyo']) / max(1, sum(1 for v in por_etiqueta.values() if v['apoyo']))
    completas = sum(es_exacta(f, l) for f, l in zip(filas, lecturas))
    return {
        'filas': n,
        'intencion_exactitud': acierto_int / n if n else 0.0,
        'banderas': banderas,
        'banderas_f1_media': sum(v['f1'] for v in banderas.values()) / len(banderas),
        'banderas_f1_micro': _f1(btp, bfp, bfn)[2],
        'tramos_micro': {'precision': p, 'recall': r, 'f1': f1},
        'tramos_macro_f1': macro,
        'tramos_por_etiqueta': por_etiqueta,
        'lectura_exacta': completas / n if n else 0.0,
    }


def es_exacta(f: dict, l: dict) -> bool:
    """La fila entera bien: intención, las tres banderas y todos los tramos."""
    return (f['intencion'] == l['intencion']
            and all(f['banderas'][b] == l['banderas'][b] for b in BANDERAS)
            and {(t['inicio'], t['fin'], t['etiqueta']) for t in f['tramos']}
            == {(t['inicio'], t['fin'], t['etiqueta']) for t in l['tramos']})


def puntaje(m: dict) -> float:
    """Un solo número para elegir la mejor época: la media de las tres cabezas.

    Las banderas entran con su F1 conjunto (las tres juntas) y no con la media
    de cada una, porque validación trae una sola fila con apuro: su F1 suelto
    salta de 0 a 1 por un caso y haría elegir la época a la suerte.
    """
    return (m['intencion_exactitud'] + m['banderas_f1_micro'] + m['tramos_micro']['f1']) / 3


def marcado(texto: str, tramos: list[dict]) -> str:
    """El texto con cada tramo entre corchetes, como en ESQUEMA.md."""
    out, i = [], 0
    for t in sorted(tramos, key=lambda t: t['inicio']):
        out.append(texto[i:t['inicio']])
        out.append(f"[{texto[t['inicio']:t['fin']]}]{t['etiqueta']}")
        i = t['fin']
    out.append(texto[i:])
    return ''.join(out)
