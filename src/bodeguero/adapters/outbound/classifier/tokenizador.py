"""
El tokenizador del lector, en Python puro y sin la librería `tokenizers`.

¿Por qué no usar el BertTokenizer de Hugging Face tal cual? Porque el Express
tiene que reproducirlo en TypeScript sin dependencias nativas, y eso sólo es
seguro si el algoritmo entero cabe en una página y no depende de detalles
internos de una librería. Aquí está esa página. Lo que entrena el modelo y lo
que compara el TypeScript (ejemplos-tokenizacion.jsonl) sale de este archivo.

El algoritmo, en cuatro pasos:

1. **Partir el texto en unidades** recorriendo los puntos de código (no las
   unidades UTF-16 de JavaScript):
   - letra (\\p{L}) o cifra (\\p{N}) abre o continúa una PALABRA;
   - una marca combinante (\\p{M}) continúa la palabra si hay una abierta, y
     si no se descarta (así el U+FE0F de «❤️» no forma palabra);
   - separador (\\p{Z}), control (\\p{Cc}) o formato (\\p{Cf}, como el U+200D
     de los emojis compuestos) cierra la palabra y se descarta;
   - cualquier otro punto de código (signos, emojis) cierra la palabra y es
     una unidad SÍMBOLO de un solo punto de código.
   Es la misma división en palabras que usa `generar/comun.py` para el BIO,
   salvo la marca suelta, así que los tramos caen siempre en límites de
   unidad.
2. **Normalizar cada unidad**: minúsculas si `minusculas`; luego, si
   `quitar_tildes`, NFD y fuera las \\p{Mn}; si no, NFC (para que una «é»
   escrita como «e» + tilde suelta se encuentre en el vocabulario).
3. **WordPiece** sobre cada unidad: el trozo más largo del vocabulario desde
   el principio; los siguientes llevan «##» delante. Si algún trozo no está,
   o la unidad pasa de `max_letras_por_palabra` puntos de código, la unidad
   entera es [UNK].
4. **Armar la secuencia**: [CLS] + piezas + [SEP]. Si no caben en
   `largo_maximo`, se cortan las piezas del final (las palabras cortadas no
   reciben etiqueta: el Express las da por O).
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field


@dataclass
class Unidad:
    inicio: int          # en puntos de código sobre el texto tal cual
    fin: int             # excluido
    palabra: bool        # False para los símbolos (signos, emojis)


@dataclass
class Codificado:
    input_ids: list[int]
    # Para cada unidad PALABRA, el índice de su primera pieza en input_ids
    # (con el [CLS] contado), o -1 si la cortó el largo máximo o no dio piezas.
    primera_pieza: list[int]
    palabras: list[Unidad]
    piezas: list[str] = field(default_factory=list)


def _cat(c: str) -> str:
    return unicodedata.category(c)


def partir(texto: str) -> list[Unidad]:
    """El paso 1: el texto en unidades PALABRA y SÍMBOLO."""
    res: list[Unidad] = []
    abierta = -1  # inicio de la palabra abierta, o -1
    for i, c in enumerate(texto):
        g = _cat(c)
        if g[0] in ('L', 'N'):
            if abierta < 0:
                abierta = i
            continue
        if g[0] == 'M':
            # La marca continúa la palabra abierta; suelta, se descarta.
            if abierta < 0:
                continue
            continue
        if abierta >= 0:
            res.append(Unidad(abierta, i, True))
            abierta = -1
        if g[0] == 'Z' or g in ('Cc', 'Cf'):
            continue
        res.append(Unidad(i, i + 1, False))
    if abierta >= 0:
        res.append(Unidad(abierta, len(texto), True))
    return res


class Tokenizador:
    def __init__(self, vocab: list[str], minusculas: bool, quitar_tildes: bool,
                 largo_maximo: int = 64, max_letras_por_palabra: int = 100,
                 cls: str = '[CLS]', sep: str = '[SEP]', unk: str = '[UNK]', pad: str = '[PAD]'):
        self.vocab = vocab
        self.ids = {p: i for i, p in enumerate(vocab)}
        self.minusculas = minusculas
        self.quitar_tildes = quitar_tildes
        self.largo_maximo = largo_maximo
        self.max_letras = max_letras_por_palabra
        self.cls, self.sep, self.unk, self.pad = cls, sep, unk, pad
        for t in (cls, sep, unk, pad):
            assert t in self.ids, f'falta {t} en el vocabulario'

    # ── carga y descripción ────────────────────────────────────────────────

    @classmethod
    def desde_vocab(clase, ruta_vocab: str, **kw) -> Tokenizador:
        with open(ruta_vocab, encoding='utf-8') as f:
            # Una pieza por línea; el id es el número de línea (desde 0). Se
            # quita sólo el salto de línea: hay piezas que son espacios raros.
            vocab = [linea.rstrip('\n').rstrip('\r') for linea in f]
        return clase(vocab, **kw)

    def descripcion(self) -> dict:
        """Lo que va a tokenizador.json para el TypeScript."""
        return {
            'tipo': 'wordpiece',
            'vocabulario': 'vocab.txt (una pieza por línea; el id es el número de línea desde 0)',
            'tamano_vocabulario': len(self.vocab),
            'minusculas': self.minusculas,
            'quitar_tildes': self.quitar_tildes,
            'normalizacion_si_no_se_quitan_tildes': 'NFC',
            'largo_maximo': self.largo_maximo,
            'max_letras_por_palabra': self.max_letras,
            'prefijo_continuacion': '##',
            'tokens_especiales': {
                'cls': {'pieza': self.cls, 'id': self.ids[self.cls]},
                'sep': {'pieza': self.sep, 'id': self.ids[self.sep]},
                'unk': {'pieza': self.unk, 'id': self.ids[self.unk]},
                'pad': {'pieza': self.pad, 'id': self.ids[self.pad]},
            },
            'particion': {
                'palabra': 'tira de \\p{L} y \\p{N}; una \\p{M} continúa la palabra abierta y suelta se descarta',
                'descartado': '\\p{Z}, \\p{Cc}, \\p{Cf} y las \\p{M} sueltas',
                'simbolo': 'cualquier otro punto de código, uno por unidad',
                'recorrido': 'por puntos de código (en JavaScript: for…of o Array.from, nunca por índice UTF-16)',
            },
            'truncado': 'si [CLS] + piezas + [SEP] pasa de largo_maximo, se cortan las piezas del final',
            # La revisión del v1 tuvo que adivinarlo: no estaba escrito (22/10/2026).
            'primera_pieza': ('índice en input_ids (con el [CLS] contado) de la primera pieza de cada palabra; '
                              '-1 si el truncado la dejó fuera entera o no dio piezas: esa palabra va como O. '
                              'Si su primera pieza cabe y las siguientes no, lleva el índice de la primera'),
            'etiqueta_de_palabra': 'la de su PRIMERA pieza; los símbolos y las demás piezas no llevan etiqueta',
        }

    # ── los pasos ──────────────────────────────────────────────────────────

    def normalizar(self, unidad: str) -> str:
        t = unidad.lower() if self.minusculas else unidad
        if self.quitar_tildes:
            t = unicodedata.normalize('NFD', t)
            t = ''.join(c for c in t if _cat(c) != 'Mn')
        else:
            t = unicodedata.normalize('NFC', t)
        return t

    def wordpiece(self, palabra: str) -> list[str]:
        if not palabra:
            return []
        if len(palabra) > self.max_letras:
            return [self.unk]
        piezas, inicio, n = [], 0, len(palabra)
        while inicio < n:
            fin, encontrada = n, None
            while inicio < fin:
                trozo = palabra[inicio:fin]
                if inicio > 0:
                    trozo = '##' + trozo
                if trozo in self.ids:
                    encontrada = trozo
                    break
                fin -= 1
            if encontrada is None:
                return [self.unk]
            piezas.append(encontrada)
            inicio = fin
        return piezas

    def codificar(self, texto: str) -> Codificado:
        unidades = partir(texto)
        piezas: list[str] = [self.cls]
        palabras: list[Unidad] = []
        primera: list[int] = []
        cabe = self.largo_maximo - 1  # deja sitio al [SEP]
        for u in unidades:
            ps = self.wordpiece(self.normalizar(texto[u.inicio:u.fin]))
            if u.palabra:
                palabras.append(u)
                primera.append(len(piezas) if ps and len(piezas) < cabe else -1)
            for p in ps:
                if len(piezas) < cabe:
                    piezas.append(p)
        piezas.append(self.sep)
        return Codificado([self.ids[p] for p in piezas], primera, palabras, piezas)


def cargar(ruta_json: str, ruta_vocab: str) -> Tokenizador:
    """Carga el tokenizador desde los artefactos (tokenizador.json + vocab.txt)."""
    with open(ruta_json, encoding='utf-8') as f:
        d = json.load(f)
    e = d['tokens_especiales']
    return Tokenizador.desde_vocab(
        ruta_vocab, minusculas=d['minusculas'], quitar_tildes=d['quitar_tildes'],
        largo_maximo=d['largo_maximo'], max_letras_por_palabra=d['max_letras_por_palabra'],
        cls=e['cls']['pieza'], sep=e['sep']['pieza'], unk=e['unk']['pieza'], pad=e['pad']['pieza'])


def tramos_de_bio(etiquetas: list[str], palabras: list[tuple[int, int]]) -> list[dict]:
    """Junta el BIO por palabra en tramos por caracteres, como seqeval.

    Un tramo empieza en B-X, o en I-X si lo anterior no era X (seqeval en su
    modo por defecto hace lo mismo), y sigue mientras vengan I-X.
    """
    res, cur = [], None
    for (a, b), e in zip(palabras, etiquetas, strict=False):
        if e == 'O':
            if cur:
                res.append(cur)
            cur = None
            continue
        p, t = e.split('-', 1)
        if p == 'B' or cur is None or cur['etiqueta'] != t:
            if cur:
                res.append(cur)
            cur = {'inicio': a, 'fin': b, 'etiqueta': t}
        else:
            cur['fin'] = b
    if cur:
        res.append(cur)
    return res
