"""
Arma la carpeta modelos/lector con todo lo que necesita el bot.

Uso:
    python artefactos.py --corrida <carpeta de entrenar.py> --exportado <carpeta de exportar.py> \\
        --destino ../../modelos/lector

Escribe lector.onnx, vocab.txt, tokenizador.json, etiquetas.json,
ejemplos-tokenizacion.jsonl y ejemplos-salida.jsonl. El LEEME.md lo escribe
leeme.py después, porque lleva el sha256 de todo lo demás.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from datos import BANDERAS, BIO, INTENCIONES, ETIQUETAS, lectura, leer, preparar
from entrenar import carpeta_base
import onnxruntime as ort

from exportar import CAMINOS, alimentar, sesion
from tokenizador import Tokenizador

# Casos de borde para que el tokenizador de TypeScript se rompa aquí y no en
# producción: emojis pegados y compuestos, tildes sueltas (NFD), espacios
# raros, mayúsculas, ñ, palabras larguísimas y frases que no caben.
BORDES = [
    '',
    '   ',
    'ceviche🔥chicharrón',
    '❤️❤️ hola causa',
    '👨‍👩‍👧 algo para la familia',
    'quiero un café de altura',
    'QUIERO UN CHAUFA ESPECIAL!!!',
    '1/4 de pollo a las 10:30pm',
    "pa' 4 personas, máx S/ 120.50",
    'ñoquis con piña para año nuevo',
    'a' * 101 + ' ceviche',
    'em‌panada con aji de gallina',
    '¿¡Qué!? ¿tienen ají de gallina?',
    'hola\tcausa\nquiero\r\nun lomo',
    'İstanbul kebab en Miraflores',
    'ｃｅｖｉｃｈｅ de pescado',
    'çaña über crème brûlée',
    '日本料理 sushi en San Isidro',
    'oe causaaaa un chaufaaaa 🍜🍜🍜 pa ya!!!',
    'quiero ' + ' '.join(['ceviche'] * 70),
    'Buenas noches, ¿me podría recomendar un restaurante de comida criolla tradicional con '
    'estacionamiento, terraza, juegos para niños y música en vivo en Miraflores, Barranco, San Isidro, '
    'Surco o La Molina para una cena de cumpleaños de mi mamá el sábado a las ocho y media de la noche?',
    # Lo que la revisión del v1 echó de menos (22/10/2026): mayúsculas con tilde
    # y Ñ (la minúscula de «Á» es «á», no «a»), tonos de piel (un modificador
    # pegado al emoji), banderas (dos indicadores regionales) y el ZWJ con tono.
    'ÁNGEL QUIERE UN CAFÉ CON PIÑA EN ÑAÑA',
    'Ñoquis o ÑOQUIS, ÉL PIDE AJÍ DE GALLINA Y ÚNICO POSTRE',
    'dale 👍🏽 un chaufa',
    '🇵🇪 comida peruana 🇮🇹 o italiana',
    '👩🏻‍🍳 recomiéndame algo 🙏🏿',
]

# Los siete casos difíciles del encargo: van primero en ejemplos-salida.
DIFICILES = [
    '¿a cuánto vendo el aceite?',
    '¿cuánto vendí de aceite?',
    '¿cuánto está el dólar?',
    '¿debería subir el precio de la leche?',
    '¿tengo gaseosa de litro y medio?',
    'oe causa una inca kola gordita cuanto',
    'hola, ¿qué puedes hacer?',
]


def u16(texto: str, i: int) -> int:
    """El índice UTF-16 (el de JavaScript) del punto de código i."""
    return len(texto[:i].encode('utf-16-le')) // 2


def fila_tokenizacion(tok: Tokenizador, texto: str, id_: str) -> dict:
    c = tok.codificar(texto)
    return {
        'id': id_, 'texto': texto, 'input_ids': c.input_ids, 'piezas': c.piezas,
        # [inicio, fin] en puntos de código y [inicio, fin] en UTF-16, y el
        # índice de la primera pieza de cada palabra (-1 si se cortó).
        'palabras': [[u.inicio, u.fin, u16(texto, u.inicio), u16(texto, u.fin)] for u in c.palabras],
        'primera_pieza': c.primera_pieza,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corrida', required=True)
    ap.add_argument('--exportado', required=True)
    ap.add_argument('--destino', required=True)
    a = ap.parse_args()
    os.makedirs(a.destino, exist_ok=True)

    with open(os.path.join(a.corrida, 'config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    with open(os.path.join(a.exportado, 'medicion.json'), encoding='utf-8') as f:
        med = json.load(f)
    base = carpeta_base(cfg['base'])
    t = cfg['tokenizador']
    tok = Tokenizador.desde_vocab(os.path.join(base, 'vocab.txt'), minusculas=t['minusculas'],
                                  quitar_tildes=t['quitar_tildes'], largo_maximo=t['largo_maximo'])

    shutil.copyfile(os.path.join(a.exportado, 'lector.onnx'), os.path.join(a.destino, 'lector.onnx'))
    shutil.copyfile(os.path.join(base, 'vocab.txt'), os.path.join(a.destino, 'vocab.txt'))

    desc = tok.descripcion()
    desc['modelo_base'] = cfg['base']
    desc['entradas_del_onnx'] = med['entradas']
    desc['nota'] = ('attention_mask va en 1 en cada pieza real; con lote de 1 no hay relleno. '
                    + ('token_type_ids va todo en 0. ' if 'token_type_ids' in med['entradas'] else '')
                    + 'Los tensores son int64.')
    with open(os.path.join(a.destino, 'tokenizador.json'), 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False, indent=2)
        f.write('\n')

    etiquetas = {
        'intenciones': INTENCIONES,
        'banderas': BANDERAS,
        'etiquetas': ETIQUETAS,
        'bio': BIO,
        'salidas': {
            'intencion': f'[lote, {len(INTENCIONES)}] probabilidades (softmax) en el orden de «intenciones»',
            'banderas': f'[lote, {len(BANDERAS)}] probabilidad de cada bandera (sigmoide), encendida si ≥ 0,5',
            'tramos': f'[lote, largo, {len(BIO)}] probabilidades BIO por pieza (softmax) en el orden de «bio»',
        },
        'umbral_bandera': 0.5,
        'onnxruntime': {
            'version_probada': ort.__version__,
            'hilos': 1,
            # En arm64 (el Mac) onnxruntime usa por defecto otros núcleos int8
            # (KleidiAI) que en x86_64 (Fly). Apagarlos hace que el Mac lea
            # exactamente como producción; en x86_64 la entrada no cambia nada.
            'config_sesion': CAMINOS['clasico'],
        },
        'como_se_lee_el_bio': ('La etiqueta de cada palabra es el argmax en su PRIMERA pieza. Un tramo empieza '
                               'en B-X, o en I-X si la palabra anterior no era X, y sigue con las I-X que '
                               'vengan. El tramo va del inicio de su primera palabra al fin de la última.'),
    }
    with open(os.path.join(a.destino, 'etiquetas.json'), 'w', encoding='utf-8') as f:
        json.dump(etiquetas, f, ensure_ascii=False, indent=2)
        f.write('\n')

    # 500 filas: todos los casos de borde y, antes, las primeras frases de
    # prueba que quepan (en el v1 eran 479 + 21; desde el v2 los bordes van
    # siempre enteros, aunque la prueba crezca).
    pru = leer('prueba')
    filas_tok = [fila_tokenizacion(tok, f['texto'], f['id']) for f in pru[:500 - len(BORDES)]]
    filas_tok += [fila_tokenizacion(tok, x, f'borde-{i + 1:02d}') for i, x in enumerate(BORDES)]
    with open(os.path.join(a.destino, 'ejemplos-tokenizacion.jsonl'), 'w', encoding='utf-8') as f:
        for r in filas_tok:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 50 salidas esperadas: los siete difíciles y 43 de prueba repartidas
    # por intención (siempre las mismas, para que el archivo no cambie solo).
    # Salen por el camino clásico, el de producción (ver CAMINOS en exportar.py).
    s = sesion(os.path.join(a.destino, 'lector.onnx'), config=CAMINOS['clasico'])
    elegidas = [{'id': f'dificil-{i + 1}', 'texto': x} for i, x in enumerate(DIFICILES)]
    por_int = {n: [f for f in pru if f['intencion'] == n] for n in INTENCIONES}
    k = 0
    while len(elegidas) < 50:
        for n in INTENCIONES:
            if k < len(por_int[n]) and len(elegidas) < 50:
                elegidas.append({'id': por_int[n][k]['id'], 'texto': por_int[n][k]['texto']})
        k += 1
    vacias = [{'texto': e['texto'], 'intencion': 'fuera', 'banderas': {b: False for b in BANDERAS},
               'tramos': [], 'id': e['id'], 'origen': 'traza'} for e in elegidas]
    with open(os.path.join(a.destino, 'ejemplos-salida.jsonl'), 'w', encoding='utf-8') as f:
        for ej in preparar(vacias, tok):
            pi, pb, pt = s.run(['intencion', 'banderas', 'tramos'], alimentar(ej.input_ids, med['entradas']))
            bio_ids = pt[0].argmax(-1).tolist()
            lec = lectura(ej, pi[0].tolist(), pb[0].tolist(), bio_ids)
            texto = ej.fila['texto']
            f.write(json.dumps({
                'id': ej.fila['id'], 'texto': texto, 'input_ids': ej.input_ids,
                'intencion': [round(float(x), 6) for x in pi[0]],
                'banderas': [round(float(x), 6) for x in pb[0]],
                'tramos_argmax': bio_ids,
                'tramos_prob_max': [round(float(x), 6) for x in pt[0].max(-1)],
                'lectura': {
                    'intencion': lec['intencion'],
                    'banderas': lec['banderas'],
                    'tramos': [{'etiqueta': t['etiqueta'], 'texto': texto[t['inicio']:t['fin']],
                                'inicio': t['inicio'], 'fin': t['fin'],
                                'inicio_utf16': u16(texto, t['inicio']), 'fin_utf16': u16(texto, t['fin'])}
                               for t in lec['tramos']],
                },
            }, ensure_ascii=False) + '\n')
    print(f'artefactos en {a.destino}: {len(filas_tok)} tokenizaciones, {len(elegidas)} salidas')


if __name__ == '__main__':
    main()
