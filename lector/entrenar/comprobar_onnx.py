"""
Comprueba los artefactos de backend/modelos/lector sin PyTorch: sólo
onnxruntime, numpy y el tokenizador en Python puro.

Sirve para dos cosas:
  - correrlo en otra máquina u otra arquitectura (el Mac es arm64 y Fly es
    x86_64: la cuantización int8 usa caminos de CPU distintos en cada una) y
    ver que la lectura no cambia;
  - ser la referencia de lo que tiene que hacer el lector en TypeScript.

Uso: python comprobar_onnx.py <carpeta de artefactos> [--onnx otro.onnx] [--volcar lecturas.jsonl]

Con --volcar escribe la lectura de cada frase de prueba, para comparar dos
máquinas fila por fila (diff de los dos archivos).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import onnxruntime as ort

from datos import lectura, leer, metricas, preparar
from tokenizador import cargar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('carpeta')
    ap.add_argument('--onnx')
    ap.add_argument('--volcar')
    ap.add_argument('--config', action='append', default=[],
                    help='entrada de configuración de la sesión, clave=valor (p. ej. mlas.disable_kleidiai=1)')
    ap.add_argument('--camino-mac', action='store_true',
                    help='no aplicar la config_sesion de etiquetas.json (deja los núcleos KleidiAI del Mac)')
    a = ap.parse_args()
    d = a.carpeta
    tok = cargar(os.path.join(d, 'tokenizador.json'), os.path.join(d, 'vocab.txt'))
    with open(os.path.join(d, 'tokenizador.json'), encoding='utf-8') as f:
        entradas = json.load(f)['entradas_del_onnx']
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    # Por defecto, la misma sesión que pide etiquetas.json (la de producción).
    with open(os.path.join(d, 'etiquetas.json'), encoding='utf-8') as f:
        config = {} if a.camino_mac else dict(json.load(f).get('onnxruntime', {}).get('config_sesion', {}))
    config.update(kv.split('=', 1) for kv in a.config)
    for k, v in config.items():
        so.add_session_config_entry(k, v)
    s = ort.InferenceSession(a.onnx or os.path.join(d, 'lector.onnx'), so, providers=['CPUExecutionProvider'])
    print(f'máquina {platform.machine()} · onnxruntime {ort.__version__}' + (f' · {config}' if config else ''))

    def correr(ids):
        x = np.array([ids], dtype=np.int64)
        f = {'input_ids': x, 'attention_mask': np.ones_like(x)}
        if 'token_type_ids' in entradas:
            f['token_type_ids'] = np.zeros_like(x)
        return s.run(['intencion', 'banderas', 'tramos'], f)

    # 1. La tokenización de los 500 ejemplos.
    malas = 0
    with open(os.path.join(d, 'ejemplos-tokenizacion.jsonl'), encoding='utf-8') as f:
        filas = [json.loads(l) for l in f]
    for r in filas:
        c = tok.codificar(r['texto'])
        if c.input_ids != r['input_ids'] or c.primera_pieza != r['primera_pieza']:
            malas += 1
    print(f'tokenización: {len(filas) - malas}/{len(filas)} iguales')

    # 2. Las 50 salidas esperadas.
    with open(os.path.join(d, 'ejemplos-salida.jsonl'), encoding='utf-8') as f:
        salidas = [json.loads(l) for l in f]
    distintas, dif = 0, 0.0
    for r in salidas:
        pi, pb, pt = correr(r['input_ids'])
        arg = pt[0].argmax(-1).tolist()
        mismo = (arg == r['tramos_argmax'] and int(np.argmax(pi[0])) == int(np.argmax(r['intencion']))
                 and [x >= 0.5 for x in pb[0]] == [x >= 0.5 for x in r['banderas']])
        distintas += not mismo
        dif = max(dif, float(np.abs(pi[0] - np.array(r['intencion'])).max()),
                  float(np.abs(pb[0] - np.array(r['banderas'])).max()))
    print(f'salidas: {len(salidas) - distintas}/{len(salidas)} con la misma lectura; '
          f'mayor diferencia de probabilidad {dif:.6f}')

    # 3. Las métricas en prueba y la latencia en esta máquina.
    pru = preparar(leer('prueba'), tok)
    lect, ms = [], []
    for e in pru:
        t0 = time.perf_counter()
        pi, pb, pt = correr(e.input_ids)
        ms.append((time.perf_counter() - t0) * 1000)
        lect.append(lectura(e, pi[0].tolist(), pb[0].tolist(), pt[0].argmax(-1).tolist()))
    m = metricas([e.fila for e in pru], lect)
    if a.volcar:
        with open(a.volcar, 'w', encoding='utf-8') as f:
            for e, l in zip(pru, lect):
                f.write(json.dumps({'id': e.fila['id'], **l}, ensure_ascii=False, sort_keys=True) + '\n')
    ms.sort()
    print(f"prueba: intención {m['intencion_exactitud']:.4f} · banderas "
          + ' '.join(f"{b} {v['f1']:.4f}" for b, v in m['banderas'].items())
          + f" · tramos F1 {m['tramos_micro']['f1']:.4f} · lectura exacta {m['lectura_exacta']:.4f}")
    print(f'latencia aquí (un hilo): p50 {ms[len(ms) // 2]:.2f} ms · p95 {ms[int(0.95 * len(ms))]:.2f} ms')


if __name__ == '__main__':
    main()
