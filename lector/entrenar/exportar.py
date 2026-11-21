"""
Exporta una corrida a ONNX, la cuantiza a int8 dinámico y la mide.

Uso:
    python exportar.py --corrida <carpeta de entrenar.py> --salida <carpeta>

Deja en --salida: lector-fp32.onnx, lector.onnx (int8) y medicion.json con
las métricas de PyTorch, del .onnx fp32 y del .onnx int8 en validación y en
prueba, la diferencia entre ellos, el peso y la latencia (p50 y p95, un hilo).

La puerta («el int8 no cae más de un punto contra PyTorch») se mide en
VALIDACIÓN desde el v2 (22/10/2026). En el v1 se medía en prueba, y como
reentrenar.sh la usa para aceptar o descartar un modelo, cada reentrenamiento
volvía a mirar la prueba: la prueba sólo se informa, no decide nada.

¿Por qué int8 dinámico y no estático? Porque el dinámico no pide un conjunto
de calibración ni toca las activaciones al exportar: sólo guarda los pesos en
8 bits y cuantiza las activaciones al vuelo. Pesa la cuarta parte y en CPU
va más rápido, que es lo que importa dentro del Express.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from datos import BANDERAS, BIO, INTENCIONES, lectura, leer, metricas, preparar
from entrenar import carpeta_base, informe_ronda, predecir
from modelo import Lector, LectorParaExportar
from tokenizador import Tokenizador

OPSET = 17


def cargar_corrida(corrida: str):
    with open(os.path.join(corrida, 'config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    base = carpeta_base(cfg['base'])
    t = cfg['tokenizador']
    tok = Tokenizador.desde_vocab(os.path.join(base, 'vocab.txt'), minusculas=t['minusculas'],
                                  quitar_tildes=t['quitar_tildes'], largo_maximo=t['largo_maximo'])
    modelo = Lector(base, len(INTENCIONES), len(BANDERAS), len(BIO), atencion='eager', pesos_base=False)
    modelo.load_state_dict(torch.load(os.path.join(corrida, 'mejor.pt'), map_location='cpu'))
    modelo.eval()
    return cfg, base, tok, modelo


def exportar(modelo: Lector, ruta: str):
    envoltura = LectorParaExportar(modelo).eval()
    ids = torch.tensor([[4, 100, 200, 300, 5]], dtype=torch.long)
    mask = torch.ones_like(ids)
    entradas = ['input_ids', 'attention_mask'] + (['token_type_ids'] if modelo.usa_tipos else [])
    args = (ids, mask) + ((torch.zeros_like(ids),) if modelo.usa_tipos else ())
    ejes = {n: {0: 'lote', 1: 'largo'} for n in entradas}
    ejes.update({'intencion': {0: 'lote'}, 'banderas': {0: 'lote'}, 'tramos': {0: 'lote', 1: 'largo'}})
    torch.onnx.export(envoltura, args, ruta, dynamo=False, opset_version=OPSET, input_names=entradas,
                      output_names=['intencion', 'banderas', 'tramos'], dynamic_axes=ejes,
                      do_constant_folding=True)
    m = onnx.load(ruta)
    onnx.checker.check_model(m)
    return entradas


def sesion(ruta: str, hilos: int = 1, config: dict | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    for k, v in (config or {}).items():
        so.add_session_config_entry(k, v)
    so.intra_op_num_threads = hilos
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(ruta, so, providers=['CPUExecutionProvider'])


def alimentar(ids: list[int], entradas: list[str]) -> dict:
    a = np.array([ids], dtype=np.int64)
    d = {'input_ids': a, 'attention_mask': np.ones_like(a)}
    if 'token_type_ids' in entradas:
        d['token_type_ids'] = np.zeros_like(a)
    return d


def predecir_onnx(s: ort.InferenceSession, ejemplos, entradas):
    res, crudo = [], []
    for e in ejemplos:
        pi, pb, pt = s.run(['intencion', 'banderas', 'tramos'], alimentar(e.input_ids, entradas))
        res.append(lectura(e, pi[0].tolist(), pb[0].tolist(), pt[0].argmax(-1).tolist()))
        crudo.append((pi[0], pb[0], pt[0]))
    return res, crudo


def latencia(s: ort.InferenceSession, frases: list[list[int]], entradas, vueltas: int = 5) -> dict:
    for ids in frases[:30]:
        s.run(None, alimentar(ids, entradas))
    ms = []
    for _ in range(vueltas):
        for ids in frases:
            f = alimentar(ids, entradas)
            t0 = time.perf_counter()
            s.run(None, f)
            ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    q = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))]
    return {'corridas': len(ms), 'p50_ms': round(q(0.50), 3), 'p95_ms': round(q(0.95), 3),
            'p99_ms': round(q(0.99), 3), 'media_ms': round(statistics.fmean(ms), 3)}


def resumen(m: dict) -> dict:
    return {
        'intencion_exactitud': m['intencion_exactitud'],
        **{f'bandera_{b}_f1': m['banderas'][b]['f1'] for b in BANDERAS},
        'tramos_micro_f1': m['tramos_micro']['f1'],
        **{f'tramo_{e}_f1': v['f1'] for e, v in m['tramos_por_etiqueta'].items()},
        'lectura_exacta': m['lectura_exacta'],
    }


# Las dos maneras en que onnxruntime hace las cuentas en int8. En un Mac
# (arm64) usa por defecto los núcleos KleidiAI de ARM; en Fly (x86_64) usa
# los «clásicos» de MLAS. Dan probabilidades algo distintas y a veces otra
# etiqueta, así que el int8 se mide por los dos caminos y el criterio de no
# caer más de un punto tiene que cumplirse en ambos. En el Mac, el camino
# clásico se obtiene apagando KleidiAI (mlas.disable_kleidiai=1): se comprobó
# que así da lo mismo que onnxruntime en x86_64 (Rosetta) salvo empates.
CAMINOS = {'mac': {}, 'clasico': {'mlas.disable_kleidiai': '1'}}

# Las métricas de cabecera: sobre ellas va el «no más de un punto». Las F1
# por etiqueta se informan también, pero con apoyos de 7 a 43 tramos un solo
# tramo mueve más de dos puntos y no sirven de criterio.
CABECERA = ['intencion_exactitud', *(f'bandera_{b}_f1' for b in BANDERAS), 'tramos_micro_f1', 'lectura_exacta']


def misma_lectura(x: dict, y: dict) -> bool:
    return (x['intencion'] == y['intencion'] and x['banderas'] == y['banderas']
            and {(t['inicio'], t['fin'], t['etiqueta']) for t in x['tramos']}
            == {(t['inicio'], t['fin'], t['etiqueta']) for t in y['tramos']})


def comparar(filas: list[dict], l_torch: list[dict], l_int8: list[dict]) -> dict:
    """El int8 contra PyTorch en un conjunto: métricas, caída en puntos y filas que cambian."""
    m_t, m_8 = metricas(filas, l_torch), metricas(filas, l_int8)
    rt, r8 = resumen(m_t), resumen(m_8)
    d = {k: round(100 * (r8[k] - rt[k]), 2) for k in rt}
    return {
        'metricas': r8,
        'puntos_menos_pytorch': d,
        'peor_caida_cabecera': min(d[k] for k in CABECERA),
        'cumple_un_punto': all(d[k] >= -1.0 for k in CABECERA),
        # Cuántas filas lee distinto que PyTorch, y cuántos casos cambia en
        # cada bandera: con pocas filas de una bandera, un solo caso cambiado
        # ya mueve su F1 varios puntos, y conviene verlo antes de descartar.
        'filas_con_otra_lectura': sum(not misma_lectura(x, y) for x, y in zip(l_torch, l_int8)),
        'casos_de_bandera_cambiados': {b: sum(x['banderas'][b] != y['banderas'][b] for x, y in zip(l_torch, l_int8))
                                       for b in BANDERAS},
        'metricas_completas': m_8,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corrida', required=True)
    ap.add_argument('--salida', required=True)
    ap.add_argument('--por-tensor', action='store_true',
                    help='una escala por matriz en vez de una por columna (más pérdida en el camino clásico)')
    ap.add_argument('--datos', default=None, help='carpeta de los conjuntos (por defecto lector/datos)')
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    torch.set_num_threads(4)

    cfg, base, tok, modelo = cargar_corrida(a.corrida)
    val = preparar(leer('validacion', a.datos), tok)
    pru = preparar(leer('prueba', a.datos), tok)
    filas_val = [e.fila for e in val]
    filas = [e.fila for e in pru]

    fp32 = os.path.join(a.salida, 'lector-fp32.onnx')
    pre = os.path.join(a.salida, 'lector-pre.onnx')
    int8 = os.path.join(a.salida, 'lector.onnx')
    entradas = exportar(modelo, fp32)
    # El preproceso (inferencia de formas y plegado de constantes) es lo que
    # recomienda onnxruntime antes de cuantizar: sin él algunos MatMul quedan
    # sin forma conocida y se saltan. La inferencia «simbólica» no termina con
    # la máscara de atención de transformers 5, así que basta la de onnx.
    quant_pre_process(fp32, pre, skip_symbolic_shape=True)
    # Pesos int8 con signo y una escala por columna: por tensor, el camino
    # clásico perdía 1,3 puntos de intención; por columna queda dentro del punto
    # en los dos caminos y pesa lo mismo (las escalas son 0,2 MB).
    # reduce_range (24/10/2026): en x86 sin VNNI (el AMD EPYC de Fly sólo tiene AVX2) el producto u8×s8 de MLAS
    # suma en int16 y se desborda con pesos de 8 bits enteros: en Fly el v2 leía «local» con 0,32 de confianza para
    # todo, mientras en la Mac (ARM) leía bien. Con 7 bits no se desborda en ninguna CPU.
    quantize_dynamic(pre, int8, weight_type=QuantType.QInt8, per_channel=not a.por_tensor, reduce_range=True)
    os.remove(pre)

    lt_val = predecir(modelo, val, tok.ids[tok.pad], torch.device('cpu'))
    lt_pru = predecir(modelo, pru, tok.ids[tok.pad], torch.device('cpu'))
    m_torch_val = metricas(filas_val, lt_val)
    m_torch = metricas(filas, lt_pru)
    s32 = sesion(fp32)
    l32, c32 = predecir_onnx(s32, pru, entradas)
    m32 = metricas(filas, l32)
    l32_val, _ = predecir_onnx(s32, val, entradas)

    def delta(x, y):
        rx, ry = resumen(x), resumen(y)
        return {k: round(100 * (ry[k] - rx[k]), 2) for k in rx}

    rng = np.random.default_rng(22)
    frases = [e.input_ids for e in pru if len(e.input_ids) <= 48]
    largas = [[tok.ids[tok.cls]] + rng.integers(1000, len(tok.vocab), 46).tolist() + [tok.ids[tok.sep]] for _ in range(60)]
    lat = {'fp32_prueba': latencia(s32, frases, entradas), 'fp32_48_piezas': latencia(s32, largas, entradas, vueltas=8)}
    por_camino = {}
    for nombre, config in CAMINOS.items():
        s8 = sesion(int8, config=config)
        l8, c8 = predecir_onnx(s8, pru, entradas)
        l8_val, _ = predecir_onnx(s8, val, entradas)
        ronda8, lineas8 = informe_ronda(filas, l8)
        with open(os.path.join(a.salida, f'fallos_ronda_int8_{nombre}.tsv'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lineas8) + '\n')
        with open(os.path.join(a.salida, f'lecturas_prueba_int8_{nombre}.jsonl'), 'w', encoding='utf-8') as f:
            for e, l in zip(pru, l8):
                f.write(json.dumps({'id': e.fila['id'], **l}, ensure_ascii=False) + '\n')
        en_val = comparar(filas_val, lt_val, l8_val)
        en_pru = comparar(filas, lt_pru, l8)
        por_camino[nombre] = {
            'config': config,
            # La puerta: validación. La prueba va aparte y no decide nada.
            'validacion': {k: v for k, v in en_val.items() if k != 'metricas_completas'},
            'prueba': {k: v for k, v in en_pru.items() if k not in ('metricas_completas', 'cumple_un_punto')},
            'cumple_un_punto': en_val['cumple_un_punto'],
            'mayor_diferencia_probabilidad_vs_fp32': {
                'intencion': max(float(np.abs(x[0] - y[0]).max()) for x, y in zip(c32, c8)),
                'banderas': max(float(np.abs(x[1] - y[1]).max()) for x, y in zip(c32, c8)),
                'tramos': max(float(np.abs(x[2] - y[2]).max()) for x, y in zip(c32, c8)),
            },
            'ronda': ronda8,
            'metricas_completas': en_pru['metricas_completas'],
            'metricas_completas_validacion': en_val['metricas_completas'],
        }
        # Latencia: un hilo, lote de 1, las frases de prueba tal cual (todas
        # bajo 48 piezas) y aparte frases de exactamente 48, el peor caso pedido.
        lat[f'int8_{nombre}_prueba'] = latencia(s8, frases, entradas)
        lat[f'int8_{nombre}_48_piezas'] = latencia(s8, largas, entradas, vueltas=8)
    lat.update({'largo_medio_prueba': round(statistics.fmean(len(f) for f in frases), 1),
                'hilos': 1, 'lote': 1, 'onnxruntime': ort.__version__})

    med = {
        'corrida': cfg['nombre'], 'base': cfg['base'], 'opset': OPSET, 'entradas': entradas,
        'cuantizacion': 'dinámica, pesos QInt8 con reduce_range ' + ('por tensor' if a.por_tensor else 'por canal'),
        'ir_version': onnx.load(int8, load_external_data=False).ir_version,
        'megabytes': {'fp32': round(os.path.getsize(fp32) / 1e6, 2), 'int8': round(os.path.getsize(int8) / 1e6, 2)},
        'puerta': {'conjunto': 'validacion', 'filas': len(filas_val), 'metricas': CABECERA,
                   'limite_puntos': -1.0, 'cumple': all(c['cumple_un_punto'] for c in por_camino.values())},
        'validacion_pytorch': resumen(m_torch_val),
        'validacion_onnx_fp32_filas_con_otra_lectura': sum(not misma_lectura(x, y) for x, y in zip(lt_val, l32_val)),
        'prueba_pytorch': resumen(m_torch),
        'prueba_onnx_fp32': resumen(m32),
        'puntos_fp32_menos_pytorch': delta(m_torch, m32),
        'int8': por_camino,
        'latencia': lat,
        'metricas_pytorch_completas': m_torch,
    }
    with open(os.path.join(a.salida, 'medicion.json'), 'w', encoding='utf-8') as f:
        json.dump(med, f, ensure_ascii=False, indent=2)
    print(f"{cfg['nombre']}: int8 {med['megabytes']['int8']} MB · fp32 {med['megabytes']['fp32']} MB")
    for nombre, c in por_camino.items():
        v = c['validacion']
        print(f"  int8 {nombre}: p50 {lat[f'int8_{nombre}_prueba']['p50_ms']} ms · p95 {lat[f'int8_{nombre}_prueba']['p95_ms']} ms "
              f"(48 piezas: p95 {lat[f'int8_{nombre}_48_piezas']['p95_ms']} ms) · en validación: peor caída de cabecera "
              f"{v['peor_caida_cabecera']} puntos, {v['filas_con_otra_lectura']} filas con otra lectura, "
              f"cumple: {v['cumple_un_punto']}")
    print('  validación pytorch', {k: round(x, 4) for k, x in resumen(m_torch_val).items() if k in CABECERA})
    for nombre, c in por_camino.items():
        print(f'  validación {nombre:7}', {k: round(x, 4) for k, x in c['validacion']['metricas'].items() if k in CABECERA})
    print('  prueba pytorch    ', {k: round(x, 4) for k, x in resumen(m_torch).items() if k in CABECERA})
    for nombre, c in por_camino.items():
        print(f'  prueba {nombre:11}', {k: round(x, 4) for k, x in c['prueba']['metricas'].items() if k in CABECERA})


if __name__ == '__main__':
    main()
