"""
Afina un codificador base como lector: entrena con `entrenamiento`, elige la
mejor época por `validacion` y mide en `prueba`.

Uso (desde lector/entrenar, con HF_HOME apuntando a donde están los modelos):

    python entrenar.py --base Geotrend/distilbert-base-es-cased --nombre geotrend \\
        --minusculas si --quitar-tildes no --salida <carpeta fuera del repo>

Deja en --salida: mejor.pt (los pesos de la mejor época), config.json (con
qué se entrenó y la configuración del tokenizador), historial.json (cada
época en validación), metricas_prueba.json y fallos_ronda.tsv.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn

from datos import (BIO, DATOS, IGNORAR, INTENCIONES, BANDERAS, Ejemplo, es_exacta, lectura, leer, marcado,
                   metricas, preparar, puntaje)
from modelo import Lector
from tokenizador import Tokenizador


def sha256_de(ruta: str) -> str:
    import hashlib
    with open(ruta, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def carpeta_base(repo: str) -> str:
    """La carpeta local del modelo base (ya descargado en HF_HOME)."""
    if os.path.isdir(repo):
        return repo
    # Se bajan sólo los archivos que hacen falta (sin los pesos de TF ni de
    # Flax, que no se usan), así que la caché nunca está «completa» para
    # huggingface_hub: se busca la instantánea a mano.
    hub = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub')
    snaps = os.path.join(hub, 'models--' + repo.replace('/', '--'), 'snapshots')
    if os.path.isdir(snaps) and os.listdir(snaps):
        return os.path.join(snaps, sorted(os.listdir(snaps))[0])
    return snapshot_download(repo)


def lotes(ejemplos: list[Ejemplo], tam: int, pad_id: int, barajar: bool, rng: random.Random | None = None):
    orden = list(range(len(ejemplos)))
    if barajar:
        rng.shuffle(orden)
    for k in range(0, len(orden), tam):
        sel = [ejemplos[i] for i in orden[k:k + tam]]
        L = max(len(e.input_ids) for e in sel)
        ids = torch.full((len(sel), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(sel), L), dtype=torch.long)
        tram = torch.full((len(sel), L), IGNORAR, dtype=torch.long)
        for j, e in enumerate(sel):
            n = len(e.input_ids)
            ids[j, :n] = torch.tensor(e.input_ids)
            mask[j, :n] = 1
            tram[j, :n] = torch.tensor(e.tramos)
        yield sel, {
            'input_ids': ids, 'attention_mask': mask, 'token_type_ids': torch.zeros_like(ids),
            'intencion': torch.tensor([e.intencion for e in sel]),
            'banderas': torch.tensor([e.banderas for e in sel]),
            'tramos': tram,
        }


@torch.no_grad()
def predecir(modelo: Lector, ejemplos: list[Ejemplo], pad_id: int, disp, tam: int = 64):
    """Las lecturas de cada ejemplo con el modelo en PyTorch."""
    modelo.eval()
    res = []
    for sel, b in lotes(ejemplos, tam, pad_id, barajar=False):
        i, ba, t = modelo(b['input_ids'].to(disp), b['attention_mask'].to(disp), b['token_type_ids'].to(disp))
        pi = torch.softmax(i, -1).cpu().numpy()
        pb = torch.sigmoid(ba).cpu().numpy()
        pt = t.argmax(-1).cpu().numpy()
        for j, e in enumerate(sel):
            res.append(lectura(e, pi[j].tolist(), pb[j].tolist(), pt[j].tolist()))
    return res


def informe_ronda(filas: list[dict], lecturas: list[dict]) -> tuple[dict, list[str]]:
    """Los casos de las rondas, uno por uno: cuántos salen enteros y cuáles no."""
    pares = [(f, l) for f, l in zip(filas, lecturas) if f['origen'] == 'ronda']
    n = len(pares)
    lineas = ['id\tqué falló\ttexto\tesperado\tleído']
    fallos = 0
    for f, l in pares:
        if es_exacta(f, l):
            continue
        fallos += 1
        que = []
        if f['intencion'] != l['intencion']:
            que.append('intención')
        if any(f['banderas'][b] != l['banderas'][b] for b in BANDERAS):
            que.append('banderas')
        if {(t['inicio'], t['fin'], t['etiqueta']) for t in f['tramos']} != {(t['inicio'], t['fin'], t['etiqueta']) for t in l['tramos']}:
            que.append('tramos')

        def resumen(x):
            bs = ','.join(b for b in BANDERAS if x['banderas'][b]) or '-'
            return f"{x['intencion']} · {bs} · {marcado(f['texto'], x['tramos'])}"
        lineas.append('\t'.join([f['id'], '+'.join(que), f['texto'].replace('\t', ' '), resumen(f), resumen(l)]))
    datos = {
        'casos': n,
        'enteros': n - fallos,
        'exactitud_entera': (n - fallos) / n if n else 0.0,
        'intencion_exactitud': sum(f['intencion'] == l['intencion'] for f, l in pares) / n if n else 0.0,
        'banderas_exactas': sum(all(f['banderas'][b] == l['banderas'][b] for b in BANDERAS) for f, l in pares) / n if n else 0.0,
        'tramos_exactos': sum({(t['inicio'], t['fin'], t['etiqueta']) for t in f['tramos']} == {(t['inicio'], t['fin'], t['etiqueta']) for t in l['tramos']} for f, l in pares) / n if n else 0.0,
    }
    return datos, lineas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--nombre', required=True)
    ap.add_argument('--minusculas', choices=['si', 'no'], required=True)
    ap.add_argument('--quitar-tildes', choices=['si', 'no'], required=True)
    ap.add_argument('--salida', required=True)
    ap.add_argument('--epocas', type=int, default=10)
    ap.add_argument('--paciencia', type=int, default=3)
    ap.add_argument('--lote', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--lr-cabezas', type=float, default=5e-4)
    ap.add_argument('--largo-maximo', type=int, default=64)
    ap.add_argument('--semilla', type=int, default=22)
    ap.add_argument('--dispositivo', choices=['auto', 'mps', 'cpu'], default='auto',
                    help='en MPS (la GPU del Mac) dos corridas iguales se separan desde la 5.ª época; '
                         'en CPU dan los mismos pesos, pero tarda varias veces más')
    ap.add_argument('--repetir-humanas', type=int, default=1,
                    help='cuántas veces entra en cada época lo escrito por personas (redactada y traza)')
    ap.add_argument('--datos', default=None,
                    help='carpeta con entrenamiento/validacion/prueba.jsonl (por defecto lector/datos); '
                         'la validación cruzada de la traza pasa aquí cada pliegue')
    # En CPU el resultado depende también de cuántos hilos suman: con otro
    # número cambia el orden de las sumas y los pesos ya no salen al bit.
    ap.add_argument('--hilos', type=int, default=10, help='hilos de PyTorch en CPU (fijo, para pesos repetibles)')
    a = ap.parse_args()

    random.seed(a.semilla)
    np.random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    os.makedirs(a.salida, exist_ok=True)
    if a.dispositivo == 'auto':
        disp = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        disp = torch.device(a.dispositivo)
    if disp.type == 'cpu':
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(a.hilos)

    base = carpeta_base(a.base)
    tok = Tokenizador.desde_vocab(os.path.join(base, 'vocab.txt'), minusculas=a.minusculas == 'si',
                                  quitar_tildes=a.quitar_tildes == 'si', largo_maximo=a.largo_maximo)
    pad_id = tok.ids[tok.pad]
    ent, val, pru = (preparar(leer(n, a.datos), tok) for n in ('entrenamiento', 'validacion', 'prueba'))
    # Lo sintético es el 84 % del entrenamiento y la prueba son mensajes de
    # personas: repetir lo escrito a mano le da más peso sin inventar nada.
    ent = ent + [e for e in ent if e.fila['origen'] != 'sintetica'] * (a.repetir_humanas - 1)
    print(f'{a.nombre}: {len(ent)} entrenamiento, {len(val)} validación, {len(pru)} prueba · {disp}', flush=True)

    modelo = Lector(base, len(INTENCIONES), len(BANDERAS), len(BIO), atencion='eager').to(disp)
    n_param = sum(p.numel() for p in modelo.parameters())
    cabezas = [p for n, p in modelo.named_parameters() if not n.startswith('codificador.')]
    sin_decaer = ('bias', 'LayerNorm.weight', 'layer_norm.weight', 'norm.weight')
    cod_dec = [p for n, p in modelo.codificador.named_parameters() if not any(s in n for s in sin_decaer)]
    cod_nodec = [p for n, p in modelo.codificador.named_parameters() if any(s in n for s in sin_decaer)]
    opt = torch.optim.AdamW([
        {'params': cod_dec, 'lr': a.lr, 'weight_decay': 0.01},
        {'params': cod_nodec, 'lr': a.lr, 'weight_decay': 0.0},
        {'params': cabezas, 'lr': a.lr_cabezas, 'weight_decay': 0.01},
    ])
    pasos_por_epoca = math.ceil(len(ent) / a.lote)
    total = pasos_por_epoca * a.epocas
    calentar = int(0.06 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / calentar if s < calentar else max(0.0, (total - s) / max(1, total - calentar)))
    ce = nn.CrossEntropyLoss()
    ce_tramos = nn.CrossEntropyLoss(ignore_index=IGNORAR)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(a.semilla)

    historial, mejor, sin_mejorar, t0 = [], -1.0, 0, time.time()
    for ep in range(1, a.epocas + 1):
        modelo.train()
        suma, k = 0.0, 0
        for _, b in lotes(ent, a.lote, pad_id, barajar=True, rng=rng):
            b = {n: v.to(disp) for n, v in b.items()}
            i, ba, t = modelo(b['input_ids'], b['attention_mask'], b['token_type_ids'])
            perdida = ce(i, b['intencion']) + bce(ba, b['banderas']) + ce_tramos(t.reshape(-1, t.shape[-1]), b['tramos'].reshape(-1))
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            suma += perdida.item()
            k += 1
        m = metricas([e.fila for e in val], predecir(modelo, val, pad_id, disp))
        p = puntaje(m)
        historial.append({'epoca': ep, 'perdida': suma / k, 'puntaje': p, 'intencion': m['intencion_exactitud'],
                          'banderas_f1_micro': m['banderas_f1_micro'], 'tramos_f1': m['tramos_micro']['f1'],
                          'segundos': round(time.time() - t0, 1)})
        print(f"  época {ep}: pérdida {suma / k:.4f} · val intención {m['intencion_exactitud']:.3f} "
              f"banderas {m['banderas_f1_micro']:.3f} tramos {m['tramos_micro']['f1']:.3f} → {p:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        if p > mejor:
            mejor, sin_mejorar = p, 0
            torch.save(modelo.state_dict(), os.path.join(a.salida, 'mejor.pt'))
        else:
            sin_mejorar += 1
            if sin_mejorar >= a.paciencia:
                print('  sin mejorar en validación: paro', flush=True)
                break

    modelo.load_state_dict(torch.load(os.path.join(a.salida, 'mejor.pt'), map_location=disp))
    mejor_ep = max(historial, key=lambda h: h['puntaje'])['epoca']
    m_val = metricas([e.fila for e in val], predecir(modelo, val, pad_id, disp))
    lect = predecir(modelo, pru, pad_id, disp)
    m_pru = metricas([e.fila for e in pru], lect)
    # Lo leído en prueba, fila por fila: la validación cruzada de la traza junta
    # aquí lo de cada pliegue, y el informe compara con otras versiones.
    with open(os.path.join(a.salida, 'lecturas_prueba.jsonl'), 'w', encoding='utf-8') as f:
        for e, l in zip(pru, lect):
            f.write(json.dumps({'id': e.fila['id'], **l}, ensure_ascii=False) + '\n')
    ronda, lineas = informe_ronda([e.fila for e in pru], lect)
    with open(os.path.join(a.salida, 'fallos_ronda.tsv'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')
    cfg = {
        'nombre': a.nombre, 'base': a.base, 'parametros': n_param, 'mejor_epoca': mejor_ep,
        'segundos_entrenando': round(time.time() - t0, 1), 'dispositivo': str(disp),
        'hiperparametros': {k: v for k, v in vars(a).items() if k not in ('salida',)},
        'tokenizador': tok.descripcion(), 'validacion': m_val, 'prueba': m_pru, 'ronda': ronda,
        'datos_sha256': {n: sha256_de(os.path.join(a.datos or DATOS, f'{n}.jsonl'))
                         for n in ('entrenamiento', 'validacion', 'prueba')},
        'hilos': torch.get_num_threads() if disp.type == 'cpu' else None,
    }
    with open(os.path.join(a.salida, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(a.salida, 'historial.json'), 'w', encoding='utf-8') as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)
    print(f"{a.nombre}: mejor época {mejor_ep} · val {puntaje(m_val):.4f} · prueba intención "
          f"{m_pru['intencion_exactitud']:.3f} banderas {m_pru['banderas_f1_media']:.3f} tramos "
          f"{m_pru['tramos_micro']['f1']:.3f} · ronda entera {ronda['enteros']}/{ronda['casos']}", flush=True)


if __name__ == '__main__':
    main()
