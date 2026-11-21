#!/bin/bash
# Copia del reentrenamiento del lector de restaurant-system, para la bodega.
# Antes de correrlo: python3 ../generar/sintetica.py (escribe lector/datos).
# Reentrena el lector de punta a punta y deja los artefactos en
# modelos/lector. Desde el v2 (22/10/2026) entrena en CPU por
# defecto, porque en la GPU del Mac (MPS) dos corridas iguales se separan desde
# la quinta época: en CPU, con la misma semilla y los mismos hilos, los pesos
# salen iguales al bit. Tarda unos 15 minutos en un Mac M4 Pro y necesita unos
# 2 GB libres FUERA del repo, en $LECTOR_TMP.
# Para escribir en otra carpeta, DESTINO=….
#
# Uso:
#   LECTOR_TMP=/private/tmp/lector ./reentrenar.sh
#
# Al terminar, $LECTOR_TMP se puede borrar entero: nada del repo depende de él.
#
# ¿Cuándo reentrenar? Cuando cambien los conjuntos de lector/datos (otra
# ronda etiquetada, etiquetas nuevas en generar/comun.py). Después hay que
# volver a correr las pruebas del tokenizador de TypeScript contra el
# ejemplos-tokenizacion.jsonl nuevo, aunque el vocabulario sea el mismo.
set -euo pipefail
: "${LECTOR_TMP:?falta LECTOR_TMP (una carpeta fuera del repo)}"
AQUI="$(cd "$(dirname "$0")" && pwd)"
DESTINO="${DESTINO:-$AQUI/../../modelos/lector}"
PY311="${PY311:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11}"

# El modelo elegido y su revisión exacta en Hugging Face (ver LEEME.md).
BASE="Geotrend/distilbert-base-es-cased"
REVISION="1db4091250e535236cc5e2b8b487e1a075fa738b"
MINUSCULAS="si"
QUITAR_TILDES="no"
REPETIR_HUMANAS="${REPETIR_HUMANAS:-1}"
SEMILLA="${SEMILLA:-22}"
DISPOSITIVO="${DISPOSITIVO:-cpu}"    # «auto» usa la GPU del Mac: más rápido, pero no repetible
HILOS="${HILOS:-10}"                 # en CPU, con otro número de hilos cambian las sumas y los pesos

mkdir -p "$LECTOR_TMP"
if [ ! -x "$LECTOR_TMP/venv/bin/python" ]; then
  "$PY311" -m venv "$LECTOR_TMP/venv"
  "$LECTOR_TMP/venv/bin/pip" install --no-cache-dir -q --upgrade pip
  "$LECTOR_TMP/venv/bin/pip" install --no-cache-dir -q -r "$AQUI/requirements.txt"
fi
PY="$LECTOR_TMP/venv/bin/python"
export HF_HOME="$LECTOR_TMP/hf" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1

# Sólo lo que hace falta: sin los pesos de TensorFlow ni de Flax.
"$PY" -I -c "
from huggingface_hub import snapshot_download
snapshot_download('$BASE', revision='$REVISION',
                  allow_patterns=['config.json', 'vocab.txt', 'tokenizer_config.json', 'model.safetensors', 'README.md'])
"
export HF_HUB_OFFLINE=1
cd "$AQUI"
"$PY" entrenar.py --base "$BASE" --nombre lector --minusculas "$MINUSCULAS" --quitar-tildes "$QUITAR_TILDES" \
  --repetir-humanas "$REPETIR_HUMANAS" --semilla "$SEMILLA" --dispositivo "$DISPOSITIVO" --hilos "$HILOS" \
  --salida "$LECTOR_TMP/corridas/lector"
"$PY" exportar.py --corrida "$LECTOR_TMP/corridas/lector" --salida "$LECTOR_TMP/exportado/lector"

# El int8 no puede caer más de un punto contra PyTorch por ninguno de los dos
# caminos de cálculo (ver LEEME), medido en VALIDACIÓN: la prueba no decide
# nada (en el v1 decidía, y cada reentrenamiento volvía a mirarla). Si cae, NO
# se tocan los artefactos que ya están en el repo: hay que mirar
# casos_de_bandera_cambiados en medicion.json o probar otra semilla.
"$PY" -I -c "
import json, sys
m = json.load(open('$LECTOR_TMP/exportado/lector/medicion.json'))
cab = m['puerta']['metricas']
malos = {c: {k: v for k, v in x['validacion']['puntos_menos_pytorch'].items() if k in cab and v < -1}
         for c, x in m['int8'].items()}
if any(malos.values()):
    sys.exit(f'el int8 cae más de un punto en validación: {malos}. No se tocan los artefactos del repo.')
print('el int8 cumple en validación por los dos caminos')
"
"$PY" artefactos.py --corrida "$LECTOR_TMP/corridas/lector" --exportado "$LECTOR_TMP/exportado/lector" --destino "$DESTINO"
"$PY" leeme.py "$DESTINO"
"$PY" comprobar_onnx.py "$DESTINO"
echo "listo: $DESTINO (actualiza a mano las cifras del LEEME con $LECTOR_TMP/exportado/lector/medicion.json)"
