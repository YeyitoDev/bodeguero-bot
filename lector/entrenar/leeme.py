"""
Pone al día el bloque de sha256 del LEEME.md de backend/modelos/lector.

El LEEME se escribe a mano (lleva el porqué de cada cosa), pero las huellas
no: las calcula este guion entre las marcas <!-- sha256 --> y
<!-- /sha256 -->, para que nunca digan otra cosa que los archivos que hay.

Uso: python leeme.py <carpeta de artefactos>
"""
from __future__ import annotations

import hashlib
import os
import re
import sys


def sha256(ruta: str) -> str:
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for trozo in iter(lambda: f.read(1 << 20), b''):
            h.update(trozo)
    return h.hexdigest()


def main():
    d = sys.argv[1]
    leeme = os.path.join(d, 'LEEME.md')
    filas = ['| archivo | bytes | sha256 |', '|---|---:|---|']
    for n in sorted(os.listdir(d)):
        if n == 'LEEME.md' or n.startswith('.'):
            continue
        p = os.path.join(d, n)
        filas.append(f'| `{n}` | {os.path.getsize(p):,} | `{sha256(p)}` |'.replace(',', ' '))
    bloque = '<!-- sha256 -->\n' + '\n'.join(filas) + '\n<!-- /sha256 -->'
    print('\n'.join(filas))
    if not os.path.exists(leeme):
        print(f'(no hay {leeme}: sólo se imprimen las huellas)')
        return
    with open(leeme, encoding='utf-8') as f:
        texto = f.read()
    nuevo, n = re.subn(r'<!-- sha256 -->.*?<!-- /sha256 -->', bloque, texto, flags=re.S)
    if n != 1:
        sys.exit('el LEEME no tiene las marcas <!-- sha256 --> … <!-- /sha256 -->')
    with open(leeme, 'w', encoding='utf-8') as f:
        f.write(nuevo)


if __name__ == '__main__':
    main()
