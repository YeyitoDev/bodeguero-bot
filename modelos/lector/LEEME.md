# El lector del bodeguero

Clasificador chico (DistilBERT en español, int8, ONNX) que lee la pregunta del
bodeguero y devuelve:

- **la intención**: `precio`, `stock`, `ventas`, `consejo`, `saludo`, `cortesia` o `fuera`;
- **la bandera** `usted`;
- **los tramos**: `PRODUCTO`, `MARCA`, `MEDIDA`, `CANTIDAD`, `PERIODO`.

No redacta nada: `AnswerQuestion` usa la intención para elegir la rama del grafo
y los tramos para buscar el producto en el catálogo. Las etiquetas se definen en
`lector/generar/comun.py`; los datos salen de `lector/generar/sintetica.py`.

Reentrenar (unos minutos en CPU):

```bash
python3 lector/generar/sintetica.py
LECTOR_TMP=/private/tmp/lector-bodega lector/entrenar/reentrenar.sh
```

<!-- sha256 -->
| archivo | bytes | sha256 |
|---|---:|---|
| `LICENSE` | 11 358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `NOTICE` | 735 | `9a8c20994e1b61fb6c485c5790f9ac415f5b1169cbe617298cc5c784c8a637db` |
| `ejemplos-salida.jsonl` | 31 371 | `e6a96114fe1b761efff15086fc6451f99cf89f054afdf6e60fabc89ed47c40dc` |
| `ejemplos-tokenizacion.jsonl` | 236 727 | `5708636a2e196550a23e48e39232db624f26f26b19ca9ae86274367b8bf2129f` |
| `etiquetas.json` | 1 155 | `cd084ab900771c89a1aab2e05c6fa3efc8894d3bfad4dfc5ce2eecae0c8a1752` |
| `lector.onnx` | 63 798 485 | `0427d0aa1723baffdd5a0c6e5c0a1646c3f4382b3fc66f5d1137f20f55dbb07b` |
| `tokenizador.json` | 1 655 | `de65549215a6146802b61b3ed0d5f8b1af223ae67c361d2d8ca343757398640d` |
| `vocab.txt` | 173 939 | `47e91a9d732b83263c724c9f89aeb25d5ac038802c3a072a1bffafb09b5b708f` |
<!-- /sha256 -->
