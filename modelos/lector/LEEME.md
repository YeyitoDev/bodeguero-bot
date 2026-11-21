# El lector del bodeguero

Clasificador chico (DistilBERT en español, int8, ONNX) que lee la pregunta del
bodeguero y devuelve:

- **la intención**: `precio`, `stock`, `inventario` (listar el catálogo o una categoría), `ventas`, `consejo`, `saludo`, `cortesia` o `fuera`;
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
| `ejemplos-salida.jsonl` | 30 234 | `59c5cfc9febb205a8172d95ba684be267412cfd388c4e44c35d9e2d9361864e4` |
| `ejemplos-tokenizacion.jsonl` | 227 107 | `2d615326b741b9c38ddae693b1c138bc16baeffaa1e19554393cbdc8b0d0ce0e` |
| `etiquetas.json` | 1 173 | `7a48273bfb9699c0c16dd5e2f8fe763b8fe7caff8d9a0c89e26a25e0a8ccd0bb` |
| `lector.onnx` | 63 799 262 | `9b4ec79c75da0777b5b6b72b6ba98955b4691f783635bc5cdd18a066777d2eeb` |
| `lector.onnx.sha256` | 78 | `adb2d0efc8409f461ddc9d655c7dcc017bc49b0ef86be160d4b030ec108c2ac0` |
| `tokenizador.json` | 1 655 | `de65549215a6146802b61b3ed0d5f8b1af223ae67c361d2d8ca343757398640d` |
| `vocab.txt` | 173 939 | `47e91a9d732b83263c724c9f89aeb25d5ac038802c3a072a1bffafb09b5b708f` |
<!-- /sha256 -->
