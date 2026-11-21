"""
Lo que predice el lector de la bodega: la definición única de las etiquetas.
`entrenar/datos.py` lee de aquí; si una etiqueta cambia, hay que reentrenar.
"""

# La primera es la que se usa para rutear: precio y stock van a la
# herramienta del catálogo, ventas y consejo al RAG, el resto a respuestas fijas.
INTENCIONES = ['precio', 'stock', 'ventas', 'consejo', 'saludo', 'cortesia', 'fuera']

# El modelo necesita al menos una bandera; «usted» sirve para contestar con el mismo trato.
BANDERAS = ['usted']

ETIQUETAS = ['PRODUCTO', 'MARCA', 'MEDIDA', 'CANTIDAD', 'PERIODO']
