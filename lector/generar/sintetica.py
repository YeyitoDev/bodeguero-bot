"""
Datos sintéticos del lector de la bodega: escribe lector/datos/{entrenamiento,validacion,prueba}.jsonl.

Cada frase se arma por segmentos (texto, etiqueta|None), así los tramos salen
exactos aunque después se le quiten tildes o el «¿». La prueba usa marcas y
plantillas que el entrenamiento nunca ve, para medir si generaliza a un
catálogo distinto del de los ejemplos.

    python sintetica.py            # semilla 7

Además suma las frases escritas a mano de datos/manual.tsv (origen «manual»):
dos de cada tres al entrenamiento y una a la prueba.
"""
from __future__ import annotations

import json
import os
import random
import re
import unicodedata

from comun import BANDERAS, ETIQUETAS, INTENCIONES

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(os.path.dirname(AQUI), 'datos')

# (producto, género, unidad de stock con su género, marcas, medidas)
PRODUCTOS = [
    ('arroz', 'm', ('bolsas', 'f'), ['Costeño', 'Paisana', 'Faraón', 'Valle Norte'], ['1 kg', '5 kg', 'medio kilo', 'un kilo']),
    ('azúcar', 'f', ('bolsas', 'f'), ['Cartavio', 'Paramonga', 'Casa Grande'], ['1 kg', 'medio kilo', '5 kg']),
    ('aceite', 'm', ('botellas', 'f'), ['Primor', 'Cil', 'Ideal', 'Sao', 'Cocinero'], ['900 ml', '1 litro', 'medio litro']),
    ('leche', 'f', ('tarros', 'm'), ['Gloria', 'Laive', 'Pura Vida', 'Ideal'], ['tarro azul', '400 g', 'caja de 1 litro', 'tarro']),
    ('fideos', 'm', ('paquetes', 'm'), ['Don Vittorio', 'Molitalia', 'Lavaggi', 'Nicolini'], ['450 g', '1 kg', 'medio kilo']),
    ('tallarín', 'm', ('paquetes', 'm'), ['Don Vittorio', 'Molitalia', 'Lavaggi'], ['450 g', '1 kg']),
    ('huevos', 'm', ('huevos', 'm'), ['La Calera', 'Avinka'], ['la docena', 'media docena', 'a granel']),
    ('gaseosa', 'f', ('unidades', 'f'), ['Inca Kola', 'Coca-Cola', 'Pepsi', 'Sprite', 'Fanta', 'Guaraná', 'Kola Real', 'Big Cola'],
     ['500 ml', '1.5 litros', 'litro y medio', '3 litros', 'personal', 'gordita', '2 litros', '625 ml']),
    ('agua', 'f', ('unidades', 'f'), ['San Luis', 'Cielo', 'San Mateo', 'Vida'], ['625 ml', '2.5 litros', '7 litros', 'sin gas', 'con gas']),
    ('cerveza', 'f', ('botellas', 'f'), ['Pilsen', 'Cristal', 'Cusqueña', 'Corona', 'Tres Cruces'], ['630 ml', 'lata', 'six pack', 'de litro']),
    ('pan', 'm', ('unidades', 'f'), ['Bimbo', 'Unión'], ['francés', 'de yema', 'integral', 'molde']),
    ('atún', 'm', ('latas', 'f'), ['Florida', 'A1', 'Campomar', 'Real'], ['en lata', 'en trozos', '170 g']),
    ('galletas', 'f', ('paquetes', 'm'), ['Soda Field', 'Casino', 'Oreo', 'Morochas', 'Margarita', 'Ritz', 'Club Social'], ['de 6 unidades', 'paquete chico', 'familiar']),
    ('detergente', 'm', ('bolsas', 'f'), ['Ariel', 'Opal', 'Ace', 'Bolívar', 'Marsella'], ['500 g', '1 kg', '150 g', '4 kg']),
    ('jabón', 'm', ('unidades', 'f'), ['Bolívar', 'Marsella', 'Protex', 'Camay'], ['en barra', '200 g']),
    ('papel higiénico', 'm', ('paquetes', 'm'), ['Elite', 'Suave', 'Noble', 'Paracas'], ['de 4 rollos', 'de 2 rollos', 'doble hoja']),
    ('chocolate', 'm', ('unidades', 'f'), ['Sublime', 'Princesa', 'Triángulo', 'Cua Cua'], ['clásico', 'grande']),
    ('mantequilla', 'f', ('unidades', 'f'), ['Laive', 'Gloria', 'Dorina'], ['200 g', 'con sal']),
    ('sal', 'f', ('bolsas', 'f'), ['Emsal', 'Marina', 'Lobos'], ['1 kg', 'de mesa']),
    ('avena', 'f', ('bolsas', 'f'), ['3 Ositos', 'Quaker', 'Santa Catalina'], ['900 g', 'tarro', 'bolsita']),
    ('café', 'm', ('frascos', 'm'), ['Altomayo', 'Nescafé', 'Cafetal'], ['50 g', 'sobre', 'frasco chico']),
    ('lentejas', 'f', ('bolsas', 'f'), ['Costeño', 'Paisana'], ['500 g', '1 kg', 'medio kilo']),
    ('hot dog', 'm', ('paquetes', 'm'), ['San Fernando', 'Otto Kunz', 'Razzeto'], ['paquete de 10', 'de pollo']),
    ('yogurt', 'm', ('botellas', 'f'), ['Gloria', 'Laive', 'Yoleit'], ['1 litro', 'de fresa', 'personal']),
    ('pilas', 'f', ('paquetes', 'm'), ['Duracell', 'Panasonic', 'Energizer'], ['AA', 'AAA', 'doble A']),
    ('fósforos', 'm', ('cajas', 'f'), ['Inti', 'Llama'], ['caja chica', 'paquete de 10']),
    ('balón de gas', 'm', ('balones', 'm'), ['Solgas', 'Llama Gas', 'Zeta Gas'], ['10 kilos', '5 kilos']),
    ('cigarros', 'm', ('cajetillas', 'f'), ['Hamilton', 'Lucky Strike', 'Pall Mall'], ['cajetilla', 'suelto']),
    ('caramelos', 'm', ('bolsas', 'f'), ['Halls', 'Mentitas', 'Chick'], ['bolsa', 'sueltos']),
    ('queso', 'm', ('kilos', 'm'), ['Bonlé', 'Laive', 'Gloria'], ['fresco', 'edam', 'medio kilo']),
]
CANTIDADES = ['2', '3', '5', '10', 'dos', 'tres', 'cuatro', 'seis', 'una docena', 'media docena', 'un ciento']
PERIODOS = ['hoy', 'ayer', 'esta semana', 'la semana pasada', 'el mes pasado', 'este mes', 'en octubre',
            'la primera semana de octubre', 'el sábado', 'el fin de semana', 'en la quincena', 'en septiembre',
            'del 1 al 7 de octubre', 'el domingo', 'anteayer', 'este año']

# Plantillas: {obj} con artículo, {nom} sin artículo, {uni}/{cuantas} unidad de stock,
# {cant}, {per}, {prod} producto genérico en plural. La «u» al final marca el trato de usted.
# Las últimas dos de cada intención sólo aparecen en la prueba.
PLANTILLAS = {
    'precio': [
        '¿cuánto cuesta {obj}?', '¿a cuánto está {obj}?', 'precio de {obj}', '¿a cuánto vendo {obj}?',
        '¿cuánto vale {obj}?', '¿cuánto es {obj}?', '¿en cuánto dejo {obj}?', '¿a cómo está {obj}?',
        '¿cuánto me sale {cant} de {nom}?', '¿qué precio tiene {obj}?', 'dime el precio de {obj}',
        '¿cuánto cobro por {obj}?', '¿a cuánto lo tengo {obj}?', '{nom} ¿cuánto?', 'precio {nom}',
        '¿cuánto están {cant} {nom}?', '¿a cuánto está {obj} ahorita?',
        '¿me podría decir cuánto cuesta {obj}?u', 'disculpe, ¿a cuánto está {obj}?u',
        '¿a cuánto sale {obj}?', '¿cuánto le pongo a {obj}?',
    ],
    'stock': [
        '¿cuántas {uni} de {nom} me quedan?', '¿tengo {nom}?', '¿hay {nom}?', '¿queda {nom}?',
        'stock de {obj}', '¿se acabó {obj}?', '¿cuánto {nom} tengo?', '¿cuánto me queda de {obj}?',
        '¿todavía hay {nom}?', '¿me quedan {nom}?', '¿cuánto stock hay de {obj}?', '¿tengo que reponer {obj}?',
        'quedan {nom}?', '¿cuántos {nom} hay?', '¿ya no hay {nom}?', '{nom} ¿hay?',
        '¿sabe si queda {nom}?u', '¿me podría decir cuánto stock tengo de {obj}?u',
        '¿cuánto inventario tengo de {obj}?', '¿me alcanza {obj} para la semana?',
    ],
    'ventas': [
        '¿cuánto vendí {per}?', '¿qué fue lo que más se vendió {per}?', '¿cuánto gané {per}?',
        '¿cuántas {uni} de {nom} vendí {per}?', '¿cómo me fue {per}?', '¿cuánto entró a la caja {per}?',
        '¿cuánto me deben de fiado?', '¿cuánto fiado di {per}?', 'total de ventas {per}',
        '¿cuál fue mi mejor día {per}?', '¿qué producto salió más {per}?', '¿cuánto vendí de {nom} {per}?',
        '¿cuánto gasté en reponer {per}?', '¿cuántos clientes me deben?', 'ventas {per}',
        '¿se vendió bien {obj} {per}?', '¿me podría decir cuánto vendí {per}?u',
        '¿cuál fue la ganancia {per}?', '¿qué tal estuvieron las ventas {per}?',
    ],
    'consejo': [
        'dame un consejo para vender más {prod}', '¿cómo puedo vender más?', '¿cuál es el límite de fiado?',
        '¿cada cuánto cobro el fiado?', '¿a quién le puedo fiar?', '¿me conviene aceptar yape?',
        '¿cómo hago para que no se me venzan {los} {prod}?', '¿qué combos puedo armar?', '¿dónde pongo {obj}?',
        '¿qué hago si un cliente no me paga el fiado?', '¿debería subir el precio de {obj}?', '¿cómo ordeno la bodega?',
        'una idea para atraer más clientes', '¿cómo vendo más {prod} en verano?', '¿me recomiendas aceptar plin?',
        '¿qué hago con {los} {prod} que están por vencer?', '¿qué me aconseja para vender más?u',
        '¿cómo puedo ganar más con mi bodega?', '¿me conviene fiar a un vecino nuevo?',
    ],
    'saludo': [
        'hola', 'buenas', 'buenos días', 'buenas tardes, ¿me ayudas?', 'hola causa', 'oe', '¿estás ahí?',
        'hola, ¿qué puedes hacer?', 'holi', 'buenas noches', 'hola bot', 'qué tal',
        'buenas noches, ¿me podría ayudar?u', 'buen día, ¿con quién hablo?u', 'hola, ¿cómo funciona esto?', 'alo',
    ],
    'cortesia': [
        'gracias', 'ok', 'listo', 'chau', 'ya', 'perfecto, gracias', 'muchas gracias causa', 'ya está, bye',
        'gracias 👍', 'ok gracias', 'genial', 'chévere', 'le agradezco muchou',
        'muy amable, graciasu', 'nos vemos', 'ya entendí, gracias',
    ],
    'fuera': [
        '¿quién ganó el partido de ayer?', 'cuéntame un chiste', '¿qué hora es?', '¿va a llover mañana?',
        '¿cuál es la capital de Francia?', 'escríbeme un poema', '¿cuánto está el dólar?', '¿cómo está el tráfico?',
        'pon música', '¿quién es el presidente?', '¿cuántos años tienes?', 'hazme la tarea de matemáticas',
        '¿me podría recomendar una película?u', '¿qué opina del clima?u', '¿cuándo juega Perú?', '¿cómo se hace un ceviche?',
    ],
}
PREFIJOS = ['', '', '', 'oe ', 'causa, ', 'una pregunta, ', 'oye ', 'ya, ']
SUFIJOS = ['', '', '', ' pe', ' porfa', ' 🙏', ' causa', ' ??']


def _quitar_tildes(t: str) -> str:
    # Carácter por carácter para no cambiar el largo (los tramos ya están contados).
    return ''.join(unicodedata.normalize('NFD', c)[0] for c in t)


def _articulo(genero: str, plural: bool) -> str:
    return {('m', False): 'el', ('f', False): 'la', ('m', True): 'los', ('f', True): 'las'}[(genero, plural)]


def _es_plural(nombre: str) -> bool:
    return nombre.split()[0].endswith('s')


def _plural(nombre: str) -> str:
    if _es_plural(nombre):
        return nombre
    primera, *resto = nombre.split()
    primera = _quitar_tildes(primera) + 'es' if primera[-1] in 'nlr' else primera + 's'
    return ' '.join([primera, *resto])


def objeto(rng: random.Random, marcas: set[str], articulo: bool) -> list[tuple[str, str | None]]:
    """El producto del que se habla: producto, marca y medida en alguna combinación."""
    prod, gen, _uni, mars, meds = rng.choice(PRODUCTOS)
    mars = [m for m in mars if m in marcas] or mars
    marca, medida = rng.choice(mars), rng.choice(meds)
    forma = rng.choice(['p', 'pm', 'pmd', 'pd', 'm', 'md', 'pdm'])
    segs: list[tuple[str, str | None]] = []
    if articulo:
        segs.append((_articulo(gen, _es_plural(prod)) + ' ', None))
    piezas = {'p': (prod, 'PRODUCTO'), 'm': (marca, 'MARCA'), 'd': (medida, 'MEDIDA')}
    for i, k in enumerate(forma):
        if i:
            sep = rng.choice([' de ', ' ']) if k == 'd' else rng.choice([' ', ' ', ' de '])
            segs.append((sep, None))
        segs.append(piezas[k])
    return segs


def frase(intencion: str, plantilla: str, rng: random.Random, marcas: set[str]) -> dict:
    usted = plantilla.endswith('u')
    plantilla = plantilla[:-1] if usted else plantilla
    prod, gen, (uni, gen_uni), _m, _d = rng.choice(PRODUCTOS)
    segs: list[tuple[str, str | None]] = [(rng.choice(PREFIJOS) if not usted else '', None)]
    for i, parte in enumerate(re.split(r'\{(\w+)\}', plantilla)):
        if i % 2 == 0:
            segs.append((parte, None))
        elif parte == 'obj':
            segs += objeto(rng, marcas, articulo=True)
        elif parte == 'nom':
            segs += objeto(rng, marcas, articulo=False)
        elif parte == 'uni':
            segs.append((uni, None))
        elif parte == 'cant':
            segs.append((rng.choice(CANTIDADES), 'CANTIDAD'))
        elif parte == 'per':
            segs.append((rng.choice(PERIODOS), 'PERIODO'))
        elif parte == 'prod':
            segs.append((_plural(prod), 'PRODUCTO'))
        elif parte == 'los':
            segs.append((_articulo(gen, True), None))
    if not usted:
        segs.append((rng.choice(SUFIJOS), None))

    sin_tildes = rng.random() < 0.4
    sin_abre = rng.random() < 0.5
    texto, tramos = '', []
    for t, etiqueta in segs:
        if sin_tildes:
            t = _quitar_tildes(t)
        if sin_abre and etiqueta is None:
            t = t.replace('¿', '')
        if etiqueta:
            tramos.append({'inicio': len(texto), 'fin': len(texto) + len(t), 'etiqueta': etiqueta})
        texto += t
    # Los espacios del prefijo/sufijo pueden quedar dobles o en los bordes: se limpian corriendo los tramos.
    while texto.startswith(' '):
        texto = texto[1:]
        tramos = [{**x, 'inicio': x['inicio'] - 1, 'fin': x['fin'] - 1} for x in tramos]
    texto = texto.rstrip()
    return {'texto': texto, 'intencion': intencion, 'banderas': {'usted': usted}, 'tramos': tramos}


def validar(f: dict) -> None:
    t = f['texto']
    assert f['intencion'] in INTENCIONES and set(f['banderas']) == set(BANDERAS), f
    fin_anterior = 0
    for x in f['tramos']:
        a, b = x['inicio'], x['fin']
        assert x['etiqueta'] in ETIQUETAS and fin_anterior <= a < b <= len(t), f
        assert t[a].isalnum() and t[b - 1].isalnum(), f
        assert (a == 0 or not t[a - 1].isalnum()) and (b == len(t) or not t[b].isalnum()), f
        fin_anterior = b


def generar(n: int, intenciones_peso: dict, plantillas: dict, marcas: set[str], rng: random.Random, prefijo: str) -> list[dict]:
    filas, vistos = [], set()
    nombres = list(intenciones_peso)
    pesos = [intenciones_peso[i] for i in nombres]
    intentos = 0
    while len(filas) < n and intentos < n * 50:
        intentos += 1
        intencion = rng.choices(nombres, pesos)[0]
        f = frase(intencion, rng.choice(plantillas[intencion]), rng, marcas)
        if f['texto'].lower() in vistos:
            continue
        vistos.add(f['texto'].lower())
        validar(f)
        f['id'] = f'{prefijo}-{len(filas) + 1:05d}'
        f['origen'] = 'sintetica'
        filas.append(f)
    return filas


def manuales() -> list[dict]:
    """Las frases escritas a mano de datos/manual.tsv, con sus tramos pasados a caracteres."""
    filas = []
    with open(os.path.join(DATOS, 'manual.tsv'), encoding='utf-8') as fh:
        for linea in fh:
            if not linea.strip() or linea.startswith('#'):
                continue
            intencion, marcado, *trato = linea.rstrip('\n').split('\t')
            texto, tramos, k = '', [], 0
            for m in re.finditer(r'\[([^\]]+)\]([A-Z]+)', marcado):
                texto += marcado[k:m.start()]
                tramos.append({'inicio': len(texto), 'fin': len(texto) + len(m.group(1)), 'etiqueta': m.group(2)})
                texto += m.group(1)
                k = m.end()
            texto += marcado[k:]
            f = {'texto': texto, 'intencion': intencion, 'banderas': {'usted': trato == ['u']}, 'tramos': tramos,
                 'id': f'man-{len(filas) + 1:03d}', 'origen': 'manual'}
            validar(f)
            filas.append(f)
    return filas


def main(semilla: int = 7) -> None:
    rng = random.Random(semilla)
    todas = sorted({m for p in PRODUCTOS for m in p[3]})
    marcas_prueba = set(todas[::5])  # una de cada cinco marcas sólo en la prueba
    marcas_ent = set(todas) - marcas_prueba
    ent_plant = {k: v[:-2] for k, v in PLANTILLAS.items()}
    peso = {'precio': 3, 'stock': 3, 'ventas': 2, 'consejo': 2, 'saludo': 1, 'cortesia': 1, 'fuera': 1}

    ent = generar(4000, peso, ent_plant, marcas_ent, rng, 'ent')
    vistos = {f['texto'].lower() for f in ent}
    val = [f for f in generar(500, peso, ent_plant, marcas_ent, rng, 'val') if f['texto'].lower() not in vistos][:400]
    vistos |= {f['texto'].lower() for f in val}
    pru = [f for f in generar(800, peso, PLANTILLAS, set(todas), rng, 'pru') if f['texto'].lower() not in vistos][:600]

    # Una de cada tres frases escritas a mano sólo va a la prueba: mide lo que las plantillas no cubren.
    man = manuales()
    ent += [f for i, f in enumerate(man) if i % 3 != 2]
    pru += [f for i, f in enumerate(man) if i % 3 == 2]

    os.makedirs(DATOS, exist_ok=True)
    for nombre, filas in (('entrenamiento', ent), ('validacion', val), ('prueba', pru)):
        with open(os.path.join(DATOS, f'{nombre}.jsonl'), 'w', encoding='utf-8') as fh:
            for f in filas:
                fh.write(json.dumps(f, ensure_ascii=False) + '\n')
        print(nombre, len(filas))


if __name__ == '__main__':
    main()
