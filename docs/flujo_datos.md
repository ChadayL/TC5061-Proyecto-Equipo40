# Trabajo con datos y stages

## Las piezas

- `dataset.py` lee los archivos y devuelve las tablas `viajes` y `estaciones`.
- `data_preparation/` contiene las funciones que transforman los datos.
- Cada módulo de preparación tiene un `run(datos)` que agrupa sus transformaciones.
- `pipeline.py` define el orden en `STAGES`; `load_stage()` ejecuta hasta el punto solicitado.
- Los notebooks prueban, comparan e interpretan resultados.

## Cargar un punto común

```python
from mibici.pipeline import load_stage

datos = load_stage("structure")
viajes = datos["viajes"]
estaciones = datos["estaciones"]
```

| Solicitud | Ejecución actual |
|---|---|
| `raw` | Cargar originales. |
| `structure` | Originales → structure. |
| `missing` | Originales → structure → missing. |
| `common` | Todas las etapas compartidas de `STAGES`. |

`common` es el final de la lista, no otra transformación. Si se modifica una etapa, también cambia el resultado de las posteriores. Una etapa sin tratamientos devuelve los datos sin cambios.

Cada llamada reconstruye desde los originales: por ahora no hay caché ni guardado automático. `nrows=5` limita a cinco viajes por archivo y conserva el catálogo completo; es una muestra para probar código, no para concluir sobre la calidad de todo el conjunto.

## Probar antes de integrar

1. Registrar el hallazgo y su evidencia.
2. Cargar el stage anterior al tratamiento que se investiga.
3. Probar sobre una copia y comparar el resultado.
4. Implementar la función reutilizable en el módulo correspondiente.
5. Cuando el tratamiento esté justificado, incorporarlo al `run()` de su etapa.
6. Documentar la verificación y abrir un PR para revisión.

```python
datos = load_stage("structure")
prueba = {nombre: df.copy() for nombre, df in datos.items()}

# Aplicar aquí una función propuesta y comparar datos con prueba.
```

No es necesario crear un stage por hallazgo. Varias funciones relacionadas pueden formar parte de una etapa. Para diagnosticar la calidad original, cargar `raw`.

## Contrato de run()

Cada `run(datos)` recibe y devuelve un diccionario con `viajes` y `estaciones`. Conserva las tablas que no transforma y evita modificar accidentalmente las entradas.

Ejemplo mínimo de etapa todavía sin tratamientos:

```python
def run(datos):
    return datos
```

`run()` no carga archivos ni llama a etapas anteriores. El orden lo controla exclusivamente `pipeline.py`:

```python
STAGES = [
    ("structure", structure.run),
    ("missing", missing.run),
]
```

Al agregar una etapa, revisar sus columnas y tipos de entrada y salida. Usar nombres únicos; `raw` y `common` están reservados. Las etapas se ejecutan en orden, independientemente de cómo aparezcan los archivos en el explorador.

## Después de common

Las transformaciones específicas de P1, P2 o P3 se construyen a partir de `common` y quedan fuera de la lista compartida. Las variables se implementan en `features.py`.

Los imputadores que calculan estadísticas, escaladores y selectores se ajustan solo con entrenamiento dentro del pipeline del modelo. Los predictores temporales utilizan únicamente información disponible al predecir; no se decide la preparación explorando el periodo de prueba reservado.

## Colaboración

Actualizar el repositorio antes de empezar; trabajar en una rama por tarea y coordinar quién edita cada notebook. Un cambio solo llega al resto cuando se integra y los compañeros actualizan sus copias.

Después de modificar módulos, reiniciar el kernel y ejecutar nuevamente el notebook. Para reproducir un resultado se necesitan los mismos archivos, configuración, commit y dependencias. El nombre `common` por sí solo no identifica una versión de datos.

Consultar la [guía de hallazgos](../references/hallazgos/README.md) para registrar decisiones.
