# Hallazgos

Un archivo YAML por hallazgo, guardado en la etapa donde se detectó:

- `01_data_understanding/`
- `02_data_preparation/`
- `03_modeling/p1_regression/`
- `03_modeling/p2_classification/`
- `03_modeling/p3_clustering/`

## Registrar

1. Revisar si el problema ya está registrado.
2. Copiar [plantilla.yaml](plantilla.yaml) a la carpeta correspondiente.
3. Reemplazar todos los datos del ejemplo con evidencia propia.
4. Nombrar el archivo `H-iniciales-numero-descripcion.yaml`.

Cada autor lleva su consecutivo. El nombre y la ubicación se conservan al resolverlo. `autor` identifica a quien lo detectó; `responsable`, a quien lo atiende.

## Seguimiento

- `pendiente`: falta atenderlo.
- `en_revision`: se investiga o trabaja en la solución.
- `resuelto`: se documentó la decisión y se verificó el resultado.

El responsable actualiza acción, resolución y estado mediante PR. Coordinar las ediciones sobre el mismo archivo; archivos independientes reducen conflictos, pero no los eliminan.

En la evidencia indicar notebook y sección, archivo o versión de datos y el resultado observado. Evitar incluir identificadores personales o registros completos de usuarios.

La solución se implementa en el código reutilizable. En `resolucion` explicar qué se hizo, cómo se verificó y, si aplica, la función o PR. También puede resolverse conservando los datos sin cambios, con justificación.

La plantilla es un ejemplo pendiente, no un hallazgo que deba contarse automáticamente. No requiere un sistema adicional: empezar completando los campos esenciales.
