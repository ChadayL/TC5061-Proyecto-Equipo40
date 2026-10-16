# Registro de hallazgos

Cada hallazgo se documenta en un archivo YAML independiente.

## Agregar un hallazgo

1. Revisar si ya existe uno sobre el mismo problema.
2. Copiar plantilla.yaml dentro de pendientes/.
3. Nombrarlo H-iniciales-numero-descripcion.yaml.
4. Completar la descripción, evidencia y versión de los datos.
5. Abrir un Pull Request para incorporarlo.

Cada integrante administra su numeración.
Ejemplo: H-co-001-genero-faltante.yaml.

No incluir datos personales ni registros completos de usuarios.

## Estados

- pendiente: detectado, sin decisión o tratamiento completo.
- en_revision: se está investigando, implementando o verificando.
- resuelto: existe una decisión documentada y verificada.

La carpeta y el campo estado deben coincidir.
Al cambiar de estado, mover el archivo sin cambiar su nombre.

## Atender un hallazgo

1. Asignar un responsable y moverlo a en_revision/.
2. Documentar las acciones propuestas.
3. Implementar el tratamiento en el módulo correspondiente.
4. Referenciar el ID del hallazgo en el Pull Request.
5. Registrar qué cambió y cómo se verificó.
6. Moverlo a resueltos/ cuando termine la revisión.

Resolver no implica necesariamente modificar los datos.
También puede decidirse conservarlos, justificando la decisión.

## Relación con el código

El YAML documenta el problema, las decisiones y su verificación.
Las transformaciones se implementan en el código del proyecto.
Los notebooks muestran la evidencia y analizan los resultados.