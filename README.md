# MiBici

Proyecto del equipo 40 de TC5061 con datos abiertos de MiBici. Estima las salidas diarias por estación (P1), aproxima el riesgo de desbalance (P2) y agrupa estaciones por patrones de uso (P3).

Para empezar desde un clon, sigue la [guía de desarrollo con Docker](docs/jupyter-contenedor.md) y ejecuta la [ingesta de datos](docs/datos-mibici.md#ingesta-de-datos).

## Documentación

- [Datos de MiBici](docs/datos-mibici.md): fuentes, formatos, particiones y notebooks.
- [Entrenamiento y experimentos con MLflow](docs/experimentos-mlflow.md): corridas, búsquedas y Model Registry.
- [Reproducir el modelo seleccionado de P1](docs/experimentos-mlflow.md#configuración-seleccionada-para-p1): comando, parámetros, métricas y corrida original.
- [Despliegue de P1 con FastAPI y Docker](docs/servir-p1.md): descarga del modelo y prueba de inferencia.
- [Monitoreo y comparación A/B de P1](docs/monitoreo-p1.md): Evidently, resultados y decisión sobre la candidata.
- [Calidad y pruebas](docs/calidad-y-pruebas.md): PyTest, Deepchecks y comprobaciones de CI.
