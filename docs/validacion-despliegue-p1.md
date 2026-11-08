# Validación del despliegue P1 — MiBici

## 1. Objetivo

Documentar la validación local, de extremo a extremo, del despliegue de un modelo P1 de predicción de salidas de bicicletas, utilizando MLflow Model Registry, MLflow Model Serving, Docker Compose y una API REST implementada con FastAPI y Pydantic.

## 2. Entrenamiento y evaluación

El modelo se entrenó con el comando:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run python -m mibici.modeling.train --problem p1 --model linear_regression
```

| Indicador | Resultado |
|---|---:|
| Algoritmo | Regresión lineal |
| Viajes cargados en entrenamiento | 6,311,111 |
| Viajes cargados en validación | 1,231,240 |
| Validation MAE | 7.110 |
| Validation RMSE | 13.1798 |
| Baseline RMSE | 16.002 |
| Reducción relativa de RMSE frente al baseline | 17.6 % |

La reducción relativa se calculó como `(16.002 - 13.1798) / 16.002 × 100`.

- Experimento: `mibici-equipo40`.
- Run ID: `b3c72f322286474791bf1e054f6983e0`.
- URI del modelo generado: `models:/m-48ae953dc15c46e2850f1391600ae0bf`.

Los números de viajes corresponden a los registros cargados durante el entrenamiento y la validación; no se presentan como el número final de filas del conjunto de características.

## 3. Registro y promoción en MLflow

El modelo se promovió con:

```bash
uv run python -m mibici.modeling.registry --problem p1
```

El registro devolvió:

| Campo | Valor |
|---|---|
| Nombre | `mibici-p1` |
| Versión | `1` |
| Alias | `production` |
| Etapa | `Production` |
| Corrida fuente | `b3c72f322286474791bf1e054f6983e0` |
| Validation RMSE | `13.1798` |

El servidor de seguimiento y el registro utilizados para esta validación fueron locales (`http://127.0.0.1:5000`). La API de etapas de MLflow mostró una advertencia de obsolescencia, pero el registro y la asignación del alias finalizaron correctamente.

## 4. Arquitectura de inferencia

```text
Cliente HTTP
    |
    | POST http://127.0.0.1:8000/predict
    v
FastAPI + Pydantic (contenedor api)
    |
    | HTTP a inference:8080/invocations
    v
MLflow Model Serving (contenedor inference)
    |
    v
Modelo registrado mibici-p1, versión 1
    |
    v
Respuesta JSON: predicted_departures
```

El archivo `docker-compose.serving.yml` define los servicios `api` e `inference`. Los puertos se publicaron exclusivamente en la interfaz local: `127.0.0.1:8000` para FastAPI y `127.0.0.1:8080` para MLflow Serving.

## 5. Preparación de artefactos

Se descargó el modelo desde el registro hacia `build/serving`:

```python
from pathlib import Path
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
destination = Path("build/serving")
destination.mkdir(parents=True, exist_ok=True)
mlflow.artifacts.download_artifacts(
    artifact_uri="models:/mibici-p1/1",
    dst_path=str(destination),
)
```

La descarga reportó 38 artefactos y se comprobó la existencia de:

- `build/serving/MLmodel`.
- `build/serving/requirements.txt`.

Los artefactos del modelo se emplearon para construir la imagen de inferencia; no es necesario incorporarlos al repositorio Git. La carpeta `.local-mlflow/` se excluyó mediante `.gitignore` para evitar subir la base de datos y los artefactos del servidor local.

## 6. Construcción y ejecución con Docker

```bash
docker compose -f docker-compose.serving.yml up -d --build
docker compose -f docker-compose.serving.yml ps
```

La construcción terminó correctamente y produjo las imágenes:

- `mibici-api:local`.
- `mibici-inference:p1-v1`.

Se levantaron los contenedores `mibici-serving-api-1` y `mibici-serving-inference-1`. Durante la primera ejecución, el puerto 8000 estaba ocupado por el contenedor previo de pruebas `mibici-api-qa`. Se detuvo dicho contenedor y se recreó el servicio API. La comprobación final de `docker compose ps` mostró ambos servicios en estado `Up`, con los puertos publicados en `127.0.0.1`.

## 7. Prueba HTTP real de extremo a extremo

### Solicitud

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": 2,
    "day_of_week": 1,
    "departures_lag_1": 76.0,
    "departures_lag_7": 21.0,
    "departures_mean_7": 55.857142857142854
  }'
```

### Resultado observado

**Estado HTTP:** `200 OK`.

```json
{
  "predicted_departures": 46.298935294225515
}
```

Este resultado demuestra que, para la solicitud presentada, FastAPI recibió los datos, invocó el servicio de inferencia en Docker y devolvió una predicción numérica generada por el modelo. La predicción es una estimación, no un conteo observado ni una validación de exactitud para ese caso particular.

**Evidencia versionada:** [`evidence/p1-docker-prediction.json`](../evidence/p1-docker-prediction.json).

## 8. Pruebas automatizadas

Se ejecutó la suite del repositorio:

```bash
uv run pytest -v
```

**Resultado reportado:** 34 pruebas aprobadas. La suite incluye pruebas de validación y manejo de errores de FastAPI, inferencia simulada y pruebas HTTP de integración. Algunas pruebas de integración están configuradas para omitirse si no hay un servicio API disponible; por ello, la solicitud `curl` anterior constituye evidencia independiente de la inferencia real con el modelo desplegado.

## 9. Alcance y conclusiones

Se comprobó exitosamente un despliegue **local** de extremo a extremo:

1. Entrenamiento del modelo P1 y evaluación frente a un baseline.
2. Registro de `mibici-p1` versión 1 y promoción al alias `production`.
3. Descarga de artefactos del registro para el servicio de inferencia.
4. Construcción de imágenes OCI y ejecución de los dos contenedores con Docker Compose.
5. Respuesta HTTP `200 OK` y predicción real desde `POST /predict`.
6. Conservación de evidencia JSON y pruebas automatizadas del repositorio.

Esta validación no equivale a un despliegue en infraestructura de producción externa, ni demuestra por sí sola disponibilidad, escalabilidad o monitoreo continuo.

## 10. Referencias

- [Repositorio MiBici — Equipo 40](https://github.com/ChadayL/TC5061-Proyecto-Equipo40)
- [Pull Request #13](https://github.com/ChadayL/TC5061-Proyecto-Equipo40/pull/13)
- Configuración: `docker-compose.serving.yml`.
- Documentación de serving: `docs/servir-p1.md`.
- Evidencia: `evidence/p1-docker-prediction.json`.
