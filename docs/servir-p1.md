# Servir P1 con MLflow Model Serving

Desde la raíz del repositorio y con el entorno de desarrollo activo, descargar la versión 1 del Registry.

```powershell
docker compose exec -T ds_workspace python -c 'from pathlib import Path; import mlflow; p=Path("build/serving"); p.mkdir(parents=True, exist_ok=True); mlflow.artifacts.download_artifacts(artifact_uri="models:/mibici-p1/1", dst_path=str(p))'
```

El artefacto queda en build/serving, una carpeta ignorada por Git. La imagen incorpora el modelo y su código. El servidor carga esa copia al arrancar.

```powershell
docker compose -f docker-compose.serving.yml up -d --build
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/invocations -ContentType application/json -InFile build/serving/serving_input_example.json
```

El puerto 8080 está publicado en localhost para la prueba directa. FastAPI
llama a http://inference:8080/invocations por la red de Compose. Para dejar
únicamente FastAPI accesible desde el host, retirar el bloque ports de inference.

## Probar FastAPI

Compose construye la etapa api y publica http://127.0.0.1:8000/docs.
En `POST /predict` enviar un JSON con las cinco variables de P1.

```powershell
$body = '{"station_id":2,"day_of_week":1,"departures_lag_1":76.0,"departures_lag_7":21.0,"departures_mean_7":55.857142857142854}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -Body $body
```

La respuesta de P1 v1 es `{"predicted_departures":46.29893898270091}`.
Si están activos los contenedores temporales de la prueba, detener
mibici-api-p1-test y mibici-inference-p1-test antes de arrancar Compose
para liberar los puertos 8000 y 8080.

## Preparar otra versión u otro modelo

Descargar primero una versión concreta en su propia carpeta, construir la imagen
y probarla con el ejemplo que trae el artefacto. Este ejemplo usa P1 versión 2.
Debe existir en el Registry antes de ejecutar la descarga. Usar el mismo comando
con mibici-p2 o mibici-p3 para preparar clasificación o clustering.

```powershell
$modelName = 'mibici-p1'
$modelVersion = '2'
$env:MODEL_DIR = "build/serving/$modelName-v$modelVersion"
$env:MODEL_TAG = "p1-v$modelVersion"
$env:INFERENCE_PORT = '8081'

docker compose exec -T ds_workspace mkdir -p $env:MODEL_DIR
docker compose exec -T ds_workspace mlflow artifacts download --artifact-uri "models:/$modelName/$modelVersion" --dst-path $env:MODEL_DIR
docker compose -p mibici-preview -f docker-compose.serving.yml up -d --build inference
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8081/invocations -ContentType application/json -InFile "$env:MODEL_DIR/serving_input_example.json"
```

Comprobar el resultado de cada comando antes de continuar. Usar un directorio
nuevo al exportar otra versión para evitar mezclar archivos. Después de probar
P1 v2, ejecutar `docker compose -p mibici-preview -f docker-compose.serving.yml down`,
cambiar INFERENCE_PORT a 8080 y ejecutar `docker compose -f docker-compose.serving.yml up -d`.
Compose reemplaza el contenedor de P1 por la imagen probada.

## Servir P1, P2 y P3 a la vez

Cada instancia de MLflow Model Serving carga un artefacto. Usar el mismo
Dockerfile y Compose para crear tres imágenes y tres contenedores. Repetir
la descarga y el arranque con los valores siguientes y sustituir
`-p mibici-preview` por el nombre del proyecto de la fila. Las versiones son ejemplos.

| Proyecto Compose | Modelo en el Registry | MODEL_DIR | MODEL_TAG | INFERENCE_PORT |
|---|---|---|---|---|
| mibici-serving | mibici-p1/1 | build/serving/mibici-p1-v1 | p1-v1 | 8080 |
| mibici-serving-p2 | mibici-p2/1 | build/serving/mibici-p2-v1 | p2-v1 | 8081 |
| mibici-serving-p3 | mibici-p3/1 | build/serving/mibici-p3-v1 | p3-v1 | 8082 |

Probar cada puerto con su propio serving_input_example.json. Los tres servicios
exponen /invocations y cada uno usa la firma de su modelo. Cuando se integre
FastAPI, se pueden declarar los tres servicios en un Compose compartido,
retirar sus puertos publicados y dirigir cada petición al servicio correspondiente.
