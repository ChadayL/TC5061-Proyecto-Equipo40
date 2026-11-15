# Pruebas, calidad de datos y seguridad

Dentro del [contenedor de trabajo](jupyter-contenedor.md), instala el grupo de calidad (PyTest, Deepchecks, Bandit y pip-audit) con las versiones fijadas en `uv.lock`:

```bash
uv sync --locked --extra quality
```

Corre la suite automatizada desde la raíz del repositorio:

```bash
uv run pytest -q
```

Las pruebas unitarias cubren funciones de P1, P2 y P3, datos límite y la consistencia de predicciones del Random Forest con la misma semilla. Las pruebas que necesitan Parquet local se omiten en un clon limpio; para habilitarlas, primero ejecuta la [ingesta de datos](datos-mibici.md#ingesta-de-datos). No subas `data/` a Git.

Ejecuta los chequeos de Deepchecks con muestras deterministas de hasta 100,000 viajes por partición:

```bash
uv run --extra quality python -m mibici.validation
```

El comando genera `reports/data_quality/integridad_train.html`, `comparacion_train_validation.html` y `resumen.json`. Deepchecks revisa integridad en train y compara train contra validation para buscar problemas de partición y drift. El resumen reporta tipos, nulos y duplicados; los viajes, identificadores de usuario y reportes generados no se versionan. Si los datos locales aún no están ingeridos, la validación indica los archivos que faltan.

En una ejecución de referencia con 1,000 filas (--sample-rows 1000), PyTest terminó con **19 pruebas aprobadas**; Deepchecks ejecutó las suites de integridad y comparación (12 chequeos cada una). En esa muestra se observaron nulos en Genero y filas duplicadas; son hallazgos reportados para análisis, no filas eliminadas automáticamente. Bandit terminó sin hallazgos de seguridad reportables; hay supresiones explícitas para el acceso al host fijo MiBici y el comando fijo git rev-parse (sin shell ni entrada del usuario).

Escanea vulnerabilidades conocidas de dependencias y patrones de seguridad en el código:

```bash
uv run --extra quality pip-audit
uv run --extra quality bandit -r mibici
```

`pip-audit` puede encontrar vulnerabilidades en versiones fijadas: revisa el paquete y el aviso antes de actualizar y vuelve a ejecutar las pruebas. Bandit puede señalar patrones que requieren inspección manual. El repositorio ignora `.env`, bases MLflow, datos y reportes locales; no pongas tokens ni contraseñas en el código. La integración de GitHub Actions ejecuta PyTest, Bandit y pip-audit en pushes y pull requests.

Para la evidencia de entrega, guarda una captura de la salida de `uv run pytest -q`, de los dos escaneos de seguridad y del reporte HTML de Deepchecks abierto en el navegador. En GitHub, la pestaña **Actions** muestra el resultado automatizado del workflow `quality`.
