"""Prueba de carga para la API de predicción P1 de MiBici.

Ejemplo:
    locust -f locustfile.py --host http://127.0.0.1:8000

Ejecución sin interfaz:
    locust -f locustfile.py --host http://127.0.0.1:8000 \
        --headless -u 10 -r 2 -t 2m --csv locust_results --html locust_report.html

Los resultados deben medirse con los contenedores en ejecución; este archivo
no genera ni presupone métricas de desempeño.
"""

from locust import HttpUser, between, task


class MiBiciPredictionUser(HttpUser):
    """Simula clientes que solicitan predicciones de salidas de bicicletas."""

    wait_time = between(1, 3)

    @task
    def predict_departures(self):
        payload = {
            "station_id": 2,
            "day_of_week": 1,
            "departures_lag_1": 76.0,
            "departures_lag_7": 21.0,
            "departures_mean_7": 55.857142857142854,
        }

        with self.client.post(
            "/predict",
            json=payload,
            name="POST /predict",
            catch_response=True,
            timeout=30,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return

            try:
                result = response.json()
            except ValueError:
                response.failure("La respuesta no es JSON válido")
                return

            prediction = result.get("predicted_departures")
            if (
                isinstance(prediction, bool)
                or not isinstance(prediction, (int, float))
            ):
                response.failure("Falta predicted_departures numérico")
            else:
                response.success()
