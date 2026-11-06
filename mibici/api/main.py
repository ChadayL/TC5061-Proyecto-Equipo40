import os
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError

app = FastAPI(title="MiBici P1")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8080/invocations")
NonnegativeFloat = Annotated[FiniteFloat, Field(ge=0)]


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/docs")


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    station_id: Annotated[int, Field(gt=0, le=2**63 - 1)]
    day_of_week: Annotated[int, Field(ge=0, le=6)]
    departures_lag_1: NonnegativeFloat
    departures_lag_7: NonnegativeFloat
    departures_mean_7: NonnegativeFloat


class InferenceResponse(BaseModel):
    predictions: Annotated[list[FiniteFloat], Field(min_length=1, max_length=1)]


class PredictionOutput(BaseModel):
    predicted_departures: FiniteFloat


@app.post("/predict", response_model=PredictionOutput)
def predict(features: PredictionInput) -> PredictionOutput:
    payload = {"dataframe_records": [features.model_dump()]}
    try:
        response = httpx.post(INFERENCE_URL, json=payload, timeout=10.0)
        response.raise_for_status()
        result = InferenceResponse.model_validate_json(response.content)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "El servidor de inferencia tardó demasiado") from exc
    except (httpx.HTTPError, ValidationError) as exc:
        raise HTTPException(502, "El servidor de inferencia no devolvió una predicción válida") from exc
    return PredictionOutput(predicted_departures=result.predictions[0])
