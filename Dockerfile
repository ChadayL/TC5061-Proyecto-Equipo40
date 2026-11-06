FROM python:3.10-slim AS jupyter
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv
ENV UV_PROJECT_ENVIRONMENT=/opt/mibici-venv \
    UV_LINK_MODE=copy \
    PATH="/opt/mibici-venv/bin:$PATH"
WORKDIR /workspace/mibici
EXPOSE 8888
CMD ["sh", "-c", "uv sync --locked && exec uv run --locked jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/workspace/mibici"]

FROM python:3.10-slim AS mlflow
RUN pip install --no-cache-dir mlflow==3.17.1 boto3
WORKDIR /mlflow
EXPOSE 5000

FROM golang:1.24.8-bookworm AS minio-build
RUN GOBIN=/out CGO_ENABLED=0 go install github.com/minio/minio@RELEASE.2025-10-15T17-29-55Z

FROM debian:bookworm-slim AS minio
COPY --from=minio-build /out/minio /usr/local/bin/minio
ENTRYPOINT ["minio"]

FROM python:3.10-slim AS inference
ARG MODEL_DIR=build/serving
COPY ${MODEL_DIR}/requirements.txt /tmp/model-requirements.txt
RUN pip install --no-cache-dir pip==26.2.1 \
    && pip install --no-cache-dir --resume-retries 5 -c /tmp/model-requirements.txt \
    mlflow scikit-learn pandas numpy scipy cloudpickle loguru python-dotenv typer
COPY ${MODEL_DIR}/ /opt/model/
EXPOSE 8080
CMD ["mlflow", "models", "serve", "-m", "/opt/model", "--host", "0.0.0.0", "--port", "8080", "--env-manager", "local"]

FROM python:3.10-slim AS api
WORKDIR /app
COPY mibici/api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt
COPY mibici/api/ /app/mibici/api/
EXPOSE 8000
CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
