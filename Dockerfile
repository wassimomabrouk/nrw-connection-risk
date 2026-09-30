# Live prediction API (docs/SERVING.md). The collector keeps running outside the container
# (systemd); the API reads its data read-only through a volume.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Pickled models only load with the scikit-learn minor version they were saved with:
# build with the version of the machine that fits the bundle (py -m pip show scikit-learn).
ARG SKLEARN_VERSION=1.9.1

WORKDIR /app
# Dependencies first, from pyproject.toml alone (with an empty package), so this slow layer
# stays cached when only the code changes; then the code itself without dependencies.
COPY pyproject.toml ./
RUN mkdir -p src/nrw_connection_risk && touch src/nrw_connection_risk/__init__.py \
    && pip install ".[serve]" "scikit-learn==${SKLEARN_VERSION}" \
    && pip uninstall -y nrw-connection-risk && rm -rf src
COPY src ./src
RUN pip install --no-deps .
COPY config ./config

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/collector /app/data/serving /app/models \
    && chown -R app /app/data/serving
USER app

ENV NRW_ROOT=/app
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "nrw_connection_risk.serving.api:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
