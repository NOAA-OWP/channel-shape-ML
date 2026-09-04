# Environment Builder
FROM mambaorg/micromamba:1.5.8-bookworm-slim AS builder

USER root

WORKDIR /tmp/env
COPY environment.yml /tmp/env/environment.yml

# Install dependencies directly into micromamba's pre-configured base environment (/opt/conda)
RUN micromamba install -y -n base -f /tmp/env/environment.yml \
    && micromamba clean --all --yes \
    && find /opt/conda -name '__pycache__' -type d -exec rm -rf {} + \
    && find /opt/conda -name '*.a' -delete \
    && find /opt/conda -name '*.pyc' -delete

# Production Runtime
FROM debian:bookworm-slim AS runtime

LABEL maintainer="RIver ML Pipeline Engineering"
LABEL description="RIver ML Sequential Inference Engine (TW -> Y -> r -> n)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/conda/bin:$PATH" \
    GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.gpkg,.parquet,.json \
    AWS_S3_MAX_CONCURRENT_REQUESTS=64 \
    PYARROW_IGNORE_TIMEZONE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda

RUN groupadd -g 1001 river_ml && \
    useradd -u 1001 -g river_ml -m -s /bin/bash river_ml_user

WORKDIR /app

RUN mkdir -p /app/data /app/models /app/outputs /tmp/river_ml && \
    chmod -R 777 /app /tmp/river_ml

COPY src/ /app/src/
COPY deployment/ /app/deployment/
COPY run_inference.py /app/run_inference.py
COPY entrypoint.sh /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh

USER root

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["--help"]