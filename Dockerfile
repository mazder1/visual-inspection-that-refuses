# CPU-only serving image. The CUDA wheel alone is ~2.5 GB; the CPU wheel keeps
# this image around 1 GB, and the target platform (Cloud Run) bills CPU.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VINSPECT_BUNDLES=/app/bundles \
    # Matches the Cloud Run --cpu setting; torch saturates what is there.
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

# Torch CPU wheel first, pinned to its own index, so the layer caches well and
# no CUDA dependency ever sneaks in.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0

COPY pyproject.toml ./
COPY src ./src
RUN pip install .[serve]

# The frozen chain: model weights + calibrator + thresholds, per category.
# Produced by `python -m vinspect.serve.export` and copied in at build time so
# the container is self-contained -- no dataset, no runs directory.
COPY bundles ./bundles

EXPOSE 8080
# Cloud Run injects PORT; default to 8080 locally. One worker: the model is
# CPU-bound and memory is the scarce resource.
CMD ["sh", "-c", "uvicorn vinspect.serve.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
