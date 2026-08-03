# Torch-free INT8 serving image. Torch existed in the previous image only to
# run the model; ONNX Runtime replaces it at ~50 MB against ~1.3 GB, taking
# the image from 1.95 GB to a few hundred MB. The scoring path is numpy/scipy
# and shared verbatim with the evaluated chain; the INT8 graph plus a chain
# refit on its own validation scores is the stage-3 arm-B configuration.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    VINSPECT_BUNDLES=/app/bundles \
    OMP_NUM_THREADS=4

WORKDIR /app

# Serving dependencies only -- deliberately NOT `pip install .`, whose core
# dependencies include torch. The source tree rides along on PYTHONPATH.
RUN pip install \
    "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9" \
    "numpy>=1.24" "scipy>=1.10" "pillow>=10.0" "onnxruntime>=1.18"

COPY src ./src
# chain.json + model.int8.onnx per category; weights and fp32 graphs are
# excluded in .dockerignore.
COPY bundles ./bundles

EXPOSE 8080
CMD ["sh", "-c", "uvicorn vinspect.serve.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
