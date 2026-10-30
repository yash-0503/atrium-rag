# Atrium — container image.
#
# The image installs from pyproject.toml, not requirements.txt: the
# distribution metadata is the source of truth for what "atrium" is,
# and requirements.txt is a dev convenience that had already drifted
# from it (pinned floors vs. ranges).
#
#   docker build -t atrium-rag .
#   docker build -t atrium-rag --build-arg EXTRAS="ocr,rerank" .
#
# The runtime image needs libmagic (python-magic's native library) and
# nothing else from the OS. Tesseract is deliberately NOT installed:
# it is a host-level binary, the `ocr` extra is opt-in, and shipping it
# would add ~30 MB to every image for a feature most deployments do
# not use. If you want OCR in the container, either use the
# `rapidocr` provider (pure pip) or build a derived image that adds
# `apt-get install tesseract-ocr`.

FROM python:3.12-slim AS base

# libmagic: python-magic's native library. Without it the MIME
# detector falls back to the extension table, which is fine but
# weaker.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Optional extras, e.g. --build-arg EXTRAS="ocr,rerank".
ARG EXTRAS=""

# Dependencies first: this layer is cached until pyproject.toml
# changes, so a code-only edit doesn't reinstall the world.
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY atrium ./atrium
RUN if [ -n "$EXTRAS" ]; then \
        pip install --no-cache-dir ".[$EXTRAS]"; \
    else \
        pip install --no-cache-dir .; \
    fi

# Non-root. The API writes one file (the SQLite job queue) and needs
# a writable HOME for model caches.
RUN useradd --create-home --uid 10001 atrium \
    && mkdir -p /data \
    && chown -R atrium:atrium /app /data
USER atrium

# The job queue lives here. Mount a volume to keep job history across
# container restarts, and to share the queue with a worker container.
ENV JOB_DB_PATH=/data/jobs.db \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    QDRANT_HOST=qdrant

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-m", "atrium.healthcheck"]

CMD ["uvicorn", "atrium.main:app", "--host", "0.0.0.0", "--port", "8000"]
