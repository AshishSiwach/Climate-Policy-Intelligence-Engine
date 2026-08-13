# CPIE — Streamlit chat UI container
#
# Build:  docker build -t cpie .
# Run:    docker compose up          (recommended — wires Postgres + Grafana)
#         docker run -p 8501:8501 \
#           -e OPENAI_API_KEY=sk-... \
#           -e POSTGRES_HOST=host.docker.internal \
#           -v $(pwd)/data:/app/data \
#           cpie
#
# Notes:
#   - Uses CPU-only torch (~500 MB) via requirements.txt override.
#     Local dev uses CUDA (RTX 4050) — container targets portability.
#   - data/raw/ and data/processed/ are NOT baked in. Mount them as volumes
#     or run ingestion inside the container after first boot.
#   - Secrets come from environment variables, never from the image.

FROM python:3.12-slim-bookworm

# System deps for PyMuPDF (libGL) and psycopg (libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — good practice; Streamlit doesn't need root
RUN useradd -m -u 1000 cpie

WORKDIR /app

# Install Python deps first (layer-cached until requirements.txt changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source — after deps so code changes don't bust the pip cache
COPY --chown=cpie:cpie . .

# Ensure log and data dirs exist with correct ownership
RUN mkdir -p logs data/raw data/processed && chown -R cpie:cpie logs data

USER cpie

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# --server.headless true: suppress the browser-open prompt in container
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
