FROM python:3.9-slim

# LightGBM needs the OpenMP runtime, which the slim image doesn't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install dependencies first so Docker can cache this layer
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what the API needs at runtime
COPY src/ src/
COPY app/ app/
COPY models/ models/

# Don't run as root
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
