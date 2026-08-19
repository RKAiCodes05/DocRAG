FROM python:3.12-slim

# ============================================================
# Environment
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# Working directory
# ============================================================

WORKDIR /app

# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Application
# ============================================================

COPY api ./api
COPY generation ./generation
COPY ingestion ./ingestion
COPY retrieval ./retrieval
COPY rag ./rag

# ============================================================
# Runtime
# ============================================================

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]