FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 1) OS packages - Tesseract with the dataset's languages (eng/spa/deu) + Poppler.
#    apt lists removed in the same layer to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa tesseract-ocr-deu \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# unprivileged user; UID matches the VM user so the bind-mounted folders are writable
ARG APP_UID=1000
RUN useradd --create-home --uid ${APP_UID} appuser

WORKDIR /app

# 2) Python dependencies - only reinstalled when requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) application code - lives at /app/backend so "from backend.xxx import ..." works
COPY --chown=appuser:appuser . /app/backend/

USER appuser
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
