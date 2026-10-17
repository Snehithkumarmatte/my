# streamlit-ui: slim base, Streamlit + HTTP client + PostgreSQL driver only
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# curl is used only by the compose healthcheck; apt lists removed in the same layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# dependencies first, so editing app.py does not reinstall them
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY --chown=appuser:appuser .streamlit/ .streamlit/
COPY --chown=appuser:appuser app.py ./

USER appuser
EXPOSE 8501
# must bind 0.0.0.0, otherwise it is unreachable from the host
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
