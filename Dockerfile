# Multi-stage-lite build: keep it simple but production-sane.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker layer caching skips this on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + pre-trained model artifact
COPY app/ ./app/
COPY model/churn_model.joblib ./model/churn_model.joblib

EXPOSE 8000

# Basic container-level health check so orchestrators (Render/Railway/ECS/k8s)
# know when the app is actually ready to serve traffic
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
