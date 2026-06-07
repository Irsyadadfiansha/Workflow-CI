FROM python:3.12.7-slim

WORKDIR /app

# Copy MLProject files
COPY MLProject/ /app/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose MLflow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000')" || exit 1

# Default command
CMD ["mlflow", "models", "serve", "-m", "model", "--port", "5000"]