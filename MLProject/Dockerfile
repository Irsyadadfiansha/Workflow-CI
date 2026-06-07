FROM python:3.12.7-slim

WORKDIR /app

COPY MLProject/ /app/

RUN pip install --upgrade pip && \
    pip install mlflow

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "model", "--port", "5000"]