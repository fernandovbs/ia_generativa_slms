FROM python:3.11-slim

# Evita geração de arquivos .pyc e garante logs em tempo real no Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ONNX_MODEL_DIR=/app/model_onnx \
    ORT_PROVIDER=CPUExecutionProvider

WORKDIR /app

# Instala dependências de sistema mínimas necessárias para o ONNX Runtime (libgomp1 para OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API FastAPI
COPY app/ ./app/

# Expõe a porta 8000 (Swagger UI ficará em http://localhost:8000/docs)
EXPOSE 8000

# Comando de inicialização com Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
