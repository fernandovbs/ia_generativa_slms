FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TVM_NDK_CC=aarch64-linux-gnu-gcc

WORKDIR /workspace

# Instala toolchain de cross-compilação ARM64 (Android aarch64) e libs base (~50 MB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libx11-6 \
    libzstd1 \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

# 1. Instala PyTorch CPU-only (~180 MB usando --index-url para NÃO baixar pacotes CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Instala dependências leves do MLC LLM (com numpy<2 para compatibilidade com TVM DataType)
RUN pip install --no-cache-dir \
    "numpy<2" safetensors tqdm pydantic shortuuid typing_extensions requests \
    attrs cloudpickle psutil ml_dtypes decorator scipy tornado \
    fastapi uvicorn openai tiktoken prompt_toolkit pyyaml transformers sentencepiece protobuf pytest

# 3. Instala apache-tvm-ffi==0.1.11 e os binários CPU estáveis do Apache TVM e MLC LLM
RUN pip install --no-cache-dir "apache-tvm-ffi==0.1.11" \
    && pip install --no-cache-dir --no-deps --pre -f https://mlc.ai/wheels \
    mlc-ai-cpu \
    mlc-llm-cpu \
    && printf '#!/bin/sh\nexec python -m mlc_llm "$@"\n' > /usr/local/bin/mlc_llm \
    && chmod +x /usr/local/bin/mlc_llm

CMD ["mlc_llm", "--help"]
