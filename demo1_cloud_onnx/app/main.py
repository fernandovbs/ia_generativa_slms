import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# Diretório onde o modelo ONNX exportado pelo notebook está localizado
MODEL_DIR = Path(os.getenv("ONNX_MODEL_DIR", "/app/model_onnx"))
PROVIDER = os.getenv("ORT_PROVIDER", "CPUExecutionProvider")
NUM_THREADS = int(os.getenv("ORT_NUM_THREADS", "4"))

# Variáveis globais de estado do modelo carregado (carregado apenas 1 vez!)
state: Dict[str, object] = {
    "ort_session": None,
    "tokenizer": None,
    "config": None,
    "onnx_file": None,
}


class InputData(BaseModel):
    prompt: str = Field(
        ...,
        description="Pergunta ou instrução enviada para o Small Language Model",
        examples=["Explique em 3 frases o que é um Small Language Model (SLM) e suas vantagens no deploy."],
    )
    system_prompt: str = Field(
        default="Você é um assistente especialista em Inteligência Artificial Generativa. Responda de forma clara, concisa e em português.",
        description="Instrução de sistema para guiar o comportamento do modelo",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Número máximo de novos tokens gerados",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Temperatura de amostragem (0.0 = determinístico/greedy)",
    )


class OutputData(BaseModel):
    generated_text: str = Field(..., description="Resposta gerada pelo modelo ONNX")
    model_dir: str = Field(..., description="Caminho do modelo ONNX carregado")
    execution_provider: str = Field(..., description="Provedor de execução do ONNX Runtime")
    new_tokens: int = Field(..., description="Quantidade de novos tokens gerados")
    inference_time_ms: float = Field(..., description="Tempo total de inferência em milissegundos")
    tokens_per_second: float = Field(..., description="Velocidade de geração em tokens/segundo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa a sessão do ONNX Runtime (sem PyTorch!) no carregamento da API."""
    onnx_file = MODEL_DIR / "model.onnx"
    if not onnx_file.exists():
        onnx_candidates = list(MODEL_DIR.glob("*.onnx"))
        if not onnx_candidates:
            raise RuntimeError(
                f"Nenhum arquivo .onnx encontrado em {MODEL_DIR.resolve()}. "
                "Execute o passo de exportação no Jupyter Notebook antes de subir o container."
            )
        onnx_file = onnx_candidates[0]

    config_path = MODEL_DIR / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    print(f"[ONNX Runtime] Inicializando InferenceSession pura a partir de: {onnx_file}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = NUM_THREADS

    # Carrega o grafo ONNX apenas UMA vez em memória (sem depender do PyTorch!)
    ort_session = ort.InferenceSession(
        str(onnx_file),
        sess_options=sess_options,
        providers=[PROVIDER],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    state["ort_session"] = ort_session
    state["tokenizer"] = tokenizer
    state["config"] = model_config
    state["onnx_file"] = str(onnx_file)
    print(f"[ONNX Runtime] Modelo carregado com sucesso! Providers ativos: {ort_session.get_providers()}")
    yield
    state.clear()


app = FastAPI(
    title="Demo 1 - SLM Deploy com FastAPI + ONNX Runtime (Zero-PyTorch)",
    description=(
        "API REST para servir um Small Language Model (Qwen2.5-0.5B-Instruct) "
        "otimizado no formato ONNX dentro de um container Docker leve (apenas ONNX Runtime + NumPy)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Monitoramento"])
async def health_check():
    """Verifica se a API e a sessão do ONNX Runtime estão prontas."""
    ort_session: Optional[ort.InferenceSession] = state.get("ort_session")  # type: ignore
    return {
        "status": "ok" if ort_session is not None else "uninitialized",
        "onnx_runtime_version": ort.__version__,
        "active_providers": ort_session.get_providers() if ort_session else [],
        "onnx_file": state.get("onnx_file"),
    }


@app.get("/model-info", tags=["Inspeção ONNX"])
async def model_info():
    """Exibe metadados do grafo ONNX (tensores de entrada, saída e provedores de execução)."""
    ort_session: Optional[ort.InferenceSession] = state.get("ort_session")  # type: ignore
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Sessão ONNX não inicializada.")

    inputs: List[Dict[str, object]] = [
        {"name": inp.name, "shape": inp.shape, "type": inp.type}
        for inp in ort_session.get_inputs()
    ]
    outputs: List[Dict[str, object]] = [
        {"name": out.name, "shape": out.shape, "type": out.type}
        for out in ort_session.get_outputs()
    ]
    return {
        "onnx_file": state.get("onnx_file"),
        "providers": ort_session.get_providers(),
        "num_inputs": len(inputs),
        "inputs_preview": inputs[:8],
        "outputs_preview": outputs[:4],
    }


def _sample_next_token(logits: np.ndarray, temperature: float) -> int:
    """Seleciona o próximo token a partir dos logits usando Greedy (temp=0) ou Softmax com temperatura."""
    if temperature <= 1e-5:
        return int(np.argmax(logits))
    scaled = logits.astype(np.float64) / float(temperature)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


@app.post("/generate", response_model=OutputData, tags=["Inferência"])
async def generate(data: InputData) -> OutputData:
    """Executa inferência autorregressiva com KV-Cache diretamente na onnxruntime.InferenceSession."""
    ort_session: Optional[ort.InferenceSession] = state.get("ort_session")  # type: ignore
    tokenizer = state.get("tokenizer")
    config: Optional[Dict[str, object]] = state.get("config")  # type: ignore

    if ort_session is None or tokenizer is None or config is None:
        raise HTTPException(status_code=503, detail="Modelo ONNX não carregado.")

    messages = [
        {"role": "system", "content": data.system_prompt},
        {"role": "user", "content": data.prompt},
    ]

    # Aplica o template de chat oficial (ChatML) e tokeniza diretamente para arrays NumPy
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt_text, return_tensors="np")
    input_ids: np.ndarray = encoded["input_ids"].astype(np.int64)
    seq_len = input_ids.shape[1]

    # Dimensões do KV-Cache do modelo (ex: Qwen2.5-0.5B tem 24 camadas, 2 KV heads, head_dim=64)
    num_layers = int(config["num_hidden_layers"])
    num_kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])

    # Descobre o tipo de float esperado pelo KV-Cache no grafo ONNX (float32 ou float16)
    kv_ort_type = "tensor(float)"
    input_names = set()
    for inp in ort_session.get_inputs():
        input_names.add(inp.name)
        if inp.name.startswith("past_key_values"):
            kv_ort_type = inp.type
    np_kv_dtype = np.float16 if "float16" in kv_ort_type else np.float32

    # Inicializa o KV-Cache vazio (past_sequence_length = 0) para o passo de prefill
    empty_kv = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np_kv_dtype)
    past_kv: Dict[str, np.ndarray] = {}
    for layer_idx in range(num_layers):
        past_kv[f"past_key_values.{layer_idx}.key"] = empty_kv
        past_kv[f"past_key_values.{layer_idx}.value"] = empty_kv

    output_names = [out.name for out in ort_session.get_outputs()]
    eos_token_ids = {tokenizer.eos_token_id}
    if isinstance(config.get("eos_token_id"), list):
        eos_token_ids.update(config["eos_token_id"])
    elif config.get("eos_token_id") is not None:
        eos_token_ids.add(int(config["eos_token_id"]))

    generated_tokens: List[int] = []
    cur_input_ids = input_ids
    total_len = seq_len

    start_time = time.perf_counter()
    for step in range(data.max_new_tokens):
        feed: Dict[str, np.ndarray] = {
            "input_ids": cur_input_ids,
            "attention_mask": np.ones((1, total_len), dtype=np.int64),
            **past_kv,
        }
        if "position_ids" in input_names:
            if step == 0:
                feed["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
            else:
                feed["position_ids"] = np.array([[total_len - 1]], dtype=np.int64)

        ort_outputs = ort_session.run(output_names, feed)
        logits = ort_outputs[0][0, -1, :]

        next_token = _sample_next_token(logits, data.temperature)
        generated_tokens.append(next_token)

        if next_token in eos_token_ids:
            break

        # Atualiza o KV-Cache com os tensores 'present.X.key/value' retornados pelo ONNX Runtime
        for idx, out_name in enumerate(output_names[1:], start=1):
            if out_name.startswith("present."):
                past_name = out_name.replace("present.", "past_key_values.", 1)
                past_kv[past_name] = ort_outputs[idx]

        cur_input_ids = np.array([[next_token]], dtype=np.int64)
        total_len += 1

    elapsed_s = time.perf_counter() - start_time
    num_new_tokens = len(generated_tokens)
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return OutputData(
        generated_text=response_text,
        model_dir=str(MODEL_DIR),
        execution_provider=ort_session.get_providers()[0],
        new_tokens=num_new_tokens,
        inference_time_ms=round(elapsed_s * 1000, 2),
        tokens_per_second=round(num_new_tokens / elapsed_s, 2) if elapsed_s > 0 else 0.0,
    )
