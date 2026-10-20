import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

# NumPy é a espinha dorsal de dados: representa tensores contíguos em memória C para o ONNX Runtime sem overhead
import numpy as np

# ONNX Runtime (ORT): motor de inferência compilado em C++ de alta performance (Zero dependência de PyTorch!)
import onnxruntime as ort

# FastAPI: framework web assíncrono para expor o modelo como microsserviço RESTful de baixa latência
from fastapi import FastAPI, HTTPException

# Pydantic: validação automática de esquemas de entrada e saída com geração de documentação Swagger/OpenAPI
from pydantic import BaseModel, Field

# AutoTokenizer do Hugging Face: implementado em Rust, converte texto <-> IDs de tokens sem carregar o PyTorch
from transformers import AutoTokenizer

# Caminho do diretório contendo model.onnx, config.json e arquivos do tokenizer
MODEL_DIR = Path(os.getenv("ONNX_MODEL_DIR", "/app/model_onnx"))

# Provedor de execução do ONNX Runtime (padrão CPU; pode ser chaveado para CUDAExecutionProvider sem alterar o código)
PROVIDER = os.getenv("ORT_PROVIDER", "CPUExecutionProvider")

# Número de threads dedicadas para paralelismo intra-operador (multiplicação de matrizes com SIMD AVX2/AVX-512)
NUM_THREADS = int(os.getenv("ORT_NUM_THREADS", "4"))

# Dicionário de estado global da aplicação: mantém a sessão e o tokenizer vivos durante todo o ciclo do container
state: Dict[str, object] = {
    "ort_session": None,
    "tokenizer": None,
    "config": None,
    "onnx_file": None,
}


class InputData(BaseModel):
    """Esquema de entrada da requisição POST /generate."""
    prompt: str = Field(
        ...,
        description="Pergunta ou instrução enviada para o Small Language Model",
        examples=["Explique em 3 frases o que é um Small Language Model (SLM) e suas vantagens no deploy."],
    )
    system_prompt: str = Field(
        default="Você é um assistente especialista em Inteligência Artificial Generativa. Responda de forma clara, concisa e em português.",
        description="Instrução de sistema para guiar o tom e formato da resposta",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Limite máximo de novos tokens gerados no loop autorregressivo",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Temperatura de amostragem (0.0 = determinístico/greedy, valores maiores = mais criativo)",
    )


class OutputData(BaseModel):
    """Esquema de resposta da requisição POST /generate com métricas de telemetria."""
    generated_text: str = Field(..., description="Texto gerado pelo modelo ONNX decodificado")
    model_dir: str = Field(..., description="Caminho do modelo ONNX carregado no container")
    execution_provider: str = Field(..., description="Provedor de execução ativo no ONNX Runtime (ex: CPUExecutionProvider)")
    new_tokens: int = Field(..., description="Quantidade total de tokens gerados")
    inference_time_ms: float = Field(..., description="Tempo total de inferência decorrido em milissegundos")
    tokens_per_second: float = Field(..., description="Velocidade média de geração em tokens por segundo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciador de Ciclo de Vida (Lifespan) do FastAPI:
    Executa exatamente UMA VEZ durante o startup do servidor Uvicorn (Cold-Start Único).
    Carrega o grafo ONNX compilado em memória para que as requisições HTTP tenham resposta instantânea.
    """
    # 1. Localiza o arquivo .onnx exportado no diretório montado
    onnx_file = MODEL_DIR / "model.onnx"
    if not onnx_file.exists():
        onnx_candidates = list(MODEL_DIR.glob("*.onnx"))
        if not onnx_candidates:
            raise RuntimeError(
                f"Nenhum arquivo .onnx encontrado em {MODEL_DIR.resolve()}. "
                "Execute o passo de exportação no Jupyter Notebook antes de subir o container."
            )
        onnx_file = onnx_candidates[0]

    # 2. Lê as configurações arquiteturais do modelo (número de camadas, cabeças de atenção e dimensão dos heads)
    config_path = MODEL_DIR / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    print(f"[ONNX Runtime] Inicializando InferenceSession pura a partir de: {onnx_file}")
    
    # 3. Configura as opções da sessão C++ do ONNX Runtime
    sess_options = ort.SessionOptions()
    
    # ORT_ENABLE_ALL: ativa fusão de operadores (ex: LayerNorm, GeLU, MatMul+Add) e eliminação de constantes redundantes
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Define o número de threads da CPU para processar as multiplicações matriciais com instruções SIMD
    sess_options.intra_op_num_threads = NUM_THREADS

    # 4. Inicializa o motor C++ InferenceSession (carrega os 1.89 GB em memória sem qualquer dependência de PyTorch!)
    ort_session = ort.InferenceSession(
        str(onnx_file),
        sess_options=sess_options,
        providers=[PROVIDER],
    )
    
    # 5. Carrega o Tokenizer nativo em Rust (Hugging Face tokenizers)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    # 6. Salva as instâncias no estado compartilhado da aplicação
    state["ort_session"] = ort_session
    state["tokenizer"] = tokenizer
    state["config"] = model_config
    state["onnx_file"] = str(onnx_file)
    print(f"[ONNX Runtime] Modelo carregado com sucesso! Providers ativos: {ort_session.get_providers()}")
    
    # Libera a execução para a API receber requisições
    yield
    
    # Limpa as referências de memória quando a API for encerrada
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
    """Endpoint de sondagem (Healthcheck): verifica se o modelo está pronto para receber tráfego."""
    ort_session: Optional[ort.InferenceSession] = state.get("ort_session")  # type: ignore
    return {
        "status": "ok" if ort_session is not None else "uninitialized",
        "onnx_runtime_version": ort.__version__,
        "active_providers": ort_session.get_providers() if ort_session else [],
        "onnx_file": state.get("onnx_file"),
    }


@app.get("/model-info", tags=["Inspeção ONNX"])
async def model_info():
    """Inspeciona o grafo ONNX carregado: lista as 51 entradas (inputs + KV-cache) e 49 saídas."""
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
    """
    Amostragem de Próximo Token implementada 100% em NumPy (sem PyTorch):
    - Se temperature <= 0: Greedy Decoding (escolhe o índice com maior valor via np.argmax)
    - Se temperature > 0: Softmax com estabilidade numérica e amostragem multinomial ponderada
    """
    # 1. Greedy Search determinístico: escolhe imediatamente o token mais provável
    if temperature <= 1e-5:
        return int(np.argmax(logits))
    
    # 2. Escala os logits brutos pela temperatura desejada (valores menores deixam a curva mais pontiaguda)
    scaled = logits.astype(np.float64) / float(temperature)
    
    # 3. Estabilidade Numérica: subtrai o valor máximo antes da exponenciação para evitar estouro (Overflow/Inf)
    scaled -= np.max(scaled)
    
    # 4. Calcula o Softmax: probabilidades entre 0.0 e 1.0 cuja soma é exatamente 1.0
    probs = np.exp(scaled)
    probs /= np.sum(probs)
    
    # 5. Sorteia um índice do vocabulário baseado na distribuição de probabilidades calculada
    return int(np.random.choice(len(probs), p=probs))


@app.post("/generate", response_model=OutputData, tags=["Inferência"])
async def generate(data: InputData) -> OutputData:
    """
    Endpoint principal de inferência autorregressiva:
    Executa a geração token a token mantendo o KV-Cache em NumPy e alimentando a InferenceSession do ONNX Runtime.
    """
    # Recupera os componentes carregados no lifespan
    ort_session: Optional[ort.InferenceSession] = state.get("ort_session")  # type: ignore
    tokenizer = state.get("tokenizer")
    config: Optional[Dict[str, object]] = state.get("config")  # type: ignore

    if ort_session is None or tokenizer is None or config is None:
        raise HTTPException(status_code=503, detail="Modelo ONNX não carregado.")

    # --------------------------------------------------------------------------
    # ETAPA 1: Formatação do ChatML e Tokenização Direta para NumPy
    # --------------------------------------------------------------------------
    messages = [
        {"role": "system", "content": data.system_prompt},
        {"role": "user", "content": data.prompt},
    ]

    # Aplica o Chat Template oficial do Qwen (<|im_start|>system... <|im_start|>user...)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # O tokenizer converte o texto em IDs inteiros e retorna diretamente um array NumPy int64 (sem PyTorch!)
    encoded = tokenizer(prompt_text, return_tensors="np")
    input_ids: np.ndarray = encoded["input_ids"].astype(np.int64)
    seq_len = input_ids.shape[1]

    # --------------------------------------------------------------------------
    # ETAPA 2: Extração das Dimensões Arquiteturais do KV-Cache
    # --------------------------------------------------------------------------
    # Qwen2.5-0.5B: 24 camadas, 2 KV heads (Grouped-Query Attention), head_dim = 896 / 14 = 64
    num_layers = int(config["num_hidden_layers"])
    num_kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])

    # Descobre se o grafo ONNX foi exportado esperando KV-Cache em float32 ou float16
    kv_ort_type = "tensor(float)"
    input_names = set()
    for inp in ort_session.get_inputs():
        input_names.add(inp.name)
        if inp.name.startswith("past_key_values"):
            kv_ort_type = inp.type
    np_kv_dtype = np.float16 if "float16" in kv_ort_type else np.float32

    # --------------------------------------------------------------------------
    # ETAPA 3: Inicialização do KV-Cache Explícito Vazio (Passo de Prefill)
    # --------------------------------------------------------------------------
    # Dimensão [batch_size=1, num_kv_heads=2, past_sequence_length=0, head_dim=64]
    empty_kv = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np_kv_dtype)
    past_kv: Dict[str, np.ndarray] = {}
    for layer_idx in range(num_layers):
        past_kv[f"past_key_values.{layer_idx}.key"] = empty_kv
        past_kv[f"past_key_values.{layer_idx}.value"] = empty_kv

    # Nomes dos tensores que a sessão ONNX deve calcular e retornar (logits + 48 tensores present.X)
    output_names = [out.name for out in ort_session.get_outputs()]
    
    # Identifica os IDs dos tokens de parada (<|im_end|> = 151645 e <|endoftext|> = 151643)
    eos_token_ids = {tokenizer.eos_token_id}
    if isinstance(config.get("eos_token_id"), list):
        eos_token_ids.update(config["eos_token_id"])
    elif config.get("eos_token_id") is not None:
        eos_token_ids.add(int(config["eos_token_id"]))

    generated_tokens: List[int] = []
    cur_input_ids = input_ids  # No primeiro passo, passamos todos os tokens do prompt inicial
    total_len = seq_len

    # --------------------------------------------------------------------------
    # ETAPA 4: Loop Autorregressivo de Decodificação (Token a Token)
    # --------------------------------------------------------------------------
    start_time = time.perf_counter()
    for step in range(data.max_new_tokens):
        # 1. Monta o feed dictionary com arrays NumPy contíguos esperados pelo ONNX Runtime
        feed: Dict[str, np.ndarray] = {
            "input_ids": cur_input_ids,
            "attention_mask": np.ones((1, total_len), dtype=np.int64),
            **past_kv,  # Desempacota as 48 chaves/valores de cache do histórico
        }
        
        # Adiciona os position_ids (necessários para o cálculo do RoPE em grafos ONNX modernos)
        if "position_ids" in input_names:
            if step == 0:
                feed["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
            else:
                feed["position_ids"] = np.array([[total_len - 1]], dtype=np.int64)

        # 2. Executa a inferência de alta performance em C++ no ONNX Runtime
        ort_outputs = ort_session.run(output_names, feed)
        
        # 3. Fatia os logits correspondentes ao último token da sequência (shape: [151936])
        logits = ort_outputs[0][0, -1, :]

        # 4. Amostra o próximo token usando a função NumPy vetorizada
        next_token = _sample_next_token(logits, data.temperature)
        generated_tokens.append(next_token)

        # 5. Condição de parada: encerra imediatamente se o modelo gerou o token de finalização
        if next_token in eos_token_ids:
            break

        # 6. Atualiza o KV-Cache: os tensores 'present.X' retornados viram o 'past_key_values.X' da próxima rodada
        for idx, out_name in enumerate(output_names[1:], start=1):
            if out_name.startswith("present."):
                past_name = out_name.replace("present.", "past_key_values.", 1)
                past_kv[past_name] = ort_outputs[idx]

        # 7. Prepara para a próxima iteração: envia APENAS o último token gerado (shape: [1, 1])!
        # Graças ao KV-cache, a complexidade não é O(N^2), mas O(N) por token gerado.
        cur_input_ids = np.array([[next_token]], dtype=np.int64)
        total_len += 1

    # --------------------------------------------------------------------------
    # ETAPA 5: Pós-processamento e Cálculo de Throughput
    # --------------------------------------------------------------------------
    elapsed_s = time.perf_counter() - start_time
    num_new_tokens = len(generated_tokens)
    
    # Decodifica a lista de inteiros de volta para texto em português
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return OutputData(
        generated_text=response_text,
        model_dir=str(MODEL_DIR),
        execution_provider=ort_session.get_providers()[0],
        new_tokens=num_new_tokens,
        inference_time_ms=round(elapsed_s * 1000, 2),
        tokens_per_second=round(num_new_tokens / elapsed_s, 2) if elapsed_s > 0 else 0.0,
    )
