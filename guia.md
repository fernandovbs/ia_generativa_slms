# 📘 Guia Técnico e Arquitetural: Deploy de Small Language Models (SLMs)
### Fundamentos de Otimização, Formatos de Precisão e Engenharia de Inferência (Cloud, Local e Mobile)

Este documento reúne os conceitos teóricos, matemáticos e de engenharia de software utilizados no projeto de deploy do modelo **`Qwen2.5-0.5B-Instruct`** (e sua variante **`Qwen2.5-1.5B-Instruct`**). O objetivo deste guia é fornecer uma referência completa e independente para desenvolvedores, pesquisadores e engenheiros que queiram entender e reproduzir as três arquiteturas de inferência implementadas no repositório:

1. **☁️ Nuvem / Microsserviço de Produção (Demo 1):** ONNX Runtime (`CPUExecutionProvider`) + FastAPI + Docker.
2. **💻 Edge Local / Desktop (Demo 2):** Quantização GGUF (`Q8_0`) via `llama.cpp` + Ollama + Open WebUI.
3. **📱 Borda Extrema / Smartphone Android (Demo 3):** Compilação nativa via MLC LLM e Apache TVM (`q4f16_1` + OpenCL) + App Nativo (APK) com provisionamento offline.

---

## 📌 Parte 1: Precisões Numéricas e Técnicas de Quantização
*(Modelos abordados: `FP32`, `BF16`, `Q8_0` e `q4f16_1`)*

Uma arquitetura Transformer como o `Qwen2.5-0.5B-Instruct` possui aproximadamente **490 milhões de parâmetros (pesos)** — valores reais distribuídos em matrizes lineares, projeções de atenção e camadas de feed-forward. O consumo de memória RAM/VRAM e o espaço em disco dependem diretamente do número de bits dedicados à representação de cada peso:

$$\text{Tamanho dos Pesos em Memória} \approx \text{Número de Parâmetros} \times \text{Bytes por Parâmetro}$$

### 1. `FP32` (*Floating Point 32-bit* / Precisão Simples) — **~1890 MB** (Demo 1 - ONNX)
- **Definição:** Padrão clássico de ponto flutuante do padrão IEEE 754 (1 bit de sinal, 8 bits de expoente e 23 bits de mantissa = **32 bits = 4 bytes por parâmetro**).
- **Cálculo de escala:** $490 \times 10^6 \times 4\text{ bytes} \approx 1,96\text{ GB}$ (resultando em um arquivo `model.onnx_data` de $\approx 1889\text{ MB}$).
- **Aplicação no projeto:** Utilizado na exportação ONNX para execução em **CPU de servidores x86/ARM (`CPUExecutionProvider`)**. Na CPU, operações vetoriais em `FP32` são suportadas nativamente por conjuntos de instruções como AVX2 e AVX-512, garantindo precisão numérica total e compatibilidade universal sem necessidade de aceleração por GPU.

### 2. `BF16` (*Brain Floating Point 16-bit*) — **~942 MB** (Pesos Originais Hugging Face)
- **Definição:** Formato de ponto flutuante de **16 bits (2 bytes por parâmetro)** criado pelo Google Brain especificamente para Deep Learning.
- **Diferença entre `FP16` e `BF16`:**
  - O `FP16` tradicional reduz o expoente para 5 bits e a mantissa para 10 bits. Isso diminui o alcance dinâmico, facilitando problemas de *underflow/overflow* (valores muito pequenos viram zero e valores altos viram infinito).
  - O **`BF16` mantém exatamente os mesmos 8 bits de expoente do `FP32`** (preservando o alcance dinâmico de magnitudes), reduzindo apenas a precisão das casas decimais da mantissa (de 23 para 7 bits).
- **Cálculo de escala:** Com 2 bytes por parâmetro, o arquivo original `model.safetensors` ocupa exatamente a metade do tamanho em relação ao FP32: **~942,3 MB**.

### 3. `Q8_0` (*Quantização Inteira Simétrica de 8 bits por Bloco*) — **~506 MB** (Demo 2 - GGUF / Ollama)
- **Definição:** Mapeamento de matrizes de ponto flutuante em **inteiros de 8 bits com sinal (`int8`, faixa de `-127` a `+127` = 1 byte por peso)**.
- **O significado de "quantização em bloco" (`_0`):**
  - Se um único fator de escala fosse aplicado à matriz inteira, a presença de tensores discrepantes (*outliers*) degradaria a representação dos demais pesos.
  - Para evitar essa distorção, o `llama.cpp` subdivide as matrizes em **pequenos blocos contíguos de 32 números**. Para cada bloco de 32 inteiros `int8`, armazena-se **1 fator de escala (`delta`) em `FP16`**.
  - Durante o cálculo, a aproximação é dada por: $\text{Peso Real} \approx \text{int8} \times \text{delta}$. O sufixo `_0` indica quantização simétrica em torno do zero (sem offset adicional).
- **Impacto prático:** Redução do consumo de disco e RAM para **~506 MB** (-73% em relação ao `FP32`) com degradação de perplexidade praticamente imperceptível.

### 4. `q4f16_1` (*Quantização em Grupo de 4 bits com Ativações FP16*) — **~265 MB** (Demo 3 - Android MLC)
- **Significado da nomenclatura:**
  - **`q4`**: Os pesos da rede são quantizados em inteiros de **apenas 4 bits** (`int4`, permitindo 16 valores possíveis por peso = **0,5 byte por parâmetro**).
  - **`f16`**: O fator de escala de cada grupo e os cálculos dinâmicos das ativações do Transformer rodam em ponto flutuante de 16 bits (`float16`).
  - **`_1`**: Formato de empacotamento otimizado do MLC LLM, onde **8 pesos de 4 bits são agrupados dentro de um único inteiro de 32 bits (`uint32`)**.
- **Relevância em dispositivos móveis:** Em smartphones, a capacidade de cálculo bruto (*FLOPS*) das GPUs mobile é frequentemente limitada pela **largura de banda da memória RAM compartilhada (LPDDR)**. Carregar 265 MB de pesos da memória para os registradores da GPU é **7 vezes mais rápido** do que carregar 1890 MB, resultando em menor consumo energético e taxas de geração fluidas.

---

## 📌 Parte 2: Arquitetura de Atenção GQA (*Grouped-Query Attention*)

Nos primeiros modelos Transformer (*Multi-Head Attention* — MHA), para cada cabeça de consulta (*Query* - $Q$), existia um par correspondente de cabeça de chave (*Key* - $K$) e cabeça de valor (*Value* - $V$). Em modelos generativos longos, armazenar todas as chaves e valores na memória durante o diálogo (*KV-Cache*) tornava-se o principal vilão de consumo de RAM.

O `Qwen2.5-0.5B` implementa **Grouped-Query Attention (GQA)**:
* **Cabeças de Consulta ($Q$):** Mantém **14 cabeças**, garantindo expressividade e capacidade de atenção refinada.
* **Cabeças de Chave e Valor ($KV$):** Agrupa as 14 consultas para compartilharem **apenas 2 cabeças de $K$ e 2 cabeças de $V$**.

```
[Q0, Q1, Q2, Q3, Q4, Q5, Q6] ──────► Compartilham KV_Head_0
[Q7, Q8, Q9, Q10, Q11, Q12, Q13] ───► Compartilham KV_Head_1
```

* **Vantagem arquitetural:** O tamanho do KV-Cache é reduzido em **7 vezes** ($14 / 2 = 7$), tornando a retenção de contexto viável em servidores com CPUs modestas e em dispositivos com memória restrita.

---

## 📌 Parte 3: Arquitetura da Demo 1 (Nuvem: ONNX Runtime + FastAPI)

### 1. Separação entre o Grafo `.onnx` e o Tokenizer Rust
Um pipeline de inferência de LLM em produção é desacoplado em três etapas:
```
[Texto] ──► (1. Tokenizer BPE em Rust) ──► [IDs Numéricos] ──► (2. Grafo ONNX C++) ──► [Logits] ──► (3. Detokenizer) ──► [Texto]
```
1. **Pré-processamento (Tokenizer):** Algoritmo determinístico baseado em dicionário (*Byte-Level BPE*) e aplicação do template ChatML (`<|im_start|>user...`). Não envolve operações tensoriais pesadas. A biblioteca `tokenizers` da Hugging Face é implementada em Rust nativo e não depende do PyTorch.
2. **Grafo Computacional (`model.onnx`):** Função matemática pura e compilada em C++ que recebe vetores inteiros (`input_ids`) e gera as probabilidades do vocabulário (`logits`).
3. **Pós-processamento (Detokenizer):** Mapeia os índices de maior probabilidade de volta para caracteres textuais.

Essa separação permite que a imagem Docker de produção não precise carregar o framework PyTorch (economizando de 2 a 4 GB no container), operando exclusivamente com `onnxruntime` e `numpy`.

### 2. Ciclo de Inferência: Fase *Prefill* vs. Fase *Decode*
A inferência de um modelo de linguagem ocorre em duas fases com características computacionais distintas:
* **Fase A — *Prefill* (Processamento do Prompt):**
  - O prompt completo (por exemplo, 50 tokens) é conhecido antecipadamente.
  - O modelo processa todos os 50 tokens em paralelo em uma única execução matricial contígua (*MatMul* $M \times K \times N$), gerando as ativações de atenção de todas as camadas e prevendo o primeiro token de resposta.
  - Comportamento: **Compute-Bound** (paralelizável nas threads da CPU).
* **Fase B — *Decode* (Geração Autorregressiva):**
  - Para prever o segundo token, o modelo depende do primeiro; para o terceiro, depende do segundo.
  - A execução passa a ser sequencial: a cada passo alimenta-se **apenas 1 token novo** (`shape=[1, 1]`).
  - Comportamento: **Memory-Bound** (o gargalo passa a ser a velocidade de leitura dos pesos e do histórico na memória).

### 3. Entradas e Saídas Explícitas de KV-Cache
Como um grafo ONNX exportado é estático e **sem estado interno (*stateless*)**, a memória acumulada dos tokens passados precisa ser fornecida e devolvida explicitamente a cada passo:
* **Entradas (`past_key_values.X.key/value`):** 48 tensores (24 camadas $\times$ 2 tensores) com o histórico de chaves e valores. No passo inicial (*prefill*), passam-se tensores vazios (`shape=[1, 2, 0, 64]`).
* **Saídas (`present.X.key/value`):** 48 tensores atualizados pelo grafo com as informações do token recém-processado.
* **Loop Autorregressivo:** A aplicação Python apenas redireciona `past_kv = present_kv`. Desta forma, o custo de geração de cada novo token permanece $O(1)$ em vez de recalcular todo o passado em $O(N^2)$.

### 4. Fusão de Operadores (*Operator Fusion*)
Em frameworks interpretados, uma camada de normalização e ativação tradicional executa múltiplas leituras e escritas consecutivas na memória RAM (`Linear` $\to$ RAM $\to$ `BiasAdd` $\to$ RAM $\to$ `SiLU` $\to$ RAM).
Ao habilitar `GraphOptimizationLevel.ORT_ENABLE_ALL` no ONNX Runtime:
* Nós adjacentes são fundidos em um único kernel de baixo nível (ex.: `FusedMatMulAddSiLU`).
* Os dados intermediários permanecem nos registradores rápidos do processador, minimizando a latência de acesso à RAM.

---

## 📌 Parte 4: Arquitetura da Demo 2 (Edge Local: GGUF + llama.cpp + Ollama)

### 1. Mapeamento de Memória via `mmap` (*Memory-Mapped File*)
* Tradicionalmente, o carregamento de modelos envolve ler os bytes do disco via `fread()` e copiá-los para novos buffers alocados na RAM.
* O formato **`.gguf`** armazena os tensores alinhados em páginas de memória diretamente compatíveis com as estruturas de hardware.
* O motor `llama.cpp` utiliza a chamada de sistema **`mmap()`**: os dados do arquivo no SSD são mapeados diretamente no espaço de endereçamento virtual do processo.
* **Benefícios:**
  - Inicialização quase instantânea (*Zero-Copy*).
  - O sistema operacional gerencia as páginas de memória em cache de forma transparente, permitindo que processos compartilhem recursos sem duplicação de dados.

### 2. Aceleração Vetorial SIMD (AVX2 e ARM NEON)
* **SIMD** (*Single Instruction, Multiple Data*): capacidade do processador de executar a mesma operação matemática sobre múltiplos dados simultaneamente.
* Em modelos quantizados (`Q8_0`), instruções como **AVX2** (x86) ou **NEON** (ARM) carregam blocos de números de 8 bits em registradores largos (ex.: 256 bits) e realizam produtos escalares paralelizados em um único ciclo de clock, permitindo altas taxas de geração mesmo sem GPU dedicada.

### 3. Orquestração em Containers Docker Isolados
Na Demo 2, o ambiente é desacoplado em dois containers integrados por rede virtual interna:
* **`slm-ollama` (Backend):** Executa o daemon do Ollama na porta do host `11435` (evitando conflito caso o host já possua uma instância padrão na porta `11434`). Carrega o modelo definido via [`Modelfile`](file:///c:/Users/Pichau/projects/ia_generativa/demo2_local_gguf/Modelfile) (`FROM /models/qwen2.5-0.5b-instruct.gguf`).
* **`slm-open-webui` (Frontend):** Interface de usuário moderna exposta em `http://localhost:3000`, conectada diretamente ao container do Ollama via `http://ollama:11434`.
* **Configuração de Contexto no `Modelfile`:** Parâmetros como `PARAMETER num_ctx 16384` garantem que o modelo suporte histórico longo de mensagens e payloads gerados pela interface sem truncamento de respostas.

---

## 📌 Parte 5: Arquitetura da Demo 3 (Mobile: MLC LLM + Apache TVM)

### 1. Compilação de Redes Neurais vs. Motores Interpretados
* Enquanto o ONNX Runtime e o `llama.cpp` são executáveis genéricos que interpretam arquivos de modelo, o **MLC LLM (baseado no Apache TVM Relax)** atua como um **compilador de ponta a ponta**.
* O compilador analisa o grafo estrutural do modelo, gera código nativo em C++ para a CPU (`aarch64-linux-android`) e sintetiza shaders customizados em **OpenCL** para a GPU mobile (Qualcomm Adreno ou ARM Mali), empacotando o runtime na biblioteca nativa `libtvm4j_runtime_packed.so`.

### 2. Otimização de Memória e *Chunked Prefill*
Em ambientes mobile, o sistema operacional Android conta com o *Low Memory Killer* (LMK), que finaliza processos que demandam picos excessivos de alocação de memória.
* Modelos configurados para servidores costumam pré-alocar buffers para janelas de até 32K tokens, demandando mais de 5 GB de memória temporária.
* Na compilação mobile:
  - Limita-se a janela de contexto para valores adequados ao uso em celular (`context_window_size = 4096`).
  - Aplica-se **Chunked Prefill** (`prefill_chunk_size = 512`): prompts longos são fatiados e computados em blocos de 512 tokens.
* **Resultado:** O consumo de buffer temporário diminui drasticamente, permitindo que a alocação total (pesos + buffers + KV-Cache) fique em torno de **~628 MB**, viabilizando a execução em smartphones convencionais.

### 3. Ponte JNI e Provisionamento Offline
* **Comunicação Kotlin $\leftrightarrow$ C++:** A interface visual do app (`MLCChat`, desenvolvida em Kotlin com Jetpack Compose) delega as chamadas de inferência ao motor C++ (`libtvm4j_runtime_packed.so`) através da camada JNI (*Java Native Interface*).
* **Compatibilidade com *Scoped Storage* (Android 14/15):** Arquivos transferidos via `adb push` pertencem originalmente ao usuário de sistema `shell` (`UID 2000`), impedindo a leitura direta pelo aplicativo. O provisionamento é realizado copiando os arquivos com o utilitário nativo `run-as ai.mlc.mlcchat` para a pasta privada do aplicativo (`application.filesDir`), permitindo funcionamento 100% offline (inclusive em Modo Avião).

---

## 📌 Parte 6: ONNX Runtime Mobile vs. MLC LLM

| Dimensão | ONNX Runtime Mobile (`onnxruntime-android`) | MLC LLM + Apache TVM |
| :--- | :--- | :--- |
| **Abordagem** | Runtime genérico portável | Compilação especializada ponta a ponta |
| **Aceleração de Hardware** | NNAPI do Android ou Qualcomm QNN (NPU Hexagon) | Kernels OpenCL/Vulkan sob medida para GPU Mobile |
| **Padronização de Artefatos** | Permite usar o mesmo formato `.onnx` da nuvem ao celular | Exige compilação dedicada gerando bibliotecas de modelo |
| **Cenário de Escolha** | Equipes que buscam pipeline único de deploy multiplataforma | Casos onde se busca a máxima taxa de decodificação na GPU mobile |

---

## 🛠️ Guia de Reprodução Rápida das Demos

Para reproduzir as três demonstrações de ponta a ponta a partir do notebook [`apresentacao_slms_deploy.ipynb`](file:///c:/Users/Pichau/projects/ia_generativa/apresentacao_slms_deploy.ipynb):

1. **Validação de Ambiente (Bloco 0):** Execute a verificação inicial para confirmar o interpretador Python da `.venv`, a disponibilidade do Docker e as versões das bibliotecas instaladas.
2. **Demo 1 — Nuvem (Passos 1.1 a 1.6):**
   - Execute a exportação ONNX e verifique a geração dos arquivos `model.onnx` e `model.onnx_data`.
   - Suba o container Docker (`docker compose up -d --build`) e aguarde o healthcheck em `http://localhost:8000/health`.
   - Explore o Swagger UI em `http://localhost:8000/docs` e valide a inferência via `POST /generate`.
3. **Demo 2 — Local (Passos 2.1 a 2.5):**
   - Execute o download dos pesos e a quantização para `Q8_0` via `convert_hf_to_gguf.py`.
   - Suba os containers do Ollama e Open WebUI (`docker compose up -d`) e registre o modelo via `ollama create`.
   - Acesse a interface web em `http://localhost:3000` e teste a API REST na porta `11435`.
4. **Demo 3 — Android (Passos 3.1 a 3.4):**
   - Gere o `mlc-chat-config.json` e execute a compilação TVM (`mlc_llm compile --device android`).
   - Compile o APK nativo com o Gradle Wrapper (`gradlew.bat assembleDebug`).
   - Conecte o smartphone Android com depuração USB ativada e execute o provisionamento offline para validar o chat local no aparelho.
