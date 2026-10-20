## 📌 Parte 1: O que significam as Precisões Numéricas e Quantizações?
*(Presentes no Diagrama 1: `FP32`, `BF16`, `Q8_0`, `q4f16_1`)*

Uma rede neural como o `Qwen2.5-0.5B-Instruct` possui cerca de **490 milhões de parâmetros (pesos)** — que nada mais são do que números reais dentro de matrizes que multiplicam os vetores de entrada. O **tamanho do modelo no disco e na memória RAM** depende diretamente de **quantos bits usamos para guardar cada um desses 490 milhões de números**:

```
490 milhões de pesos × Bytes por peso ≈ Tamanho do Modelo
```

### 1. `FP32` (*Floating Point 32-bit* / Precisão Simples) — **~1890 MB** (Demo 1 - ONNX)
- **O que é:** É o formato padrão de ponto flutuante da computação científica (1 bit de sinal, 8 bits de expoente, 23 bits de mantissa/fração = **32 bits = 4 bytes por peso**).
- **Conta rápida:** $490 \times 10^6 \text{ pesos} \times 4\text{ bytes} \approx 1,96\text{ GB}$ ($\approx 1890\text{ MB}$).
- **Por que aparece na Demo 1?** Quando exportamos o modelo para `.onnx` visando rodar em **CPU de servidor (`CPUExecutionProvider`)**, o `FP32` é o formato nativo mais universal das unidades vetoriais (AVX2/AVX-512) de qualquer processador x86 sem depender de instruções especiais de placa de vídeo.

### 2. `BF16` (*Brain Floating Point 16-bit*) — **~942 MB** (Modelo Base Original na Hugging Face)
- **O que é:** Formato de **16 bits (2 bytes por peso)** criado pelo Google Brain especificamente para Deep Learning.
- **Qual a diferença entre `FP16` e `BF16`?**
  - O `FP16` tradicional reduz o expoente para 5 bits (o que causa *overflow/underflow*, isto é, números muito grandes viram `Inf` durante o treino).
  - O **`BF16` mantém os mesmos 8 bits de expoente do `FP32`** (mesmo alcance dinâmico!), cortando apenas as casas decimais da mantissa (de 23 para 7 bits).
- **Conta rápida:** Como usa 2 bytes por peso (metade do `FP32`), o arquivo `model.safetensors` original da Hugging Face tem exatamente a metade do tamanho: **~942,3 MB**.

### 3. `Q8_0` (*Quantização Inteira de 8 bits por Bloco*) — **~506 MB** (Demo 2 - GGUF / Ollama)
- **O que é:** Em vez de guardar números com vírgula (ponto flutuante de 16 ou 32 bits), arredondamos os pesos para **números inteiros de 8 bits com sinal (`int8`, que vão de `-127` a `+127` = 1 byte por peso)**.
- **O que significa o `_0` e "por bloco" no `Q8_0`?**
  - Se usássemos uma única escala para a matriz inteira, um único valor extremo (*outlier*) estragaria a precisão de todos os outros pesos.
  - Por isso, o `llama.cpp` divide os pesos em **pequenos blocos de 32 números consecutivos**. Para cada bloco de 32 números `int8`, ele guarda **1 único fator de escala (`delta` em `FP16`)**.
  - Na hora de calcular, o processador multiplica o inteiro pelo fator de escala: $\text{Peso Real} \approx \text{int8} \times \text{escala}$. (O sufixo `_0` significa quantização *simétrica*, sem ponto-zero/offset adicional).
- **Resultado prático:** Reduz o tamanho para **~506 MB** (73% menor que o `FP32`) com **perda de qualidade virtualmente zero**.

### 4. `q4f16_1` (*Quantização em Grupo de 4 bits com Ativações FP16*) — **~265 MB** (Demo 3 - Android MLC)
- **O que o nome `q4f16_1` significa letra por letra?**
  - **`q4`**: Os pesos da rede são comprimidos em inteiros de **apenas 4 bits** (`int4`, apenas 16 valores possíveis por peso = **0,5 byte por peso**!).
  - **`f16`**: O fator de escala de cada grupo de 32 pesos e as **ativações/cálculos em tempo de execução** na GPU do celular rodam em `float16` (`f16`).
  - **`_1`**: Esquema de quantização por grupo assimétrico/otimizado do MLC LLM onde **8 pesos de 4 bits são empacotados dentro de um único inteiro de 32 bits (`uint32`)**.
- **Por que fazemos isso no celular?** Em smartphones, o maior gargalo **não é a velocidade de cálculo da GPU, e sim a largura de banda da memória RAM** (ler os pesos da memória para o chip). Ler 265 MB (`q4f16_1`) da memória do celular é **7 vezes mais rápido** do que ler 1890 MB (`FP32`), além de economizar bateria e não travar o Android!

---

## 📌 Parte 2: O que é `GQA` (Grouped-Query Attention: 14 Q-Heads / 2 KV-Heads)?
*(Presente no cartão laranja do Modelo Base no Diagrama 1 e na inspeção da Demo 1)*

- No mecanismo de **Atenção (*Self-Attention*)** de um Transformer, cada palavra gera três vetores:
  - **Q (*Query* / Consulta):** O que o token atual está procurando.
  - **K (*Key* / Chave):** O "rótulo" de cada token passado.
  - **V (*Value* / Valor):** O conteúdo de informação de cada token passado.
- Nos primeiros Transformers (*Multi-Head Attention — MHA*), se o modelo tinha 14 cabeças de consulta (`Q`), ele também tinha 14 cabeças de chave (`K`) e 14 de valor (`V`). Isso gastava muita memória de **KV-Cache**.
- O `Qwen2.5-0.5B` usa **GQA (*Grouped-Query Attention*)**: ele mantém **14 cabeças de *Query*** (preservando a capacidade de raciocínio), mas agrupa todas elas para compartilharem **apenas 2 cabeças de *Key* e 2 de *Value***.
- **Por que isso é importante citar na apresentação?** Porque reduz o tamanho da memória de conversa (**KV-Cache**) em **7 vezes** ($14 / 2 = 7$), o que viabiliza rodar janelas de contexto grandes em CPU e celular!

---

## 📌 Parte 3: Conceitos da Demo 1 (ONNX Runtime + FastAPI)

### 1. Por que convertemos o Modelo para `.onnx`, mas usamos o `Tokenizer` nativo da Hugging Face? Como eles funcionam juntos?
Essa é uma das perguntas mais importantes de engenharia de IA!

- **Uma LLM em produção é dividida em 3 partes distintas:**
  ```
  [Texto Humano] ──► (1. Tokenizer) ──► [IDs Numéricos] ──► (2. Rede Neural ONNX) ──► [Logits/IDs] ──► (3. Detokenizer) ──► [Texto Humano]
  ```
  1. **Pré-processamento (Tokenizer):** É um algoritmo de dicionário e regras de texto (chamado *Byte-Level BPE — Byte-Pair Encoding*) junto com o template **ChatML** (`<|im_start|>user...`). Ele **não tem pesos de rede neural nem multiplicação de matrizes**; ele apenas converte a string `"Olá mundo"` em uma lista de números inteiros: `[151644, 872, 198, ...]`.
  2. **O Cérebro (Rede Neural / Grafo `.onnx`):** O arquivo `model.onnx` **não sabe o que é uma letra ou uma palavra**. Ele é uma função puramente matemática que recebe matrizes de números inteiros (`input_ids`) e devolve um vetor de probabilidades (`logits`) para os 151.936 números do vocabulário.
  3. **Pós-processamento (Detokenizer):** Pega o número inteiro escolhido (ex.: `ID 45912`) e olha na tabela `tokenizer.json` a qual sílaba/palavra ele corresponde.
- **Por que mantemos o `AutoTokenizer` da Hugging Face junto com o `onnxruntime`?**
  - Porque o `tokenizers` da Hugging Face já é escrito internamente em **Rust nativo super-rápido**, ocupa poucos megabytes e **não depende do PyTorch**!
  - Assim, no container de produção (`main.py`), o `AutoTokenizer` faz a ponte `Texto ↔ Números Inteiros`, enquanto o `onnxruntime.InferenceSession` faz todo o trabalho pesado de rede neural (`Números → Probabilidades do Próximo Token`).

### 2. O que são "Fase A: Prefill" e "Fase B: Decode"?
*(Presentes no Diagrama da Demo 1)*

Quando você envia uma pergunta para um modelo gerador de texto, a inferência acontece em **duas fases com comportamentos matemáticos completamente diferentes**:

- **Fase A — *Prefill* (Preenchimento / Leitura do Prompt):**
  - Imagine que seu prompt tem **50 tokens** (`"Explique em 3 tópicos por que SLMs são ideais..."`).
  - Na primeira chamada ao modelo (`step == 0` no `main.py`), você já conhece todos esses 50 tokens de uma vez!
  - Portanto, passamos uma matriz `[1, 50]` para o grafo ONNX. O modelo processa os **50 tokens em paralelo** (operação *Compute-Bound*, dominada por multiplicação Matriz × Matriz), preenche a memória das 24 camadas com as chaves (`K`) e valores (`V`) desses 50 tokens e gera apenas o **1º token da resposta**.
- **Fase B — *Decode* (Decodificação Autoregressiva, 1 a 1):**
  - Para gerar o **2º token da resposta**, o modelo depende do 1º token que acabou de sair. Para gerar o 3º, depende do 2º, e assim por diante (*Autoregressivo*).
  - Portanto, na Fase B, nós fazemos um `for` onde passamos **apenas 1 único token novo por vez** (`input_ids` com shape `[1, 1]`), consultando o histórico já salvo no **KV-Cache**!

### 3. O que são "Entradas e Saídas Explícitas de KV-Cache" (`past_key_values` → `present`)?
- **Sem KV-Cache:** Se o modelo já escreveu 200 palavras, para gerar a palavra 201 ele teria que ler e recalcular todas as matrizes das 200 palavras anteriores de novo! O tempo cresceria de forma quadrática $O(N^2)$ e o servidor travaria após algumas frases.
- **Com KV-Cache Explícito no ONNX:**
  - Como um grafo `.onnx` é **estático e sem estado interno oculto (*stateless*)**, a memória das palavras anteriores precisa entrar e sair explicitamente como parâmetros da função!
  - Por isso, nosso grafo tem **51 entradas** e **49 saídas**:
    - Nas entradas, passamos `past_key_values.0.key` até `past_key_values.23.value` (48 tensores contendo o histórico das 24 camadas até o passo anterior).
    - Nas saídas, o ONNX Runtime nos devolve `present.0.key` até `present.23.value` (os 48 tensores atualizados com o token que acabou de ser lido).
    - No nosso loop em Python/NumPy (`main.py`), basta fazer `past_kv = present_kv` a cada volta do loop! Assim, cada passo do *Decode* processa **apenas 1 token** ($O(1)$ de cálculo linear), tornando a geração muito rápida.

### 4. O que é "Fusão de Operadores" (*Operator Fusion*) e `ORT_ENABLE_ALL`?
*(Presente nas Demos 1 e 3)*

- Em uma camada Transformer escrita em PyTorch puro, uma operação simples como `Linear + Bias + Ativação SiLU + Soma Residual` executa **4 operações separadas**:
  1. A CPU lê a matriz da memória RAM, multiplica e grava o resultado temporário `T1` na RAM.
  2. A CPU lê `T1` da RAM, soma o *bias* e grava `T2` na RAM.
  3. A CPU lê `T2` da RAM, aplica a função `SiLU` e grava `T3` na RAM.
- Esse "vai-e-vem" na memória RAM é lentíssimo!
- Quando ativamos **`GraphOptimizationLevel.ORT_ENABLE_ALL`** no ONNX Runtime (ou quando compilamos com o **Apache TVM** na Demo 3), o otimizador analisa o grafo inteiro e **funde múltiplos nós em um único kernel C++/OpenCL** (por exemplo: `FusedMatMulAddActivation`).
- **Resultado:** O dado entra nos registradores ultra-rápidos do processador/GPU uma única vez, sofre todas as operações seguidas sem voltar para a memória RAM, e só grava o resultado final!

---

## 📌 Parte 4: Conceitos da Demo 2 (GGUF + `llama.cpp` + Ollama + Open WebUI)

### 1. O que é `mmap` (*Memory-Mapped File*) no formato `.gguf`?
*(Presente no Diagrama da Demo 2)*

- Na forma tradicional de carregar um arquivo, o programa aloca 506 MB de memória RAM vazia e chama `read()` para copiar todos os bytes do disco para a RAM antes de poder fazer qualquer conta.
- O formato **`.gguf`** organiza os tensores dentro do arquivo binário já alinhados exatamente como a CPU espera na memória.
- Assim, o `llama.cpp` (motor interno do Ollama) usa a chamada de sistema **`mmap`** do Linux/Windows: ele apenas aponta a tabela de páginas da memória virtual direto para o arquivo no SSD!
- **Vantagens para explicar na apresentação:**
  1. **Carregamento quase instantâneo (*Zero-Copy*):** O modelo fica pronto em fração de segundo.
  2. **Menor pressão na RAM:** O próprio sistema operacional gerencia o cache de páginas do disco; se faltar RAM no notebook, o SO não precisa duplicar os pesos na área de *swap*.

### 2. O que são instruções `SIMD` (`AVX2` / `ARM NEON`)?
- **SIMD** significa *Single Instruction, Multiple Data* (Uma Instrução, Múltiplos Dados).
- Em vez de o processador multiplicar 1 peso por 1 ativação de cada vez, instruções vetoriais como **AVX2** (em processadores Intel/AMD) ou **ARM NEON** (em celulares e Macs M1/M2/M3) carregam **8, 16 ou 32 números quantizados (`Q8_0`) de uma só vez** em um registrador largo de 256 bits e multiplicam todos em **um único ciclo de clock**.

### 3. Como funciona a arquitetura dos 2 Containers Docker na Demo 2 (`slm-ollama` + `slm-open-webui`)?
- No [`demo2_local_gguf/docker-compose.yml`](file:///c:/Users/Pichau/projects/ia_generativa/demo2_local_gguf/docker-compose.yml), subimos dois serviços dentro da mesma rede virtual do Docker:
  1. **`slm-ollama` (Backend / Engine C++):** Roda o servidor do Ollama onde importamos nosso `.gguf` com o [`Modelfile`](file:///c:/Users/Pichau/projects/ia_generativa/demo2_local_gguf/Modelfile) (`ollama create qwen25-slm-custom -f /models/Modelfile`). Mapeamos a porta `11435` do Windows para a `11434` do container para nunca conflitar caso o PC já tenha um Ollama instalado.
  2. **`slm-open-webui` (Frontend Web):** É a interface estilo ChatGPT rodando na porta `3000`. Como ambos estão na mesma rede Docker, o Open WebUI conversa com o Ollama pelo endereço interno `http://ollama:11434` sem passar por fora.
- **Por que colocamos `PARAMETER num_ctx 16384` e `PARAMETER num_predict 2048` no `Modelfile`?**
  - Quando você digita apenas `"Oi"` no Open WebUI, a interface web envia por trás todo o histórico da conversa e faz chamadas automáticas pedindo para o modelo gerar um título e sugestões de perguntas. Com `num_ctx 16384` (16K tokens de contexto), garantimos que o modelo nunca trave nem corte respostas por falta de janela!

---

## 📌 Parte 5: Conceitos da Demo 3 (Mobile Android Nativo com MLC LLM / Apache TVM)

### 1. O que é o Apache TVM / TVM Relax e qual a diferença dele para o ONNX Runtime e o `llama.cpp`?
*(Presente no Diagrama da Demo 3)*

- O **ONNX Runtime** e o **`llama.cpp`** são **interpretadores/motores genéricos**: eles vêm pré-compilados com uma coleção de funções prontas e leem o arquivo do modelo (`.onnx` ou `.gguf`) em tempo de execução.
- O **MLC LLM (baseado no compilador Apache TVM Relax)** vai um passo além: ele é um **compilador de redes neurais (*Machine Learning Compiler*)**.
  - Ele pega a arquitetura específica do `Qwen2.5`, funde a desquantização de 4 bits (`q4f16_1`) com a multiplicação de matrizes (`fused_dequantize_NT_matmul`) e **gera código de máquina customizado (`.o` / `.so`) + *shaders/kernels* OpenCL sob medida** para a GPU do celular!

### 2. O que significa `prefill_chunk_size = 512` e por que a memória caiu de `5365 MB` para `628.8 MB`?
*(Destaque verde na caixa 1 do Diagrama da Demo 3 — ótimo ponto para impressionar a banca!)*

- Quando rodamos a primeira compilação com os valores padrões de servidor do Qwen2.5 (`context_window_size = 32768` e `prefill_chunk_size = 8192`), o compilador TVM reservou **5052 MB (5 GB!) só de buffer temporário** para conseguir processar 8192 tokens de uma só vez. Em um celular, tentar alocar 5 GB de uma vez faz o Android matar o app na hora (*Out Of Memory*).
- Ao configurarmos no Passo 3.1:
  - `--context-window-size 4096` (janela de 4K tokens, mais do que suficiente para chat no celular)
  - `--prefill-chunk_size 512` (*Chunked Prefill*: se o usuário colar um texto grande de 2000 tokens, a GPU do celular mastiga esse texto em pedacinhos de 512 em 512 tokens)
- **O resultado medido pelo compilador TVM no Passo 3.2:**
  - **Pesos do modelo (`Parameters`):** `265.12 MB`
  - **Buffer temporário (`Temporary buffer`):** caiu de `5052 MB` para **`315.75 MB`**!
  - **KV-Cache para 4096 tokens:** `48.00 MB` (`0.01 MB` por token)
  - **Total absoluto de RAM/VRAM no celular:** **`628.87 MB`** (uma redução de **88% de memória**, permitindo rodar liso em qualquer celular intermediário!).

### 3. O que são os arquivos `lib0.o`, `devc.o`, `libtvm4j_runtime_packed.so` e JNI?
*(Presentes nas caixas 2 e 3 do Diagrama da Demo 3)*

- **`lib0.o` e `qwen2_q4f16_1_devc.o` (dentro do `.tar`):**
  - `lib0.o`: Código nativo compilado para o processador ARM64 do celular (`aarch64-linux-android`) que orquestra o modelo.
  - `devc.o`: Contém o bytecode da máquina virtual do TVM (*Relax VM*) e o código-fonte dos kernels **OpenCL** que rodam dentro da placa de vídeo (GPU Adreno/Mali) do smartphone.
- **JNI (*Java Native Interface*) e `libtvm4j_runtime_packed.so`:**
  - Aplicativos Android têm sua interface visual escrita em **Kotlin / Jetpack Compose** (que roda na máquina virtual Java/ART do Android), mas o motor de IA é escrito em **C++ / OpenCL**.
  - O módulo `:mlc4j` usa **JNI** para fazer a ponte entre o Kotlin (`MLCEngine.kt` → `JSONFFIEngine.java` → `org.apache.tvm.LibInfo`) e a biblioteca C++ nativa `libtvm4j_runtime_packed.so` empacotada dentro do `.apk`.

### 4. Como funciona o Provisionamento 100% Offline via USB (`adb`) e o *Scoped Storage* do Android 14/15?
*(Presente na caixa 4 do Diagrama da Demo 3)*

- **O desafio de apresentar ao vivo:** Em uma apresentação de mestrado, depender do Wi-Fi da sala para o celular baixar 800 MB de pesos do Hugging Face é arriscado. Além disso, embutir centenas de megabytes comprimidos dentro do `.apk` dobra o gasto de armazenamento no celular.
- **Como resolvemos no Passo 3.4:**
  1. O script instala o aplicativo ([`qwen2.5-mlc-chat.apk`](file:///c:/Users/Pichau/projects/ia_generativa/demo3_android_mlc/qwen2.5-mlc-chat.apk)) via cabo USB (`adb install -r`).
  2. Envia os shards quantizados (`params_shard_*.bin` e `mlc-chat-config.json`) via USB (`adb push`) para a pasta temporária `/data/local/tmp/` do Android.
  3. **O pulo do gato do Android 14/15 (`run-as ai.mlc.mlcchat`):** No Android moderno, o sistema de segurança (*Scoped Storage* / SELinux) proíbe que um aplicativo leia arquivos que foram criados pelo usuário `shell` do `adb push` (gerando o erro `EACCES: Permission denied`). Ao executarmos `adb shell run-as ai.mlc.mlcchat cp -r ... files/`, o Android assume a identidade do próprio usuário interno do aplicativo (`u0_a500`) e copia os pesos para o armazenamento interno privado do app (`application.filesDir`).
  4. Assim, quando o app abre, ele encontra todos os arquivos já indexados com permissão total e libera o botão de **Chat 100% Offline** imediatamente (funcionando até em **Modo Avião**)!

---

## 🎤 Roteiro Rápido (Passo a Passo) para o Dia da Apresentação

1. **Abertura (Célula 0 e Bloco 0):**
   - Mostre o **Diagrama 1 (`fluxo_geral_slms.svg`)** explicando a tese central: *"Vamos pegar um único SLM de 490 milhões de parâmetros e provar como técnicas de compilação e quantização permitem rodá-lo na Nuvem (FP32), no Notebook (GGUF 8-bit) e dentro de um Smartphone Android offline (MLC 4-bit), sempre com Zero PyTorch em produção."*
2. **Demo 1 — Cloud / Microsserviço (Células 3 a 17):**
   - Mostre o **Diagrama da Demo 1**, rode a inspeção do grafo ONNX (mostrando as 51 entradas do KV-Cache e o GQA), suba o container `slm-onnx-api`, mostre o **Swagger UI em `http://localhost:8000/docs`** e dispare o `POST /generate` no notebook.
3. **Demo 2 — Notebook Local (Células 18 a 28):**
   - Mostre o **Diagrama da Demo 2**, destaque a queda de tamanho de **1890 MB para 506 MB** na conversão para `.gguf` (`Q8_0`), suba os containers `slm-ollama` e `slm-open-webui`, rode a célula do `/api/chat` e abra a interface visual estilo ChatGPT em **`http://localhost:3000`**.
4. **Demo 3 — Android Nativo Offline (Células 29 a 38):**
   - Mostre o **Diagrama da Demo 3**, rode a tabela comparativa (chegando aos **265 MB** em 4 bits `q4f16_1`), mostre a compilação TVM reduzindo a RAM para **628 MB**, gere/valide o `.apk` com o Gradle e, com o celular plugado no USB (ou já instalado), mostre o chat rodando diretamente na tela do seu telefone em **Modo Avião**!
5. **Fechamento (Célula 38):**
   - Comente brevemente a alternativa com `onnxruntime-android` (para equipes que querem manter um único formato `.onnx` em tudo) vs. `MLC LLM` (para máxima performance OpenCL em GPUs mobile) e encerre com a tabela comparativa final.
