Aqui está o **guia passo a passo** planejado para a execução prática das suas três demonstrações, organizado por ambiente de execução e acompanhado dos links de referência oficiais de cada ecossistema.

---

### **Demo 1: Servidor Externo / Produção (FastAPI + ONNX Runtime via Docker)**
* **Objetivo:** Mostrar o servimento de um modelo otimizado em ONNX exposto por uma API REST em Python dentro de um container Docker.

#### **Passo a Passo de Execução:**
1. **Exportar e Otimizar o Modelo para ONNX:**
   * Converta o modelo PyTorch/Hugging Face para o formato `.onnx` usando a biblioteca `optimum` ou `torch.onnx`.
   * Aplique otimizações de grafo com `onnxruntime.transformers.optimizer`.
2. **Criar a API com FastAPI (`main.py`):**
   * Monte uma estrutura com Pydantic para os esquemas de entrada (`InputData`) e saída (`OutputData`).
   * No carregamento da API, inicialize a sessão do ONNX Runtime (`onnxruntime.InferenceSession`) configurando o provedor de execução (`CPUExecutionProvider` ou `CUDAExecutionProvider`).
   * Crie uma rota assíncrona POST (`/generate`) para processar requisições e retornar o texto gerado.
3. **Empacotar em Container Docker (`Dockerfile`):**
   * Use uma imagem base Python (ex: `python:3.10-slim`), copie o `requirements.txt` (com `fastapi`, `uvicorn`, `onnxruntime`) e os arquivos de código.
   * Exponha a porta `8000` e defina o comando de inicialização com Uvicorn: `uvicorn main:app --host 0.0.0.0 --port 8000`.
4. **Executar e Demonstrar:**
   * Inicie o container: `docker run -p 8000:8000 meu-servico-onnx`.
   * Abra a documentação automática interativa do Swagger UI em `http://localhost:8000/docs` para disparar chamadas de teste ao vivo.

🔗 **Links de Referência:**
* Documentação do FastAPI: [[suspicious link removed]](https://[suspicious link removed]/)
* ONNX Runtime Python API: [[suspicious link removed]]([suspicious link removed])
* Hugging Face Optimum (ONNX Export): [[suspicious link removed]]([suspicious link removed])

---

### **Demo 2: Notebook / Laptop Local (Portabilidade para GGUF + Ollama / LM Studio)**
* **Objetivo:** Demonstrar a conversão de um modelo Hugging Face para o formato quantizado GGUF e sua execução offline no computador via linha de comando (Ollama) ou interface gráfica (LM Studio).

#### **Passo a Passo de Execução:**
1. **Converter o Modelo HF para GGUF (via `llama.cpp`):**
   * Clone o repositório do `llama.cpp` e instale as dependências:
     ```bash
     git clone https://github.com/ggerganov/llama.cpp.git
     cd llama.cpp && pip install -r requirements.txt
     ```
   * Execute o script de conversão definindo o tipo de quantização desejado (ex: `q4_k_m`):
     ```bash
     python convert_hf_to_gguf.py ../pasta-do-modelo-hf/ --outfile ../modelo-quantizado.gguf --outtype q4_k_m
     ```
2. **Criar e Importar o Modelo no Ollama:**
   * Crie um arquivo `Modelfile` no mesmo diretório indicando o arquivo `.gguf` e o template de conversa:
     ```dockerfile
     FROM ./modelo-quantizado.gguf
     PARAMETER num_ctx 4096
     TEMPLATE """<|system|>Você é um assistente útil.<|user|>{{ .Prompt }}<|assistant|>"""
     ```
   * Compile o modelo no servidor local do Ollama:
     ```bash
     ollama create meu-modelo-custom -f Modelfile
     ```
3. **Executar a Inferência Local:**
   * **Via Terminal (Ollama):** Execute `ollama run meu-modelo-custom` para abrir o chat interativo.
   * **Via Interface Gráfica (LM Studio):** Abra o LM Studio, carregue o arquivo `modelo-quantizado.gguf` baixado/convertido e inicie a conversa pela interface.

🔗 **Links de Referência:**
* Repositório `llama.cpp` (Conversão GGUF): [[suspicious link removed]](https://[suspicious link removed])
* Guia do Ollama `Modelfile`: [[suspicious link removed]]([suspicious link removed])
* LM Studio: [[suspicious link removed]](https://[suspicious link removed]/)

---

### **Demo 3: Dispositivo Android Nativo (MLC LLM)**
* **Objetivo:** Apresentar a compilação nativa de um SLM e seu empacotamento em um aplicativo móvel (APK) para Android via MLC LLM (sem dependência de Termux ou emuladores).

#### **Passo a Passo de Execução:**
1. **Converter Pesos e Gerar Configuração de Chat:**
   * Converta os pesos do modelo para o formato do MLC LLM especificando a quantização (ex: `q4f16_1`):
     ```bash
     mlc_llm convert_weight ./meu-modelo-hf/ --quantization q4f16_1 -o dist/meu-modelo-mlc/
     ```
   * Gere o arquivo de configuração `mlc-chat-config.json`:
     ```bash
     mlc_llm gen_config ./meu-modelo-hf/ --quantization q4f16_1 --conv-template phi-3 -o dist/meu-modelo-mlc/
     ```
2. **Compilar a Biblioteca Nativa Android (`.so`):**
   * Tendo o Android NDK e CMake configurados no seu ambiente de desenvolvimento, compile a biblioteca estática:
     ```bash
     mlc_llm compile ./dist/meu-modelo-mlc/mlc-chat-config.json --device android -o dist/libs/[suspicious link removed]
     ```
3. **Empacotar no Android Studio e Gerar o APK:**
   * Clone o projeto de template nativo para Android mantido pelo MLC LLM.
   * Copie as bibliotecas geradas (`.so`) e a pasta de pesos para o diretório de ativos (`assets`) do projeto Android.
   * Abra o projeto no **Android Studio** e faça o build do aplicativo (`Build > Build Bundle(s) / APK(s) > Build APK`).
   * Instale o arquivo `.apk` gerado no celular Android para rodar o modelo 100% offline via GPU/OpenCL do aparelho.
4. *(Citação Oral durante a Apresentação)*: Comente brevemente que, para equipes que padronizam pipelines exclusivamente em ONNX, a alternativa de mercado para integrar modelos diretamente no código Java/Kotlin do Android é a biblioteca **ONNX Runtime Android** (`onnxruntime-android`).

🔗 **Links de Referência:**
* Guia do MLC LLM para Android: [[suspicious link removed]]([suspicious link removed])
* Repositório MLC LLM: [[suspicious link removed]]([suspicious link removed])
* ONNX Runtime Android (para citação): [[suspicious link removed]]([suspicious link removed])

---

📱 Se precisar de ajuda para revisar os scripts de conversão ou preparar os arquivos de configuração (`Dockerfile` ou `Modelfile`) antes de rodar os testes, é só avisar!