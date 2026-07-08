import os
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

app = Flask(__name__)

print("[INFO] Baixando a versão Q4_K_M do Gemma 4 12B (Aprox. 7.6GB)...")
# Baixa o modelo da instância de armazenamento GGUF do Bartowski ou do seu próprio repo
model_path = hf_hub_download(
    repo_id="bartowski/gemma-4-12B-it-GGUF",
    filename="gemma-4-12B-it-Q4_K_M.gguf"
)

print("[INFO] Carregando o Gemma 4 12B na memória RAM do Space...")
# Configurado para CPU pura (n_gpu_layers=0), ideal para o plano gratuito do HF
llm = Llama(
    model_path=model_path,
    n_ctx=2048,        # Limite de contexto equilibrado para não estourar RAM
    n_threads=2,       # Quantidade exata de vCPUs disponíveis no HF Free
    n_gpu_layers=0
)
print("[INFO] Modelo carregado com sucesso e pronto para uso!")

@app.route('/v1/chat', methods=['POST'])
def chat():
    dados = request.json
    prompt_usuario = dados.get("prompt", "")
    
    # Prompt estruturado seguindo o padrão nativo do Gemma 4
    prompt_final = f"<|im_start|>user\n{prompt_usuario}<|im_end|>\n<|im_start|>assistant\n"
    
    # Executando a inferência via llama.cpp
    resposta_llm = llm(
        prompt_final,
        max_tokens=256,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    )
    
    texto_gerado = resposta_llm["choices"][0]["text"]
    return jsonify({"resposta": texto_gerado})

if __name__ == '__main__':
    # O HF Spaces exige que a aplicação rode na porta 7860
    app.run(host='0.0.0.0', port=7860)