from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import time
import uuid
import base64
import io
import gc
import traceback

# ── Autenticação ──────────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

app = FastAPI(title="THC LLM API")

# ── Registro de modelos de TEXTO ──────────────────────────────────────────────
TEXT_MODELS = {
    "gemma-1b": {
        "backend": "transformers",
        "id": "google/gemma-3-1b-it",
        "label": "Gemma 3 1B",
        "desc": "Rápido • conversação geral",
    },
    "gemma-4b": {
        "backend": "gguf",
        "repo": "bartowski/google_gemma-3-4b-it-GGUF",
        "file": "google_gemma-3-4b-it-Q4_K_M.gguf",
        "label": "Gemma 3 4B",
        "desc": "⚡ Rápido • bem mais forte que o 1B • ~2.5GB",
    },
    "gemma-12b": {
        "backend": "gguf",
        "repo": "bartowski/google_gemma-3-12b-it-GGUF",
        "file": "google_gemma-3-12b-it-Q4_K_M.gguf",
        "label": "Gemma 3 12B",
        "desc": "🚀 Gemma mais forte • ~7.3GB",
    },
    "qwen-coder-3b": {
        "backend": "transformers",
        "id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "label": "Qwen2.5-Coder 3B",
        "desc": "Especialista em código",
    },
    "llama32-3b": {
        "backend": "gguf",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "label": "Llama 3.2 3B",
        "desc": "⚡ Rápido • teste e bate-papo • ~2GB",
    },
    "llama31-8b": {
        "backend": "gguf",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "label": "Llama 3.1 8B",
        "desc": "Código + matemática • ~5GB",
    },
    "hermes3-8b": {
        "backend": "gguf",
        "repo": "NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
        "file": "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf",
        "label": "Nous-Hermes 3 8B",
        "desc": "Escrita criativa • ~5GB",
    },
    "qwen-14b-gguf": {
        "backend": "gguf",
        "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "label": "Qwen2.5 14B GGUF",
        "desc": "🔥 Mais forte • mais lento (~9GB)",
    },
    "deepseek-r1-distill-14b": {
        "backend": "gguf",
        "repo": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "file": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "label": "DeepSeek R1 Distill 14B",
        "desc": "🧮 Raciocínio avançado • ~9GB GGUF",
    },
}
DEFAULT_MODEL_KEY = "gemma-1b"

# ── Instrução de idioma — sempre aplicada, em todos os modelos e modos ───────
LANGUAGE_INSTRUCTION = (
    "Responda SEMPRE em português do Brasil, independentemente do idioma da "
    "pergunta. Nunca responda em chinês, inglês, japonês ou qualquer outro "
    "idioma, mesmo que seu raciocínio interno use outro idioma — a resposta "
    "final deve ser 100% em português do Brasil."
)

# ── Cache de UM modelo de texto por vez ───────────────────────────────────────
_current = {"key": None, "tokenizer": None, "model": None, "backend": None}

def unload_current():
    if _current.get("model") is not None:
        print(f"[THC LLM] Descarregando modelo anterior ({_current['key']})...")
    _current["key"] = None
    _current["tokenizer"] = None
    _current["model"] = None
    _current["backend"] = None
    gc.collect()

def get_text_model(key: str):
    if key not in TEXT_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo desconhecido: {key}")

    if _current.get("key") == key and _current.get("model") is not None:
        return _current

    unload_current()

    cfg = TEXT_MODELS[key]
    backend = cfg["backend"]

    try:
        if backend == "transformers":
            model_id = cfg["id"]
            print(f"[THC LLM] Carregando modelo (transformers): {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                device_map="cpu",
                token=hf_token,
                trust_remote_code=True,
            )
            model.eval()
            _current.update({"key": key, "tokenizer": tokenizer, "model": model, "backend": "transformers"})

        elif backend == "gguf":
            from llama_cpp import Llama
            repo = cfg['repo']
            filename = cfg['file']
            print(f"[THC LLM] Carregando GGUF: {repo}/{filename}...")

            llm = Llama.from_pretrained(
                repo_id=repo,
                filename=filename,
                n_ctx=4096,
                n_threads=2,
                verbose=False,
            )
            _current.update({"key": key, "tokenizer": None, "model": llm, "backend": "gguf"})

        else:
            raise HTTPException(status_code=500, detail=f"Backend desconhecido: {backend}")

    except HTTPException:
        raise
    except Exception as e:
        unload_current()
        raise HTTPException(status_code=500, detail=f"Erro ao carregar {key}: {str(e)}") from e

    print(f"[THC LLM] Modelo {key} carregado!")
    return _current

# Pré-carrega o modelo padrão no boot
print(f"[THC LLM] Pré-carregando modelo padrão ({DEFAULT_MODEL_KEY})...")
get_text_model(DEFAULT_MODEL_KEY)

# ── Modelo de IMAGEM ──────────────────────────────────────────────────────────
IMAGE_MODEL_ID = "stabilityai/sd-turbo"
image_pipeline = None

def get_image_pipeline():
    global image_pipeline
    if image_pipeline is None:
        from diffusers import AutoPipelineForText2Image
        print(f"[THC LLM] Carregando modelo de imagem {IMAGE_MODEL_ID}...")
        image_pipeline = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=torch.float32,
            token=hf_token,
        )
        image_pipeline.to("cpu")
        print("[THC LLM] Modelo de imagem pronto!")
    return image_pipeline

# ── Schemas ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL_KEY
    mode: Optional[str] = "medium"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = 2
    size: Optional[int] = 512

# ── Endpoints — Geral ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/v1/models")
def list_models():
    return {
        "text_models": [
            {"key": k, "label": v["label"], "desc": v["desc"], "active": k == _current["key"]}
            for k, v in TEXT_MODELS.items()
        ],
        "image_model": {"id": IMAGE_MODEL_ID},
    }

# ── Aplica os modos Fast/Médio/Thinking ────────────────────────────────────────
def apply_mode(mode, max_tokens, temperature, chat):
    do_sample = temperature > 0
    if mode == "fast":
        return min(max_tokens, 256), False, None, chat
    elif mode == "thinking":
        t = min(temperature, 0.4) if temperature > 0 else 0.3
        chat = [{
            "role": "system",
            "content": LANGUAGE_INSTRUCTION + " Pense com cuidado, passo a passo, "
                        "antes de responder. Explique seu raciocínio brevemente e "
                        "depois dê a resposta final de forma clara."
        }] + chat
        return max(max_tokens, 768), True, t, chat
    else:
        return max_tokens, do_sample, (temperature if do_sample else None), chat

# ── Endpoints — Chat ────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        state = get_text_model(req.model)
        backend = state["backend"]
        chat = [{"role": m.role, "content": m.content} for m in req.messages]

        # ── Garante idioma PT-BR em qualquer modo, pra qualquer modelo ───────
        if req.mode != "thinking":
            chat = [{"role": "system", "content": LANGUAGE_INSTRUCTION}] + chat

        max_tokens, do_sample, temperature, chat = apply_mode(
            req.mode, req.max_tokens, req.temperature, chat
        )

        t0 = time.time()

        if backend == "transformers":
            tokenizer = state["tokenizer"]
            model = state["model"]

            tokenized = tokenizer.apply_chat_template(
                chat, return_tensors="pt", add_generation_prompt=True, return_dict=True,
            )
            input_ids = tokenized["input_ids"]

            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
            if do_sample and temperature:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)

            generated = output[0][input_ids.shape[-1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            prompt_tokens = input_ids.shape[-1]
            completion_tokens = len(generated)

        elif backend == "gguf":
            llm = state["model"]
            result = llm.create_chat_completion(
                messages=chat,
                max_tokens=max_tokens,
                temperature=temperature if temperature else 0.0,
            )
            text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

        else:
            raise HTTPException(status_code=500, detail="Backend inválido")

        elapsed = time.time() - t0
        print(f"[THC LLM] Resposta gerada em {elapsed:.1f}s ({backend})")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "mode": req.mode,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        err = traceback.format_exc()
        print(f"[ERRO CHAT] {err}")
        raise HTTPException(status_code=500, detail=str(e) + "\n" + err)

# ── Endpoints — Geração de Imagem ──────────────────────────────────────────────
@app.post("/v1/images/generations")
def generate_image(req: ImageRequest):
    try:
        pipe = get_image_pipeline()
        t0 = time.time()
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=0.0,
            height=req.size,
            width=req.size,
        )
        image = result.images[0]
        elapsed = time.time() - t0

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "created": int(time.time()),
            "elapsed_seconds": round(elapsed, 1),
            "data": [{"b64_json": img_b64}]
        }
    except Exception as e:
        err = traceback.format_exc()
        print(f"[ERRO IMAGEM] {err}")
        raise HTTPException(status_code=500, detail=str(e) + "\n" + err)