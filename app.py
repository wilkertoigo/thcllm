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

# ── Registro de modelos de TEXTO disponíveis ──────────────────────────────────
TEXT_MODELS = {
    "gemma-1b": {
        "id": "google/gemma-3-1b-it",
        "label": "Gemma 3 1B",
        "desc": "Rápido • conversação geral",
    },
    "qwen-coder-3b": {
        "id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "label": "Qwen2.5-Coder 3B",
        "desc": "Especialista em código",
    },
    "phi4-mini": {
        "id": "microsoft/Phi-4-mini-instruct",
        "label": "Phi-4 Mini",
        "desc": "Código + raciocínio",
    },
}
DEFAULT_MODEL_KEY = "gemma-1b"

# ── Cache de UM modelo de texto por vez (limite de RAM em CPU) ───────────────
_current = {"key": None, "tokenizer": None, "model": None}

def get_text_model(key: str):
    if key not in TEXT_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo desconhecido: {key}")

    if _current["key"] == key and _current["model"] is not None:
        return _current["tokenizer"], _current["model"]

    # Descarrega o modelo anterior pra liberar RAM
    if _current["model"] is not None:
        print(f"[THC LLM] Descarregando modelo anterior ({_current['key']})...")
        del _current["model"]
        del _current["tokenizer"]
        gc.collect()

    model_id = TEXT_MODELS[key]["id"]
    print(f"[THC LLM] Carregando modelo de texto: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()

    _current["key"] = key
    _current["tokenizer"] = tokenizer
    _current["model"] = model
    print(f"[THC LLM] Modelo {model_id} pronto!")
    return tokenizer, model

# Pré-carrega o modelo padrão no boot
print(f"[THC LLM] Pré-carregando modelo padrão ({DEFAULT_MODEL_KEY})...")
get_text_model(DEFAULT_MODEL_KEY)

# ── Modelo de IMAGEM — carregado sob demanda ──────────────────────────────────
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
    mode: Optional[str] = "medium"       # fast | medium | thinking
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

# ── Endpoints — Chat ────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        tokenizer, model = get_text_model(req.model)

        chat = [{"role": m.role, "content": m.content} for m in req.messages]

        # ── Ajustes por modo ──────────────────────────────────────────────────
        max_tokens = req.max_tokens
        do_sample = req.temperature > 0
        temperature = req.temperature if req.temperature > 0 else None

        if req.mode == "fast":
            max_tokens = min(req.max_tokens, 256)
            do_sample = False
            temperature = None
        elif req.mode == "thinking":
            max_tokens = max(req.max_tokens, 768)
            do_sample = True
            temperature = min(req.temperature, 0.4) if req.temperature > 0 else 0.3
            chat = [{
                "role": "system",
                "content": "Pense com cuidado, passo a passo, antes de responder. "
                            "Explique seu raciocínio brevemente e depois dê a resposta final de forma clara."
            }] + chat

        tokenized = tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
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
                "prompt_tokens": input_ids.shape[-1],
                "completion_tokens": len(generated),
                "total_tokens": input_ids.shape[-1] + len(generated),
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