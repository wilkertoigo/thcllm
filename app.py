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
import traceback

# ── Autenticação ──────────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

app = FastAPI(title="THC LLM API")

# ── Carrega modelo de TEXTO na inicialização ─────────────────────────────────
MODEL_ID = "google/gemma-3-1b-it"

print(f"[THC LLM] Carregando modelo de texto {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    device_map="cpu",
    token=hf_token,
)
model.eval()
print("[THC LLM] Modelo de texto pronto!")

# ── Modelo de IMAGEM — carregado sob demanda (lazy load) ─────────────────────
IMAGE_MODEL_ID = "stabilityai/sd-turbo"
image_pipeline = None

def get_image_pipeline():
    global image_pipeline
    if image_pipeline is None:
        from diffusers import AutoPipelineForText2Image
        print(f"[THC LLM] Carregando modelo de imagem {IMAGE_MODEL_ID} (primeira vez, pode demorar)...")
        image_pipeline = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=torch.float32,
            token=hf_token,
        )
        image_pipeline.to("cpu")
        print("[THC LLM] Modelo de imagem pronto!")
    return image_pipeline

# ── Schemas — Chat ────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

# ── Schemas — Imagem ──────────────────────────────────────────────────────────
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
        "object": "list",
        "data": [
            {"id": MODEL_ID, "object": "model", "type": "text"},
            {"id": IMAGE_MODEL_ID, "object": "model", "type": "image"},
        ]
    }

# ── Endpoints — Chat (texto) ───────────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        chat = [{"role": m.role, "content": m.content} for m in req.messages]

        tokenized = tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = tokenized["input_ids"]

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature if req.temperature > 0 else None,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0][input_ids.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
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
            guidance_scale=0.0,  # SD-Turbo funciona melhor sem guidance
            height=req.size,
            width=req.size,
        )
        image = result.images[0]
        elapsed = time.time() - t0

        # Converte para base64
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