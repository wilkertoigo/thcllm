from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import time
import uuid
import traceback

# ── Autenticação ──────────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

app = FastAPI(title="THC LLM API")

# ── Carrega modelo na inicialização ──────────────────────────────────────────
MODEL_ID = "google/gemma-3-1b-it"

print(f"[THC LLM] Carregando modelo {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,        # corrigido: era torch_dtype
    device_map="cpu",
    token=hf_token,
)
model.eval()
print("[THC LLM] Modelo pronto!")

# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_ID}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model"}]
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        chat = [{"role": m.role, "content": m.content} for m in req.messages]

        input_ids = tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            add_generation_prompt=True,
        )

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
        # Mostra o erro completo nos logs e retorna para o cliente
        err = traceback.format_exc()
        print(f"[ERRO] {err}")
        raise HTTPException(status_code=500, detail=str(e) + "\n" + err)