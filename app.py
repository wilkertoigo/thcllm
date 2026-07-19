from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator, constr
from typing import List, Optional, Literal
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import time
import uuid
import base64
import io
import gc
import glob
import traceback
import numpy as np
import httpx
import asyncio
import json

# ── Import Configuration and Logger ───────────────────────────────────────────
from config import TEXT_MODELS, DEFAULT_MODEL_KEY, IMAGE_MODEL_ID
from logger import logger

# ── Retry Configuration ───────────────────────────────────────────────────────
import time
from functools import wraps


def retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2):
    """Decorator para retry com backoff exponencial (funções síncronas)"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Tentativa {attempt + 1}/{max_retries} falhou: {str(e)}. "
                            f"Retrying em {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"Todas as {max_retries} tentativas falharam: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator


def async_retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2):
    """Decorator para retry com backoff exponencial (funções assíncronas)"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Tentativa {attempt + 1}/{max_retries} falhou: {str(e)}. "
                            f"Retrying em {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"Todas as {max_retries} tentativas falharam: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator

# ── Import Custom Exceptions ───────────────────────────────────────────────────
from exceptions import (
    THCError,
    ModelNotFoundError,
    ModelLoadError,
    BackendError,
    ConfigurationError,
    APIError,
    ImageGenerationError,
    ChatError,
)

# ── Autenticação ──────────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

app = FastAPI(title="THC LLM API")

LANGUAGE_INSTRUCTION = (
    "Responda SEMPRE em português do Brasil, independentemente do idioma da "
    "pergunta. Nunca responda em chinês, inglês, japonês ou qualquer outro "
    "idioma, mesmo que seu raciocínio interno use outro idioma — a resposta "
    "final deve ser 100% em português do Brasil."
)

# ═══════════════════════════════════════════════════════════════════════════
# RAG + SKILLS — base de conhecimento local (embeddings + busca por similaridade)
# ═══════════════════════════════════════════════════════════════════════════
KNOWLEDGE_DIR = "knowledge"
SKILLS_DIR = "skills"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

embed_model = None
knowledge_index = {"chunks": [], "vectors": None}
skills_index = {"chunks": [], "vectors": None}


def get_embed_model():
    global embed_model
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Carregando modelo de embeddings ({EMBED_MODEL_ID})...")
        embed_model = SentenceTransformer(EMBED_MODEL_ID)
        logger.info("Modelo de embeddings pronto!")
    return embed_model


def chunk_text(text, max_chars=600):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i + max_chars])
    return chunks


def build_index(directory):
    os.makedirs(directory, exist_ok=True)
    files = glob.glob(os.path.join(directory, "*.md")) + glob.glob(os.path.join(directory, "*.txt"))
    all_chunks = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                content = fh.read()
            for chunk in chunk_text(content):
                all_chunks.append({"text": chunk, "source": os.path.basename(f)})
        except Exception as e:
            logger.error(f"Erro lendo {f}: {e}")

    if not all_chunks:
        logger.warning(f"Nenhum arquivo .md/.txt encontrado em /{directory} — RAG dessa pasta ficará vazio.")
        return {"chunks": [], "vectors": None}

    model = get_embed_model()
    texts = [c["text"] for c in all_chunks]
    vectors = model.encode(texts, normalize_embeddings=True)
    logger.info(f"Indexado: {len(all_chunks)} trechos de /{directory}")
    return {"chunks": all_chunks, "vectors": np.array(vectors)}


def retrieve(query, index, top_k=3, min_score=0.25):
    if index["vectors"] is None or len(index["chunks"]) == 0:
        return []
    model = get_embed_model()
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = index["vectors"] @ q_vec  # cosine similarity (vetores já normalizados)
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in top_idx:
        if scores[i] >= min_score:
            results.append(index["chunks"][i])
    return results


def reload_indexes():
    global knowledge_index, skills_index
    knowledge_index = build_index(KNOWLEDGE_DIR)
    skills_index = build_index(SKILLS_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# BUSCA WEB — DuckDuckGo, sem API key
# ═══════════════════════════════════════════════════════════════════════════
@retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2)
def web_search(query, max_results=4):
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="br-pt", max_results=max_results))
        if not results:
            return ""
        lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            url = r.get("href") or r.get("url", "")
            lines.append(f"- {title}: {body} (Fonte: {url})")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Erro na busca web: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Monta o system prompt combinando idioma + RAG + skills + web + modo
# ═══════════════════════════════════════════════════════════════════════════
def build_system_prompt(user_query, mode, use_web, free_mode=False):
    now = datetime.now(ZoneInfo("America/Sao_Paulo"))
    now_str = now.strftime("%A, %d de %B de %Y, %H:%M (horário de Brasília)")

    parts = [
        LANGUAGE_INSTRUCTION,
        f"\nA data e hora atuais são: {now_str}. Esta é a fonte oficial e confiável "
        f"para qualquer pergunta sobre data ou hora atual — nunca invente outra data "
        f"nem use datas do seu treinamento para isso.",
    ]

    if not free_mode:
        knowledge_hits = retrieve(user_query, knowledge_index, top_k=3)
        if knowledge_hits:
            block = "\n".join(f"- {h['text']}" for h in knowledge_hits)
            parts.append(f"\n### Informações da loja (use se forem relevantes à pergunta):\n{block}")

        skill_hits = retrieve(user_query, skills_index, top_k=2)
        if skill_hits:
            block = "\n".join(f"- {h['text']}" for h in skill_hits)
            parts.append(f"\n### Instruções de comportamento a seguir:\n{block}")

    if use_web:
        web_results = web_search(user_query)
        if web_results:
            parts.append(
                f"\n### Resultados de busca na web (cite a fonte quando usar):\n{web_results}"
            )
        else:
            parts.append(
                "\n### Busca na web ativada, mas não retornou resultados úteis para esta "
                "pergunta. Informe isso ao usuário em vez de inventar uma resposta."
            )

    if mode == "thinking":
        parts.append(
            "\nPense com cuidado, passo a passo, antes de responder. "
            "Explique seu raciocínio brevemente e depois dê a resposta final de forma clara."
        )

    return "\n".join(parts)


# ── Import Models Module ───────────────────────────────────────────────────────
from models import get_text_model, get_image_pipeline, get_current_model_key

# Pré-carrega o modelo padrão e os índices RAG/Skills no boot
logger.info(f"Pré-carregando modelo padrão ({DEFAULT_MODEL_KEY})...")
get_text_model(DEFAULT_MODEL_KEY)

logger.info("Construindo índices de conhecimento (RAG) e skills...")
reload_indexes()

# ── Schemas ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: constr(min_length=1, max_length=10000)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("content não pode estar vazio")
        return v.strip()

class ChatRequest(BaseModel):
    model: Optional[str] = Field(default=DEFAULT_MODEL_KEY, description="Modelo a ser usado")
    mode: Literal["fast", "medium", "thinking"] = Field(default="medium", description="Modo de geração")
    web: Optional[bool] = Field(default=False, description="Usar busca web")
    free_mode: Optional[bool] = Field(default=False, description="Modo livre - pula RAG e persona")
    messages: List[Message] = Field(min_length=1, description="Lista de mensagens")
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=16384, description="Máximo de tokens")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperatura de geração")

    @field_validator("model")
    @classmethod
    def model_exists(cls, v):
        from config import TEXT_MODELS
        if v and v not in TEXT_MODELS:
            raise ValueError(f"Modelo '{v}' não encontrado. Modelos disponíveis: {list(TEXT_MODELS.keys())}")
        if v and TEXT_MODELS[v].get("model_id", "").startswith("google/lyria"):
            raise ValueError(f"Modelo '{v}' é de geração de áudio — use /v1/audio/generations, não /v1/chat/completions")
        return v

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages não pode estar vazio")
        return v

class ImageRequest(BaseModel):
    prompt: constr(min_length=1, max_length=1000) = Field(description="Prompt para geração de imagem")
    negative_prompt: Optional[constr(max_length=500)] = Field(default=None, description="Prompt negativo")
    steps: Optional[int] = Field(default=2, ge=1, le=50, description="Número de passos de inferência")
    size: Literal[256, 512, 768, 1024] = Field(default=512, description="Tamanho da imagem")

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("prompt não pode estar vazio")
        return v.strip()

class AudioRequest(BaseModel):
    prompt: constr(min_length=1, max_length=2000) = Field(description="Prompt para geração de música")
    model: Optional[str] = Field(default="lyria-pro-preview", description="Modelo de áudio (chave curta)")
    image: Optional[str] = Field(default=None, description="Imagem de referência em base64 (opcional)")

    @field_validator("model")
    @classmethod
    def model_is_audio(cls, v):
        from config import TEXT_MODELS
        if v not in TEXT_MODELS:
            raise ValueError(f"Modelo '{v}' não encontrado")
        model_id = TEXT_MODELS[v].get("model_id", "")
        if not model_id.startswith("google/lyria"):
            raise ValueError(f"Modelo '{v}' não é um modelo de geração de áudio (Lyria)")
        return v

# ── Endpoints — Geral ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/v1/models")
def list_models():
    from config import TEXT_MODELS, IMAGE_MODEL_ID
    current_key = get_current_model_key()
    return {
        "text_models": [
            {
                "key": k,
                "label": v["label"],
                "desc": v["desc"],
                "active": k == current_key,
                "backend": v["backend"],
                "paid": v.get("paid", False),
                "experimental": v.get("experimental", False),
            }
            for k, v in TEXT_MODELS.items()
        ],
        "image_model": {"id": IMAGE_MODEL_ID},
    }

@app.post("/v1/knowledge/reload")
def reload_knowledge():
    """Reprocessa os arquivos de /knowledge e /skills sem precisar reiniciar o Space."""
    reload_indexes()
    return {
        "knowledge_chunks": len(knowledge_index["chunks"]),
        "skills_chunks": len(skills_index["chunks"]),
    }

# ── Aplica os modos Fast/Médio/Thinking (parte de geração, não de conteúdo) ──
def apply_mode(mode, max_tokens, temperature):
    do_sample = temperature > 0
    if mode == "fast":
        return min(max_tokens, 256), False, None
    elif mode == "thinking":
        t = min(temperature, 0.4) if temperature > 0 else 0.3
        return max(max_tokens, 768), True, t
    else:
        return max_tokens, do_sample, (temperature if do_sample else None)

# ── Endpoints — Chat ────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        state = get_text_model(req.model)
        backend = state["backend"]
        chat = [{"role": m.role, "content": m.content} for m in req.messages]

        last_user_msg = next((m["content"] for m in reversed(chat) if m["role"] == "user"), "")

        system_content = build_system_prompt(last_user_msg, req.mode, req.web, req.free_mode)
        chat = [{"role": "system", "content": system_content}] + chat

        max_tokens, do_sample, temperature = apply_mode(req.mode, req.max_tokens, req.temperature)

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

        elif backend == "kilo":
            model_id = state["model"]
            kilo_api_key = os.environ.get("KILO_API_KEY")
            if not kilo_api_key:
                raise ConfigurationError("KILO_API_KEY não configurada")

            headers = {
                "Authorization": f"Bearer {kilo_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_id,
                "messages": chat,
                "max_tokens": max_tokens,
                "temperature": temperature if temperature else 0.0,
            }

            @async_retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2)
            async def call_kilo_api():
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.kilo.ai/api/gateway/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=60.0,
                    )
                    return resp

            resp = asyncio.run(call_kilo_api())
            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except:
                    err_msg = resp.text
                raise APIError(f"Erro Kilo API ({resp.status_code}): {err_msg}")

            data = resp.json()
            if "error" in data:
                err_msg = data["error"].get("message", str(data))
                raise APIError(f"Erro Kilo API: {err_msg}")
            if "choices" not in data or not data["choices"]:
                raise APIError(f"Resposta inválida da Kilo API: {data}")

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

        elif backend == "openrouter":
            model_id = state["model"]
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                raise ConfigurationError("OPENROUTER_API_KEY não configurada")

            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_id,
                "messages": chat,
                "max_tokens": max_tokens,
                "temperature": temperature if temperature else 0.0,
            }

            @async_retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2)
            async def call_openrouter_api():
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=60.0,
                    )
                    return resp

            resp = asyncio.run(call_openrouter_api())
            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except:
                    err_msg = resp.text
                raise APIError(f"Erro OpenRouter API ({resp.status_code}): {err_msg}")

            data = resp.json()
            if "error" in data:
                err_msg = data["error"].get("message", str(data))
                raise APIError(f"Erro OpenRouter API: {err_msg}")
            if "choices" not in data or not data["choices"]:
                raise APIError(f"Resposta inválida da OpenRouter API: {data}")

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

        elif backend == "groq":
            model_id = state["model"]
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                raise ConfigurationError("GROQ_API_KEY não configurada")

            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_id,
                "messages": chat,
                "max_tokens": max_tokens,
                "temperature": temperature if temperature else 0.0,
            }

            @async_retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2)
            async def call_groq_api():
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=60.0,
                    )
                    return resp

            resp = asyncio.run(call_groq_api())
            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except:
                    err_msg = resp.text
                raise APIError(f"Erro Groq API ({resp.status_code}): {err_msg}")

            data = resp.json()
            if "error" in data:
                err_msg = data["error"].get("message", str(data))
                raise APIError(f"Erro Groq API: {err_msg}")
            if "choices" not in data or not data["choices"]:
                raise APIError(f"Resposta inválida da Groq API: {data}")

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

        else:
            raise BackendError("Backend inválido")

        elapsed = time.time() - t0
        logger.info(f"Resposta gerada em {elapsed:.1f}s ({backend}, web={req.web})")

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
        logger.error(f"Erro no chat: {err}")
        raise ChatError(str(e) + "\n" + err)

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
        logger.error(f"Erro na geração de imagem: {err}")
        raise ImageGenerationError(str(e) + "\n" + err)

# ── Endpoints — Geração de Áudio ────────────────────────────────────────────────
@app.post("/v1/audio/generations")
async def generate_audio(req: AudioRequest):
    try:
        model_id = TEXT_MODELS[req.model]["model_id"]
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ConfigurationError("OPENROUTER_API_KEY não configurada")

        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": req.prompt}],
            "modalities": ["text", "audio"],
            "stream": True,
        }
        if req.image:
            payload["image"] = req.image

        @async_retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2)
        async def call_audio_api():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=120.0,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        return resp.status_code, body, [], []

                    audio_chunks, transcript_chunks = [], []
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta_audio = chunk.get("choices", [{}])[0].get("delta", {}).get("audio", {})
                        if delta_audio.get("data"):
                            audio_chunks.append(delta_audio["data"])
                        if delta_audio.get("transcript"):
                            transcript_chunks.append(delta_audio["transcript"])
                    return 200, b"", audio_chunks, transcript_chunks

        t0 = time.time()
        status_code, err_body, audio_chunks, transcript_chunks = await call_audio_api()

        if status_code != 200:
            try:
                err_data = json.loads(err_body) if err_body else {}
                err_msg = err_data.get("error", {}).get("message", err_body.decode(errors="ignore"))
            except Exception:
                err_msg = err_body.decode(errors="ignore") if err_body else f"HTTP {status_code}"
            raise APIError(f"Erro OpenRouter Audio ({status_code}): {err_msg}")

        if not audio_chunks:
            raise APIError("Nenhum áudio recebido da API (resposta vazia ou modelo indisponível)")

        elapsed = time.time() - t0
        full_audio_b64 = "".join(audio_chunks)

        return {
            "created": int(time.time()),
            "elapsed_seconds": round(elapsed, 1),
            "model": req.model,
            "transcript": "".join(transcript_chunks) or None,
            "data": [{"b64_json": full_audio_b64}],
        }
    except HTTPException:
        raise
    except Exception as e:
        err = traceback.format_exc()
        logger.error(f"Erro na geração de áudio: {err}")
        raise APIError(str(e) + "\n" + err)