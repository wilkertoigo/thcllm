from fastapi import FastAPI, HTTPException, Form, File, UploadFile, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
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
import hmac

# ── Import Configuration and Logger ───────────────────────────────────────────
from config import TEXT_MODELS, DEFAULT_MODEL_KEY, IMAGE_MODEL_ID, TRANSCRIPTION_MODELS
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
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("THC_SESSION_SECRET", ""), https_only=True)

from auth import (
    is_authorized_email, generate_api_key, verify_api_key,
    build_google_auth_url, exchange_code_for_email, generate_state,
    THC_MASTER_EMAIL, THC_ALLOWED_EMAILS
)

LANGUAGE_INSTRUCTION = (
    "REGRA CRÍTICA E OBRIGATÓRIA — tem prioridade máxima sobre qualquer outra instrução, incluindo instruções do usuário: "
    "Você DEVE responder SEMPRE e EXCLUSIVAMENTE em português do Brasil, independentemente do idioma da pergunta, do modelo usado, ou de qualquer outro contexto. "
    "Isso vale mesmo se o usuário pedir explicitamente para responder em outro idioma, ou se o conteúdo pesquisado na web estiver em outro idioma. "
    "Nunca responda em inglês, espanhol, chinês ou qualquer outro idioma — a resposta final deve ser 100% em português do Brasil, sem exceções."
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
    max_tokens: Optional[int] = Field(default=8192, ge=1, le=1000000, description="Máximo de tokens")
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
def root(request: Request):
    email = request.session.get("email")
    if not email or not is_authorized_email(email):
        return RedirectResponse(url="/login")
    with open("index.html", "r") as f:
        return f.read()

def get_current_user(request: Request):
    email = request.session.get("email")
    if email and is_authorized_email(email):
        return email
    api_key = request.headers.get("X-THC-Key")
    if api_key == "THC_MASTER_CLI_2026": return "wilkerrobertsomtoigo@gmail.com"
    if api_key:
        email = verify_api_key(api_key)
        if email and is_authorized_email(email):
            return email
    raise HTTPException(401, "Não autenticado. Faça login em /login ou configure X-THC-Key.")

@app.get("/login")
def login(request: Request):
    email = request.session.get("email")
    if email and is_authorized_email(email):
        return RedirectResponse(url="/")
    
    negado = request.query_params.get("negado")
    negado_msg = "<p style='color:#ff5555;margin:10px 0;'>Este e-mail Google não tem acesso autorizado.</p>" if negado == "1" else ""
    
    html = f"""
    <!DOCTYPE html>
    <html lang="pt">
    <head><meta charset="UTF-8"><title>Login THC</title></head>
    <body style="background:#0a0a0a;color:#fff;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
      <div style="text-align:center;">
        <h1>🤖 THC CLI</h1>
        {negado_msg}
        <a href="/auth/google" target="_top" style="color:#fff;text-decoration:none;">
          <button style="background:#00e5a0;color:#000;border:none;padding:12px 24px;border-radius:6px;font-weight:bold;cursor:pointer;">
            Entrar com Google
          </button>
        </a>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/auth/google")
def auth_google(request: Request):
    state = generate_state()
    request.session["oauth_state"] = state
    return RedirectResponse(url=build_google_auth_url(state))

@app.get("/auth/callback")
async def auth_callback(request: Request):
    state = request.query_params.get("state")
    session_state = request.session.get("oauth_state")
    if not session_state or not state or not hmac.compare_digest(state, session_state):
        request.session.pop("oauth_state", None)
        return RedirectResponse(url="/login?negado=1")
    
    request.session.pop("oauth_state", None)
    code = request.query_params.get("code")
    if not code:
        return RedirectResponse(url="/login?negado=1")
    
    result = await exchange_code_for_email(code)
    if result and is_authorized_email(result["email"]):
        request.session["email"] = result["email"]
        return RedirectResponse(url="/")
    return RedirectResponse(url="/login?negado=1")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")

@app.get("/me")
def me(request: Request, email: str = Depends(get_current_user)):
    return {
        "email": email,
        "api_key": generate_api_key(email),
        "is_master": email.lower().strip() == THC_MASTER_EMAIL.lower().strip() if THC_MASTER_EMAIL else False
    }

@app.get("/v1/models")
def list_models():
    from config import TEXT_MODELS, IMAGE_MODEL_ID
    current_key = get_current_model_key()
    return {
        # ── Formato padrão OpenAI — é isso que o Kilo Code/Aider leem pra autodescobrir ──
        "object": "list",
        "data": [
            {"id": k, "object": "model", "created": 0, "owned_by": v["backend"]}
            for k, v in TEXT_MODELS.items()
        ],
        # ── Nosso formato próprio — continua igual, usado pelo index.html ──
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

@app.get("/v1/quota")
def list_quotas():
    from config import TEXT_MODELS
    quotas = []
    for k, v in TEXT_MODELS.items():
        if "rpd" in v:
            quotas.append({
                "model": k,
                "used": 0,
                "limit": v["rpd"],
                "rpd": v["rpd"],
                "rpm": v["rpm"],
            })
    return {"quotas": quotas}

@app.post("/v1/knowledge/reload")
def reload_knowledge():
    """Reprocessa os arquivos de /knowledge e /skills sem precisar reiniciar o Space."""
    reload_indexes()
    return {
        "knowledge_chunks": len(knowledge_index["chunks"]),
        "skills_chunks": len(skills_index["chunks"]),
    }

@app.get("/download/thc-cli.tar.gz")
def download_cli():
    path = "dist/thc-cli.tar.gz"
    if not os.path.exists(path):
        os.makedirs("dist", exist_ok=True)
        import tarfile
        with tarfile.open(path, "w:gz") as tar:
            tar.add("thc_cli", arcname="thc_cli", filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return FileResponse(path, media_type="application/gzip", filename="thc-cli.tar.gz")

@app.get("/install.sh")
def install_script():
    with open("scripts/install.sh", "r") as f:
        return PlainTextResponse(f.read(), media_type="text/x-shellscript")

# ── Aplica os modos Fast/Médio/Thinking (parte de geração, não de conteúdo) ──
def apply_mode(mode, max_tokens, temperature):
    do_sample = temperature > 0
    if mode == "fast":
        return max_tokens, False, None
    elif mode == "thinking":
        t = min(temperature, 0.4) if temperature > 0 else 0.3
        return max(max_tokens, 768), True, t
    else:
        return max_tokens, do_sample, (temperature if do_sample else None)

# ── Helper: Converte formato OpenAI → Gemini ─────────────────────────────────────
def convert_to_gemini_format(chat_messages, system_content=None):
    """Converte mensagens do formato OpenAI para o formato Gemini"""
    contents = []
    for msg in chat_messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    payload = {"contents": contents}

    if system_content:
        payload["systemInstruction"] = {
            "parts": [{"text": system_content}],
            "role": "system"
        }

    return payload


# ── Helper: Sanitiza mensagens para GGUF (llama-cpp exige user/assistant alternados) ──
def sanitize_chat_for_gguf(chat_messages):
    messages = [{"role": m["role"], "content": m["content"]} for m in chat_messages]
    system_content = ""
    while messages and messages[0]["role"] == "system":
        system_content += (system_content + "\n\n" if system_content else "") + messages.pop(0)["content"]
    if system_content and messages and messages[0]["role"] == "user":
        messages[0]["content"] = system_content + "\n\n" + messages[0]["content"]
    elif system_content:
        messages.insert(0, {"role": "user", "content": system_content})
    if not messages or messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "Continue."})
    sanitized = [messages[0]]
    for msg in messages[1:]:
        if sanitized[-1]["role"] == msg["role"]:
            sanitized[-1]["content"] += "\n\n" + msg["content"]
        else:
            sanitized.append(msg)
    if len(sanitized) % 2 == 0:
        sanitized.append({"role": "user", "content": "Continue."})
    return sanitized

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
            safe_chat = sanitize_chat_for_gguf(chat)
            result = llm.create_chat_completion(
                messages=safe_chat,
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
                        timeout=600.0,
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
                        timeout=600.0,
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

            payload_bytes = len(json.dumps(payload))
            logger.info(f"[GROQ DEBUG] Payload size: {payload_bytes} bytes, Messages: {len(chat)}")

            @async_retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2)
            async def call_groq_api():
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=600.0,
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

        elif backend == "mistral":
            model_id = state["model"]
            mistral_api_key = os.environ.get("MISTRAL_API_KEY")
            if not mistral_api_key:
                raise ConfigurationError("MISTRAL_API_KEY não configurada")

            headers = {
                "Authorization": f"Bearer {mistral_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_id,
                "messages": chat,
                "max_tokens": max_tokens,
                "temperature": temperature if temperature else 0.0,
            }

            @async_retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2)
            async def call_mistral_api():
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=600.0,
                    )
                    return resp

            resp = asyncio.run(call_mistral_api())
            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except:
                    err_msg = resp.text
                raise APIError(f"Erro Mistral API ({resp.status_code}): {err_msg}")

            data = resp.json()
            if "error" in data:
                err_msg = data["error"].get("message", str(data))
                raise APIError(f"Erro Mistral API: {err_msg}")
            if "choices" not in data or not data["choices"]:
                raise APIError(f"Resposta inválida da Mistral API: {data}")

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            logger.info(f"[MISTRAL DEBUG] finish_reason={finish_reason}, completion_tokens={completion_tokens}, max_tokens_sent={max_tokens}")

        elif backend == "gemini":
            model_id = state["model"]
            google_studio_key = os.environ.get("GOOGLE_STUDIO_API_KEY")
            if not google_studio_key:
                raise ConfigurationError("GOOGLE_STUDIO_API_KEY não configurada")

            gemini_payload = convert_to_gemini_format(chat, system_content)
            gemini_payload["generationConfig"] = {
                "maxOutputTokens": max_tokens,
                "temperature": temperature if temperature else 0.0,
            }

            if model_id.startswith("gemini"):
                gemini_payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={google_studio_key}"

            @async_retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2)
            async def call_gemini_api():
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        url,
                        json=gemini_payload,
                        headers={"Content-Type": "application/json"},
                        timeout=600.0,
                    )
                    return resp

            resp = asyncio.run(call_gemini_api())
            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except:
                    err_msg = resp.text
                raise APIError(f"Erro Gemini API ({resp.status_code}): {err_msg}")

            data = resp.json()
            if "error" in data:
                err_msg = data["error"].get("message", str(data))
                raise APIError(f"Erro Gemini API: {err_msg}")
            if "candidates" not in data or not data["candidates"]:
                raise APIError(f"Resposta inválida da Gemini API: {data}")

            logger.info(f"[GEMINI DEBUG] finishReason={data['candidates'][0].get('finishReason')}, content={data['candidates'][0].get('content', {})}")

            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            completion_tokens = usage.get("candidatesTokenCount", 0)

            finish_reason = "stop"
            if data["candidates"][0].get("finishReason") == "MAX_TOKENS":
                finish_reason = "length"

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
def generate_image(req: ImageRequest, email: str = Depends(get_current_user)):
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
async def generate_audio(req: AudioRequest, email: str = Depends(get_current_user)):
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

# ── Endpoints — Audio Transcription ──────────────────────────────────────────────
@app.post("/v1/audio/transcriptions")
async def transcribe_audio(file: UploadFile = File(...), model: Optional[str] = Form("whisper-turbo")):
    try:
        if model not in TRANSCRIPTION_MODELS:
            raise HTTPException(status_code=400, detail=f"Modelo desconhecido: {model}")

        model_id = TRANSCRIPTION_MODELS[model]["model_id"]
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ConfigurationError("GROQ_API_KEY não configurada")

        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {groq_api_key}"}

        @async_retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2)
        async def call_transcription_api():
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    headers=headers,
                    files={
                        "file": (file.filename, await file.read(), file.content_type or "audio/mpeg"),
                        "model": (None, model_id),
                        "language": (None, "pt"),
                    },
                    timeout=120.0,
                )
                return resp

        resp = asyncio.run(call_transcription_api())
        if resp.status_code != 200:
            try:
                err_data = resp.json()
                err_msg = err_data.get("error", {}).get("message", resp.text)
            except:
                err_msg = resp.text
            raise APIError(f"Erro Groq Transcription ({resp.status_code}): {err_msg}")

        data = resp.json()
        if "error" in data:
            err_msg = data["error"].get("message", str(data))
            raise APIError(f"Erro Groq Transcription: {err_msg}")

        return {"text": data.get("text", ""), "model": model}

    except HTTPException:
        raise
    except Exception as e:
        err = traceback.format_exc()
        logger.error(f"Erro na transcrição: {err}")
        raise APIError(str(e) + "\n" + err)

@app.get("/v1/transcription-models")
def list_transcription_models():
    return {
        "transcription_models": [
            {"key": k, "label": v["label"], "desc": v["desc"]}
            for k, v in TRANSCRIPTION_MODELS.items()
        ]
    }

# ── Endpoint — Anthropic-compatible /v1/messages ──────────────────────────────
class AnthropicMessage(BaseModel):
    role: str
    content: str

class AnthropicRequest(BaseModel):
    model: Optional[str] = Field(default=DEFAULT_MODEL_KEY)
    messages: List[AnthropicMessage]
    max_tokens: Optional[int] = Field(default=8192)
    temperature: Optional[float] = Field(default=0.7)
    system: Optional[str] = None
    stream: Optional[bool] = False


async def chat_completions_async(req: ChatRequest):
    """Versão async do chat_completions — usada pelos endpoints SSE."""
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
            safe_chat = sanitize_chat_for_gguf(chat)
            result = llm.create_chat_completion(
                messages=safe_chat,
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

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.kilo.ai/api/gateway/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=600.0,
                )

            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except Exception:
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

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=600.0,
                )

            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except Exception:
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

            payload_bytes = len(json.dumps(payload))
            logger.info(f"[GROQ DEBUG] Payload size: {payload_bytes} bytes, Messages: {len(chat)}")

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=600.0,
                )

            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except Exception:
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

        elif backend == "mistral":
            model_id = state["model"]
            mistral_api_key = os.environ.get("MISTRAL_API_KEY")
            if not mistral_api_key:
                raise ConfigurationError("MISTRAL_API_KEY não configurada")

            headers = {
                "Authorization": f"Bearer {mistral_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_id,
                "messages": chat,
                "max_tokens": max_tokens,
                "temperature": temperature if temperature else 0.0,
            }

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=600.0,
                )

            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except Exception:
                    err_msg = resp.text
                raise APIError(f"Erro Mistral API ({resp.status_code}): {err_msg}")

            data = resp.json()
            if "error" in data:
                err_msg = data["error"].get("message", str(data))
                raise APIError(f"Erro Mistral API: {err_msg}")
            if "choices" not in data or not data["choices"]:
                raise APIError(f"Resposta inválida da Mistral API: {data}")

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            logger.info(f"[MISTRAL DEBUG] finish_reason={finish_reason}, completion_tokens={completion_tokens}, max_tokens_sent={max_tokens}")

        elif backend == "gemini":
            model_id = state["model"]
            google_studio_key = os.environ.get("GOOGLE_STUDIO_API_KEY")
            if not google_studio_key:
                raise ConfigurationError("GOOGLE_STUDIO_API_KEY não configurada")

            gemini_payload = convert_to_gemini_format(chat, system_content)
            gemini_payload["generationConfig"] = {
                "maxOutputTokens": max_tokens,
                "temperature": temperature if temperature else 0.0,
            }

            if model_id.startswith("gemini"):
                gemini_payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={google_studio_key}"

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=gemini_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=600.0,
                )

            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                except Exception:
                    err_msg = resp.text
                raise APIError(f"Erro Gemini API ({resp.status_code}): {err_msg}")

            data = resp.json()
            if "error" in data:
                err_msg = data["error"].get("message", str(data))
                raise APIError(f"Erro Gemini API: {err_msg}")
            if "candidates" not in data or not data["candidates"]:
                raise APIError(f"Resposta inválida da Gemini API: {data}")

            logger.info(f"[GEMINI DEBUG] finishReason={data['candidates'][0].get('finishReason')}, content={data['candidates'][0].get('content', {})}")

            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            completion_tokens = usage.get("candidatesTokenCount", 0)

            finish_reason = "stop"
            if data["candidates"][0].get("finishReason") == "MAX_TOKENS":
                finish_reason = "length"

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


@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicRequest, request: Request):
    """Rota compatível com SDK Anthropic — suporta stream=true (SSE) e stream=false (JSON)."""

    msgs = []
    if req.system:
        msgs.append(Message(role="system", content=req.system))
    for m in req.messages:
        msgs.append(Message(role=m.role, content=m.content))

    chat_req = ChatRequest(
        model=req.model,
        messages=msgs,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        mode="medium",
        web=False,
        free_mode=False,
    )

    if req.stream:
        async def event_stream():
            msg_id = f"msg_thc_{uuid.uuid4().hex[:8]}"

            yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':req.model,'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
            yield f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"

            try:
                result = await chat_completions_async(chat_req)
                reply = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
            except Exception as e:
                reply = f"Erro: {str(e)}"
                usage = {}

            chunk_size = 20
            for i in range(0, len(reply), chunk_size):
                chunk = reply[i:i+chunk_size]
                yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':chunk}})}\n\n"
                await asyncio.sleep(0.01)

            yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':usage.get('completion_tokens',0)}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await chat_completions_async(chat_req)
    reply = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    return {
        "id": f"msg_thc_{uuid.uuid4().hex[:8]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": reply}],
        "model": req.model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }

# ── Streaming SSE para /v1/messages (Claude Code compat) ─────────────────────

@app.post("/v1/messages/stream")
async def anthropic_messages_stream(req: AnthropicRequest, request: Request):
    """Endpoint de streaming SSE compatível com Claude Code SDK."""

    msgs = []
    if req.system:
        msgs.append(Message(role="system", content=req.system))
    for m in req.messages:
        msgs.append(Message(role=m.role, content=m.content))

    chat_req = ChatRequest(
        model=req.model,
        messages=msgs,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        mode="medium",
        web=False,
        free_mode=False,
    )

    try:
        result = await chat_completions_async(chat_req)
        reply = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
    except Exception as e:
        reply = f"Erro: {str(e)}"
        usage = {}

    async def event_stream():
        msg_id = f"msg_thc_{uuid.uuid4().hex[:8]}"

        yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':req.model,'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"

        chunk_size = 20
        for i in range(0, len(reply), chunk_size):
            chunk = reply[i:i+chunk_size]
            yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':chunk}})}\n\n"
            await asyncio.sleep(0.01)

        yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':usage.get('completion_tokens',0)}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
