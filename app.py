from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
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

# ── Logging Configuration ─────────────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("THC_LLM")

# ── Custom Exceptions ───────────────────────────────────────────────────────────
class THCError(HTTPException):
    """Base exception for THC LLM application"""
    def __init__(self, detail: str, status_code: int = 500):
        super().__init__(status_code=status_code, detail=detail)


class ModelNotFoundError(THCError):
    """Raised when a requested model is not found"""
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=400)


class ModelLoadError(THCError):
    """Raised when a model fails to load"""
    pass


class BackendError(THCError):
    """Raised when a backend operation fails"""
    pass


class ConfigurationError(THCError):
    """Raised when configuration is missing or invalid"""
    pass


class APIError(THCError):
    """Raised when external API calls fail"""
    pass


class ImageGenerationError(THCError):
    """Raised when image generation fails"""
    pass


class ChatError(THCError):
    """Raised when chat completion fails"""
    pass

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
    "hy3-free": {
        "backend": "kilo",
        "model_id": "tencent/hy3:free",
        "label": "Hy3 295B (free)",
        "desc": "🔥 Kilo free • 295B MoE Tencent",
    },
    "nemotron-ultra-free": {
        "backend": "kilo",
        "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "label": "Nemotron Ultra 550B (free)",
        "desc": "🚀 Kilo free • 550B NVIDIA",
    },
    "laguna-free": {
        "backend": "kilo",
        "model_id": "poolside/laguna-m.1:free",
        "label": "Laguna M.1 (free)",
        "desc": "⚡ Kilo free • Poolside",
    },
    "laguna-xs-free": {
        "backend": "kilo",
        "model_id": "poolside/laguna-xs-2.1:free",
        "label": "Laguna XS 2.1 (free)",
        "desc": "⚡ Kilo free • coding agent 33B",
    },
}
DEFAULT_MODEL_KEY = "gemma-1b"

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
def build_system_prompt(user_query, mode, use_web):
    now = datetime.now(ZoneInfo("America/Sao_Paulo"))
    now_str = now.strftime("%A, %d de %B de %Y, %H:%M (horário de Brasília)")

    parts = [
        LANGUAGE_INSTRUCTION,
        f"\nA data e hora atuais são: {now_str}. Esta é a fonte oficial e confiável "
        f"para qualquer pergunta sobre data ou hora atual — nunca invente outra data "
        f"nem use datas do seu treinamento para isso.",
    ]

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


# ── Cache de UM modelo de texto por vez ───────────────────────────────────────
_current = {"key": None, "tokenizer": None, "model": None, "backend": None}

def unload_current():
    if _current.get("model") is not None:
        logger.info(f"Descarregando modelo anterior ({_current['key']})...")
    _current["key"] = None
    _current["tokenizer"] = None
    _current["model"] = None
    _current["backend"] = None
    gc.collect()

def get_text_model(key: str):
    if key not in TEXT_MODELS:
        raise ModelNotFoundError(f"Modelo desconhecido: {key}")

    if _current.get("key") == key and _current.get("model") is not None:
        return _current

    unload_current()

    cfg = TEXT_MODELS[key]
    backend = cfg["backend"]

    try:
        if backend == "transformers":
            model_id = cfg["id"]
            logger.info(f"Carregando modelo (transformers): {model_id}...")
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
            logger.info(f"Carregando GGUF: {repo}/{filename}...")

            llm = Llama.from_pretrained(
                repo_id=repo,
                filename=filename,
                n_ctx=8192,
                n_threads=2,
                verbose=False,
            )
            _current.update({"key": key, "tokenizer": None, "model": llm, "backend": "gguf"})

        elif backend == "kilo":
            model_id = cfg["model_id"]
            logger.info(f"Backend HTTP (kilo): {model_id}...")
            _current.update({"key": key, "tokenizer": None, "model": model_id, "backend": "kilo"})

        else:
            raise BackendError(f"Backend desconhecido: {backend}")

    except HTTPException:
        raise
    except Exception as e:
        unload_current()
        raise ModelLoadError(f"Erro ao carregar {key}: {str(e)}") from e

    logger.info(f"Modelo {key} carregado!")
    return _current

# Pré-carrega o modelo padrão e os índices RAG/Skills no boot
logger.info(f"Pré-carregando modelo padrão ({DEFAULT_MODEL_KEY})...")
get_text_model(DEFAULT_MODEL_KEY)

logger.info("Construindo índices de conhecimento (RAG) e skills...")
reload_indexes()

# ── Modelo de IMAGEM ──────────────────────────────────────────────────────────
IMAGE_MODEL_ID = "stabilityai/sd-turbo"
image_pipeline = None

def get_image_pipeline():
    global image_pipeline
    if image_pipeline is None:
        from diffusers import AutoPipelineForText2Image
        logger.info(f"Carregando modelo de imagem {IMAGE_MODEL_ID}...")
        image_pipeline = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=torch.float32,
            token=hf_token,
        )
        image_pipeline.to("cpu")
        logger.info("Modelo de imagem pronto!")
    return image_pipeline

# ── Schemas ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL_KEY
    mode: Optional[str] = "medium"
    web: Optional[bool] = False
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

        system_content = build_system_prompt(last_user_msg, req.mode, req.web)
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
                raise APIError(f"Erro Kilo API: {resp.text}")

            data = resp.json()
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