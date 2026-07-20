# ── Models Module ─────────────────────────────────────────────────────────────
"""Módulo para gerenciamento de modelos de linguagem e imagem"""

import torch
import gc
from typing import Dict, Any
from diffusers import AutoPipelineForText2Image
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

# Importar logger e configuração (sem imports circulares)
from logger import logger
from config import TEXT_MODELS, DEFAULT_MODEL_KEY, IMAGE_MODEL_ID

# ── Authentication ───────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# ── Cache de Modelos ───────────────────────────────────────────────────────────
_current: Dict[str, Any] = {"key": None, "tokenizer": None, "model": None, "backend": None}
image_pipeline = None


# ── Text Model Management ─────────────────────────────────────────────────────
def unload_current():
    """Descarrega o modelo atual da memória"""
    if _current.get("model") is not None:
        logger.info(f"Descarregando modelo anterior ({_current['key']})...")
    _current["key"] = None
    _current["tokenizer"] = None
    _current["model"] = None
    _current["backend"] = None
    gc.collect()


def get_text_model(key: str) -> Dict[str, Any]:
    """Carrega e retorna um modelo de texto pelo key"""
    # Importar exceções do módulo exceptions para evitar import circular
    from exceptions import ModelNotFoundError, ModelLoadError, BackendError
    from fastapi import HTTPException
    
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

        elif backend == "openrouter":
            model_id = cfg["model_id"]
            logger.info(f"Backend HTTP (openrouter): {model_id}...")
            _current.update({"key": key, "tokenizer": None, "model": model_id, "backend": "openrouter"})

        elif backend == "groq":
            model_id = cfg["model_id"]
            logger.info(f"Backend HTTP (groq): {model_id}...")
            _current.update({"key": key, "tokenizer": None, "model": model_id, "backend": "groq"})

        elif backend == "mistral":
            model_id = cfg["model_id"]
            logger.info(f"Backend HTTP (mistral): {model_id}...")
            _current.update({"key": key, "tokenizer": None, "model": model_id, "backend": "mistral"})

        else:
            raise BackendError(f"Backend desconhecido: {backend}")

    except HTTPException:
        raise
    except Exception as e:
        unload_current()
        raise ModelLoadError(f"Erro ao carregar {key}: {str(e)}") from e

    logger.info(f"Modelo {key} carregado!")
    return _current


# ── Image Model Management ─────────────────────────────────────────────────────
def get_image_pipeline():
    """Carrega e retorna o pipeline de geração de imagens"""
    global image_pipeline
    if image_pipeline is None:
        logger.info(f"Carregando modelo de imagem {IMAGE_MODEL_ID}...")
        image_pipeline = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=torch.float32,
            token=hf_token,
        )
        image_pipeline.to("cpu")
        logger.info("Modelo de imagem pronto!")
    return image_pipeline


def get_current_model_key() -> str:
    """Retorna a key do modelo atualmente carregado"""
    return _current.get("key", None)
