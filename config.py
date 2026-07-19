# ── Configuration Module ───────────────────────────────────────────────────────
"""Configuração independente sem imports circulares"""

TEXT_MODELS = {
    # ── KILO BACKEND (do maior para o menor porte) ────────────────────────────────────────
    "nemotron-ultra-free": {
        "backend": "kilo",
        "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "label": "Nemotron Ultra 550B (free)",
        "desc": "🚀 Kilo free • 550B NVIDIA",
    },
    "hy3-free": {
        "backend": "kilo",
        "model_id": "tencent/hy3:free",
        "label": "Hy3 295B (free)",
        "desc": "🔥 Kilo free • 295B MoE Tencent",
    },
    "nemotron-super-free": {
        "backend": "kilo",
        "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "label": "Nemotron Super 120B (free)",
        "desc": "🚀 Kilo free • 120B NVIDIA",
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
    "nemotron-nano-omni-free": {
        "backend": "kilo",
        "model_id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "label": "Nemotron Nano Omni 30B (free)",
        "desc": "⚡ Kilo free • multimodal + reasoning",
    },
    "north-mini-code-free": {
        "backend": "kilo",
        "model_id": "cohere/north-mini-code:free",
        "label": "North Mini Code (free)",
        "desc": "💻 Kilo free • Cohere coding",
    },
    "gemma4-31b-free": {
        "backend": "openrouter",
        "model_id": "google/gemma-4-31b-it:free",
        "label": "Gemma 4 31B (free)",
        "desc": "🔥 OpenRouter free • Google",
    },
    "gptoss-20b-free": {
        "backend": "openrouter",
        "model_id": "openai/gpt-oss-20b:free",
        "label": "GPT-OSS 20B (free)",
        "desc": "🤖 OpenRouter free • OpenAI open-weight",
    },
    "nemotron-vl-free": {
        "backend": "openrouter",
        "model_id": "nvidia/nemotron-nano-12b-v2-vl:free",
        "label": "Nemotron Nano 12B VL (free)",
        "desc": "👁️ OpenRouter free • visão + texto",
    },
    "openrouter-auto-free": {
        "backend": "kilo",
        "model_id": "openrouter/free",
        "label": "OpenRouter Auto (free)",
        "desc": "🎲 Kilo free • roteador automático",
    },
    # ── TRANSFORMERS BACKEND ──────────────────────────────────────────────────────────────
    "gemma-1b": {
        "backend": "transformers",
        "id": "google/gemma-3-1b-it",
        "label": "Gemma 3 1B",
        "desc": "Rápido • conversação geral",
    },
    # ── GGUF BACKEND ────────────────────────────────────────────────────────────────────────
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
IMAGE_MODEL_ID = "stabilityai/sd-turbo"