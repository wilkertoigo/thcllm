# ── Configuration Module ───────────────────────────────────────────────────────
"""Configuração independente sem imports circulares"""

TEXT_MODELS = {
    # ── KILO BACKEND (do maior para o menor porte) ──────────────────────────
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
    "openrouter-auto-free": {
        "backend": "kilo",
        "model_id": "openrouter/free",
        "label": "OpenRouter Auto (free)",
        "desc": "🎲 Kilo free • roteador automático",
    },

    # ── OPENROUTER BACKEND (texto, do maior para o menor porte) ─────────────
    "gemma4-31b-free": {
        "backend": "openrouter",
        "model_id": "google/gemma-4-31b-it:free",
        "label": "Gemma 4 31B (free)",
        "desc": "🔥 OpenRouter free • Google",
    },
    "nemotron-nano30b-free": {
        "backend": "openrouter",
        "model_id": "nvidia/nemotron-3-nano-30b-a3b:free",
        "label": "Nemotron Nano 30B (free)",
        "desc": "⚡ OpenRouter free • rápido",
    },
    "gemma4-26b-free": {
        "backend": "openrouter",
        "model_id": "google/gemma-4-26b-a4b-it:free",
        "label": "Gemma 4 26B A4B (free)",
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
        "desc": "👁 OpenRouter free • visão + texto",
    },
    "nemotron-nano9b-free": {
        "backend": "openrouter",
        "model_id": "nvidia/nemotron-nano-9b-v2:free",
        "label": "Nemotron Nano 9B (free)",
        "desc": "⚡ OpenRouter free • ultra rápido",
    },

    # ── OPENROUTER BACKEND (áudio — geração de música, PAGO) ────────────────
    "lyria-pro-preview": {
        "backend": "openrouter",
        "model_id": "google/lyria-3-pro-preview",
        "label": "Lyria 3 Pro (áudio)",
        "desc": "🎵 OpenRouter • geração de música (pago: $0.08/música)",
        "paid": True,
    },
    "lyria-clip-preview": {
        "backend": "openrouter",
        "model_id": "google/lyria-3-clip-preview",
        "label": "Lyria 3 Clip (áudio)",
        "desc": "🎵 OpenRouter • geração de música curta (pago: $0.04/clipe)",
        "paid": True,
    },

    # ── GROQ BACKEND ────────────────────────────────────────────────────────────────────────────────
    "llama33-70b-groq": {
        "backend": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "label": "Llama 3.3 70B (Groq)",
        "desc": "⚡ Groq free • ultra rápido, LPU",
    },
    "gptoss120b-groq": {
        "backend": "groq",
        "model_id": "openai/gpt-oss-120b",
        "label": "GPT-OSS 120B (Groq)",
        "desc": "🔥 Groq free • open-weight OpenAI",
    },
    "gptoss20b-groq": {
        "backend": "groq",
        "model_id": "openai/gpt-oss-20b",
        "label": "GPT-OSS 20B (Groq)",
        "desc": "⚡ Groq free • rápido",
    },
    "qwen36-27b-groq": {
        "backend": "groq",
        "model_id": "qwen/qwen3.6-27b",
        "label": "Qwen3.6 27B (Groq)",
        "desc": "🚀 Groq free • Alibaba via LPU",
    },
    "llama31-8b-groq": {
        "backend": "groq",
        "model_id": "llama-3.1-8b-instant",
        "label": "Llama 3.1 8B (Groq)",
        "desc": "⚡ Groq free • instant",
    },
    "compound-groq": {
        "backend": "groq",
        "model_id": "groq/compound",
        "label": "Compound (Groq)",
        "desc": "🛠️ Groq free • agente com tool-use nativo",
    },

    # ── MISTRAL BACKEND ────────────────────────────────────────────────────────────────────────────────
    "mistral-small-mst": {
        "backend": "mistral",
        "model_id": "mistral-small-latest",
        "label": "Mistral Small (free)",
        "desc": "🇫🇷 Mistral free • geral",
    },
    "open-nemo-mst": {
        "backend": "mistral",
        "model_id": "open-mistral-nemo",
        "label": "Mistral Nemo (free)",
        "desc": "🇫🇷 Mistral free • open-weight",
    },
    "ministral8b-mst": {
        "backend": "mistral",
        "model_id": "ministral-8b-latest",
        "label": "Ministral 8B (free)",
        "desc": "⚡ Mistral free • rápido",
    },
    "ministral3b-mst": {
        "backend": "mistral",
        "model_id": "ministral-3b-latest",
        "label": "Ministral 3B (free)",
        "desc": "⚡ Mistral free • ultra rápido",
    },
    "devstral-mst": {
        "backend": "mistral",
        "model_id": "devstral-latest",
        "label": "Devstral (free)",
        "desc": "💻 Mistral free • especialista código",
    },
    "magistral-small-mst": {
        "backend": "mistral",
        "model_id": "magistral-small-latest",
        "label": "Magistral Small (free)",
        "desc": "🧠 Mistral free • reasoning",
    },

    # ── GEMINI BACKEND ────────────────────────────────────────────────────────────────────────
    "gemini25-flash": {
        "backend": "gemini",
        "model_id": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "desc": "✨ Google free • rápido e multimodal",
    },
    "gemini20-flash": {
        "backend": "gemini",
        "model_id": "gemini-2.0-flash",
        "label": "Gemini 2.0 Flash",
        "desc": "✨ Google free • estável",
    },
    "gemini25-flash-lite": {
        "backend": "gemini",
        "model_id": "gemini-2.5-flash-lite",
        "label": "Gemini 2.5 Flash Lite",
        "desc": "⚡ Google free • leve, cota maior",
    },
    "gemini20-flash-lite": {
        "backend": "gemini",
        "model_id": "gemini-2.0-flash-lite",
        "label": "Gemini 2.0 Flash Lite",
        "desc": "⚡ Google free • leve",
    },

    # ── TRANSFORMERS BACKEND ─────────────────────────────────────────────────
    "gemma-1b": {
        "backend": "transformers",
        "id": "google/gemma-3-1b-it",
        "label": "Gemma 3 1B",
        "desc": "Rápido • conversação geral",
    },
    "qwen-coder-3b": {
        "backend": "transformers",
        "id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "label": "Qwen2.5-Coder 3B",
        "desc": "Especialista em código",
    },

    # ── GGUF BACKEND ─────────────────────────────────────────────────────────
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
