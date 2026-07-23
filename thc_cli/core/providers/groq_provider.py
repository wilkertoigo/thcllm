import httpx
from typing import Optional

from .base import BaseProvider


GROQ_MODELS = [
    {"key": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B", "desc": "Modelo versatile da Meta via Groq"},
    {"key": "openai/gpt-oss-120b", "label": "GPT-OSS 120B", "desc": "Modelo grande open source via Groq"},
    {"key": "openai/gpt-oss-20b", "label": "GPT-OSS 20B", "desc": "Modelo compacto open source via Groq"},
    {"key": "qwen/qwen3.6-27b", "label": "Qwen3 6-27B", "desc": "Modelo Qwen via Groq"},
    {"key": "llama-3.1-8b-instant", "label": "Llama 3.1 8B Instant", "desc": "Modelo rápido e leve via Groq"},
    {"key": "groq/compound", "label": "Groq Compound", "desc": "Modelo compound Groq"},
]


class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY não configurada. Configure em ~/.thcrc ou variável de ambiente.")
        self.api_key = api_key
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def chat_completion(self, messages: list[dict], model: str, max_tokens: int = 8192,
                        temperature: float = 0.7) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def list_models(self) -> list[dict]:
        return list(GROQ_MODELS)
