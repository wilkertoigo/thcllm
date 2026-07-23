import httpx

from .base import BaseProvider


OPENROUTER_MODELS = [
    {"key": "google/gemma-4-31b-it:free", "label": "Gemma 4 31B Free", "desc": "Modelo Gemma 4 31B free via OpenRouter"},
    {"key": "nvidia/nemotron-3-nano-30b-a3b:free", "label": "Nemotron 3 Nano 30B Free", "desc": "Modelo Nemotron 3 Nano free via OpenRouter"},
    {"key": "google/gemma-4-26b-a4b-it:free", "label": "Gemma 4 26B Free", "desc": "Modelo Gemma 4 26B free via OpenRouter"},
    {"key": "openai/gpt-oss-20b:free", "label": "GPT-OSS 20B Free", "desc": "Modelo GPT-OSS 20B free via OpenRouter"},
    {"key": "nvidia/nemotron-nano-12b-v2-vl:free", "label": "Nemotron Nano 12B v2 VL Free", "desc": "Modelo Nemotron Nano 12B free via OpenRouter"},
    {"key": "nvidia/nemotron-nano-9b-v2:free", "label": "Nemotron Nano 9B v2 Free", "desc": "Modelo Nemotron Nano 9B free via OpenRouter"},
]


class OpenRouterProvider(BaseProvider):
    name = "openrouter"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY não configurada. Configure em ~/.thcrc ou variável de ambiente.")
        self.api_key = api_key
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

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
        return list(OPENROUTER_MODELS)
