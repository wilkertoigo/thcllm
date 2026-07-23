import httpx

from .base import BaseProvider


MISTRAL_MODELS = [
    {"key": "mistral-small-latest", "label": "Mistral Small Latest", "desc": "Modelo pequeno e rápido Mistral"},
    {"key": "open-mistral-nemo", "label": "Open Mistral Nemo", "desc": "Modelo open Nemo via Mistral"},
    {"key": "ministral-8b-latest", "label": "Ministral 8B", "desc": "Modelo leve 8B Mistral"},
    {"key": "ministral-3b-latest", "label": "Ministral 3B", "desc": "Modelo ultra leve 3B Mistral"},
    {"key": "devstral-latest", "label": "Devstral Latest", "desc": "Modelo para código/developers Mistral"},
    {"key": "magistral-small-latest", "label": "Magistral Small", "desc": "Modelo pequeno magistral Mistral"},
]


class MistralProvider(BaseProvider):
    name = "mistral"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("MISTRAL_API_KEY não configurada. Configure em ~/.thcrc ou variável de ambiente.")
        self.api_key = api_key
        self.endpoint = "https://api.mistral.ai/v1/chat/completions"

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
        return list(MISTRAL_MODELS)
