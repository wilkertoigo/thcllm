import json
import urllib.parse
from typing import Optional

import httpx

from .base import BaseProvider


GEMINI_MODELS = [
    {"key": "gemini-3.5-flash", "label": "Gemini 3.5 Flash", "desc": "Modelo flash Gemini"},
    {"key": "gemini-3.1-flash-lite", "label": "Gemini 3.1 Flash Lite", "desc": "Modelo flash lite Gemini"},
    {"key": "gemini-flash-lite-latest", "label": "Gemini Flash Lite Latest", "desc": "Última versão flash lite Gemini"},
    {"key": "gemini-3-flash-preview", "label": "Gemini 3 Flash Preview", "desc": "Preview do modelo flash Gemini"},
    {"key": "gemma-4-26b-a4b-it", "label": "Gemma 4 26B", "desc": "Modelo Gemma 4 26B via Gemini API"},
    {"key": "gemma-4-31b-it", "label": "Gemma 4 31B", "desc": "Modelo Gemma 4 31B via Gemini API"},
]


def _convert_to_gemini_format(messages: list[dict]) -> dict:
    contents = []
    system_content = None
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_content = content
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({
            "role": gemini_role,
            "parts": [{"text": content}]
        })
    payload = {"contents": contents}
    if system_content:
        payload["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": system_content}]
        }
    return payload


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY não configurada. Configure em ~/.thcrc ou variável de ambiente.")
        self.api_key = api_key
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

    def chat_completion(self, messages: list[dict], model: str, max_tokens: int = 8192,
                        temperature: float = 0.7) -> dict:
        payload = _convert_to_gemini_format(messages)
        payload["generationConfig"] = {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{self.endpoint}/{model}:generateContent?key={self.api_key}"
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def list_models(self) -> list[dict]:
        return list(GEMINI_MODELS)
