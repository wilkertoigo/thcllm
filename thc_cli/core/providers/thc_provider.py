from typing import Optional

from ..client import THCClient
from .base import BaseProvider


class ThcProvider(BaseProvider):
    name = "thc"

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.endpoint = endpoint or "https://hulktoigo-thcllm.hf.space"
        self.api_key = api_key
        self.client = THCClient({"endpoint": self.endpoint, "api_key": self.api_key})

    def chat_completion(self, messages: list[dict], model: str, max_tokens: int = 8192,
                        temperature: float = 0.7, mode: str = "medium", web: bool = False) -> dict:
        return self.client.chat_completion(
            messages=messages,
            model=model,
            mode=mode,
            web=web,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def list_models(self) -> list[dict]:
        data = self.client.list_models()
        return data.get("text_models", [])
