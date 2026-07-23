from .base import BaseProvider
from .thc_provider import ThcProvider
from .groq_provider import GroqProvider
from .mistral_provider import MistralProvider
from .gemini_provider import GeminiProvider
from .openrouter_provider import OpenRouterProvider

PROVIDER_CLASSES = {
    "thc": ThcProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}


def get_provider(name: str, config: dict) -> BaseProvider:
    provider_keys = config.get("provider_keys", {})
    if name == "thc":
        return ThcProvider(
            api_key=config.get("api_key"),
            endpoint=config.get("endpoint"),
        )
    cls = PROVIDER_CLASSES.get(name)
    if cls is None:
        valid = ", ".join(sorted(PROVIDER_CLASSES))
        raise ValueError(f"Provider '{name}' desconhecido. Opções válidas: {valid}")
    api_key = provider_keys.get(name) or ""
    return cls(api_key=api_key)
