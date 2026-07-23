import json
import os
from pathlib import Path

DEFAULT_ENDPOINT = "https://hulktoigo-thcllm.hf.space"
CONFIG_PATH = Path.home() / ".thcrc"
DEFAULT_PROVIDER_FALLBACK_ORDER = ["openrouter", "mistral", "groq", "gemini"]
_VALID_PROVIDERS = {"thc", "groq", "mistral", "gemini", "openrouter"}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            file_config = json.load(f)
    else:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_config = {}
        with open(CONFIG_PATH, "w") as f:
            json.dump({"endpoint": DEFAULT_ENDPOINT}, f)
    
    endpoint = os.environ.get("THC_ENDPOINT") or file_config.get("endpoint") or DEFAULT_ENDPOINT
    api_key = os.environ.get("THC_API_KEY") or file_config.get("api_key")
    provider = os.environ.get("THC_PROVIDER") or file_config.get("provider", DEFAULT_PROVIDER_FALLBACK_ORDER[0])
    provider_keys = dict(file_config.get("provider_keys", {}))
    for key_name, env_name in [
        ("groq", "GROQ_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ]:
        if env_name in os.environ:
            provider_keys[key_name] = os.environ[env_name]
    
    raw_fallback_order = file_config.get("provider_fallback_order", DEFAULT_PROVIDER_FALLBACK_ORDER)
    provider_fallback_order = [p for p in raw_fallback_order if p in _VALID_PROVIDERS]
    
    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "provider": provider,
        "provider_keys": provider_keys,
        "provider_fallback_order": provider_fallback_order,
    }