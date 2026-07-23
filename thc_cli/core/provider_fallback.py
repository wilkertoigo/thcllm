from typing import Optional, Callable, List

from .providers import get_provider, PROVIDER_CLASSES
from .providers.base import ProviderRateLimitError


def chat_completion_with_fallback(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    config: dict,
    fallback_order: list[str],
    provider_name: str,
    on_fallback: Optional[Callable[[str, str], None]] = None,
) -> tuple[dict, str]:
    tried: List[str] = []
    order = [provider_name] + [p for p in fallback_order if p != provider_name]
    last_error: Optional[ProviderRateLimitError] = None

    for name in order:
        if name in tried:
            continue
        tried.append(name)
        try:
            provider = get_provider(name, config)
            result = provider.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if name != provider_name and on_fallback:
                on_fallback(provider_name, name)
            return result, name
        except ProviderRateLimitError as e:
            last_error = e
            continue
        except ValueError as e:
            if "desconhecido" in str(e):
                continue
            raise
        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            is_rate = status_code == 429 or any(k in msg for k in [
                "rate limit", "quota", "too many requests", "resource_exhausted", "resourseexhausted"
            ])
            if is_rate:
                last_error = ProviderRateLimitError(
                    provider_name=name,
                    status_code=status_code,
                    message=str(e),
                )
                continue
            raise

    raise last_error or ProviderRateLimitError(
        provider_name=provider_name,
        message="Todos os providers da lista de fallback falharam",
    )
