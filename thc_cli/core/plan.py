from typing import Optional

from .agent import _provider_chat_completion


def generate_plan(
    provider,
    prompt: str,
    model: Optional[str] = None,
    mode: str = "medium",
    web: bool = False,
    max_tokens: int = 8192,
    temperature: float = 0.7,
) -> str:
    system_prompt = (
        "Antes de executar qualquer ação, liste em formato numerado (1. 2. 3.) "
        "os passos que você pretende seguir para resolver a tarefa do usuário. "
        "Não execute nada ainda, apenas planeje."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    result = _provider_chat_completion(provider, messages, model, mode, web, max_tokens, temperature)
    return result["choices"][0]["message"]["content"]
