import json
import re
from typing import Optional

from .memory import MemoryStore
from .system_prompt import build_base_system_prompt
from .tools import ALL_TOOLS, TOOLS_BY_NAME


TOOL_CALL_PATTERN = re.compile(
    r"```tool_call\s*\n(.*?)\n```",
    re.DOTALL,
)

FALLBACK_TOOL_CALL_PATTERN = re.compile(
    r"```(\w[\w_-]*)\s*\n(\{.*?\})\s*\n```",
    re.DOTALL,
)

RAW_TOOL_CALL_PATTERN = re.compile(
    r'^\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}$',
    re.MULTILINE | re.DOTALL,
)


def _build_tools_prompt(tools=None):
    registry = tools if tools is not None else ALL_TOOLS
    payload = []
    for tool in registry:
        payload.append({
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        })
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_tool_call(text: str) -> Optional[dict]:
    if not text or not text.strip():
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    name = data.get("name") or data.get("tool_name")
    arguments = data.get("arguments") or data.get("parameters") or data.get("input") or {}
    if not name or name not in TOOLS_BY_NAME:
        return None
    return {"name": name, "arguments": arguments}


def _extract_tool_calls(text: str) -> list[dict]:
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        raw = match.group(1).strip()
        parsed = _parse_tool_call(raw)
        if parsed:
            calls.append(parsed)
    if not calls:
        for match in FALLBACK_TOOL_CALL_PATTERN.finditer(text):
            lang = match.group(1).strip().lower()
            if lang == "tool_call":
                continue
            raw = match.group(2).strip()
            parsed = _parse_tool_call(raw)
            if parsed:
                calls.append(parsed)
    if not calls:
        for match in RAW_TOOL_CALL_PATTERN.finditer(text):
            raw = match.group(0).strip()
            parsed = _parse_tool_call(raw)
            if parsed:
                calls.append(parsed)
    return calls


def _truncate_history(messages: list[dict]) -> list[dict]:
    if len(messages) <= 40:
        return messages
    return [messages[0]] + messages[-(30):]


def _is_thc_provider(provider) -> bool:
    name = getattr(provider, "name", None)
    return name == "thc"


def _provider_chat_completion(provider, messages, model, mode, web, max_tokens, temperature):
    if _is_thc_provider(provider):
        return provider.chat_completion(
            messages=messages,
            model=model,
            mode=mode,
            web=web,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return provider.chat_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def run_agent(
    provider,
    messages: list[dict],
    model: Optional[str] = None,
    mode: str = "medium",
    web: bool = False,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    max_rounds: int = 10,
    on_tool_call: Optional[callable] = None,
    on_tool_result: Optional[callable] = None,
    on_thinking: Optional[callable] = None,
    on_round_complete: Optional[callable] = None,
    skill: Optional[dict] = None,
) -> str:
    tools_to_use = TOOLS_BY_NAME
    if skill and skill.get("tools_allowed"):
        allowed = set(skill["tools_allowed"])
        tools_to_use = {k: v for k, v in TOOLS_BY_NAME.items() if k in allowed}
    tools_description = _build_tools_prompt(tools=list(tools_to_use.values()))
    memory_store = MemoryStore()
    pinned = memory_store.get_pinned()
    system_prompt = (
        build_base_system_prompt(pinned_memories=pinned if pinned else None) + "\n\n" +
        "Você é um assistente agente. Para usar uma ferramenta, responda EXATAMENTE neste formato:\n"
        "```tool_call\n"
        "{\"name\": \"<nome_da_tool>\", \"arguments\": {<args_json>}}\n"
        "```\n"
        f"Tools disponíveis:\n{tools_description}\n"
        "Quando a pergunta for respondida, retorne a resposta final em texto livre, sem tool_call.\n"
    )
    if skill and skill.get("system_prompt_extra"):
        system_prompt += "\n\n## Skill ativa: " + skill["name"] + "\n" + skill["system_prompt_extra"]
    messages = [{"role": "system", "content": system_prompt}] + list(messages)
    for _ in range(max_rounds):
        messages = _truncate_history(messages)
        result = _provider_chat_completion(provider, messages, model, mode, web, max_tokens, temperature)
        assistant_message = result["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_message})
        thinking = TOOL_CALL_PATTERN.split(assistant_message)[0].strip()
        if thinking and on_thinking:
            on_thinking(thinking)

        tool_calls = _extract_tool_calls(assistant_message)

        has_tool_call_blocks = bool(TOOL_CALL_PATTERN.search(assistant_message))
        if has_tool_call_blocks and not tool_calls:
            retry_message = assistant_message
            for retry in range(2):
                error_msg = "Seu último tool_call não é JSON válido ou está malformado. Responda novamente APENAS com o bloco ```tool_call``` corrigido."
                messages.append({"role": "user", "content": error_msg})
                retry_result = _provider_chat_completion(provider, messages, model, mode, web, max_tokens, temperature)
                retry_message = retry_result["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": retry_message})
                tool_calls = _extract_tool_calls(retry_message)
                if tool_calls:
                    break
            if not tool_calls:
                if on_tool_result:
                    on_tool_result("parser", "Erro: tool_call malformado após 2 tentativas de correção.")
                messages.append({"role": "user", "content": "Tool_call descartado: JSON inválido após múltiplas tentativas. Responda sem usar essa tool, ou gere um tool_call válido se realmente for necessário."})
                continue

        if not tool_calls:
            return assistant_message
        tool_results = []
        for call in tool_calls:
            if on_tool_call:
                allowed = on_tool_call(call["name"], call.get("arguments", {}))
                if allowed is False:
                    output = "Usuário negou permissão para executar esta ação."
                    tool_results.append({
                        "name": call["name"],
                        "arguments": call.get("arguments", {}),
                        "output": output,
                    })
                    if on_tool_result:
                        on_tool_result(call["name"], output)
                    continue
            tool = tools_to_use[call["name"]]
            output = tool.run(**call.get("arguments", {}))
            if on_tool_result:
                on_tool_result(call["name"], output)
            tool_results.append({
                "name": call["name"],
                "arguments": call.get("arguments", {}),
                "output": output,
            })
        tool_feedback = json.dumps(tool_results, ensure_ascii=False, indent=2)
        messages.append({"role": "user", "content": "[Resultado das tools]\n" + tool_feedback})
        if on_round_complete:
            should_continue = on_round_complete()
            if should_continue is False:
                return "Execução interrompida pelo usuário durante o Plan Mode (passo a passo)."
    return "Limite de interações do agente atingido."
