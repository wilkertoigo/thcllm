import argparse
import re
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core import print_thinking_spinner, print_diff, confirm
from ..core.agent import run_agent
from ..core.plan import generate_plan
from ..core.providers import get_provider
from ..core.session import list_sessions, load_session, save_session
from ..core.system_prompt import build_base_system_prompt
from ..core.tokens import count_tokens, count_tokens_messages
from ..core.tools import DESTRUCTIVE_TOOLS, TOOLS_BY_NAME

console = Console()


LOCATION_SENSITIVE_KEYWORDS = [
    "tempo", "clima", "previsão", "temperatura", "chuva", "chove",
    "perto de mim", "próximo", "aqui", "local", "cidade",
]


def _is_thc_provider(provider) -> bool:
    return hasattr(provider, "name") and provider.name == "thc"


def _build_web_search_system_message(query: str) -> dict | None:
    search_query = query
    query_lower = query.lower()
    has_location_keyword = any(kw in query_lower for kw in LOCATION_SENSITIVE_KEYWORDS)
    mentions_place = (
        "lages" in query_lower
        or "santa catarina" in query_lower
        or "brasil" in query_lower
        or re.search(r"\bsc\b", query_lower) is not None
    )
    if has_location_keyword and not mentions_place:
        search_query = f"{query} em Lages, Santa Catarina, Brasil"
    search_tool = TOOLS_BY_NAME["web_search"]
    search_result = search_tool.run(query=search_query, max_results=4)
    if search_result.startswith("Erro"):
        return None
    return {
        "role": "system",
        "content": (
            f"Resultados de busca na web para a pergunta do usuário:\n\n{search_result}\n\n"
            "Use essas informações se forem relevantes para responder. Se não "
            "forem relevantes ou não responderem à pergunta, ignore-as e responda "
            "normalmente."
        ),
    }


def _banner(model: str, mode: str, web: bool, agent: bool, plan: bool, provider_name: str):
    title = Text()
    title.append("🤖 ", style="bold green")
    title.append("THC CLI", style="bold cyan")
    body = (
        f"provider: [bold green]{provider_name}[/bold green]\n"
        f"modelo: [bold green]{model or 'padrão'}[/bold green]\n"
        f"modo: [bold cyan]{mode}[/bold cyan]\n"
        f"web: [bold {'green' if web else 'red'}]{'on' if web else 'off'}[/bold {'green' if web else 'red'}]\n"
        f"agente: [bold {'green' if agent else 'red'}]{'on' if agent else 'off'}[/bold {'green' if agent else 'red'}]\n"
        f"plano: [bold {'green' if plan else 'red'}]{'on' if plan else 'off'}[/bold {'green' if plan else 'red'}]\n\n"
        "[dim]comandos: /model, /mode, /web, /tools, /clear, /agent, /plan, /provider, /save, /resume, /sessions, /sair[/dim]"
    )
    console.print(Panel(body, title=title, border_style="green", expand=False))


def _short_args(args: dict, max_chars: int = 60) -> str:
    text = str(args)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def _provider_chat(provider, messages, model, mode, web, max_tokens=8192, temperature=0.7):
    if hasattr(provider, "name") and provider.name == "thc":
        return provider.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            mode=mode,
            web=web,
        )
    return provider.chat_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _format_elapsed(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs:02d}s"


def register(subparsers):
    parser = subparsers.add_parser("chat", help="Chat interativo com TUI rica")
    parser.add_argument("prompt", nargs="?", help="Texto da pergunta")
    parser.add_argument("--model", help="Modelo a usar")
    parser.add_argument("--mode", choices=["fast", "medium", "thinking"], default="medium", help="Modo de raciocínio")
    parser.add_argument("--web", action="store_true", help="Habilitar busca na web")
    parser.add_argument("--agent", action="store_true", help="Modo agente com tools")
    parser.add_argument("--provider", help="Provider a usar (thc, groq, mistral, gemini, openrouter)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Máximo de tokens na resposta")


def run(args, config):
    provider_name = args.provider or config.get("provider", "thc")
    try:
        provider = get_provider(provider_name, config)
    except Exception as e:
        print(f"Erro ao inicializar provider '{provider_name}': {e}", file=sys.stderr)
        sys.exit(1)

    model = args.model
    mode = args.mode
    web = args.web
    agent_mode = args.agent
    plan_mode = False
    plan_mode_step_confirm = config.get("plan_mode_step_confirm", False)
    max_tokens = args.max_tokens or 2048
    history = []
    current_session_id = None
    session_start_time = time.time()
    session_tokens_used = 0

    def _render_reply(text: str):
        console.print(Markdown(text))

    def on_thinking(text: str):
        if text:
            console.print(f"[dim italic]💭 {text}[/dim italic]")

    def on_tool_call(name: str, arguments: dict):
        if name in DESTRUCTIVE_TOOLS:
            if name == "write_file":
                tool = TOOLS_BY_NAME["write_file"]
                original, new, error = tool.preview(
                    path=arguments.get("path", ""),
                    content=arguments.get("content", ""),
                )
                if error:
                    console.print(f"[red]{error}[/red]")
                    return False
                print_diff(original, new, arguments.get("path", ""))
            elif name == "str_replace":
                tool = TOOLS_BY_NAME["str_replace"]
                original, new, error = tool.preview(
                    path=arguments.get("path", ""),
                    old_str=arguments.get("old_str", ""),
                    new_str=arguments.get("new_str", ""),
                )
                if error:
                    console.print(f"[red]{error}[/red]")
                    return False
                print_diff(original, new, arguments.get("path", ""))
            panel_text = f"[bold yellow]{name}[/bold yellow]\n\n{arguments}"
            console.print(Panel(panel_text, title="Tool destrutiva", border_style="yellow"))
            return confirm(f"Permitir execução de '{name}'?")
        console.print(f"[dim]🔧 {name}({_short_args(arguments)})[/dim]")
        return None

    def on_tool_result(name: str, output: str):
        if output.startswith("Erro:"):
            console.print(f"[red]✗ {name}: {output[:150]}[/red]")
        else:
            console.print(f"[green]✓ {name}: {output[:150]}[/green]")

    if args.prompt is not None:
        if agent_mode:
            messages = [{"role": "user", "content": args.prompt}]
            try:
                reply = run_agent(
                    provider=provider,
                    messages=messages,
                    model=model,
                    mode=mode,
                    web=web,
                    max_tokens=max_tokens,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                    on_thinking=on_thinking,
                )
            except Exception as e:
                print(f"Erro: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            messages = [
                {"role": "system", "content": build_base_system_prompt()},
            ]
            if web and not _is_thc_provider(provider):
                web_msg = _build_web_search_system_message(args.prompt)
                if web_msg:
                    messages.append(web_msg)
            messages.append({"role": "user", "content": args.prompt})
            try:
                with print_thinking_spinner():
                    result = _provider_chat(
                        provider=provider,
                        messages=messages,
                        model=model,
                        mode=mode,
                        web=web,
                        max_tokens=max_tokens,
                    )
                reply = result["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Erro: {e}", file=sys.stderr)
                sys.exit(1)
        _render_reply(reply)
        session_tokens_used += count_tokens(args.prompt or "") + count_tokens(reply)
        console.print(f"[dim]🪙 {session_tokens_used} tokens · ⏱ {_format_elapsed(time.time() - session_start_time)}[/dim]")
        return

    _banner(model, mode, web, agent_mode, plan_mode, provider_name)
    while True:
        try:
            user_input = console.input("[bold green]❯ [/bold green]").strip()
        except EOFError:
            console.print()
            break
        except KeyboardInterrupt:
            console.print()
            break

        cmd = user_input.lower()
        if cmd in ("/sair", "/exit"):
            console.print("👋 Até logo!")
            break
        if cmd == "/clear":
            history = []
            console.print("[dim]Histórico limpo.[/dim]")
            continue
        if cmd == "/tools":
            from thc_cli.core.tools import ALL_TOOLS
            lines = [f"- [bold green]{t.name}[/bold green]: {t.description}" for t in ALL_TOOLS]
            console.print(Panel("\n".join(lines), title="Tools", border_style="cyan"))
            continue
        if cmd.startswith("/model "):
            model = user_input.split(" ", 1)[1].strip() or None
            console.print(f"[dim]Modelo alterado para: {model or 'padrão'}[/dim]")
            _banner(model, mode, web, agent_mode, plan_mode, provider_name)
            continue
        if cmd.startswith("/mode "):
            mode = user_input.split(" ", 1)[1].strip()
            console.print(f"[dim]Modo alterado para: {mode}[/dim]")
            _banner(model, mode, web, agent_mode, plan_mode, provider_name)
            continue
        if cmd == "/web":
            web = not web
            console.print(f"[dim]Web: {'on' if web else 'off'}[/dim]")
            _banner(model, mode, web, agent_mode, plan_mode, provider_name)
            continue
        if cmd == "/agent":
            agent_mode = not agent_mode
            console.print(f"[dim]Agente: {'on' if agent_mode else 'off'}[/dim]")
            _banner(model, mode, web, agent_mode, plan_mode, provider_name)
            continue
        if cmd == "/plan":
            plan_mode = not plan_mode
            console.print(f"[dim]Plano: {'on' if plan_mode else 'off'}[/dim]")
            _banner(model, mode, web, agent_mode, plan_mode, provider_name)
            continue
        if cmd.startswith("/provider "):
            new_name = user_input.split(" ", 1)[1].strip().lower()
            try:
                new_provider = get_provider(new_name, config)
                provider_name = new_name
                provider = new_provider
                console.print(f"[dim]Provider alterado para: {provider_name}[/dim]")
                _banner(model, mode, web, agent_mode, plan_mode, provider_name)
            except Exception as e:
                console.print(f"[red]Erro ao trocar provider: {e}[/red]")
            continue
        if cmd == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]Nenhuma sessão salva.[/dim]")
            else:
                table = Table(title="Sessões")
                table.add_column("session_id", style="cyan")
                table.add_column("updated_at", style="dim")
                table.add_column("provider", style="green")
                table.add_column("model", style="green")
                table.add_column("preview", style="dim")
                for s in sessions:
                    table.add_row(
                        s.get("session_id", ""),
                        s.get("updated_at", ""),
                        s.get("provider", ""),
                        s.get("model", ""),
                        s.get("preview", ""),
                    )
                console.print(table)
            continue
        if cmd == "/status":
            status_lines = [
                f"provider: [bold green]{provider_name}[/bold green]",
                f"modelo: [bold green]{model or 'padrão'}[/bold green]",
                f"modo: [bold cyan]{mode}[/bold cyan]",
                f"web: [bold {'green' if web else 'red'}]{'on' if web else 'off'}[/bold {'green' if web else 'red'}]",
                f"agente: [bold {'green' if agent_mode else 'red'}]{'on' if agent_mode else 'off'}[/bold {'green' if agent_mode else 'red'}]",
                f"plano: [bold {'green' if plan_mode else 'red'}]{'on' if plan_mode else 'off'}[/bold {'green' if plan_mode else 'red'}]",
                f"tokens: [bold yellow]{session_tokens_used}[/bold yellow]",
                f"tempo: [bold yellow]{_format_elapsed(time.time() - session_start_time)}[/bold yellow]",
                f"mensagens: [bold yellow]{len(history)}[/bold yellow]",
                f"session_id: [bold cyan]{current_session_id or 'nenhuma'}[/bold cyan]",
            ]
            console.print(Panel("\n".join(status_lines), title="Status", border_style="cyan"))
            continue
        if cmd == "/save":
            try:
                saved_id = save_session(current_session_id, history, provider_name, model or "", mode or "", agent_mode=agent_mode)
                current_session_id = saved_id
                console.print(f"[dim]Sessão salva: {current_session_id}[/dim]")
            except FileNotFoundError:
                console.print("[red]Sessão atual inválida/corrompida. Criando nova sessão.[/red]")
                saved_id = save_session(None, history, provider_name, model or "", mode or "", agent_mode=agent_mode)
                current_session_id = saved_id
                console.print(f"[dim]Sessão salva: {current_session_id}[/dim]")
            continue
        if cmd.startswith("/resume "):
            target_session_id = user_input.split(" ", 1)[1].strip()
            if not target_session_id:
                console.print("[red]Uso: /resume <session_id>[/red]")
                continue
            try:
                session_data = load_session(target_session_id)
            except FileNotFoundError:
                console.print(f"[red]Sessão não encontrada: {target_session_id}[/red]")
                continue
            except Exception as e:
                console.print(f"[red]Erro ao carregar sessão: {e}[/red]")
                continue
            if "agent_mode" not in session_data:
                console.print("[yellow]Sessão sem modo salvo (formato antigo); mantendo modo atual.[/yellow]")
            history = session_data.get("messages", [])
            model = session_data.get("model") or None
            mode = session_data.get("mode") or mode
            agent_mode = session_data.get("agent_mode", agent_mode)
            saved_provider = session_data.get("provider")
            if saved_provider:
                try:
                    new_provider = get_provider(saved_provider, config)
                    provider_name = saved_provider
                    provider = new_provider
                except Exception:
                    console.print(f"[yellow]Provider salvo '{saved_provider}' indisponível. Mantendo '{provider_name}'.[/yellow]")
            current_session_id = session_data.get("session_id") or target_session_id
            if history:
                console.print(f"[dim]Sessão restaurada: {current_session_id}[/dim]")
            else:
                console.print(f"[yellow]Sessão restaurada ({current_session_id}), mas sem mensagens.[/yellow]")
            _banner(model, mode, web, agent_mode, plan_mode, provider_name)
            continue
        if not user_input:
            continue

        if plan_mode:
            plan_text = generate_plan(provider, user_input, model, mode, web, max_tokens)
            console.print(Panel(plan_text, title="Plano proposto", border_style="cyan"))
            if not confirm("Aprovar este plano e iniciar execução?"):
                continue
            history.append({"role": "user", "content": user_input})

            def _on_round_complete():
                if plan_mode_step_confirm:
                    return confirm("Continuar para o próximo passo?")
                return None
        else:
            history.append({"role": "user", "content": user_input})

            def _on_round_complete():
                return None

        try:
            if agent_mode:
                reply = run_agent(
                    provider=provider,
                    messages=history,
                    model=model,
                    mode=mode,
                    web=web,
                    max_tokens=max_tokens,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                    on_thinking=on_thinking,
                    on_round_complete=_on_round_complete,
                )
                history.append({"role": "assistant", "content": reply})
            else:
                with print_thinking_spinner():
                    messages_to_send = [{"role": "system", "content": build_base_system_prompt()}] + history
                    if web and not _is_thc_provider(provider):
                        last_user_message = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
                        if last_user_message:
                            web_msg = _build_web_search_system_message(last_user_message)
                            if web_msg:
                                messages_to_send.insert(1, web_msg)
                    result = _provider_chat(
                        provider=provider,
                        messages=messages_to_send,
                        model=model,
                        mode=mode,
                        web=web,
                        max_tokens=max_tokens,
                    )
                reply = result["choices"][0]["message"]["content"]
                history.append({"role": "assistant", "content": reply})
            _render_reply(reply)
            session_tokens_used += count_tokens(user_input or "") + count_tokens(reply)
            console.print(f"[dim]🪙 {session_tokens_used} tokens · ⏱ {_format_elapsed(time.time() - session_start_time)}[/dim]")
        except Exception as e:
            console.print(f"[red]Erro: {e}[/red]")
