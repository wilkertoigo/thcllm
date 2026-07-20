import argparse
import sys

from ..core import print_banner, print_assistant_reply, print_thinking_spinner


def register(subparsers):
    parser = subparsers.add_parser("chat", help="Envia uma mensagem ao modelo")
    parser.add_argument("prompt", nargs="?", help="Texto da pergunta")
    parser.add_argument("--model", help="Modelo a usar")
    parser.add_argument("--mode", choices=["fast", "medium", "thinking"], default="medium", help="Modo de raciocínio")
    parser.add_argument("--web", action="store_true", help="Habilitar busca na web")


def run(args, client):
    if args.prompt is None:
        print_banner()
        history = []
        while True:
            try:
                user_input = input("Você> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                break
            
            if user_input in ["/sair", "/exit"]:
                print("👋 Até logo!")
                break
            
            if not user_input:
                continue
            
            history.append({"role": "user", "content": user_input})
            
            try:
                with print_thinking_spinner():
                    result = client.chat_completion(
                        messages=history,
                        model=args.model,
                        mode=args.mode,
                        web=args.web
                    )
                reply = result["choices"][0]["message"]["content"]
                history.append({"role": "assistant", "content": reply})
                print_assistant_reply(reply)
            except Exception as e:
                print(f"[red]Erro: {e}[/red]")
    else:
        try:
            messages = [{"role": "user", "content": args.prompt}]
            with print_thinking_spinner():
                result = client.chat_completion(
                    messages=messages,
                    model=args.model,
                    mode=args.mode,
                    web=args.web
                )
            print_assistant_reply(result["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"Erro: {e}", file=sys.stderr)
            sys.exit(1)