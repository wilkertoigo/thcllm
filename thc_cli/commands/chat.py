import argparse
import sys


def register(subparsers):
    parser = subparsers.add_parser("chat", help="Envia uma mensagem ao modelo")
    parser.add_argument("prompt", help="Texto da pergunta")
    parser.add_argument("--model", help="Modelo a usar")
    parser.add_argument("--mode", choices=["fast", "medium", "thinking"], default="medium", help="Modo de raciocínio")
    parser.add_argument("--web", action="store_true", help="Habilitar busca na web")


def run(args, client):
    try:
        messages = [{"role": "user", "content": args.prompt}]
        result = client.chat_completion(
            messages=messages,
            model=args.model,
            mode=args.mode,
            web=args.web
        )
        print(result["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)