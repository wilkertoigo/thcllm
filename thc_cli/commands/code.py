import argparse
import sys
import os

from ..core import print_thinking_spinner, print_diff, confirm


def register(subparsers):
    parser = subparsers.add_parser("code", help="Edita arquivos de código")
    parser.add_argument("arquivo", help="Caminho do arquivo a editar")
    parser.add_argument("instrucao", help="Instrução para editar o arquivo")
    parser.add_argument("--model", help="Modelo a usar")
    parser.add_argument("--mode", choices=["fast", "medium", "thinking"], default="thinking", help="Modo de raciocínio")


def run(args, client):
    if not os.path.exists(args.arquivo):
        print(f"Erro: Arquivo não encontrado: {args.arquivo}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.arquivo, "r") as f:
        original_content = f.read()
    
    system_prompt = (
        "Você recebe um arquivo e uma instrução. Responda APENAS com o conteúdo completo e final do arquivo já modificado, "
        "sem explicações, sem markdown, sem comentários extras — apenas o código puro pronto para salvar."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Arquivo: {args.arquivo}\n\nInstrução: {args.instrucao}\n\nConteúdo atual:\n{original_content}"}
    ]
    
    try:
        with print_thinking_spinner():
            result = client.chat_completion(
                messages=messages,
                model=args.model,
                mode=args.mode or "thinking"
            )
        new_content = result["choices"][0]["message"]["content"]
        
        print_diff(original_content, new_content, args.arquivo)
        
        if confirm("Aplicar essas mudanças?"):
            with open(args.arquivo, "w") as f:
                f.write(new_content)
            print("✅ Arquivo atualizado.")
        else:
            print("❌ Alterações descartadas.")
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)