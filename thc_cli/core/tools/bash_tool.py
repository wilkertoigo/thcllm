import subprocess
from typing import Optional


class BashTool:
    name = "bash"
    description = "Executa um comando bash e retorna stdout + stderr"
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Comando bash a executar",
            },
            "cwd": {
                "type": "string",
                "description": "Diretório de trabalho (opcional)",
            },
        },
        "required": ["command"],
    }

    def run(self, command: str = "", cwd: Optional[str] = None) -> str:
        if not command:
            return "Erro: parâmetro 'command' é obrigatório"
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            parts = []
            if result.stdout:
                parts.append(result.stdout.rstrip())
            if result.stderr:
                parts.append("[stderr]\n" + result.stderr.rstrip())
            parts.append(f"[exit code {result.returncode}]")
            return "\n\n".join(parts)
        except subprocess.TimeoutExpired:
            return "Erro: comando excedeu timeout de 30s"
        except Exception as e:
            return f"Erro ao executar bash: {e}"
