from __future__ import annotations

import os
import glob as glob_mod
import re
from typing import Optional


class BaseTool:
    name: str = ""
    description: str = ""
    input_schema: dict = {}

    def run(self, **kwargs) -> str:
        raise NotImplementedError

    def preview(self, **kwargs):
        raise NotImplementedError


class FileReadTool(BaseTool):
    name = "read_file"
    description = "Lê o conteúdo completo de um arquivo e retorna como string"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Caminho absoluto ou relativo do arquivo a ler",
            }
        },
        "required": ["path"],
    }

    def run(self, path: str = "") -> str:
        if not path:
            return "Erro: parâmetro 'path' é obrigatório"
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content:
                return f"(Arquivo vazio: {path})"
            return content
        except FileNotFoundError:
            return f"Erro: arquivo não encontrado: {path}"
        except PermissionError:
            return f"Erro: sem permissão para ler: {path}"
        except Exception as e:
            return f"Erro ao ler {path}: {e}"


class FileWriteTool(BaseTool):
    name = "write_file"
    description = "Cria um novo arquivo ou sobrescreve um existente com o conteúdo informado"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Caminho absoluto ou relativo do arquivo a criar/sobrescrever",
            },
            "content": {
                "type": "string",
                "description": "Conteúdo completo a ser escrito no arquivo",
            },
        },
        "required": ["path", "content"],
    }

    def run(self, path: str = "", content: str = "") -> str:
        if not path:
            return "Erro: parâmetro 'path' é obrigatório"
        if content is None:
            return "Erro: parâmetro 'content' é obrigatório"
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Arquivo criado/atualizado: {path} ({len(content)} bytes)"
        except PermissionError:
            return f"Erro: sem permissão para escrever em: {path}"
        except Exception as e:
            return f"Erro ao escrever {path}: {e}"

    def preview(self, path: str = "", content: str = "") -> tuple[str, str, str | None]:
        if not path:
            return ("", "", "Erro: parâmetro 'path' é obrigatório")
        original = ""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    original = f.read()
            except PermissionError:
                return ("", "", f"Erro: sem permissão para ler: {path}")
        return (original, content, None)


class FileEditTool(BaseTool):
    name = "str_replace"
    description = (
        "Substitui a primeira ocorrência de old_str por new_str dentro de um arquivo. "
        "Use apenas quando o conteúdo exato do trecho for conhecido."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Caminho do arquivo a editar",
            },
            "old_str": {
                "type": "string",
                "description": "Texto exato a ser substituído",
            },
            "new_str": {
                "type": "string",
                "description": "Novo texto que substituirá old_str",
            },
        },
        "required": ["path", "old_str", "new_str"],
    }

    def run(self, path: str = "", old_str: str = "", new_str: str = "") -> str:
        if not path:
            return "Erro: parâmetro 'path' é obrigatório"
        if old_str == "":
            return "Erro: parâmetro 'old_str' não pode ser vazio"
        if new_str == "":
            return "Erro: parâmetro 'new_str' não pode ser vazio; para remover texto use new_str='(vazio)' ou outra representação explícita"
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if old_str not in content:
                return f"Erro: old_str não encontrado em {path}"
            count = content.count(old_str)
            if count > 1:
                return f"Erro: old_str aparece {count} vezes em {path}; forneça um trecho mais específico para editar apenas uma ocorrência."
            content = content.replace(old_str, new_str, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Ediçao aplicada em {path}"
        except FileNotFoundError:
            return f"Erro: arquivo não encontrado: {path}"
        except PermissionError:
            return f"Erro: sem permissão para editar: {path}"
        except Exception as e:
            return f"Erro ao editar {path}: {e}"

    def preview(self, path: str = "", old_str: str = "", new_str: str = "") -> tuple[str, str, str | None]:
        if not path:
            return ("", "", "Erro: parâmetro 'path' é obrigatório")
        if old_str == "":
            return ("", "", "Erro: parâmetro 'old_str' não pode ser vazio")
        if new_str == "":
            return ("", "", "Erro: parâmetro 'new_str' não pode ser vazio; para remover texto use new_str='(vazio)' ou outra representação explícita")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if old_str not in content:
                return (content, "", f"Erro: old_str não encontrado em {path}")
            count = content.count(old_str)
            if count > 1:
                return (content, "", f"Erro: old_str aparece {count} vezes em {path}; forneça um trecho mais específico para editar apenas uma ocorrência.")
            new_content = content.replace(old_str, new_str, 1)
            return (content, new_content, None)
        except FileNotFoundError:
            return ("", "", f"Erro: arquivo não encontrado: {path}")
        except PermissionError:
            return ("", "", f"Erro: sem permissão para editar: {path}")
        except Exception as e:
            return ("", "", f"Erro ao editar {path}: {e}")


class GlobTool(BaseTool):
    name = "glob"
    description = "Busca arquivos e diretórios por padrão glob (ex: '**/*.py', 'src/**/*.ts')"
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Padrão glob estilo .gitignore (ex: '**/*.py', 'src/**/*.ts')",
            }
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str = "") -> str:
        if not pattern:
            return "Erro: parâmetro 'pattern' é obrigatório"
        try:
            matches = sorted(glob_mod.glob(pattern, recursive=True))
            if not matches:
                return f"Nenhum caminho encontrado para o padrão: {pattern}"
            return "\n".join(matches)
        except Exception as e:
            return f"Erro ao executar glob '{pattern}': {e}"


class GrepTool(BaseTool):
    name = "grep"
    description = "Busca recursiva por expressão regular em arquivos, retornando caminho:linha:conteúdo"
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Expressão regular a buscar",
            },
            "path": {
                "type": "string",
                "description": "Diretório ou arquivo base para busca (default: diretório atual)",
            },
            "include": {
                "type": "string",
                "description": "Glob de arquivos a incluir (ex: '*.py', '*.json')",
            },
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str = "", path: str = ".", include: str = "") -> str:
        if not pattern:
            return "Erro: parâmetro 'pattern' é obrigatório"
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return f"Erro: regex inválida '{pattern}': {e}"
        try:
            base = os.path.abspath(path)
            if not os.path.exists(base):
                return f"Erro: caminho não encontrado: {path}"
            results = []
            if os.path.isfile(base):
                targets = [base]
            else:
                walk_root = base
                targets = []
                for root, dirs, files in os.walk(walk_root):
                    dirs[:] = [d for d in dirs if d != "__pycache__"]
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        if include:
                            if not glob_mod.fnmatch.fnmatch(filename, include):
                                continue
                        targets.append(filepath)
            for filepath in targets:
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, start=1):
                            if compiled.search(line):
                                rel = os.path.relpath(filepath, base)
                                results.append(f"{rel}:{lineno}:{line.rstrip()}")
                except Exception:
                    continue
            if not results:
                return f"Nenhuma correspondência para '{pattern}' em {path}"
            max_results = 200
            if len(results) > max_results:
                results = results[:max_results]
                results.append(f"... ({len(results) - max_results} linhas adicionais suprimidas)")
            return "\n".join(results)
        except Exception as e:
            return f"Erro no grep '{pattern}': {e}"
