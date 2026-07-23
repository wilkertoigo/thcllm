import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


BUILTIN_SKILLS = [
    {
        "name": "git_workflow",
        "description": "Auxilia no fluxo git: status, diff, commits semânticos, push",
        "system_prompt_extra": (
            "Você é especialista em git. Sempre rode 'git status' e 'git diff' antes de "
            "qualquer operação. Escreva commits semânticos (feat/fix/chore/docs/refactor). "
            "Nunca force-push sem confirmação explícita do usuário."
        ),
        "tools_allowed": ["bash", "read_file"],
    },
    {
        "name": "code_review",
        "description": "Revisão de código: legibilidade, bugs, segurança, performance",
        "system_prompt_extra": (
            "Você é um revisor de código experiente. Leia o código inteiro antes de comentar. "
            "Aponte problemas por ordem de severidade: bugs > segurança > performance > estilo. "
            "Seja específico: cite linha e explique o problema."
        ),
        "tools_allowed": ["read_file", "glob", "grep"],
    },
    {
        "name": "write_tests",
        "description": "Escreve testes unitários para código Python existente",
        "system_prompt_extra": (
            "Você é especialista em testes Python com unittest. Sempre leia o código antes de "
            "escrever testes. Prefira testes isolados com mocks para dependências externas. "
            "Cada teste deve ter um nome descritivo começando com test_."
        ),
        "tools_allowed": ["read_file", "write_file", "str_replace", "glob", "grep", "bash"],
    },
    {
        "name": "refactor",
        "description": "Refatora código mantendo comportamento externo idêntico",
        "system_prompt_extra": (
            "Você é especialista em refatoração segura. Antes de qualquer mudança: leia o código, "
            "identifique os testes existentes, rode-os para confirmar que passam. Faça mudanças "
            "incrementais. Rode os testes após cada mudança. Nunca altere a interface pública sem "
            "avisar explicitamente o usuário."
        ),
        "tools_allowed": ["read_file", "write_file", "str_replace", "glob", "grep", "bash"],
    },
]


class SkillStore:
    def __init__(self, skills_dir: str = None):
        if skills_dir is None:
            skills_dir = str(Path.home() / ".thc" / "skills")
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def _skill_path(self, name: str) -> Path:
        return self.skills_dir / f"{name}.json"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def add(self, name: str, description: str, system_prompt_extra: str = "", tools_allowed: list[str] = None) -> dict:
        if not re.fullmatch(r"[a-z0-9_-]+", name):
            raise ValueError(f"Nome de skill inválido: {name!r}. Use apenas [a-z0-9_-].")
        path = self._skill_path(name)
        data = {
            "name": name,
            "description": description,
            "system_prompt_extra": system_prompt_extra or "",
            "tools_allowed": tools_allowed or [],
            "created_at": self._now(),
            "updated_at": self._now(),
        }
        if path.exists():
            existing = self._load_path(path)
            data["created_at"] = existing.get("created_at", data["created_at"])
        self._save_path(path, data)
        return data

    def get(self, name: str) -> Optional[dict]:
        path = self._skill_path(name)
        if not path.exists():
            return None
        return self._load_path(path)

    def list_all(self) -> list[dict]:
        entries = []
        for path in sorted(self.skills_dir.glob("*.json")):
            try:
                entries.append(self._load_path(path))
            except Exception:
                continue
        entries.sort(key=lambda s: s.get("name", ""))
        return entries

    def delete(self, name: str) -> bool:
        path = self._skill_path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def seed(self) -> None:
        for skill in BUILTIN_SKILLS:
            path = self._skill_path(skill["name"])
            if path.exists():
                continue
            data = dict(skill)
            data["created_at"] = self._now()
            data["updated_at"] = self._now()
            self._save_path(path, data)

    def _load_path(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_path(self, path: Path, data: dict) -> None:
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
