import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from thc_cli.core.agent import run_agent
from thc_cli.core.memory import MemoryStore
from thc_cli.core.skills import BUILTIN_SKILLS, SkillStore
from thc_cli.core.tools import TOOLS_BY_NAME


class StubProvider:
    def __init__(self):
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}


class TestSkillStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SkillStore(skills_dir=os.path.join(self.tmp.name, "skills"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_cria_skill(self):
        skill = self.store.add("minha_skill", "desc", system_prompt_extra="prompt", tools_allowed=["bash"])
        self.assertEqual(skill["name"], "minha_skill")
        self.assertEqual(skill["description"], "desc")
        self.assertEqual(skill["system_prompt_extra"], "prompt")
        self.assertEqual(skill["tools_allowed"], ["bash"])
        self.assertTrue(Path(self.store._skill_path("minha_skill")).exists())

    def test_add_valida_nome_invalido(self):
        with self.assertRaises(ValueError):
            self.store.add("minha skill", "desc")

    def test_add_valida_nome_com_espacos(self):
        with self.assertRaises(ValueError):
            self.store.add("skill inválida", "desc")

    def test_get_retorna_skill_existente(self):
        self.store.add("skill1", "desc1")
        skill = self.store.get("skill1")
        self.assertIsNotNone(skill)
        self.assertEqual(skill["name"], "skill1")

    def test_get_retorna_none_se_nao_existe(self):
        self.assertIsNone(self.store.get("nao_existe"))

    def test_list_all_vazia(self):
        self.assertEqual(self.store.list_all(), [])

    def test_list_all_retorna_ordenado_alfabeticamente(self):
        self.store.add("zebra", "z")
        self.store.add("alpha", "a")
        names = [s["name"] for s in self.store.list_all()]
        self.assertEqual(names, ["alpha", "zebra"])

    def test_delete_existente_retorna_true(self):
        self.store.add("skill1", "desc1")
        self.assertTrue(self.store.delete("skill1"))
        self.assertIsNone(self.store.get("skill1"))

    def test_delete_inexistente_retorna_false(self):
        self.assertFalse(self.store.delete("nao_existe"))

    def test_seed_cria_builtin_se_nao_existem(self):
        self.store.seed()
        names = [s["name"] for s in self.store.list_all()]
        for skill in BUILTIN_SKILLS:
            self.assertIn(skill["name"], names)

    def test_seed_nao_sobrescreve_customizacao(self):
        self.store.add("write_tests", "custom", system_prompt_extra="custom extra", tools_allowed=["bash"])
        self.store.seed()
        skill = self.store.get("write_tests")
        self.assertEqual(skill["system_prompt_extra"], "custom extra")
        self.assertEqual(skill["tools_allowed"], ["bash"])

    def test_escrita_atomica(self):
        from unittest.mock import patch
        self.store.add("skill1", "desc1")
        path = self.store._skill_path("skill1")
        with patch("os.replace", side_effect=OSError("falha simulada")):
            with self.assertRaises(OSError):
                self.store.add("skill1", "desc2")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(data["name"], "skill1")
        self.assertEqual(data["description"], "desc1")


class TestSkillIntegration(unittest.TestCase):
    def _extract_tools(self, system_msg: str) -> set[str]:
        prefix = "Tools disponíveis:\n"
        idx = system_msg.index(prefix) + len(prefix)
        rest = system_msg[idx:]
        depth = 0
        for i, ch in enumerate(rest):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return {t["name"] for t in json.loads(rest[: i + 1])}
        self.fail("Could not locate tools JSON block in system prompt")

    def test_skill_extra_prompt_aparece_no_system_prompt(self):
        provider = StubProvider()
        skill = {
            "name": "git_workflow",
            "system_prompt_extra": "Sempre rode git status.",
        }
        result = run_agent(
            provider=provider,
            messages=[{"role": "user", "content": "oi"}],
            max_tokens=16,
            max_rounds=1,
            skill=skill,
        )
        system_msg = provider.calls[0]["messages"][0]["content"]
        self.assertIn("Skill ativa: git_workflow", system_msg)
        self.assertIn("Sempre rode git status.", system_msg)

    def test_skill_none_usa_todas_as_tools(self):
        provider = StubProvider()
        result = run_agent(
            provider=provider,
            messages=[{"role": "user", "content": "oi"}],
            max_tokens=16,
            max_rounds=1,
            skill=None,
        )
        system_msg = provider.calls[0]["messages"][0]["content"]
        self.assertEqual(self._extract_tools(system_msg), set(TOOLS_BY_NAME.keys()))

    def test_skill_filtra_tools_allowed(self):
        provider = StubProvider()
        skill = {
            "name": "code_review",
            "system_prompt_extra": "Seja específico.",
            "tools_allowed": ["read_file"],
        }
        result = run_agent(
            provider=provider,
            messages=[{"role": "user", "content": "oi"}],
            max_tokens=16,
            max_rounds=1,
            skill=skill,
        )
        system_msg = provider.calls[0]["messages"][0]["content"]
        self.assertEqual(self._extract_tools(system_msg), {"read_file"})

    def test_tools_allowed_vazio_usa_todas(self):
        provider = StubProvider()
        skill = {
            "name": "generic",
            "system_prompt_extra": "prompt",
            "tools_allowed": [],
        }
        result = run_agent(
            provider=provider,
            messages=[{"role": "user", "content": "oi"}],
            max_tokens=16,
            max_rounds=1,
            skill=skill,
        )
        system_msg = provider.calls[0]["messages"][0]["content"]
        self.assertEqual(self._extract_tools(system_msg), set(TOOLS_BY_NAME.keys()))

    def test_tools_allowed_invalido_ignorado(self):
        provider = StubProvider()
        skill = {
            "name": "weird",
            "system_prompt_extra": "prompt",
            "tools_allowed": ["read_file", "tool_que_nao_existe"],
        }
        result = run_agent(
            provider=provider,
            messages=[{"role": "user", "content": "oi"}],
            max_tokens=16,
            max_rounds=1,
            skill=skill,
        )
        system_msg = provider.calls[0]["messages"][0]["content"]
        self.assertEqual(self._extract_tools(system_msg), {"read_file"})


class TestSkillCLI(unittest.TestCase):
    def test_skill_inexistente_retorna_none_do_store(self):
        store = SkillStore(skills_dir=os.path.join(tempfile.mkdtemp(), "skills"))
        self.assertIsNone(store.get("nao_existe"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
