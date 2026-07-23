import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thc_cli.core.agent import (
    FALLBACK_TOOL_CALL_PATTERN,
    TOOL_CALL_PATTERN,
    _extract_tool_calls,
    _parse_tool_call,
    run_agent,
)
from thc_cli.core.tools.file_tools import FileEditTool, FileWriteTool
from thc_cli.core.tools.todo_tool import TodoWriteTool
from thc_cli.core import session as session_mod
from thc_cli.core.ui import print_diff


class TestToolCallParser(unittest.TestCase):
    def test_tool_call_formato_correto(self):
        text = "```tool_call\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/x\"}}\n```"
        calls = _extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(calls[0]["arguments"]["path"], "/tmp/x")

    def test_fallback_nome_tool_como_linguagem(self):
        text = "```read_file\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/x\"}}\n```"
        calls = _extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")

    def test_fallback_nome_invalido_nao_aceito(self):
        text = "```read_file_xyz\n{\"name\": \"read_file_xyz\", \"arguments\": {\"path\": \"/tmp/x\"}}\n```"
        calls = _extract_tool_calls(text)
        self.assertEqual(len(calls), 0)

    def test_fallback_json_malformado_nao_aceito(self):
        text = "```read_file\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/x\",}}\n```"
        calls = _extract_tool_calls(text)
        self.assertEqual(len(calls), 0)

    def test_tool_call_duplo_na_mesma_resposta(self):
        text = (
            "```tool_call\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/a\"}}\n```"
            "```tool_call\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"/tmp/b\", \"content\": \"x\"}}\n```"
        )
        calls = _extract_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(calls[1]["name"], "write_file")

    def test_bloco_codigo_legitimo_nao_vira_tool_call(self):
        text = "```python\n{\"algum\": \"json qualquer\"}\n```"
        calls = _extract_tool_calls(text)
        self.assertEqual(len(calls), 0)

    def test_fallback_nao_duplica_tool_call(self):
        text = "```tool_call\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/x\"}}\n```"
        calls = _extract_tool_calls(text)
        self.assertEqual(len(calls), 1)


class TestRetryMalformado(unittest.TestCase):
    def test_nao_crash_com_bloco_sem_tool_call_valido(self):
        provider = MagicMock()
        provider.name = "thc"
        provider.chat_completion.return_value = {
            "choices": [{"message": {"content": "apenas texto normal sem tool_call"}}]
        }
        result = run_agent(
            provider=provider,
            messages=[{"role": "user", "content": "oi"}],
            max_rounds=2,
        )
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")


class TestSessionPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.tmp.name) / ".thc" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _patch_session_dir(self):
        patcher_dir = patch.object(session_mod, "SESSIONS_DIR", self.session_dir)
        patcher_path = patch.object(session_mod, "_session_path", lambda sid: self.session_dir / f"{sid}.json")
        patcher_dir.start()
        patcher_path.start()
        self.addCleanup(patcher_dir.stop)
        self.addCleanup(patcher_path.stop)

    def test_save_cria_nova_sessao(self):
        self._patch_session_dir()
        sid = session_mod.save_session(
            session_id=None,
            messages=[{"role": "user", "content": "oi"}],
            provider="thc",
            model="m",
            mode="medium",
        )
        self.assertTrue((self.session_dir / f"{sid}.json").exists())

    def test_save_resume_existente(self):
        self._patch_session_dir()
        sid = session_mod.save_session(
            session_id=None,
            messages=[{"role": "user", "content": "oi"}],
            provider="thc",
            model="m",
            mode="medium",
        )
        session_mod.save_session(
            session_id=sid,
            messages=[{"role": "user", "content": "ola"}],
            provider="thc",
            model="m2",
            mode="fast",
        )
        data = json.loads((self.session_dir / f"{sid}.json").read_text())
        self.assertEqual(data["provider"], "thc")
        self.assertEqual(data["model"], "m2")
        self.assertEqual(data["mode"], "fast")
        self.assertEqual(len(data["messages"]), 1)

    def test_save_resume_inexistente_levanta(self):
        self._patch_session_dir()
        with self.assertRaises(FileNotFoundError):
            session_mod.save_session(
                session_id="session-nao-existe",
                messages=[{"role": "user", "content": "oi"}],
                provider="thc",
                model="m",
                mode="medium",
            )

    def test_load_inexistente_levanta(self):
        self._patch_session_dir()
        with self.assertRaises(FileNotFoundError):
            session_mod.load_session("session-nao-existe")

    def test_list_sessions_vazia_nao_crash(self):
        self._patch_session_dir()
        for p in self.session_dir.glob("*.json"):
            p.unlink()
        result = session_mod.list_sessions()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_resume_corrompido_levanta_json_decode_error(self):
        self._patch_session_dir()
        sid = session_mod.save_session(
            session_id=None,
            messages=[{"role": "user", "content": "oi"}],
            provider="thc",
            model="m",
            mode="medium",
        )
        path = self.session_dir / f"{sid}.json"
        path.write_text("json quebrado {", encoding="utf-8")
        with self.assertRaises(Exception):
            session_mod.load_session(sid)

    def test_save_resume_preserva_agent_mode(self):
        self._patch_session_dir()
        sid = session_mod.save_session(
            session_id=None,
            messages=[{"role": "user", "content": "oi"}],
            provider="thc",
            model="m",
            mode="medium",
            agent_mode=True,
        )
        data = json.loads((self.session_dir / f"{sid}.json").read_text())
        self.assertTrue(data.get("agent_mode", False))

        session_mod.save_session(
            session_id=sid,
            messages=[{"role": "user", "content": "ola"}],
            provider="thc",
            model="m",
            mode="thinking",
            agent_mode=False,
        )
        data = json.loads((self.session_dir / f"{sid}.json").read_text())
        self.assertFalse(data.get("agent_mode", True))
        self.assertEqual(data["mode"], "thinking")

    def test_sessao_antiga_sem_agent_mode_nao_quebra(self):
        self._patch_session_dir()
        sid = session_mod.save_session(
            session_id=None,
            messages=[{"role": "user", "content": "oi"}],
            provider="thc",
            model="m",
            mode="medium",
        )
        path = self.session_dir / f"{sid}.json"
        data = json.loads(path.read_text())
        del data["agent_mode"]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        loaded = session_mod.load_session(sid)
        self.assertNotIn("agent_mode", loaded)

    def test_save_resume_sessao_grande(self):
        self._patch_session_dir()
        msgs = [{"role": "user", "content": "msg " + str(i)} for i in range(250)]
        sid = session_mod.save_session(
            session_id=None,
            messages=msgs,
            provider="thc",
            model="m",
            mode="medium",
            agent_mode=True,
        )
        loaded = session_mod.load_session(sid)
        self.assertEqual(len(loaded["messages"]), 250)
        self.assertEqual(loaded["messages"][0]["content"], "msg 0")
        self.assertEqual(loaded["messages"][-1]["content"], "msg 249")
        self.assertTrue(loaded.get("agent_mode", False))


class TestPreviewTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_preview_arquivo_novo_nao_crash(self):
        tool = FileWriteTool()
        original, new, error = tool.preview(path=str(self.dir / "novo.txt"), content="ola")
        self.assertEqual(original, "")
        self.assertEqual(new, "ola")
        self.assertIsNone(error)

    def test_write_preview_path_vazio_erro(self):
        tool = FileWriteTool()
        _, _, error = tool.preview()
        self.assertIsNotNone(error)
        self.assertIn("path", error.lower())

    def test_write_preview_arquivo_existente_le_conteudo(self):
        path = self.dir / "existente.txt"
        path.write_text("antigo", encoding="utf-8")
        tool = FileWriteTool()
        original, new, error = tool.preview(path=str(path), content="novo")
        self.assertIsNone(error)
        self.assertEqual(original, "antigo")
        self.assertEqual(new, "novo")

    def test_edit_preview_old_str_nao_encontrado(self):
        path = self.dir / "f.txt"
        path.write_text("abc")
        tool = FileEditTool()
        _, _, error = tool.preview(path=str(path), old_str="x", new_str="y")
        self.assertIsNotNone(error)
        self.assertIn("old_str", error)

    def test_edit_preview_old_str_unico_aceito(self):
        path = self.dir / "unico.txt"
        path.write_text("a b a", encoding="utf-8")
        tool = FileEditTool()
        original, new, error = tool.preview(path=str(path), old_str="b", new_str="z")
        self.assertIsNone(error)
        self.assertEqual(new, "a z a")

    def test_edit_preview_old_str_duplicado_retorna_erro(self):
        path = self.dir / "dup.txt"
        path.write_text("a b a", encoding="utf-8")
        tool = FileEditTool()
        _, _, error = tool.preview(path=str(path), old_str="a", new_str="z")
        self.assertIsNotNone(error)
        self.assertIn("específico", error)

    def test_edit_preview_nao_escreve_em_disco(self):
        path = self.dir / "readonly_edit.txt"
        original_text = "linha1\nlinha2\n"
        path.write_text(original_text)
        tool = FileEditTool()
        _, _, error = tool.preview(path=str(path), old_str="linha1", new_str="linhaX")
        self.assertIsNone(error)
        self.assertEqual(path.read_text(), original_text)

    def test_edit_preview_old_str_vazio_erro(self):
        tool = FileEditTool()
        _, _, error = tool.preview(path="qualquer", old_str="", new_str="x")
        self.assertIsNotNone(error)
        self.assertIn("old_str", error)

    def test_write_preview_permission_error(self):
        tool = FileWriteTool()
        with patch("builtins.open", side_effect=PermissionError("sem permissao")):
            with patch("os.path.exists", return_value=True):
                _, _, error = tool.preview(path="/tmp/qualquer.txt", content="x")
        self.assertIsNotNone(error)
        self.assertIn("permiss", error.lower())

    def test_todo_write_sem_preview(self):
        tool = TodoWriteTool()
        self.assertFalse(hasattr(tool, "preview"))

    def test_run_old_str_duplicado_retorna_erro_sem_aplicar(self):
        path = self.dir / "dup_run.txt"
        path.write_text("a b a", encoding="utf-8")
        tool = FileEditTool()
        result = tool.run(path=str(path), old_str="a", new_str="z")
        self.assertIn("específico", result)
        self.assertEqual(path.read_text(), "a b a")


class TestDiffOutput(unittest.TestCase):
    def test_print_diff_sem_diferenca(self):
        with patch("thc_cli.core.ui.console.print") as mock_print:
            print_diff("ola", "ola", "/tmp/x.txt")
        mock_print.assert_not_called()

    def test_print_diff_com_diferenca(self):
        with patch("thc_cli.core.ui.console.print") as mock_print:
            print_diff("linha original", "linha modificada", "/tmp/x.txt")
        self.assertTrue(mock_print.called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
