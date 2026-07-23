import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from thc_cli.core.memory import MemoryStore
from thc_cli.core.system_prompt import build_base_system_prompt
from thc_cli.core.tools.memory_tools import MemoryReadTool, MemoryWriteTool


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "memory.json")
        self.store = MemoryStore(path=self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_cria_entrada_com_id_e_timestamps(self):
        entry = self.store.add("fato teste")
        self.assertIn("id", entry)
        self.assertIn("created_at", entry)
        self.assertIn("updated_at", entry)
        self.assertEqual(entry["content"], "fato teste")
        self.assertFalse(entry["pinned"])

    def test_add_pinned_true(self):
        entry = self.store.add("fato pinned", pinned=True)
        self.assertTrue(entry["pinned"])

    def test_add_tags_vazias_default(self):
        entry = self.store.add("fato sem tags")
        self.assertEqual(entry["tags"], [])

    def test_get_retorna_entrada_existente(self):
        entry = self.store.add("fato")
        found = self.store.get(entry["id"])
        self.assertIsNotNone(found)
        self.assertEqual(found["content"], "fato")

    def test_get_retorna_none_se_nao_existe(self):
        self.assertIsNone(self.store.get("uuid-inexistente"))

    def test_list_all_vazia(self):
        self.assertEqual(self.store.list_all(), [])

    def test_list_all_com_entradas(self):
        self.store.add("um")
        self.store.add("dois")
        entries = self.store.list_all()
        self.assertEqual(len(entries), 2)

    def test_search_encontra_por_palavra_em_content(self):
        self.store.add("python é legal", tags=["lang"])
        results = self.store.search("python")
        self.assertEqual(len(results), 1)

    def test_search_encontra_por_palavra_em_tags(self):
        self.store.add("fato", tags=["python"])
        results = self.store.search("python")
        self.assertEqual(len(results), 1)

    def test_search_case_insensitive(self):
        self.store.add("Python é legal")
        results = self.store.search("PYTHON")
        self.assertEqual(len(results), 1)

    def test_search_sem_resultado_retorna_lista_vazia(self):
        self.store.add("qualquer coisa")
        results = self.store.search("naoexiste")
        self.assertEqual(results, [])

    def test_search_ordena_por_relevancia(self):
        self.store.add("python e java", tags=["python"])
        self.store.add("python apenas", tags=["python", "code"])
        results = self.store.search("python")
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0]["content"].count("python"), results[1]["content"].count("python"))

    def test_update_content(self):
        entry = self.store.add("original")
        updated = self.store.update(entry["id"], content="atualizado")
        self.assertIsNotNone(updated)
        self.assertEqual(updated["content"], "atualizado")

    def test_update_pinned(self):
        entry = self.store.add("fato")
        updated = self.store.update(entry["id"], pinned=True)
        self.assertIsNotNone(updated)
        self.assertTrue(updated["pinned"])

    def test_update_retorna_none_se_nao_existe(self):
        self.assertIsNone(self.store.update("uuid-inexistente", content="x"))

    def test_delete_existente_retorna_true(self):
        entry = self.store.add("fato")
        self.assertTrue(self.store.delete(entry["id"]))

    def test_delete_inexistente_retorna_false(self):
        self.assertFalse(self.store.delete("uuid-inexistente"))

    def test_get_pinned_retorna_so_pinned(self):
        self.store.add("pinned1", pinned=True)
        self.store.add("pinned2", pinned=True)
        self.store.add("nao pinned")
        pinned = self.store.get_pinned()
        self.assertEqual(len(pinned), 2)
        self.assertTrue(all(e["pinned"] for e in pinned))

    def test_escrita_atomica_nao_corrompe_em_erro(self):
        from unittest.mock import patch
        entry = self.store.add("estavel")
        with patch("os.replace", side_effect=OSError("falha simulada")):
            with self.assertRaises(OSError):
                self.store.add("novo")
        data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        ids = [e["id"] for e in data["entries"]]
        self.assertIn(entry["id"], ids)
        self.assertEqual(len(data["entries"]), 1)

    def test_arquivo_criado_automaticamente_se_nao_existe(self):
        new_path = os.path.join(self.tmp.name, "nova", "memoria.json")
        self.assertFalse(Path(new_path).exists())
        store = MemoryStore(path=new_path)
        store.add("fato")
        self.assertTrue(Path(new_path).exists())


class TestSystemPromptMemory(unittest.TestCase):
    def test_pinned_aparece_no_system_prompt(self):
        pinned = [{"content": "usuario prefere linux", "tags": ["pref"]}]
        prompt = build_base_system_prompt(pinned_memories=pinned)
        self.assertIn("Memória persistente", prompt)
        self.assertIn("usuario prefere linux", prompt)

    def test_sem_pinned_prompt_nao_muda(self):
        prompt = build_base_system_prompt()
        self.assertNotIn("Memória persistente", prompt)

    def test_lista_vazia_prompt_nao_muda(self):
        prompt = build_base_system_prompt(pinned_memories=[])
        self.assertNotIn("Memória persistente", prompt)


class TestMemoryTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "memory.json")
        self.store = MemoryStore(path=self.path)
        self.write_tool = MemoryWriteTool()
        self.read_tool = MemoryReadTool()
        self._patcher = unittest.mock.patch(
            "thc_cli.core.tools.memory_tools.MemoryStore",
            return_value=self.store,
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.tmp.cleanup()

    def test_memory_write_tool_cria_entrada(self):
        result = self.write_tool.run(content="fato teste")
        self.assertIn("Memória salva", result)

    def test_memory_write_tool_atualiza_com_memory_id(self):
        result1 = self.write_tool.run(content="original")
        memory_id = result1.split(":")[1].strip().split()[0]
        result2 = self.write_tool.run(content="atualizado", memory_id=memory_id)
        self.assertIn("Memória atualizada", result2)

    def test_memory_read_tool_sem_parametros_retorna_tudo(self):
        self.write_tool.run(content="um")
        self.write_tool.run(content="dois")
        result = self.read_tool.run()
        self.assertIn("um", result)
        self.assertIn("dois", result)

    def test_memory_read_tool_com_query(self):
        self.write_tool.run(content="python é legal", tags=["lang"])
        result = self.read_tool.run(query="python")
        self.assertIn("python", result)

    def test_memory_read_tool_pinned_only(self):
        self.write_tool.run(content="pinned1", pinned=True)
        self.write_tool.run(content="nao pinned")
        result = self.read_tool.run(pinned_only=True)
        self.assertIn("pinned1", result)
        self.assertNotIn("nao pinned", result)

    def test_memory_read_tool_sem_resultado(self):
        result = self.read_tool.run(query="naoexiste")
        self.assertEqual(result, "Nenhuma entrada encontrada.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
