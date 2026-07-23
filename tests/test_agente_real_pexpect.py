import os
import sys
import tempfile
import unittest
from pathlib import Path

import pexpect


def _spawn_chat(*args):
    cmd = [sys.executable, "-m", "thc_cli", "chat"] + list(args)
    env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}
    return pexpect.spawn(
        " ".join(cmd),
        timeout=60,
        encoding="utf-8",
        echo=False,
        env=env,
    )


class TestAgenteRealEdit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.test_file = Path(self.tmp.name) / "teste_edit.txt"
        self.test_file.write_text("linha original\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_edicao_aceita(self):
        child = _spawn_chat(
            "--agent",
            "--provider", "thc",
            "--model", "ministral3b-mst",
            "--max-tokens", "2048",
        )
        try:
            child.expect("❯")
            prompt = f"edite o arquivo {self.test_file} substituindo 'linha original' por 'linha modificada'"
            child.sendline(prompt)
            child.expect(r"str_replace.*\? :")
            child.sendline("s")
            child.expect("❯")
            child.sendline("/sair")
            child.expect(pexpect.EOF, timeout=10)
            self.assertIn("linha modificada", self.test_file.read_text())
        finally:
            child.close(force=True)

    def test_edicao_cancelada(self):
        original = self.test_file.read_text()
        child = _spawn_chat(
            "--agent",
            "--provider", "thc",
            "--model", "ministral3b-mst",
            "--max-tokens", "2048",
        )
        try:
            child.expect("❯")
            prompt = f"Use a ferramenta str_replace para editar o arquivo {self.test_file} substituindo 'linha original' por 'linha modificada'"
            child.sendline(prompt)
            while True:
                idx = child.expect([r"❯", r".*\? :", pexpect.EOF, pexpect.TIMEOUT], timeout=40)
                if idx == 1:
                    child.sendline("n")
                else:
                    break
            self.assertEqual(self.test_file.read_text(), original)
            child.sendline("/sair")
            child.expect(pexpect.EOF, timeout=10)
        finally:
            child.close(force=True)

    def test_edicao_old_str_nao_encontrado(self):
        child = _spawn_chat(
            "--agent",
            "--provider", "thc",
            "--model", "ministral3b-mst",
            "--max-tokens", "2048",
        )
        try:
            child.expect("❯")
            prompt = f"Use a ferramenta str_replace no arquivo {self.test_file} substituindo 'trecho que nao existe' por 'outro'. Nao use outras ferramentas."
            child.sendline(prompt)
            for _ in range(8):
                idx = child.expect([r"❯", r".*\? :", pexpect.EOF, pexpect.TIMEOUT], timeout=45)
                if idx == 1:
                    if "bash" in child.after or "write_file" in child.after or "str_replace" in child.after:
                        child.sendline("n")
                    else:
                        child.sendline("s")
                elif idx == 0:
                    break
            saida = child.before
            self.assertNotIn("Permitir execução", saida)
            self.assertEqual(self.test_file.read_text(), "linha original\n")
            child.sendline("/sair")
            child.expect(pexpect.EOF, timeout=10)
        finally:
            child.close(force=True)

    def test_edicao_arquivo_binario(self):
        binario = Path(self.tmp.name) / "binario.dat"
        binario.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
        child = _spawn_chat(
            "--agent",
            "--provider", "thc",
            "--model", "ministral3b-mst",
            "--max-tokens", "2048",
        )
        try:
            child.expect("❯")
            prompt = f"sobrescreva o arquivo {binario} com o conteúdo 'texto simples'"
            child.sendline(prompt)
            child.expect("❯", timeout=50)
            saida = child.before
            self.assertNotIn("Traceback", saida)
            self.assertNotIn("Unhandled exception", saida)
            try:
                binario.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pass
            child.sendline("/sair")
            child.expect(pexpect.EOF, timeout=10)
        finally:
            child.close(force=True)

    def test_edicao_encoding_nao_utf8(self):
        latin1 = Path(self.tmp.name) / "latin1.txt"
        latin1.write_bytes("coração\n".encode("latin-1"))
        child = _spawn_chat(
            "--agent",
            "--provider", "thc",
            "--model", "ministral3b-mst",
            "--max-tokens", "2048",
        )
        try:
            child.expect("❯")
            prompt = f"Use a ferramenta str_replace para editar o arquivo {latin1} substituindo 'coração' por 'coraçao'. Nao use bash."
            child.sendline(prompt)
            for _ in range(10):
                idx = child.expect([r"❯", r".*\? :", pexpect.EOF, pexpect.TIMEOUT], timeout=40)
                if idx == 1:
                    child.sendline("s")
                elif idx == 0:
                    break
            saida = child.before
            self.assertIn("utf-8", saida, msg=f"Saída inesperada: {saida[:500]}")
        finally:
            child.close(force=True)

    def test_edicao_arquivo_novo(self):
        novo = Path(self.tmp.name) / "novo_arquivo.txt"
        self.assertFalse(novo.exists())
        child = _spawn_chat(
            "--agent",
            "--provider", "thc",
            "--model", "ministral3b-mst",
            "--max-tokens", "2048",
        )
        try:
            child.expect("❯")
            prompt = f"crie um arquivo novo em {novo} com o conteúdo 'hello world'"
            child.sendline(prompt)
            child.expect(r"write_file.*\? :")
            child.sendline("s")
            child.expect("❯", timeout=50)
            self.assertTrue(novo.exists())
            self.assertEqual(novo.read_text(encoding="utf-8"), "hello world")
            child.sendline("/sair")
            child.expect(pexpect.EOF, timeout=10)
        finally:
            child.close(force=True)

    def test_edicao_arquivo_grande_diff_nao_imprime_tudo(self):
        grande = Path(self.tmp.name) / "grande.txt"
        linhas = [f"linha {i}\n" for i in range(2000)]
        grande.write_text("".join(linhas), encoding="utf-8")
        child = _spawn_chat(
            "--agent",
            "--provider", "thc",
            "--model", "ministral3b-mst",
            "--max-tokens", "2048",
        )
        try:
            child.expect("❯")
            prompt = f"edite o arquivo {grande} trocando 'linha 1500' por 'linha MIL E QUINHENTOS'"
            child.sendline(prompt)
            child.expect(r"str_replace.*\? :", timeout=50)
            saida = child.before
            qtd_linhas = saida.count("\n")
            self.assertLess(qtd_linhas, 200)
            child.sendline("s")
            child.expect("❯", timeout=45)
            self.assertIn("linha MIL E QUINHENTOS", grande.read_text(encoding="utf-8"))
            child.sendline("/sair")
            child.expect(pexpect.EOF, timeout=10)
        finally:
            child.close(force=True)

    def test_plan_mode_aprovado_executa(self):
        child = _spawn_chat(
            "--agent",
            "--provider", "thc",
            "--model", "ministral3b-mst",
            "--max-tokens", "2048",
        )
        try:
            child.expect("❯")
            child.sendline("/plan")
            prompt = f"Use a ferramenta str_replace para editar o arquivo {self.test_file} substituindo 'linha original' por 'linha plan'"
            child.sendline(prompt)
            child.expect("Plano proposto")
            child.sendline("s")
            while True:
                idx = child.expect([r"❯", r".*\? :", pexpect.EOF, pexpect.TIMEOUT], timeout=50)
                if idx == 1:
                    child.sendline("s")
                else:
                    break
            self.assertIn("linha plan", self.test_file.read_text())
            child.sendline("/sair")
        finally:
            child.close(force=True)

    def test_plan_mode_recusado_nao_executa(self):
        original = self.test_file.read_text()
        child = _spawn_chat(
            "--agent",
            "--provider", "thc",
            "--model", "ministral3b-mst",
            "--max-tokens", "2048",
        )
        try:
            child.expect("❯")
            child.sendline("/plan")
            prompt = f"Use a ferramenta str_replace para editar o arquivo {self.test_file} substituindo 'linha original' por 'linha plan'"
            child.sendline(prompt)
            child.expect("Plano proposto")
            child.sendline("n")
            child.expect("❯", timeout=40)
            self.assertEqual(self.test_file.read_text(), original)
            child.sendline("/sair")
            child.expect(pexpect.EOF, timeout=10)
        finally:
            child.close(force=True)
    def test_alucinacao_fallback_llama_8b(self):
        import time
        prompts = [
            f"Use a ferramenta read_file para ler o arquivo {self.test_file}",
            f"Use a ferramenta write_file para criar o arquivo {self.tmp.name}/fallback1.txt com 'hello fallback'",
            f"Use a ferramenta str_replace para editar o arquivo {self.test_file} trocando 'linha original' por 'linha alterada'",
        ]
        for i, prompt in enumerate(prompts):
            if i > 0:
                time.sleep(5)
            child = _spawn_chat(
                "--agent",
                "--provider", "groq",
                "--model", "llama-3.1-8b-instant",
                "--max-tokens", "2048",
            )
            try:
                child.expect("❯", timeout=30)
                child.sendline(prompt)
                while True:
                    idx = child.expect([r"❯", r".*\? :", pexpect.EOF, pexpect.TIMEOUT], timeout=55)
                    if idx == 1:
                        child.sendline("s")
                    else:
                        break
                saida = child.before
                if "429" in saida or "Too Many Requests" in saida:
                    self.skipTest(f"Rate limit do tier free do Groq atingido no prompt: {prompt}")
                self.assertIn("✓", saida, msg=f"Sem confirmação de tool_result para prompt: {prompt}")
            finally:
                child.close(force=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)