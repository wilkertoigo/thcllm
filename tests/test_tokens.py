import sys
import unittest
from unittest.mock import MagicMock, patch

from thc_cli.core.tokens import count_tokens, count_tokens_messages


class TestCountTokens(unittest.TestCase):
    def test_string_nao_vazia_retorna_inteiro_positivo(self):
        text = "hello world"
        result = count_tokens(text)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_cresce_proporcionalmente(self):
        short = "a" * 10
        long = "a" * 100
        self.assertLess(count_tokens(short), count_tokens(long))

    def test_string_vazia_retorna_zero(self):
        self.assertEqual(count_tokens(""), 0)

    def test_fallback_sem_tiktoken_nao_crash(self):
        with patch.dict(sys.modules, {"tiktoken": None}):
            import importlib
            import thc_cli.core.tokens as tokens_mod
            importlib.reload(tokens_mod)
            result = tokens_mod.count_tokens("hello world")
            self.assertEqual(result, len("hello world") // 4)

    def test_count_tokens_messages_soma_corretamente(self):
        messages = [
            {"role": "user", "content": "primeira mensagem"},
            {"role": "assistant", "content": "segunda mensagem resposta"},
            {"role": "user", "content": "terceira mensagem"},
        ]
        total = count_tokens_messages(messages)
        expected = count_tokens("primeira mensagem") + count_tokens("segunda mensagem resposta") + count_tokens("terceira mensagem")
        self.assertEqual(total, expected)

    def test_count_tokens_messages_ignora_campos_nao_string(self):
        messages = [
            {"role": "user", "content": "ola"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": 123},
        ]
        total = count_tokens_messages(messages)
        self.assertEqual(total, count_tokens("ola"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
