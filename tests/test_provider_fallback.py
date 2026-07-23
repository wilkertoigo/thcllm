import json
import unittest
import unittest.mock

from thc_cli.core.config import load_config, DEFAULT_PROVIDER_FALLBACK_ORDER
from thc_cli.core.provider_fallback import chat_completion_with_fallback, ProviderRateLimitError
from thc_cli.core.providers import PROVIDER_CLASSES


class MockProvider:
    def __init__(self, name, fail_with_rate_limit=False, fail_with_other=False):
        self.name = name
        self._fail_with_rate_limit = fail_with_rate_limit
        self._fail_with_other = fail_with_other
        self.calls = []

    def chat_completion(self, messages, model, max_tokens=8192, temperature=0.7):
        self.calls.append({
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if self._fail_with_rate_limit:
            raise ProviderRateLimitError(self.name, status_code=429, message="Rate limited")
        if self._fail_with_other:
            raise ValueError("Some other error")
        return {
            "choices": [{"message": {"role": "assistant", "content": f"response from {self.name}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }


def _make_mocks(failures=None):
    failures = failures or {}
    mocks = {}
    for name, cls in PROVIDER_CLASSES.items():
        mock = MockProvider(
            name=name,
            fail_with_rate_limit=failures.get(name, False),
        )
        mocks[name] = mock
    return mocks


class TestFallback(unittest.TestCase):
    def test_fallback_usa_proximo_provider_em_rate_limit(self):
        mocks = _make_mocks({"openrouter": True})
        config = {"provider_fallback_order": ["openrouter", "mistral"]}
        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=lambda n, _: mocks[n]):
            result, used = chat_completion_with_fallback(
                messages=[{"role": "user", "content": "oi"}],
                model="mock-model",
                max_tokens=16,
                temperature=0.7,
                config=config,
                fallback_order=["openrouter", "mistral"],
                provider_name="openrouter",
            )
        self.assertEqual(used, "mistral")
        self.assertIn("mistral", result["choices"][0]["message"]["content"])
        self.assertEqual(len(mocks["mistral"].calls), 1)

    def test_fallback_pula_provider_ja_tentado(self):
        mocks = _make_mocks({"openrouter": True, "mistral": True})
        config = {"provider_fallback_order": ["openrouter", "mistral"]}
        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=lambda n, _: mocks[n]):
            with self.assertRaises(ProviderRateLimitError):
                chat_completion_with_fallback(
                    messages=[{"role": "user", "content": "oi"}],
                    model="mock-model",
                    max_tokens=16,
                    temperature=0.7,
                    config=config,
                    fallback_order=["openrouter", "mistral"],
                    provider_name="openrouter",
                )
        self.assertEqual(len(mocks["openrouter"].calls), 1)
        self.assertEqual(len(mocks["mistral"].calls), 1)
        self.assertEqual(len(mocks["groq"].calls), 0)

    def test_fallback_mantem_provider_original_se_sem_erro(self):
        mocks = _make_mocks()
        config = {"provider_fallback_order": ["openrouter", "mistral"]}
        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=lambda n, _: mocks[n]):
            result, used = chat_completion_with_fallback(
                messages=[{"role": "user", "content": "oi"}],
                model="mock-model",
                max_tokens=16,
                temperature=0.7,
                config=config,
                fallback_order=["openrouter", "mistral"],
                provider_name="openrouter",
            )
        self.assertEqual(used, "openrouter")
        self.assertEqual(len(mocks["openrouter"].calls), 1)
        self.assertEqual(len(mocks["mistral"].calls), 0)

    def test_fallback_todos_falham_levanta_erro_claro(self):
        mocks = _make_mocks({"openrouter": True, "mistral": True})
        config = {"provider_fallback_order": ["openrouter", "mistral"]}
        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=lambda n, _: mocks[n]):
            with self.assertRaises(ProviderRateLimitError):
                chat_completion_with_fallback(
                    messages=[{"role": "user", "content": "oi"}],
                    model="mock-model",
                    max_tokens=16,
                    temperature=0.7,
                    config=config,
                    fallback_order=["openrouter", "mistral"],
                    provider_name="openrouter",
                )

    def test_fallback_ordem_configuravel_via_config(self):
        mocks = _make_mocks({"mistral": True, "groq": True})
        config = {"provider_fallback_order": ["mistral", "groq"]}
        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=lambda n, _: mocks[n]):
            with self.assertRaises(ProviderRateLimitError):
                chat_completion_with_fallback(
                    messages=[{"role": "user", "content": "oi"}],
                    model="mock-model",
                    max_tokens=16,
                    temperature=0.7,
                    config=config,
                    fallback_order=["mistral", "groq"],
                    provider_name="mistral",
                )
        self.assertEqual(len(mocks["groq"].calls), 1)

    def test_fallback_provider_invalido_na_lista_e_ignorado(self):
        mocks = _make_mocks({"openrouter": True})
        config = {"provider_fallback_order": ["openrouter", "invalid_provider", "mistral"]}

        def _get_provider(name, cfg):
            if name not in mocks:
                raise ValueError(f"Provider desconhecido: {name}")
            return mocks[name]

        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=_get_provider):
            result, used = chat_completion_with_fallback(
                messages=[{"role": "user", "content": "oi"}],
                model="mock-model",
                max_tokens=16,
                temperature=0.7,
                config=config,
                fallback_order=["openrouter", "invalid_provider", "mistral"],
                provider_name="openrouter",
            )
        self.assertEqual(used, "mistral")
        self.assertEqual(len(mocks["mistral"].calls), 1)

    def test_fallback_contexto_mensagens_repassado_integralmente_ao_proximo_provider(self):
        mocks = _make_mocks({"openrouter": True})
        config = {"provider_fallback_order": ["openrouter", "mistral"]}
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "world"},
        ]
        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=lambda n, _: mocks[n]):
            result, used = chat_completion_with_fallback(
                messages=messages,
                model="mock-model",
                max_tokens=16,
                temperature=0.7,
                config=config,
                fallback_order=["openrouter", "mistral"],
                provider_name="openrouter",
            )
        self.assertEqual(used, "mistral")
        self.assertEqual(len(mocks["mistral"].calls), 1)
        self.assertEqual(mocks["mistral"].calls[0]["messages"], messages)

    def test_config_default_provider_e_openrouter(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"endpoint": "http://example.com"}, f)
            path = f.name
        with unittest.mock.patch("thc_cli.core.config.CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda self: path
            with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps({"endpoint": "http://example.com"}))):
                cfg = load_config()
        self.assertEqual(cfg["provider"], "openrouter")

    def test_config_le_provider_fallback_order_do_thcrc(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"provider_fallback_order": ["mistral", "groq"]}, f)
            path = f.name
        with unittest.mock.patch("thc_cli.core.config.CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda self: path
            with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps({"provider_fallback_order": ["mistral", "groq"]}))):
                cfg = load_config()
        self.assertEqual(cfg["provider_fallback_order"], ["mistral", "groq"])

    def test_config_fallback_order_default_sem_thcrc(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            path = f.name
        with unittest.mock.patch("thc_cli.core.config.CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda self: path
            with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps({}))):
                cfg = load_config()
        self.assertEqual(cfg["provider_fallback_order"], DEFAULT_PROVIDER_FALLBACK_ORDER)

    def test_fallback_http_429_via_mensagem(self):
        import httpx
        mocks = _make_mocks()
        mocks["openrouter"]._fail_with_rate_limit = False
        mocks["openrouter"]._fail_with_other = True
        
        request = httpx.Request("POST", "http://example.com")
        response = httpx.Response(429, request=request, text="Too Many Requests")
        mocks["openrouter"].chat_completion = unittest.mock.Mock(side_effect=httpx.HTTPStatusError("429", request=request, response=response))
        
        config = {"provider_fallback_order": ["openrouter", "mistral"]}
        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=lambda n, _: mocks[n]):
            result, used = chat_completion_with_fallback(
                messages=[{"role": "user", "content": "oi"}],
                model="mock-model",
                max_tokens=16,
                temperature=0.7,
                config=config,
                fallback_order=["openrouter", "mistral"],
                provider_name="openrouter",
            )
        self.assertEqual(used, "mistral")

    def test_fallback_nao_propaga_erro_nao_rate_limit(self):
        mocks = _make_mocks({"openrouter": False})
        mocks["openrouter"]._fail_with_rate_limit = False
        mocks["openrouter"]._fail_with_other = True
        mocks["openrouter"].chat_completion = unittest.mock.Mock(side_effect=ValueError("auth failed"))
        config = {"provider_fallback_order": ["openrouter", "mistral"]}
        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=lambda n, _: mocks[n]):
            with self.assertRaises(ValueError):
                chat_completion_with_fallback(
                    messages=[{"role": "user", "content": "oi"}],
                    model="mock-model",
                    max_tokens=16,
                    temperature=0.7,
                    config=config,
                    fallback_order=["openrouter", "mistral"],
                    provider_name="openrouter",
                )
        self.assertEqual(len(mocks["mistral"].calls), 0)

    def test_chat_single_shot_fallback_sucesso(self):
        from thc_cli.commands.chat import _provider_chat_with_fallback
        mocks = _make_mocks({"openrouter": True})
        config = {"provider_fallback_order": ["openrouter", "mistral"], "provider_keys": {}}
        with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=lambda n, _: mocks[n]):
            result, used = _provider_chat_with_fallback(
                provider=mocks["openrouter"],
                provider_name="openrouter",
                messages=[{"role": "user", "content": "oi"}],
                model="mock-model",
                mode="medium",
                web=False,
                max_tokens=16,
                temperature=0.7,
                config=config,
                fallback_order=["openrouter", "mistral"],
            )
        self.assertEqual(used, "mistral")
        self.assertIn("mistral", result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
