import json
import unittest
import unittest.mock

import httpx

from thc_cli.core.agent import run_agent
from thc_cli.core.provider_fallback import ProviderRateLimitError
from thc_cli.core.providers import PROVIDER_CLASSES
from thc_cli.core.providers.groq_provider import GroqProvider
from thc_cli.core.providers.mistral_provider import MistralProvider
from thc_cli.core.providers.openrouter_provider import OpenRouterProvider
from thc_cli.core.providers.gemini_provider import GeminiProvider


def _make_429_response():
    request = httpx.Request("POST", "https://example.com")
    return httpx.Response(429, request=request, text="Too Many Requests")


class TestProviderRateLimit(unittest.TestCase):
    def test_openrouter_rate_limit(self):
        provider = OpenRouterProvider(api_key="fake-key")
        with unittest.mock.patch("httpx.Client.post", return_value=_make_429_response()):
            with self.assertRaises(ProviderRateLimitError) as ctx:
                provider.chat_completion(
                    messages=[{"role": "user", "content": "oi"}],
                    model="google/gemma-4-31b-it:free",
                )
        self.assertEqual(ctx.exception.provider_name, "openrouter")
        self.assertEqual(ctx.exception.status_code, 429)

    def test_mistral_rate_limit(self):
        provider = MistralProvider(api_key="fake-key")
        with unittest.mock.patch("httpx.Client.post", return_value=_make_429_response()):
            with self.assertRaises(ProviderRateLimitError) as ctx:
                provider.chat_completion(
                    messages=[{"role": "user", "content": "oi"}],
                    model="mistral-small-latest",
                )
        self.assertEqual(ctx.exception.provider_name, "mistral")
        self.assertEqual(ctx.exception.status_code, 429)

    def test_groq_rate_limit(self):
        provider = GroqProvider(api_key="fake-key")
        with unittest.mock.patch("httpx.Client.post", return_value=_make_429_response()):
            with self.assertRaises(ProviderRateLimitError) as ctx:
                provider.chat_completion(
                    messages=[{"role": "user", "content": "oi"}],
                    model="llama-3.1-8b-instant",
                )
        self.assertEqual(ctx.exception.provider_name, "groq")
        self.assertEqual(ctx.exception.status_code, 429)

    def test_gemini_rate_limit_via_status(self):
        provider = GeminiProvider(api_key="fake-key")
        with unittest.mock.patch("httpx.Client.post", return_value=_make_429_response()):
            with self.assertRaises(ProviderRateLimitError) as ctx:
                provider.chat_completion(
                    messages=[{"role": "user", "content": "oi"}],
                    model="gemini-3.5-flash",
                )
        self.assertEqual(ctx.exception.provider_name, "gemini")
        self.assertEqual(ctx.exception.status_code, 429)

    def test_gemini_rate_limit_via_body(self):
        provider = GeminiProvider(api_key="fake-key")
        response = httpx.Response(
            200,
            request=httpx.Request("POST", "https://example.com"),
            json={
                "error": {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"},
            },
        )
        with unittest.mock.patch("httpx.Client.post", return_value=response):
            with self.assertRaises(ProviderRateLimitError) as ctx:
                provider.chat_completion(
                    messages=[{"role": "user", "content": "oi"}],
                    model="gemini-3.5-flash",
                )
        self.assertEqual(ctx.exception.provider_name, "gemini")
        self.assertEqual(ctx.exception.status_code, 429)

    def test_run_agent_fallback_success_with_config_none(self):
        fake_reply = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        fake_provider = unittest.mock.Mock()
        fake_provider.name = "openrouter"
        fake_provider.chat_completion.return_value = fake_reply

        def _fake_get_provider(name, config):
            return fake_provider

        with unittest.mock.patch("thc_cli.core.agent.get_provider", side_effect=_fake_get_provider):
            with unittest.mock.patch("thc_cli.core.provider_fallback.get_provider", side_effect=_fake_get_provider):
                with unittest.mock.patch("thc_cli.core.agent.load_config", return_value={"provider_keys": {}}):
                    with unittest.mock.patch("thc_cli.core.agent.MemoryStore"):
                        reply = run_agent(
                            provider=fake_provider,
                            messages=[{"role": "user", "content": "oi"}],
                            model="mock",
                            fallback_order=["openrouter"],
                            config=None,
                            max_rounds=1,
                        )
        self.assertIn("ok", reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
