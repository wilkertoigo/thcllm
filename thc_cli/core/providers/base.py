class ProviderRateLimitError(Exception):
    def __init__(self, provider_name: str, status_code: int = None, message: str = ""):
        self.provider_name = provider_name
        self.status_code = status_code
        super().__init__(message or f"{provider_name}: rate limit ou quota excedida")


class BaseProvider:
    name: str = ""

    def chat_completion(self, messages: list[dict], model: str, max_tokens: int = 8192,
                        temperature: float = 0.7) -> dict:
        raise NotImplementedError

    def list_models(self) -> list[dict]:
        raise NotImplementedError
