class BaseProvider:
    name: str = ""

    def chat_completion(self, messages: list[dict], model: str, max_tokens: int = 8192,
                        temperature: float = 0.7) -> dict:
        raise NotImplementedError

    def list_models(self) -> list[dict]:
        raise NotImplementedError
