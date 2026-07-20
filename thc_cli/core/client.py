import httpx


class THCAPIError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class THCClient:
    def __init__(self, config: dict):
        self.endpoint = config["endpoint"].rstrip("/")
        self.api_key = config.get("api_key")
        self.timeout = 120.0
    
    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-THC-Key"] = self.api_key
        return headers
    
    def _request(self, method: str, url: str, **kwargs) -> dict:
        kwargs.setdefault("headers", {}).update(self._get_headers())
        kwargs.setdefault("timeout", self.timeout)
        
        with httpx.Client() as client:
            response = client.request(method, url, **kwargs)
        
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except:
                detail = response.text
            raise THCAPIError(detail)
        
        return response.json()
    
    def chat_completion(
        self,
        messages: list[dict],
        model: str = None,
        mode: str = "medium",
        web: bool = False,
        max_tokens: int = 8192,
        temperature: float = 0.7
    ) -> dict:
        body = {
            "messages": messages,
            "mode": mode,
            "web": web,
            "free_mode": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model:
            body["model"] = model
        
        return self._request(
            "POST",
            f"{self.endpoint}/v1/chat/completions",
            json=body
        )
    
    def list_models(self) -> dict:
        return self._request("GET", f"{self.endpoint}/v1/models")
    
    def get_quota(self) -> dict:
        return self._request("GET", f"{self.endpoint}/v1/quota")