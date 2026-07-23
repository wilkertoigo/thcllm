import httpx


class WebFetchTool:
    name = "web_fetch"
    description = "Busca o conteúdo de texto de uma URL via HTTP GET"
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL completa a buscar (http/https)",
            }
        },
        "required": ["url"],
    }

    def run(self, url: str = "") -> str:
        if not url:
            return "Erro: parâmetro 'url' é obrigatório"
        try:
            with httpx.Client(follow_redirects=True, timeout=10) as client:
                resp = client.get(url)
            resp.raise_for_status()
            text = resp.text
            if len(text) > 12000:
                text = text[:12000] + "\n\n... (conteúdo truncado em 12000 caracteres)"
            return text
        except httpx.HTTPStatusError as e:
            return f"Erro HTTP {e.response.status_code} ao acessar {url}: {e.response.text[:500]}"
        except httpx.TimeoutException:
            return f"Erro: timeout de 10s ao acessar {url}"
        except Exception as e:
            return f"Erro ao acessar {url}: {e}"


class WebSearchTool:
    name = "web_search"
    description = "Busca na web usando DuckDuckGo, sem API key"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Termo de busca",
            },
            "max_results": {
                "type": "integer",
                "description": "Máximo de resultados (default: 4)",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def run(self, query: str = "", max_results: int = 4) -> str:
        if not query:
            return "Erro: parâmetro 'query' é obrigatório"
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region="br-pt", max_results=max_results))
            if not results:
                return f"Nenhum resultado encontrado para: {query}"
            lines = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                url = r.get("href") or r.get("url", "")
                lines.append(f"- {title}\n  {body}\n  Fonte: {url}")
            return "\n\n".join(lines)
        except ImportError:
            return "Erro: ddgs não instalado. Instale com: pip install ddgs"
        except Exception as e:
            return f"Erro na busca web: {e}"
