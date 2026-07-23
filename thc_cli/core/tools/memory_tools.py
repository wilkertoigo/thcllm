from ..memory import MemoryStore


class MemoryWriteTool:
    name = "memory_write"
    description = "Salva um fato ou contexto na memória persistente entre sessões. Use pinned=true para fatos que devem sempre estar disponíveis (preferências do usuário, contexto do projeto atual). Use pinned=false para fatos gerais consultáveis."
    input_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "O fato a salvar",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Lista de tags",
                "default": [],
            },
            "pinned": {
                "type": "boolean",
                "description": "Se true, sempre injetado no system prompt",
                "default": False,
            },
            "memory_id": {
                "type": "string",
                "description": "Se fornecido, atualiza entrada existente em vez de criar nova",
            },
        },
        "required": ["content"],
    }

    def run(self, content: str = "", tags: list = None, pinned: bool = False, memory_id: str = "") -> str:
        if not content:
            return "Erro: parâmetro 'content' é obrigatório"
        store = MemoryStore()
        if memory_id:
            entry = store.update(memory_id, content=content, tags=tags, pinned=pinned)
            if entry is None:
                return f"Erro: entrada não encontrada: {memory_id}"
            return f"Memória atualizada: {entry['id']}"
        entry = store.add(content, tags=tags, pinned=pinned)
        return f"Memória salva: {entry['id']} (pinned={pinned})"


class MemoryReadTool:
    name = "memory_read"
    description = "Consulta a memória persistente. Se query for fornecido, busca por palavras-chave. Se memory_id for fornecido, retorna entrada específica. Sem parâmetros, retorna todas as entradas."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Busca por palavras-chave",
            },
            "memory_id": {
                "type": "string",
                "description": "ID específico da memória",
            },
            "pinned_only": {
                "type": "boolean",
                "description": "Retorna só entradas pinned",
                "default": False,
            },
        },
    }

    def run(self, query: str = "", memory_id: str = "", pinned_only: bool = False) -> str:
        store = MemoryStore()
        if memory_id:
            entry = store.get(memory_id)
            if entry is None:
                return "Nenhuma entrada encontrada."
            import json
            return json.dumps([entry], ensure_ascii=False, indent=2)
        if query:
            entries = store.search(query)
        elif pinned_only:
            entries = store.get_pinned()
        else:
            entries = store.list_all()
        if not entries:
            return "Nenhuma entrada encontrada."
        import json
        return json.dumps(entries, ensure_ascii=False, indent=2)
