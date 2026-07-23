class TodoWriteTool:
    name = "todo_write"
    description = "Gerencia uma lista de tarefas (todo) em memória. Cada chamada substitui a lista completa."
    input_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "Lista completa de itens de tarefa. Cada item deve ter id, content e status.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Identificador único da tarefa"},
                        "content": {"type": "string", "description": "Descrição da tarefa"},
                        "status": {
                            "type": "string",
                            "description": "Status da tarefa",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            }
        },
        "required": ["items"],
    }

    def run(self, items=None) -> str:
        if items is None or not isinstance(items, list):
            return "Erro: parâmetro 'items' é obrigatório e deve ser uma lista."
        normalized = []
        for item in items:
            item_id = item.get("id", "")
            content = item.get("content", "")
            status = item.get("status", "pending")
            if status not in {"pending", "in_progress", "completed"}:
                status = "pending"
            normalized.append({"id": str(item_id), "content": str(content), "status": status})
        self._items = normalized
        lines = []
        for item in normalized:
            symbol = " " if item["status"] == "pending" else "~" if item["status"] == "in_progress" else "x"
            lines.append(f"[{symbol}] {item['content']}")
        return "\n".join(lines)

    _items = []
