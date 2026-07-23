import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class MemoryStore:
    def __init__(self, path: str = None):
        if path is None:
            path = str(Path.home() / ".thc" / "memory.json")
        self.path = Path(path)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._save({"entries": []})

    def _load(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def add(self, content: str, tags: list[str] = None, pinned: bool = False) -> dict:
        entry = {
            "id": str(uuid.uuid4()),
            "content": content,
            "tags": tags or [],
            "pinned": pinned,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        data = self._load()
        data["entries"].append(entry)
        self._save(data)
        return entry

    def get(self, memory_id: str) -> Optional[dict]:
        data = self._load()
        for entry in data.get("entries", []):
            if entry["id"] == memory_id:
                return entry
        return None

    def list_all(self) -> list[dict]:
        return list(self._load().get("entries", []))

    def search(self, query: str) -> list[dict]:
        words = [w.lower() for w in query.split() if w.strip()]
        if not words:
            return []
        data = self._load()
        scored = []
        for entry in data.get("entries", []):
            score = 0
            content_lower = entry.get("content", "").lower()
            tags_lower = [t.lower() for t in entry.get("tags", [])]
            for word in words:
                if word in content_lower:
                    score += 1
                if any(word in tag for tag in tags_lower):
                    score += 1
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored]

    def update(self, memory_id: str, content: str = None, tags: list[str] = None, pinned: bool = None) -> Optional[dict]:
        data = self._load()
        for entry in data.get("entries", []):
            if entry["id"] == memory_id:
                if content is not None:
                    entry["content"] = content
                if tags is not None:
                    entry["tags"] = tags
                if pinned is not None:
                    entry["pinned"] = pinned
                entry["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save(data)
                return entry
        return None

    def delete(self, memory_id: str) -> bool:
        data = self._load()
        entries = data.get("entries", [])
        for i, entry in enumerate(entries):
            if entry["id"] == memory_id:
                entries.pop(i)
                self._save(data)
                return True
        return False

    def get_pinned(self) -> list[dict]:
        return [e for e in self._load().get("entries", []) if e.get("pinned")]
