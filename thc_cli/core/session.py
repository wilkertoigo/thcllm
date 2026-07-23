import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SESSIONS_DIR = Path.home() / ".thc" / "sessions"


def _ensure_sessions_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _generate_session_id() -> str:
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = now.strftime("%f")[:4]
    return f"{timestamp}-{suffix}"


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def save_session(
    session_id: Optional[str],
    messages: list,
    provider: str,
    model: str,
    mode: str,
    agent_mode: bool = False,
) -> str:
    _ensure_sessions_dir()
    now = datetime.now(timezone.utc).isoformat()
    if session_id is None:
        session_id = _generate_session_id()
        data = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "provider": provider,
            "model": model,
            "mode": mode,
            "agent_mode": agent_mode,
            "messages": messages,
        }
    elif _session_path(session_id).exists():
        path = _session_path(session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"session_id": session_id}
        data["updated_at"] = now
        data["provider"] = provider
        data["model"] = model
        data["mode"] = mode
        data["agent_mode"] = agent_mode
        data["messages"] = messages
    else:
        raise FileNotFoundError(f"Sessão '{session_id}' não encontrada para atualizar.")
    path = _session_path(session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_id


def load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"Sessão não encontrada: {session_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions() -> list[dict]:
    _ensure_sessions_dir()
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        preview = ""
        for message in data.get("messages", []):
            if message.get("role") == "user" and message.get("content"):
                preview = message["content"][:50]
                break
        sessions.append({
            "session_id": data.get("session_id", path.stem),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "provider": data.get("provider", ""),
            "model": data.get("model", ""),
            "preview": preview,
        })
    return sessions
