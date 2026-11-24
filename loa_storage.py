"""Load and save LOA settings in the user's application config directory."""

import json
import os
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from loa_config import (
    AssistantConfig,
    DEFAULT_PROMPT,
    default_ollama_host,
)


class ChatSession(TypedDict):
    """Persisted conversation and its locally generated search metadata."""

    id: str
    title: str
    created_at: str
    updated_at: str
    summary: str
    keywords: list[str]
    messages: list[dict[str, Any]]


def config_file_path() -> Path:
    """Return the per-user path used to persist assistant settings."""
    home = Path.home()
    if sys.platform == "darwin":
        config_dir = home / "Library" / "Application Support" / "LOA"
    elif os.name == "nt":
        config_dir = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "LOA"
    else:
        config_dir = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "loa"
    return config_dir / "config.json"


def chats_directory_path() -> Path:
    """Return the private directory that stores chat session JSON files."""
    return config_file_path().parent / "chats"


def chats_index_path() -> Path:
    """Return the lightweight cross-chat metadata index path."""
    return config_file_path().parent / "chats_index.json"


def new_chat_session() -> ChatSession:
    """Create an unsaved chat record with a unique id and empty transcript."""
    timestamp = _timestamp()
    return {
        "id": uuid.uuid4().hex,
        "title": "New Chat",
        "created_at": timestamp,
        "updated_at": timestamp,
        "summary": "New conversation.",
        "keywords": [],
        "messages": [],
    }


def list_chat_sessions() -> list[ChatSession]:
    """Load saved chat sessions, newest first."""
    directory = chats_directory_path()
    if not directory.exists():
        return []
    sessions = [
        _load_chat_file(path)
        for path in directory.glob("*.json")
        if path.is_file()
    ]
    return sorted(sessions, key=lambda chat: chat["updated_at"], reverse=True)


def load_chat_session(chat_id: str) -> ChatSession:
    """Load one saved chat by its validated id."""
    return _load_chat_file(_chat_file_path(chat_id))


def save_chat_session(session: ChatSession) -> None:
    """Update local metadata and atomically persist a chat and its index entry."""
    if not _valid_chat_id(session.get("id")):
        raise ValueError("Invalid chat id.")
    messages = session.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Chat messages must be a list.")

    now = _timestamp()
    session["updated_at"] = now
    _update_chat_metadata(session)
    directory = chats_directory_path()
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        _atomic_write_json(_chat_file_path(session["id"]), session)
        _update_chat_index(session)
    except OSError as error:
        raise RuntimeError(f"Could not save chat {session['id']}: {error}") from error


def rename_chat_session(chat_id: str, title: str) -> ChatSession:
    """Rename a saved chat and refresh its compact index entry."""
    title = title.strip()
    if not title:
        raise ValueError("Chat title must not be empty.")
    session = load_chat_session(chat_id)
    session["title"] = title[:80]
    save_chat_session(session)
    return session


def delete_chat_session(chat_id: str) -> None:
    """Delete one named chat and remove it from the compact index."""
    path = _chat_file_path(chat_id)
    try:
        path.unlink(missing_ok=True)
        index = [entry for entry in _read_chat_index() if entry.get("id") != chat_id]
        _write_chat_index(index)
    except OSError as error:
        raise RuntimeError(f"Could not delete chat {chat_id}: {error}") from error


def build_chat_catalog(active_chat_id: str | None = None, limit: int = 15) -> str:
    """Format recent indexed summaries without loading full chat transcripts."""
    entries = [
        entry
        for entry in _read_chat_index()
        if isinstance(entry.get("id"), str) and entry.get("id") != active_chat_id
    ]
    entries.sort(
        key=lambda entry: str(entry.get("updated_at", "")),
        reverse=True,
    )
    entries = entries[:limit]
    if not entries:
        return "Saved past chats: none yet."
    lines = [
        "Saved past chats (use recall_past_chat(chat_id, query) when the user asks "
        "about earlier conversations):"
    ]
    for entry in entries:
        title = " ".join(str(entry.get("title") or "Untitled chat").split())
        summary = " ".join(str(entry.get("summary") or "No summary available.").split())
        lines.append(
            f"- ID: {entry['id']} | Date: {entry.get('updated_at', '')} | "
            f"Title: {title} | Summary: {summary}"
        )
    return "\n".join(lines)


def recall_past_chat(
    chat_id: str = "latest",
    query: str = "",
    exclude_chat_id: str | None = None,
) -> str:
    """Return a short, relevant excerpt from a saved conversation."""
    resolved_chat_id = _resolve_recall_chat_id(chat_id, exclude_chat_id)
    if resolved_chat_id is None:
        return f"No saved chat matches '{chat_id}'."
    session = load_chat_session(resolved_chat_id)
    query_words = set(_keywords(query))
    candidates: list[tuple[int, int, str]] = []
    for position, message in enumerate(session["messages"]):
        role = message.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        text = str(message.get("content") or "")
        display = message.get("display")
        if isinstance(display, dict):
            summary = display.get("summary")
            details = display.get("details")
            if summary:
                text = f"{summary}\n{details or text}"
        if not text.strip():
            continue
        overlap = len(query_words & set(_keywords(text)))
        candidates.append((overlap, position, f"{role}: {text.strip()}"))

    if not candidates:
        return "No readable messages were found in that saved chat."
    if query_words:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        relevant = [item for item in candidates if item[0] > 0][:6]
        if not relevant:
            relevant = sorted(candidates, key=lambda item: item[1], reverse=True)[:3]
    else:
        conversations: list[tuple[int, int, str]] = []
        for position, message in enumerate(session["messages"]):
            role = message.get("role")
            text = str(message.get("content") or "").strip()
            if role in {"user", "assistant"} and text:
                conversations.append((0, position, f"{role}: {text}"))
        if not conversations:
            return "No readable user or assistant messages were found in that saved chat."
        relevant = [
            conversations[0]
        ] if len(conversations) == 1 else [
            conversations[0],
            conversations[-1],
        ]
    excerpt = "\n".join(item[2] for item in sorted(relevant, key=lambda item: item[1]))
    header = f"Saved chat: {session['title']}\n"
    excerpt = excerpt[: max(0, 1200 - len(header))]
    return f"{header}{excerpt}"


def _resolve_recall_chat_id(
    selector: str,
    exclude_chat_id: str | None = None,
) -> str | None:
    if not isinstance(selector, str) or not selector.strip():
        selector = "latest"
    selector = selector.strip()
    normalized_selector = selector.casefold()
    entries = [
        entry
        for entry in _read_chat_index()
        if isinstance(entry.get("id"), str)
        and entry.get("id") != exclude_chat_id
    ]

    if normalized_selector in {"latest", "newest"}:
        entries.sort(
            key=lambda entry: str(entry.get("updated_at", "")),
            reverse=True,
        )
        if entries:
            return str(entries[0]["id"])
        sessions = [
            session
            for session in list_chat_sessions()
            if session["id"] != exclude_chat_id
        ]
        sessions.sort(key=lambda session: session["updated_at"], reverse=True)
        return sessions[0]["id"] if sessions else None

    if normalized_selector in {"first", "oldest"}:
        sessions = [
            session
            for session in list_chat_sessions()
            if session["id"] != exclude_chat_id
        ]
        if sessions:
            oldest = min(sessions, key=lambda session: session["created_at"])
            return oldest["id"]
        return None

    exact_match = next(
        (
            str(entry["id"])
            for entry in entries
            if str(entry["id"]).casefold() == normalized_selector
        ),
        None,
    )
    if exact_match is not None:
        return exact_match

    if _valid_chat_id(selector):
        return (
            None
            if selector.casefold() == (exclude_chat_id or "").casefold()
            else selector.lower()
        )

    title_matches = [
        entry
        for entry in entries
        if normalized_selector in str(entry.get("title", "")).casefold()
    ]
    title_matches.sort(
        key=lambda entry: str(entry.get("updated_at", "")),
        reverse=True,
    )
    if title_matches:
        return str(title_matches[0]["id"])
    if not title_matches:
        sessions = [
            session
            for session in list_chat_sessions()
            if session["id"] != exclude_chat_id
            and normalized_selector in session["title"].casefold()
        ]
        sessions.sort(key=lambda session: session["updated_at"], reverse=True)
        if sessions:
            return sessions[0]["id"]
    return None


def _chat_file_path(chat_id: str) -> Path:
    if not _valid_chat_id(chat_id):
        raise ValueError("Invalid chat id.")
    return chats_directory_path() / f"{chat_id}.json"


def _valid_chat_id(chat_id: object) -> bool:
    return isinstance(chat_id, str) and re.fullmatch(r"[a-fA-F0-9-]{32,36}", chat_id) is not None


def _load_chat_file(path: Path) -> ChatSession:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Saved chat does not exist: {path.stem}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read saved chat {path.stem}: {error}") from error
    if (
        not isinstance(data, dict)
        or not _valid_chat_id(data.get("id"))
        or not isinstance(data.get("messages"), list)
    ):
        raise RuntimeError(f"Saved chat {path.stem} is invalid.")
    return {
        "id": data["id"],
        "title": str(data.get("title") or "New Chat"),
        "created_at": str(data.get("created_at") or ""),
        "updated_at": str(data.get("updated_at") or ""),
        "summary": str(data.get("summary") or "Saved conversation."),
        "keywords": [
            str(keyword)
            for keyword in data.get("keywords", [])
            if isinstance(keyword, str)
        ],
        "messages": data["messages"],
    }


def _update_chat_metadata(session: ChatSession) -> None:
    user_messages = [
        str(message.get("content") or "").strip()
        for message in session["messages"]
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    ]
    if user_messages and session["title"] == "New Chat":
        session["title"] = _compact_text(user_messages[0], 9)[:80]
    if user_messages:
        session["summary"] = _compact_text(user_messages[-1], 20)
    all_user_text = " ".join(user_messages)
    session["keywords"] = [
        word for word, _ in Counter(_keywords(all_user_text)).most_common(8)
    ]


def _compact_text(value: str, word_limit: int) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", value.strip(), maxsplit=1)[0]
    words = first_sentence.split()
    summary = " ".join(words[:word_limit]).strip()
    if len(words) > word_limit:
        summary += "…"
    if summary and summary[-1] not in ".!?…":
        summary += "."
    return summary or "New conversation."


def _keywords(value: str) -> list[str]:
    stop_words = {
        "about", "after", "also", "and", "are", "can", "could", "from", "have",
        "help", "into", "make", "need", "please", "that", "the", "then", "this",
        "with", "would", "your",
    }
    return [
        word.lower()
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", value)
        if word.lower() not in stop_words
    ]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_chat_index() -> list[dict[str, Any]]:
    path = chats_index_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read the chat index at {path}: {error}") from error
    if not isinstance(data, list):
        raise RuntimeError(f"Chat index at {path} is invalid.")
    return [entry for entry in data if isinstance(entry, dict)]


def find_saved_chat_id(chat_id: str) -> str | None:
    """Return the canonical indexed id when the value is a 32-character hex id."""
    if re.fullmatch(r"[a-fA-F0-9]{32}", chat_id) is None:
        return None
    return next(
        (
            str(entry["id"])
            for entry in _read_chat_index()
            if isinstance(entry.get("id"), str)
            and entry["id"].casefold() == chat_id.casefold()
        ),
        None,
    )


def _update_chat_index(session: ChatSession) -> None:
    entry = {
        "id": session["id"],
        "title": session["title"],
        "updated_at": session["updated_at"],
        "summary": _compact_text(session["summary"], 15),
        "keywords": session["keywords"],
    }
    entries = [
        existing
        for existing in _read_chat_index()
        if existing.get("id") != session["id"]
    ]
    entries.append(entry)
    _write_chat_index(
        sorted(entries, key=lambda item: str(item.get("updated_at", "")), reverse=True)
    )


def _write_chat_index(entries: list[dict[str, Any]]) -> None:
    path = chats_index_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _atomic_write_json(path, entries)
    except OSError as error:
        raise RuntimeError(f"Could not save the chat index at {path}: {error}") from error


def _atomic_write_json(path: Path, data: object) -> None:
    temporary_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_config() -> AssistantConfig | None:
    """Load saved settings, returning None when the user has not set up LOA."""
    path = config_file_path()
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read LOA settings at {path}: {error}") from error

    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), str) for key in ("name", "model", "prompt")
    ):
        raise RuntimeError(
            f"LOA settings at {path} are invalid. Use /settings after startup to replace them."
        )
    stored_host = data.get("ollama_host")
    if isinstance(stored_host, str) and stored_host.strip():
        ollama_host = stored_host.strip()
        host_needs_save = ollama_host != stored_host
    else:
        ollama_host = default_ollama_host()
        host_needs_save = True

    config: AssistantConfig = {
        "name": data["name"],
        "model": data["model"],
        "prompt": data["prompt"],
        "ollama_host": ollama_host,
    }
    needs_save = host_needs_save
    if (
        "NEVER call recall_past_chat" not in config["prompt"]
        or "NEVER invent, guess, or fabricate file contents." not in config["prompt"]
    ):
        config["prompt"] = DEFAULT_PROMPT
        needs_save = True
    if needs_save:
        save_config(config)
    return config


def save_config(config: AssistantConfig) -> None:
    """Persist settings, creating the per-user config directory if needed."""
    path = config_file_path()
    host = config.get("ollama_host", "").strip()
    saved_config: AssistantConfig = {
        **config,
        "ollama_host": host or default_ollama_host(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(saved_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise RuntimeError(f"Could not save LOA settings at {path}: {error}") from error
