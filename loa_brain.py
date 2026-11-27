"""Private, fully local memory storage and lightweight token-based retrieval."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


_TOKEN_PATTERN = re.compile(r"[^\W_][\w'-]{1,}", re.UNICODE)
_STOP_WORDS = {
    "about", "after", "also", "and", "are", "because", "been", "being", "but",
    "can", "could", "did", "does", "for", "from", "had", "has", "have", "her",
    "here", "him", "his", "how", "into", "its", "just", "like", "make", "more",
    "not", "our", "please", "she", "should", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "too", "use",
    "was", "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your",
}
_DURABLE_FACT_PATTERN = re.compile(
    r"\b(?:i(?:'m| am| prefer| like| love| enjoy| use| work| live| have| want| "
    r"need| usually| always| never| plan| decided)|my (?:name|goal|routine|"
    r"computer|mac|pc|project|preference|schedule|favorite|favourite)|"
    r"ich (?:mag|liebe|bevorzuge|nutze|arbeite|wohne|habe|möchte|will|"
    r"plane|verwende)|mein(?:e|en)? (?:name|ziel|routine|computer|mac|projekt|"
    r"präferenz|lieblings)|(?:we decided|let['’]s use|i choose|we chose))\b",
    re.IGNORECASE,
)
_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z][A-Za-z0-9'-]*|[A-Z]{2,})(?:\s+(?:[A-Z][a-z][A-Za-z0-9'-]*|[A-Z]{2,}))*\b"
)
_NON_ENTITY_WORDS = {"I", "My", "The", "We", "Please", "Can", "Could", "How", "What", "User", "Assistant"}


def brain_database_path() -> Path:
    from loa_storage import config_file_path

    return config_file_path().parent / "brain.sqlite3"


def _connect() -> sqlite3.Connection:
    path = brain_database_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS brain_entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                tokens TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS brain_search
            USING fts5(entry_id UNINDEXED, content, tokens, category,
                       tokenize='unicode61')"""
        )
        path.chmod(0o600)
        return connection
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not open the local Brain database: {error}") from error


@contextmanager
def _managed_connection() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def tokenize(text: str) -> list[str]:
    """Return normalized local tokens; no model or network service is used."""
    return [
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if token.casefold() not in _STOP_WORDS
    ]


def weighted_tokens(text: str, limit: int = 20) -> list[dict[str, Any]]:
    counts = Counter(tokenize(text))
    return [
        {"token": token, "weight": count}
        for token, count in counts.most_common(limit)
    ]


def add_brain_entry(content: str, category: str = "fact") -> dict[str, Any]:
    if not isinstance(content, str) or not isinstance(category, str):
        raise ValueError("Brain entry content and category must be text.")
    content = content.strip()
    category = category.strip().lower()
    if not content or len(content) > 4000:
        raise ValueError("Brain entries must contain 1 to 4,000 characters.")
    if not category or len(category) > 40:
        raise ValueError("Brain entry category must contain 1 to 40 characters.")
    tokens = weighted_tokens(content)
    entry = {
        "id": uuid.uuid4().hex,
        "content": content,
        "category": category,
        "tokens": tokens,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        with _managed_connection() as connection:
            connection.execute(
                "INSERT INTO brain_entries VALUES (?, ?, ?, ?, ?)",
                (
                    entry["id"],
                    entry["content"],
                    entry["category"],
                    json.dumps(tokens, ensure_ascii=False),
                    entry["created_at"],
                ),
            )
            connection.execute(
                "INSERT INTO brain_search(entry_id, content, tokens, category) "
                "VALUES (?, ?, ?, ?)",
                (entry["id"], content, " ".join(item["token"] for item in tokens), category),
            )
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not save a Brain entry: {error}") from error
    return entry


def list_brain_entries(
    query: str = "", limit: int | None = 200
) -> list[dict[str, Any]]:
    if not isinstance(query, str):
        raise ValueError("Brain search query must be text.")
    query = query.strip()
    try:
        with _managed_connection() as connection:
            if query:
                terms = list(dict.fromkeys(tokenize(query)))
                if not terms:
                    return []
                match = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
                statement = """SELECT e.* FROM brain_search AS f
                    JOIN brain_entries AS e ON e.id = f.entry_id
                    WHERE brain_search MATCH ?
                    ORDER BY bm25(brain_search), e.created_at DESC, e.rowid DESC"""
                parameters: tuple[Any, ...] = (match,)
            else:
                statement = "SELECT * FROM brain_entries ORDER BY created_at DESC, rowid DESC"
                parameters = ()
            if limit is not None:
                statement += " LIMIT ?"
                parameters += (min(max(limit, 1), 500),)
            rows = connection.execute(statement, parameters).fetchall()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not search the local Brain database: {error}") from error
    return [_entry_from_row(row) for row in rows]


def search_brain(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return relevant entries using local SQLite FTS5 ranking."""
    query = query.strip()
    if not query:
        return []
    return list_brain_entries(query, limit)


def delete_brain_entry(entry_id: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{32}", entry_id):
        raise ValueError("Invalid Brain entry id.")
    try:
        with _managed_connection() as connection:
            connection.execute("DELETE FROM brain_search WHERE entry_id = ?", (entry_id,))
            connection.execute("DELETE FROM brain_entries WHERE id = ?", (entry_id,))
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not delete Brain entry {entry_id}: {error}") from error


def extract_chat_memories(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract useful text and weighted tokens locally, without an LLM call."""
    user_text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    )
    candidates: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", user_text):
        sentence = sentence.strip(" \t-*")
        words = tokenize(sentence)
        if len(words) >= 3 and sentence not in candidates:
            candidates.append(sentence)
    token_counts = Counter(tokenize(user_text))
    candidates.sort(
        key=lambda sentence: (
            10 if _DURABLE_FACT_PATTERN.search(sentence) else 0,
            sum(token_counts[token] for token in set(tokenize(sentence))),
            len(tokenize(sentence)),
        ),
        reverse=True,
    )
    selected = candidates[:20]
    if not selected:
        return []
    saved: list[dict[str, Any]] = []
    for sentence in selected:
        category = "preference" if _DURABLE_FACT_PATTERN.search(sentence) else "chat fact"
        saved.append(add_brain_entry(sentence, category))
    key_tokens = weighted_tokens(user_text, 24)
    if key_tokens:
        saved.append(
            add_brain_entry(
                "Chat keywords: " + ", ".join(item["token"] for item in key_tokens),
                "chat keywords",
            )
        )
    entities = list(dict.fromkeys(
        entity for entity in _ENTITY_PATTERN.findall(user_text)
        if entity not in _NON_ENTITY_WORDS
    ))[:20]
    if entities:
        saved.append(add_brain_entry("Entities: " + ", ".join(entities), "entities"))
    return saved


def load_about_me() -> dict[str, str]:
    from loa_storage import config_file_path

    path = config_file_path().parent / "about_me.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read the About Me profile at {path}: {error}") from error
    if not isinstance(data, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in data.items()
    ):
        raise RuntimeError(f"The About Me profile at {path} is invalid.")
    return {key: value for key, value in data.items() if value.strip()}


def save_about_me(profile: dict[str, str]) -> dict[str, str]:
    allowed = {"name", "preferences", "hardware", "routines", "goals", "notes"}
    if any(key not in allowed or not isinstance(value, str) for key, value in profile.items()):
        raise ValueError("About Me profile fields are invalid.")
    clean = {key: value.strip()[:4000] for key, value in profile.items() if value.strip()}
    from loa_storage import config_file_path

    path = config_file_path().parent / "about_me.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.chmod(0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError as error:
        raise RuntimeError(f"Could not save the About Me profile at {path}: {error}") from error
    return clean


def about_me_context() -> str:
    profile = load_about_me()
    if not profile:
        return ""
    labels = {
        "name": "Name",
        "preferences": "Preferences",
        "hardware": "Hardware and environment",
        "routines": "Routines",
        "goals": "Goals",
        "notes": "Notes",
    }
    return "\n".join(
        f"- {labels.get(key, key)}: {value}"
        for key, value in profile.items()
    )[:8000]


def _entry_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        tokens = json.loads(row["tokens"])
    except (TypeError, json.JSONDecodeError):
        tokens = []
    return {
        "id": row["id"],
        "content": row["content"],
        "category": row["category"],
        "tokens": tokens,
        "created_at": row["created_at"],
    }
