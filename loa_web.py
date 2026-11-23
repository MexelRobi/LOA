"""Authenticated, mobile-friendly web chat server for LOA."""

from __future__ import annotations

import hmac
import json
import getpass
import secrets
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import ollama
from ddgs.exceptions import DDGSException

from loa_auth import MIN_PASSWORD_LENGTH, has_web_password, save_web_password, verify_web_password
from loa_agent import (
    action_task,
    execute_approved_action,
    format_ollama_error,
    get_assistant_reply,
    public_action,
    restore_pending_action,
    sanitize_tool_content,
)
from loa_config import (
    AssistantConfig,
    build_system_prompt,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_PROMPT,
    default_config,
    default_ollama_host,
)
from loa_storage import (
    ChatSession,
    delete_chat_session,
    list_chat_sessions,
    load_chat_session,
    new_chat_session,
    rename_chat_session,
    save_chat_session,
    save_config,
)


MAX_REQUEST_BYTES = 1_000_000


def _session_transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            continue
        if role in {"user", "assistant"}:
            turn: dict[str, Any] = {
                "role": role,
                "content": (
                    sanitize_tool_content(str(message.get("content") or ""))
                    if role == "assistant"
                    else str(message.get("content") or "")
                ),
            }
            if turn["content"]:
                transcript.append(turn)
        elif role == "tool" and isinstance(message.get("ui"), dict):
            transcript.append(
                {
                    "role": "tool",
                    "content": str(message.get("content") or ""),
                    "ui": message["ui"],
                }
            )
    return transcript


def _save_active_chat(state: WebChatState) -> None:
    state.session["messages"] = [message.copy() for message in state.messages]
    save_chat_session(state.session)


class WebChatState:
    """Shared settings and conversation state for the single-user web server."""

    def __init__(self, config: AssistantConfig | None) -> None:
        self.config = config or default_config()
        self.configured = config is not None
        self.lock = threading.RLock()
        self.sessions: dict[str, float] = {}
        self.failed_logins: dict[str, list[float]] = {}
        self.session = new_chat_session()
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.config["prompt"],
                    self.session["id"],
                ),
            }
        ]
        self.session["messages"] = self.messages
        save_chat_session(self.session)
        self.transcript: list[dict[str, Any]] = []
        self.pending_action: dict[str, Any] | None = None
        self.pending_command: str | None = None
        self.pending_command_reason: str | None = None


class LoaHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying LOA's per-process chat state."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        state: WebChatState,
        secure: bool = False,
    ) -> None:
        self.state = state
        self.secure = secure
        super().__init__(address, handler)


class LoaRequestHandler(BaseHTTPRequestHandler):
    """Serve static web assets and authenticated chat API routes."""

    server: LoaHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        """Avoid logging browser sessions or other request details."""

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
            "frame-ancestors 'none'",
        )
        super().end_headers()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        asset_types = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if path in asset_types:
            name, content_type = asset_types[path]
            self._send_asset(name, content_type)
            return

        if path == "/auth/status":
            self._send_json({"password_configured": has_web_password()})
            return
        if path == "/auth/session":
            self._send_json({"authenticated": self._authorized()})
            return

        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, status=401)
            return

        with self.server.state.lock:
            if path == "/api/chats":
                self._send_json(
                    {
                        "active_chat_id": self.server.state.session["id"],
                        "chats": [
                            {
                                "id": chat["id"],
                                "title": chat["title"],
                                "updated_at": chat["updated_at"],
                                "summary": chat["summary"],
                            }
                            for chat in list_chat_sessions()
                        ],
                    }
                )
            elif path == "/api/config":
                self._send_json(
                    {
                        "name": self.server.state.config["name"],
                        "model": self.server.state.config["model"],
                        "prompt": self.server.state.config["prompt"],
                        "ollama_host": self.server.state.config["ollama_host"],
                        "default_ollama_host": DEFAULT_OLLAMA_HOST,
                        "default_prompt": DEFAULT_PROMPT,
                        "configured": self.server.state.configured,
                    }
                )
            elif path == "/api/history":
                self._send_json(
                    {
                        "transcript": self.server.state.transcript,
                        "active_chat_id": self.server.state.session["id"],
                        "pending_action": (
                            public_action(self.server.state.pending_action)
                            if self.server.state.pending_action
                            else None
                        ),
                        "pending_command": self.server.state.pending_command,
                        "command_reason": self.server.state.pending_command_reason,
                    }
                )
            else:
                self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            path = urlsplit(self.path).path
            if path == "/auth/login":
                self._login(payload)
                return
            with self.server.state.lock:
                if not self._authorized():
                    self._send_json({"error": "Unauthorized"}, status=401)
                    return
                if path == "/auth/logout":
                    self._logout()
                elif path == "/api/chats":
                    self._handle_chats(payload)
                elif path == "/api/settings":
                    self._save_settings(payload)
                elif path == "/api/chat":
                    self._send_message(payload)
                elif path == "/api/command":
                    self._handle_command(payload)
                else:
                    self._send_json({"error": "Not found"}, status=404)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
        except RuntimeError as error:
            self._send_json({"error": str(error)}, status=500)

    def _authorized(self) -> bool:
        cookie_header = self.headers.get("Cookie", "")
        cookies = {}
        for item in cookie_header.split(";"):
            key, separator, value = item.strip().partition("=")
            if separator:
                cookies[key] = value
        session = cookies.get("loa_session", "")
        now = time.monotonic()
        with self.server.state.lock:
            for saved_session, expiry in list(self.server.state.sessions.items()):
                if expiry <= now:
                    self.server.state.sessions.pop(saved_session, None)
                elif hmac.compare_digest(session, saved_session):
                    return True
            return False

    def _login(self, payload: dict[str, Any]) -> None:
        password = payload.get("password")
        if not isinstance(password, str):
            raise ValueError("Password is required.")
        client_ip = self.client_address[0]
        now = time.monotonic()
        attempts = [
            attempt
            for attempt in self.server.state.failed_logins.get(client_ip, [])
            if now - attempt < 900
        ]
        if len(attempts) >= 5:
            self.server.state.failed_logins[client_ip] = attempts
            self._send_json(
                {"error": "Too many failed attempts. Wait 15 minutes and try again."},
                status=429,
            )
            return
        if not verify_web_password(password):
            attempts.append(now)
            self.server.state.failed_logins[client_ip] = attempts
            self._send_json({"error": "Incorrect password."}, status=401)
            return

        session = secrets.token_urlsafe(32)
        self.server.state.sessions[session] = now + 12 * 60 * 60
        self.server.state.failed_logins.pop(client_ip, None)
        self._send_json({"ok": True}, cookie=session)

    def _logout(self) -> None:
        cookie_header = self.headers.get("Cookie", "")
        for item in cookie_header.split(";"):
            key, separator, value = item.strip().partition("=")
            if key == "loa_session" and separator:
                self.server.state.sessions.pop(value, None)
        self._send_json({"ok": True}, clear_cookie=True)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required.")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Invalid Content-Length.") from error
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is too large.")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("A JSON object is required.")
        return payload

    def _save_settings(self, payload: dict[str, Any]) -> None:
        config = {
            key: payload.get(key)
            for key in ("name", "model", "prompt")
        }
        if any(not isinstance(value, str) or not value.strip() for value in config.values()):
            raise ValueError(
                "Name, model, and system prompt must not be empty."
            )
        host_value = payload.get("ollama_host")
        if host_value is not None and not isinstance(host_value, str):
            raise ValueError("Ollama host URL must be text.")
        ollama_host = (
            host_value.strip()
            if isinstance(host_value, str) and host_value.strip()
            else default_ollama_host()
        )
        assistant_config: AssistantConfig = {
            "name": config["name"].strip(),
            "model": config["model"].strip(),
            "prompt": config["prompt"].strip(),
            "ollama_host": ollama_host,
        }
        save_config(assistant_config)
        state = self.server.state
        state.config = assistant_config
        state.configured = True
        if state.messages:
            state.messages[0] = {
                "role": "system",
                "content": build_system_prompt(
                    assistant_config["prompt"],
                    state.session["id"],
                ),
            }
        else:
            state.messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(
                        assistant_config["prompt"],
                        state.session["id"],
                    ),
                }
            ]
        _save_active_chat(state)
        self._send_json({"ok": True, "name": assistant_config["name"]})

    def _handle_chats(self, payload: dict[str, Any]) -> None:
        state = self.server.state
        operation = payload.get("action")
        if not isinstance(operation, str) or operation not in {
            "new",
            "open",
            "rename",
            "delete",
        }:
            raise ValueError("Choose new, open, rename, or delete for chat management.")
        if state.pending_action is not None:
            raise ValueError("Approve or cancel the pending action before managing chats.")

        if operation == "new":
            session = new_chat_session()
            self._activate_chat(session)
        else:
            chat_id = payload.get("chat_id")
            if not isinstance(chat_id, str):
                raise ValueError("A chat id is required.")
            if operation == "open":
                self._activate_chat(load_chat_session(chat_id))
            elif operation == "rename":
                title = payload.get("title")
                if not isinstance(title, str):
                    raise ValueError("A title is required.")
                renamed = rename_chat_session(chat_id, title)
                if state.session["id"] == chat_id:
                    state.session = renamed
                    _save_active_chat(state)
            elif operation == "delete":
                delete_chat_session(chat_id)
                if state.session["id"] == chat_id:
                    self._activate_chat(new_chat_session())

        self._send_json(
            {
                "ok": True,
                "active_chat_id": state.session["id"],
                "chat": {
                    "id": state.session["id"],
                    "title": state.session["title"],
                    "updated_at": state.session["updated_at"],
                    "summary": state.session["summary"],
                },
                "pending_action": (
                    public_action(state.pending_action)
                    if state.pending_action
                    else None
                ),
                "transcript": state.transcript,
            }
        )

    def _activate_chat(self, session: ChatSession) -> None:
        state = self.server.state
        state.session = session
        state.messages = [message.copy() for message in session["messages"]]
        system_prompt = build_system_prompt(
            state.config["prompt"],
            session["id"],
        )
        if state.messages and state.messages[0].get("role") == "system":
            state.messages[0]["content"] = system_prompt
        else:
            state.messages.insert(0, {"role": "system", "content": system_prompt})
        state.transcript = _session_transcript(state.messages)
        state.pending_action = restore_pending_action(state.messages)
        state.pending_command = (
            state.pending_action.get("command")
            if state.pending_action
            and state.pending_action.get("type") == "shell_command"
            else None
        )
        state.pending_command_reason = (
            str(state.pending_action.get("reason"))
            if state.pending_action
            else None
        )
        _save_active_chat(state)

    def _send_message(self, payload: dict[str, Any]) -> None:
        text = payload.get("message")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message must not be empty.")
        state = self.server.state
        if not state.configured:
            raise ValueError("Save assistant settings before sending a message.")
        if state.pending_action is not None:
            raise ValueError("Approve or cancel the pending action first.")

        text = text.strip()
        if state.messages and state.messages[0].get("role") == "system":
            state.messages[0]["content"] = build_system_prompt(
                state.config["prompt"],
                state.session["id"],
            )
        state.messages.append({"role": "user", "content": text})
        state.transcript.append({"role": "user", "content": text})
        _save_active_chat(state)
        try:
            answer = get_assistant_reply(
                state.config,
                state.messages,
                active_chat_id=state.session["id"],
            )
        except (
            ollama.ResponseError,
            httpx.HTTPError,
            DDGSException,
            OSError,
            ValueError,
        ) as error:
            state.messages.pop()
            state.transcript.pop()
            _save_active_chat(state)
            self._send_json(
                {
                    "error": (
                        "Could not complete the request: "
                        f"{format_ollama_error(error, state.config['ollama_host'])}"
                    )
                },
                status=502,
            )
            return

        state.messages = answer["history"]
        if answer["sources"]:
            assistant = next(
                (
                    message
                    for message in reversed(state.messages)
                    if message.get("role") == "assistant"
                ),
                None,
            )
            if assistant is not None:
                assistant.setdefault("ui", {})["sources"] = answer["sources"]
        action = answer["action"]
        state.pending_action = action
        command = answer["command"]
        state.pending_command = command
        state.pending_command_reason = answer["command_reason"]
        state.transcript = _session_transcript(state.messages)
        _save_active_chat(state)
        self._send_json(
            {
                "reply": answer["text"],
                "sources": answer["sources"],
                "command": command,
                "command_reason": answer["command_reason"],
                "action": public_action(action) if action else None,
                "tasks": answer["tasks"],
                "chat_id": state.session["id"],
                "transcript": state.transcript,
            }
        )

    def _handle_command(self, payload: dict[str, Any]) -> None:
        state = self.server.state
        action = state.pending_action
        if action is None:
            raise ValueError("There is no action awaiting approval.")
        approved = payload.get("approved")
        state.pending_action = None
        state.pending_command = None
        state.pending_command_reason = None
        if approved is not True:
            output = "Action was canceled; no changes were made."
        else:
            try:
                output = execute_approved_action(action)
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
            ) as error:
                output = f"Action execution failed: {error}"
        task = action_task(action, output)
        details = task["details"]
        state.messages.append(
            {
                "role": "tool",
                "tool_name": str(action["tool_name"]),
                "content": output,
                "ui": {
                    "type": "action",
                    "summary": task["label"],
                    "details": details,
                },
            }
        )
        state.transcript.append(
            {
                "role": "tool",
                "content": output,
                "ui": {
                    "type": "action",
                    "summary": task["label"],
                    "details": details,
                },
            }
        )
        action_task_entry = task
        _save_active_chat(state)
        try:
            answer = get_assistant_reply(
                state.config,
                state.messages,
                active_chat_id=state.session["id"],
            )
        except (
            ollama.ResponseError,
            httpx.HTTPError,
            DDGSException,
            OSError,
            ValueError,
        ) as error:
            assistant_error = (
                "Could not complete the follow-up: "
                f"{format_ollama_error(error, state.config['ollama_host'])}"
            )
            state.transcript.append({"role": "system", "content": assistant_error})
            self._send_json(
                {
                    "output": output,
                    "assistant_error": assistant_error,
                    "action_task": action_task_entry,
                    "tasks": [],
                    "transcript": state.transcript,
                }
            )
            return

        state.messages = answer["history"]
        if answer["sources"]:
            assistant = next(
                (
                    message
                    for message in reversed(state.messages)
                    if message.get("role") == "assistant"
                ),
                None,
            )
            if assistant is not None:
                assistant.setdefault("ui", {})["sources"] = answer["sources"]
        action = answer["action"]
        state.pending_action = action
        state.pending_command = answer["command"]
        state.pending_command_reason = answer["command_reason"]
        state.transcript = _session_transcript(state.messages)
        _save_active_chat(state)
        self._send_json(
            {
                "output": output,
                "reply": answer["text"],
                "sources": answer["sources"],
                "command": answer["command"],
                "command_reason": answer["command_reason"],
                "action": public_action(action) if action else None,
                "action_task": action_task_entry,
                "tasks": answer["tasks"],
                "chat_id": state.session["id"],
                "transcript": state.transcript,
            }
        )

    def _send_asset(self, name: str, content_type: str) -> None:
        asset_path = web_asset_directory() / name
        try:
            content = asset_path.read_bytes()
        except OSError as error:
            self._send_json({"error": f"Could not read web asset {name}: {error}"}, status=500)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(
        self,
        data: object,
        status: int = 200,
        cookie: str | None = None,
        clear_cookie: bool = False,
    ) -> None:
        content = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        if cookie is not None:
            secure_attribute = "; Secure" if self.server.secure else ""
            self.send_header(
                "Set-Cookie",
                "loa_session="
                f"{cookie}; Path=/; HttpOnly; SameSite=Strict; Max-Age=43200"
                f"{secure_attribute}",
            )
        elif clear_cookie:
            self.send_header(
                "Set-Cookie",
                "loa_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0",
            )
        self.end_headers()
        self.wfile.write(content)


def web_asset_directory() -> Path:
    """Locate web assets both in source checkouts and PyInstaller bundles."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / "web"
    return Path(__file__).resolve().parent / "web"


def lan_ip_address() -> str | None:
    """Find the preferred LAN address without sending network traffic."""
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect(("192.0.2.1", 80))
        address = connection.getsockname()[0]
        return address if not address.startswith("127.") else None
    except OSError:
        return None
    finally:
        connection.close()


def run_web_server(
    config: AssistantConfig | None,
    host: str = "0.0.0.0",
    port: int = 8765,
    reset_password: bool = False,
    certfile: str | None = None,
    keyfile: str | None = None,
) -> None:
    """Serve LOA to a mobile browser after one-time local password setup."""
    if reset_password or not has_web_password():
        print(f"Create a password of at least {MIN_PASSWORD_LENGTH} characters for the mobile web chat.")
        while True:
            password = getpass.getpass("New web password: ")
            confirmation = getpass.getpass("Confirm web password: ")
            if password != confirmation:
                print("Passwords did not match. Try again.")
                continue
            try:
                save_web_password(password)
            except (OSError, ValueError, RuntimeError) as error:
                raise RuntimeError(f"Could not save the web password: {error}") from error
            break

    if bool(certfile) != bool(keyfile):
        raise ValueError("Both a TLS certificate and private key are required for HTTPS.")
    state = WebChatState(config)
    server = LoaHTTPServer((host, port), LoaRequestHandler, state, secure=bool(certfile))
    scheme = "https" if certfile else "http"
    if certfile and keyfile:
        import ssl

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=certfile, keyfile=keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    actual_port = server.server_address[1]
    print(f"LOA web chat is listening on port {actual_port}.")
    print("Open one of these addresses on your phone or computer on the same Wi-Fi network:")
    addresses = ["127.0.0.1"]
    if host == "0.0.0.0":
        local_address = lan_ip_address()
        if local_address is not None:
            addresses.append(local_address)
        if not certfile:
            print("This server uses unencrypted HTTP. Only use it on a trusted private network.")
    for address in addresses:
        url = f"{scheme}://{address}:{actual_port}/"
        print(f"  {url}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
