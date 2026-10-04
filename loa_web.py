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
    execute_shell_command,
    extract_bash_command,
    get_assistant_reply,
)
from loa_config import AssistantConfig, default_config
from loa_storage import save_config


MAX_REQUEST_BYTES = 1_000_000


class WebChatState:
    """Shared settings and conversation state for the single-user web server."""

    def __init__(self, config: AssistantConfig | None) -> None:
        self.config = config or default_config()
        self.configured = config is not None
        self.lock = threading.RLock()
        self.sessions: dict[str, float] = {}
        self.failed_logins: dict[str, list[float]] = {}
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self.config["prompt"]}
        ]
        self.transcript: list[dict[str, Any]] = []
        self.pending_command: str | None = None


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
            if path == "/api/config":
                self._send_json(
                    {
                        "name": self.server.state.config["name"],
                        "model": self.server.state.config["model"],
                        "prompt": self.server.state.config["prompt"],
                        "configured": self.server.state.configured,
                    }
                )
            elif path == "/api/history":
                self._send_json({"transcript": self.server.state.transcript})
            else:
                self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            path = urlsplit(self.path).path
            if path == "/auth/login":
                self._login(payload)
                return
            if not self._authorized():
                self._send_json({"error": "Unauthorized"}, status=401)
                return
            if path == "/auth/logout":
                self._logout()
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
            raise ValueError("Name, model, and system prompt must not be empty.")
        assistant_config: AssistantConfig = {
            "name": config["name"].strip(),
            "model": config["model"].strip(),
            "prompt": config["prompt"].strip(),
        }
        save_config(assistant_config)
        state = self.server.state
        state.config = assistant_config
        state.configured = True
        if state.messages:
            state.messages[0] = {"role": "system", "content": assistant_config["prompt"]}
        else:
            state.messages = [{"role": "system", "content": assistant_config["prompt"]}]
        self._send_json({"ok": True, "name": assistant_config["name"]})

    def _send_message(self, payload: dict[str, Any]) -> None:
        text = payload.get("message")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message must not be empty.")
        state = self.server.state
        if not state.configured:
            raise ValueError("Save assistant settings before sending a message.")
        if state.pending_command is not None:
            raise ValueError("Approve or cancel the pending command first.")

        text = text.strip()
        state.messages.append({"role": "user", "content": text})
        state.transcript.append({"role": "user", "content": text})
        try:
            answer = get_assistant_reply(state.config, state.messages)
        except (
            ollama.ResponseError,
            httpx.HTTPError,
            DDGSException,
            OSError,
            ValueError,
        ) as error:
            state.messages.pop()
            state.transcript.pop()
            self._send_json(
                {"error": f"Could not complete the request: {error}"},
                status=502,
            )
            return

        state.messages = answer["history"]
        command = extract_bash_command(answer["text"])
        if command:
            state.pending_command = command
        turn: dict[str, Any] = {
            "role": "assistant",
            "content": answer["text"],
            "sources": answer["sources"],
        }
        state.transcript.append(turn)
        self._send_json(
            {
                "reply": answer["text"],
                "sources": answer["sources"],
                "command": command,
                "transcript": state.transcript,
            }
        )

    def _handle_command(self, payload: dict[str, Any]) -> None:
        state = self.server.state
        if state.pending_command is None:
            raise ValueError("There is no command awaiting approval.")
        approved = payload.get("approved")
        command = state.pending_command
        state.pending_command = None
        if approved is not True:
            output = "Command was not run."
        else:
            try:
                output = execute_shell_command(command)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                output = f"Command execution failed: {error}"
        state.messages.append({"role": "system", "content": output})
        state.transcript.append({"role": "system", "content": output})
        self._send_json({"output": output, "transcript": state.transcript})

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
        if certfile:
            print("HTTPS is enabled. Your browser must trust the certificate for microphone access.")
        else:
            print("This server uses unencrypted HTTP. Only use it on a trusted private network.")
    for address in addresses:
        url = f"{scheme}://{address}:{actual_port}/"
        print(f"  {url}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
