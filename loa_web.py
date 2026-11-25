"""Authenticated, mobile-friendly web chat server for LOA."""

from __future__ import annotations

import hmac
import base64
import binascii
import io
import ipaddress
import json
import getpass
import logging
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import ollama
from ddgs.exceptions import DDGSException

from loa_auth import MIN_PASSWORD_LENGTH, has_web_password, save_web_password, verify_web_password
from loa_agent import (
    AgentCancelled,
    AgentCancellation,
    BREAK_LOOP_MESSAGE,
    CANCELED_ACTION_RESPONSE,
    action_task,
    configure_runtime_tools,
    check_image_generation_readiness,
    execute_approved_action,
    format_ollama_error,
    get_assistant_reply,
    public_action,
    restore_pending_action,
    run_scheduled_prompt,
    sanitize_tool_content,
    tool_output_security_metadata,
    wrap_untrusted_tool_output,
)
from loa_brain import (
    add_brain_entry,
    delete_brain_entry,
    extract_chat_memories,
    list_brain_entries,
    load_about_me,
    save_about_me,
)
from loa_cron import (
    create_cron_job,
    cron_scheduler,
    delete_cron_job,
    get_cron_job_history,
    list_cron_jobs,
    toggle_cron_job,
    update_cron_job,
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
    generated_images_directory_path,
)
from loa_mcp import mcp_manager
from loa_skills import (
    activate_skill_command,
    create_markdown_skill,
    list_user_skills,
    read_user_skill,
    reload_skill_modules,
    resolve_skill_command,
    save_user_skill,
)


MAX_REQUEST_BYTES = 1_000_000
MAX_IMAGE_UPLOAD_BYTES = 8_000_000
WEB_LOCK_TIMEOUT = 0.5
WEB_RUNTIME_WAIT_TIMEOUT = 3.0
WEB_SHUTDOWN_TIMEOUT = 0.25
TAILNET_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("fd7a:115c:a1e0::/48"),
)


def _http_auto_login_allowed(client_ip: str) -> bool:
    """Allow HTTP link-login only from loopback or a Tailscale address."""
    try:
        address = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return address.is_loopback or any(address in network for network in TAILNET_NETWORKS)


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
    if state.session is None:
        return
    state.session["messages"] = [message.copy() for message in state.messages]
    save_chat_session(state.session)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key.replace('_', ' ').capitalize()} is required.")
    return value.strip()


def _require_reference_image_editing(config: AssistantConfig) -> None:
    image_gen = config.get("image_gen", {})
    if not image_gen.get("enabled", True) or image_gen.get("provider") != "local_flux":
        raise ValueError("Reference image uploads require enabled image generation with the local FLUX provider.")


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{key.replace('_', ' ').capitalize()} must be text.")
    return value.strip() if isinstance(value, str) else None


def _public_image_gen_settings(config: AssistantConfig) -> dict[str, Any]:
    settings = config.get("image_gen", {})
    if not isinstance(settings, dict):
        settings = {}
    provider = settings.get("provider", "openai")
    environment_name = (
        "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    )
    saved_key = settings.get("api_key", "")
    return {
        "enabled": settings.get("enabled", True),
        "provider": provider,
        "model": settings.get("model", "gpt-image-1"),
        "api_key": "",
        "api_key_configured": bool(
            (isinstance(saved_key, str) and saved_key.strip())
            or (
                provider in {"openai", "gemini"}
                and os.environ.get(environment_name, "").strip()
            )
        ),
        "size": settings.get("size", "1024x1024"),
        "quality": settings.get("quality", "high"),
    }


class WebBusyError(RuntimeError):
    """A request cannot proceed while another operation holds shared state."""


class WebChatState:
    """Shared settings and conversation state for the single-user web server."""

    def __init__(self, config: AssistantConfig | None) -> None:
        self.config = config or default_config()
        self.configured = config is not None
        self.mcp_status: list[dict[str, Any]] = []
        self._runtime_ready = threading.Event()
        self._runtime_lifecycle_lock = threading.Lock()
        self._runtime_started = False
        self._runtime_stopping = False
        self._runtime_error: Exception | None = None
        self.lock = threading.RLock()
        self.auth_lock = threading.RLock()
        self.cancellation_lock = threading.Lock()
        self.active_cancellation: AgentCancellation | None = None
        self.sessions: dict[str, float] = {}
        self.failed_logins: dict[str, list[float]] = {}
        self.session: ChatSession | None = None
        self.messages: list[dict[str, Any]] = []
        self.transcript: list[dict[str, Any]] = []
        self.pending_action: dict[str, Any] | None = None
        self.pending_command: str | None = None
        self.pending_command_reason: str | None = None

    def start_runtime(self) -> None:
        with self._runtime_lifecycle_lock:
            if self._runtime_started:
                raise RuntimeError("The web runtime has already been started.")
            self._runtime_started = True
        threading.Thread(
            target=self._initialize_runtime,
            name="loa-web-runtime",
            daemon=True,
        ).start()

    def _initialize_runtime(self) -> None:
        try:
            self.mcp_status = configure_runtime_tools(self.config)
        except Exception as error:
            if not self._runtime_stopping:
                logging.getLogger(__name__).exception("Web runtime initialization failed.")
                self._runtime_error = error
        finally:
            with self._runtime_lifecycle_lock:
                self._runtime_ready.set()

    def wait_for_runtime(self, timeout: float | None = None) -> None:
        if not self._runtime_ready.wait(timeout):
            raise WebBusyError("LOA tools are still starting. Please try again shortly.")
        if self._runtime_stopping:
            raise RuntimeError("The web server is stopping.")
        if self._runtime_error is not None:
            raise RuntimeError(
                f"Could not initialize web tools: {self._runtime_error}"
            ) from self._runtime_error

    @contextmanager
    def request_lock(self):
        if not self.lock.acquire(timeout=WEB_LOCK_TIMEOUT):
            raise WebBusyError(
                "LOA is busy with another operation. Wait for it to finish or use Break Loop."
            )
        try:
            yield
        finally:
            self.lock.release()

    def stop_runtime(self) -> None:
        with self._runtime_lifecycle_lock:
            self._runtime_stopping = True
        with self.cancellation_lock:
            if self.active_cancellation is not None:
                self.active_cancellation.event.set()
        mcp_manager.close(timeout=WEB_SHUTDOWN_TIMEOUT, permanent=True)


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
        try:
            self._handle_get()
        except WebBusyError as error:
            self._send_json({"error": str(error)}, status=503)
        except (OSError, RuntimeError, ValueError) as error:
            self._send_json({"error": str(error)}, status=500)

    def _handle_get(self) -> None:
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
            self._send_json(
                {
                    "password_configured": has_web_password(),
                    "allow_http_auto_login": (
                        self.server.secure
                        or _http_auto_login_allowed(self.client_address[0])
                    ),
                }
            )
            return
        if path == "/auth/session":
            self._send_json({"authenticated": self._authorized()})
            return

        if not self._authorized():
            self._send_json({"error": "Unauthorized"}, status=401)
            return

        if path.startswith("/api/images/"):
            image_match = re.fullmatch(r"/api/images/([a-f0-9]{32}\.png)", path)
            if image_match is None:
                self._send_json({"error": "Not found"}, status=404)
                return
            image_path = generated_images_directory_path() / image_match.group(1)
            try:
                content = image_path.read_bytes()
            except FileNotFoundError:
                self._send_json({"error": "Not found"}, status=404)
                return
            except OSError as error:
                self._send_json(
                    {"error": f"Could not read generated image: {error}"},
                    status=500,
                )
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        with self.server.state.request_lock():
            if path == "/api/chats":
                self._send_json(
                    {
                        "active_chat_id": (
                            self.server.state.session["id"]
                            if self.server.state.session
                            else None
                        ),
                        "chats": [
                            {
                                "id": chat["id"],
                                "title": chat["title"],
                                "updated_at": chat["updated_at"],
                                "summary": chat["summary"],
                                "message_count": sum(
                                    1
                                    for message in chat["messages"]
                                    if message.get("role") in {"user", "assistant"}
                                    and str(message.get("content") or "").strip()
                                ),
                            }
                            for chat in list_chat_sessions()
                        ],
                    }
                )
            elif path == "/api/config":
                state = self.server.state
                self._send_json(
                    {
                        "name": state.config["name"],
                        "model": state.config["model"],
                        "prompt": state.config["prompt"],
                        "ollama_host": state.config["ollama_host"],
                        "image_gen": _public_image_gen_settings(state.config),
                        "mcp_servers": state.config.get("mcp_servers", []),
                        "skill_states": state.config.get("skill_states", {}),
                        "skills": list_user_skills(state.config.get("skill_states", {})),
                        "mcp_status": mcp_manager.statuses(),
                        "default_ollama_host": DEFAULT_OLLAMA_HOST,
                        "default_prompt": DEFAULT_PROMPT,
                        "configured": self.server.state.configured,
                        "profile": load_about_me(),
                        "brain_entries": list_brain_entries(),
                        "cron_jobs": list_cron_jobs(),
                    }
                )
            elif path == "/api/history":
                self._send_json(
                    {
                        "transcript": self.server.state.transcript,
                        "active_chat_id": (
                            self.server.state.session["id"]
                            if self.server.state.session
                            else None
                        ),
                        "pending_action": (
                            public_action(self.server.state.pending_action)
                            if self.server.state.pending_action
                            else None
                        ),
                        "pending_command": self.server.state.pending_command,
                        "command_reason": self.server.state.pending_command_reason,
                    }
                )
            elif path == "/api/brain":
                query = parse_qs(urlsplit(self.path).query).get("q", [""])[0].strip()
                self._send_json({"entries": list_brain_entries(query)})
            elif path == "/api/profile":
                self._send_json({"profile": load_about_me()})
            elif path == "/api/cronjobs":
                self._send_json({"jobs": list_cron_jobs()})
            elif path.startswith("/api/cronjobs/") and path.endswith("/history"):
                job_id = path.removeprefix("/api/cronjobs/").removesuffix("/history").strip("/")
                self._send_json(
                    {"history": get_cron_job_history(job_id), "job_id": job_id}
                )
            else:
                self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        try:
            path = urlsplit(self.path).path
            payload = self._read_json(
                limit=12_000_000 if path == "/api/images/upload" else MAX_REQUEST_BYTES
            )
            if path == "/auth/login":
                self._login(payload)
                return
            if path == "/api/break-loop":
                if not self._authorized():
                    self._send_json({"error": "Unauthorized"}, status=401)
                    return
                self._break_loop()
                return
            if not self._authorized():
                self._send_json({"error": "Unauthorized"}, status=401)
                return
            if path == "/api/image-gen/check":
                with self.server.state.request_lock():
                    config = deepcopy(self.server.state.config)
                self._check_image_generation(payload, config)
                return
            if path in {
                "/api/settings", "/api/skills", "/api/mcp", "/api/chat",
                "/api/command", "/api/cronjobs",
            }:
                self.server.state.wait_for_runtime(timeout=WEB_RUNTIME_WAIT_TIMEOUT)
            with self.server.state.request_lock():
                if path == "/auth/logout":
                    self._logout()
                elif path == "/api/chats":
                    self._handle_chats(payload)
                elif path == "/api/settings":
                    self._save_settings(payload)
                elif path == "/api/skills":
                    self._handle_skills(payload)
                elif path == "/api/mcp":
                    self._handle_mcp(payload)
                elif path == "/api/brain":
                    self._handle_brain(payload)
                elif path == "/api/profile":
                    self._save_profile(payload)
                elif path == "/api/cronjobs":
                    self._handle_cronjobs(payload)
                elif path == "/api/chat":
                    self._send_message(payload)
                elif path == "/api/images/upload":
                    self._upload_reference_image(payload)
                elif path == "/api/command":
                    self._handle_command(payload)
                else:
                    self._send_json({"error": "Not found"}, status=404)
        except WebBusyError as error:
            self._send_json({"error": str(error)}, status=503)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
        except httpx.HTTPError as error:
            self._send_json(
                {"error": f"Image generation connection check failed: {error}"},
                status=502,
            )
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
        with self.server.state.auth_lock:
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
        auto_login = payload.get("auto_login", False)
        if not isinstance(auto_login, bool):
            raise ValueError("Auto-login flag must be a boolean.")
        if auto_login and not (
            self.server.secure or _http_auto_login_allowed(self.client_address[0])
        ):
            self._send_json(
                {
                    "error": (
                        "HTTP auto-login is allowed only over HTTPS, localhost, "
                        "or a Tailscale connection."
                    )
                },
                status=403,
            )
            return
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
        with self.server.state.auth_lock:
            self.server.state.sessions[session] = now + 12 * 60 * 60
        self.server.state.failed_logins.pop(client_ip, None)
        self._send_json({"ok": True}, cookie=session)

    def _logout(self) -> None:
        cookie_header = self.headers.get("Cookie", "")
        with self.server.state.auth_lock:
            for item in cookie_header.split(";"):
                key, separator, value = item.strip().partition("=")
                if key == "loa_session" and separator:
                    self.server.state.sessions.pop(value, None)
        self._send_json({"ok": True}, clear_cookie=True)

    def _read_json(self, limit: int = MAX_REQUEST_BYTES) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required.")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Invalid Content-Length.") from error
        if length < 0 or length > limit:
            raise ValueError("Request body is too large.")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("A JSON object is required.")
        return payload

    def _upload_reference_image(self, payload: dict[str, Any]) -> None:
        _require_reference_image_editing(self.server.state.config)
        from PIL import Image, ImageOps, UnidentifiedImageError

        encoded = payload.get("data")
        if not isinstance(encoded, str):
            raise ValueError("An encoded reference image is required.")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Invalid reference image data.") from error
        if not data or len(data) > MAX_IMAGE_UPLOAD_BYTES:
            raise ValueError("Reference image must be at most 8 MB.")
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Reference image must be PNG, JPEG, or WebP.")
                if image.width * image.height > 16_000_000:
                    raise ValueError("Reference image exceeds 16 megapixels.")
                image.load()
                converted = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError(f"Invalid reference image: {error}") from error
        directory = generated_images_directory_path()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        filename = f"{uuid.uuid4().hex}.png"
        path = directory / filename
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, "wb") as file:
                converted.save(file, format="PNG")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self._send_json({"filename": filename, "image_url": f"/api/images/{filename}"})

    def _save_settings(self, payload: dict[str, Any]) -> None:
        state = self.server.state
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
        current_image_gen = state.config.get("image_gen", {})
        if not isinstance(current_image_gen, dict):
            current_image_gen = {}
        image_gen_payload = payload.get("image_gen", current_image_gen)
        if not isinstance(image_gen_payload, dict):
            raise ValueError("Image generation settings must be an object.")
        api_key = image_gen_payload.get("api_key", "")
        clear_api_key = image_gen_payload.get("clear_api_key", False)
        if not isinstance(api_key, str) or not isinstance(clear_api_key, bool):
            raise ValueError("Image generation API key settings are invalid.")
        saved_api_key = (
            ""
            if clear_api_key
            else api_key.strip()
            or str(current_image_gen.get("api_key", ""))
        )
        image_gen = {
            "enabled": image_gen_payload.get(
                "enabled",
                current_image_gen.get("enabled", True),
            ),
            "provider": image_gen_payload.get(
                "provider",
                current_image_gen.get("provider", "openai"),
            ),
            "model": image_gen_payload.get(
                "model",
                current_image_gen.get("model", "gpt-image-1"),
            ),
            "api_key": saved_api_key,
            "size": image_gen_payload.get(
                "size",
                current_image_gen.get("size", "1024x1024"),
            ),
            "quality": image_gen_payload.get(
                "quality",
                current_image_gen.get("quality", "high"),
            ),
        }
        assistant_config: AssistantConfig = {
            "name": config["name"].strip(),
            "model": config["model"].strip(),
            "prompt": config["prompt"].strip(),
            "ollama_host": ollama_host,
            "image_gen": image_gen,
            "mcp_servers": payload.get(
                "mcp_servers",
                state.config.get("mcp_servers", []),
            ),
            "skill_states": payload.get(
                "skill_states",
                state.config.get("skill_states", {}),
            ),
        }
        mcp_changed = assistant_config["mcp_servers"] != state.config.get("mcp_servers", [])
        save_config(assistant_config)
        reload_skill_modules()
        state.mcp_status = configure_runtime_tools(
            assistant_config,
            force_mcp_reload=mcp_changed,
            reuse_mcp_connections=not mcp_changed,
        )
        state.config = assistant_config
        state.configured = True
        if state.session is not None and state.messages:
            state.messages[0] = {
                "role": "system",
                "content": build_system_prompt(
                    assistant_config["prompt"],
                    state.session["id"],
                    assistant_config["skill_states"],
                ),
            }
            _save_active_chat(state)
        self._send_json(
            {
                "ok": True,
                "name": assistant_config["name"],
                "image_gen": _public_image_gen_settings(assistant_config),
                "mcp_status": state.mcp_status,
                "skills": list_user_skills(assistant_config["skill_states"]),
            }
        )

    def _check_image_generation(
        self, payload: dict[str, Any], config: AssistantConfig
    ) -> None:
        settings = dict(config.get("image_gen", {}))
        submitted = payload.get("image_gen")
        if submitted is not None:
            if not isinstance(submitted, dict):
                raise ValueError("Image generation settings must be an object.")
            if submitted.get("clear_api_key") is True:
                settings["api_key"] = ""
            elif isinstance(submitted.get("api_key"), str) and submitted["api_key"].strip():
                settings["api_key"] = submitted["api_key"].strip()
            settings.update(
                {
                    key: submitted[key]
                    for key in ("enabled", "provider", "model", "size", "quality")
                    if key in submitted
                }
            )
        config["image_gen"] = settings
        message = check_image_generation_readiness(config)
        self._send_json({"ok": True, "message": message})

    def _handle_skills(self, payload: dict[str, Any]) -> None:
        state = self.server.state
        operation = payload.get("action")
        if operation == "list":
            self._send_json(
                {
                    "ok": True,
                    "skills": list_user_skills(state.config.get("skill_states", {})),
                    "skill_states": state.config.get("skill_states", {}),
                    "mcp_status": mcp_manager.statuses(),
                }
            )
            return
        if operation == "read":
            name = payload.get("name")
            if not isinstance(name, str):
                raise ValueError("Skill name is required.")
            self._send_json({"ok": True, **read_user_skill(name)})
            return
        if operation == "create":
            name = payload.get("name")
            content = payload.get("content")
            if not isinstance(name, str) or not isinstance(content, str):
                raise ValueError("Skill name and Markdown content are required.")
            path = create_markdown_skill(name, content)
            state.config["skill_states"].setdefault(path.name, False)
            save_config(state.config)
        elif operation == "edit":
            name = payload.get("name")
            content = payload.get("content")
            if not isinstance(name, str) or not isinstance(content, str):
                raise ValueError("Skill name and content are required.")
            save_user_skill(name, content)
        elif operation == "toggle":
            name = payload.get("name")
            enabled = payload.get("enabled")
            if not isinstance(name, str) or not isinstance(enabled, bool):
                raise ValueError("Skill name and enabled state are required.")
            if not any(skill["name"] == name for skill in list_user_skills()):
                raise ValueError(f"Skill not found: {name}")
            state.config["skill_states"][name] = enabled
            save_config(state.config)
        elif operation != "reload":
            raise ValueError("Unsupported skills operation.")

        reload_skill_modules()
        configure_runtime_tools(state.config)
        if state.session is not None and state.messages and state.messages[0].get("role") == "system":
            state.messages[0]["content"] = build_system_prompt(
                state.config["prompt"],
                state.session["id"],
                state.config.get("skill_states", {}),
            )
            _save_active_chat(state)
        self._send_json(
            {
                "ok": True,
                "skills": list_user_skills(state.config.get("skill_states", {})),
                "skill_states": state.config.get("skill_states", {}),
                "mcp_status": mcp_manager.statuses(),
            }
        )

    def _handle_mcp(self, payload: dict[str, Any]) -> None:
        state = self.server.state
        name = payload.get("name")
        enabled = payload.get("enabled")
        if not isinstance(name, str) or not isinstance(enabled, bool):
            raise ValueError("MCP server name and enabled state are required.")
        server = next(
            (
                item
                for item in state.config.get("mcp_servers", [])
                if item.get("name") == name
            ),
            None,
        )
        if server is None:
            raise ValueError(f"MCP server not found: {name}")
        server["enabled"] = enabled
        save_config(state.config)
        state.mcp_status = configure_runtime_tools(
            state.config,
            force_mcp_reload=True,
        )
        self._send_json({"ok": True, "mcp_status": state.mcp_status})

    def _handle_brain(self, payload: dict[str, Any]) -> None:
        operation = payload.get("action")
        if operation == "add":
            content = payload.get("content")
            category = payload.get("category", "fact")
            if not isinstance(content, str) or not isinstance(category, str):
                raise ValueError("Brain entry content and category must be text.")
            entry = add_brain_entry(content, category)
            self._send_json({"ok": True, "entry": entry, "entries": list_brain_entries()})
        elif operation == "delete":
            entry_id = payload.get("id")
            if not isinstance(entry_id, str):
                raise ValueError("Brain entry id is required.")
            delete_brain_entry(entry_id)
            self._send_json({"ok": True, "entries": list_brain_entries()})
        elif operation == "search":
            query = payload.get("query", "")
            if not isinstance(query, str):
                raise ValueError("Brain search query must be text.")
            self._send_json({"ok": True, "entries": list_brain_entries(query)})
        else:
            raise ValueError("Choose add, delete, or search for Brain management.")

    def _save_profile(self, payload: dict[str, Any]) -> None:
        profile = payload.get("profile")
        if not isinstance(profile, dict):
            raise ValueError("About Me profile must be an object.")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in profile.items()):
            raise ValueError("About Me profile fields must be text.")
        saved = save_about_me(profile)
        state = self.server.state
        if state.session is not None and state.messages and state.messages[0].get("role") == "system":
            state.messages[0]["content"] = build_system_prompt(
                state.config["prompt"],
                state.session["id"],
                state.config.get("skill_states", {}),
            )
            _save_active_chat(state)
        self._send_json({"ok": True, "profile": saved})

    def _handle_cronjobs(self, payload: dict[str, Any]) -> None:
        operation = payload.get("action")
        if operation == "create":
            job = create_cron_job(
                _required_string(payload, "name"),
                _required_string(payload, "schedule"),
                _required_string(payload, "payload"),
                one_time=payload.get("one_time"),
                run_once=payload.get("run_once"),
                action_type=payload.get("action_type", "prompt"),
                auto_approve=payload.get("auto_approve", True),
            )
        elif operation == "update":
            job_id = _required_string(payload, "job_id")
            job = update_cron_job(
                job_id,
                name=_optional_string(payload, "name"),
                schedule=_optional_string(payload, "schedule"),
                payload=_optional_string(payload, "payload"),
                one_time=payload.get("one_time"),
                run_once=payload.get("run_once"),
                action_type=_optional_string(payload, "action_type"),
                auto_approve=payload.get("auto_approve"),
            )
        elif operation == "toggle":
            enabled = payload.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError("Cronjob enabled state must be a boolean.")
            job = toggle_cron_job(_required_string(payload, "job_id"), enabled)
        elif operation == "delete":
            job_id = _required_string(payload, "job_id")
            delete_cron_job(job_id)
            self._send_json({"ok": True, "jobs": list_cron_jobs()})
            return
        elif operation == "run":
            output = cron_scheduler.run_now(_required_string(payload, "job_id"))
            self._send_json(
                {
                    "ok": True,
                    "output": output,
                    "history": get_cron_job_history(_required_string(payload, "job_id")),
                }
            )
            return
        else:
            raise ValueError("Choose create, update, toggle, delete, or run for cronjobs.")
        self._send_json({"ok": True, "job": job, "jobs": list_cron_jobs()})

    def _handle_chats(self, payload: dict[str, Any]) -> None:
        state = self.server.state
        operation = payload.get("action")
        if not isinstance(operation, str) or operation not in {
            "new",
            "open",
            "home",
            "rename",
            "delete",
            "vectorize",
        }:
            raise ValueError("Choose new, open, home, rename, delete, or vectorize for chat management.")
        if operation == "home":
            self._deactivate_chat()
            self._send_json(
                {
                    "ok": True,
                    "active_chat_id": None,
                    "chat": None,
                    "transcript": [],
                    "pending_action": None,
                }
            )
            return
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
                if state.session is not None and state.session["id"] == chat_id:
                    state.session = renamed
                    _save_active_chat(state)
            elif operation == "delete":
                if state.session is not None and state.session["id"] == chat_id:
                    self._deactivate_chat()
                delete_chat_session(chat_id)
            elif operation == "vectorize":
                session = load_chat_session(chat_id)
                memories = extract_chat_memories(session["messages"])
                if not memories:
                    raise ValueError("No useful user facts or tokens were found; the chat was kept.")
                if state.session is not None and state.session["id"] == chat_id:
                    self._deactivate_chat()
                delete_chat_session(chat_id)
                self._send_json(
                    {
                        "ok": True,
                        "vectorized_count": len(memories),
                        "active_chat_id": (
                            state.session["id"] if state.session else None
                        ),
                        "transcript": state.transcript,
                        "pending_action": (
                            public_action(state.pending_action)
                            if state.pending_action
                            else None
                        ),
                    }
                )
                return

        self._send_json(
            {
                "ok": True,
                "active_chat_id": state.session["id"] if state.session else None,
                "chat": (
                    {
                        "id": state.session["id"],
                        "title": state.session["title"],
                        "updated_at": state.session["updated_at"],
                        "summary": state.session["summary"],
                    }
                    if state.session
                    else None
                ),
                "pending_action": (
                    public_action(state.pending_action)
                    if state.pending_action
                    else None
                ),
                "transcript": state.transcript,
            }
        )

    def _activate_chat(self, session: ChatSession, persist: bool = True) -> None:
        state = self.server.state
        state.session = session
        state.messages = [message.copy() for message in session["messages"]]
        system_prompt = build_system_prompt(
            state.config["prompt"],
            session["id"],
            state.config.get("skill_states", {}),
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
        if persist:
            _save_active_chat(state)

    def _deactivate_chat(self) -> None:
        state = self.server.state
        state.session = None
        state.messages = []
        state.transcript = []
        state.pending_action = None
        state.pending_command = None
        state.pending_command_reason = None

    def _begin_operation(self) -> AgentCancellation:
        cancellation = AgentCancellation()
        with self.server.state.cancellation_lock:
            self.server.state.active_cancellation = cancellation
        return cancellation

    def _finish_operation(self, cancellation: AgentCancellation) -> bool:
        with self.server.state.cancellation_lock:
            if self.server.state.active_cancellation is cancellation:
                self.server.state.active_cancellation = None
            return not cancellation.event.is_set()

    def _record_canceled_action(
        self,
        action: dict[str, Any],
    ) -> dict[str, str]:
        state = self.server.state
        task = action_task(action, CANCELED_ACTION_RESPONSE)
        tool_name = str(action.get("tool_name") or action.get("type") or "action")
        entry = {
            "role": "tool",
            "tool_name": tool_name,
            "content": CANCELED_ACTION_RESPONSE,
            "ui": {
                "type": "action",
                "summary": "Request canceled by user",
                "details": task["details"],
            },
        }
        state.messages.append(entry)
        state.transcript.append(
            {"role": "tool", "content": CANCELED_ACTION_RESPONSE, "ui": entry["ui"]}
        )
        return task

    def _record_loop_break(
        self,
        action: dict[str, Any] | None = None,
    ) -> dict[str, str] | None:
        state = self.server.state
        task = self._record_canceled_action(action) if action is not None else None
        state.pending_action = None
        state.pending_command = None
        state.pending_command_reason = None
        state.messages.append({"role": "user", "content": BREAK_LOOP_MESSAGE})
        state.transcript = _session_transcript(state.messages)
        _save_active_chat(state)
        return task

    def _break_loop(self) -> None:
        state = self.server.state
        with state.cancellation_lock:
            cancellation = state.active_cancellation
            if cancellation is not None:
                cancellation.cancel()
        if cancellation is not None:
            self._send_json({"ok": True, "active": True})
            return

        with state.lock:
            with state.cancellation_lock:
                cancellation = state.active_cancellation
                if cancellation is not None:
                    cancellation.cancel()
            if cancellation is not None:
                self._send_json({"ok": True, "active": True})
                return
            action = state.pending_action
            canceled_action_task = self._record_loop_break(action)
            self._send_json(
                {
                    "ok": True,
                    "active": False,
                    "action_task": canceled_action_task,
                    "transcript": state.transcript,
                }
            )

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
        reference = payload.get("image_filename")
        if reference is not None:
            _require_reference_image_editing(state.config)
            if (
                not isinstance(reference, str)
                or re.fullmatch(r"[a-f0-9]{32}\.png", reference) is None
                or not (generated_images_directory_path() / reference).is_file()
            ):
                raise ValueError("Invalid reference image.")
            text += f"\n[Reference image for editing: {reference}]"
        skill = resolve_skill_command(text, state.config.get("skill_states", {}))
        if state.session is None:
            self._activate_chat(new_chat_session(), persist=False)
        if skill is not None:
            activate_skill_command(state.messages, skill)
        if state.messages and state.messages[0].get("role") == "system":
            state.messages[0]["content"] = build_system_prompt(
                state.config["prompt"],
                state.session["id"],
                state.config.get("skill_states", {}),
            )
        user_message = {"role": "user", "content": text}
        if reference is not None:
            user_message["ui"] = {"type": "reference_image", "filename": reference}
        state.messages.append(user_message)
        state.transcript.append({"role": "user", "content": text})
        _save_active_chat(state)
        cancellation = self._begin_operation()
        try:
            answer = get_assistant_reply(
                state.config,
                state.messages,
                active_chat_id=state.session["id"],
                cancellation=cancellation,
            )
            if not self._finish_operation(cancellation):
                raise AgentCancelled()
        except AgentCancelled:
            self._record_loop_break()
            self._send_json(
                {
                    "reply": "",
                    "canceled": True,
                    "transcript": state.transcript,
                    "chat_id": state.session["id"],
                }
            )
            return
        except (
            ollama.ResponseError,
            httpx.HTTPError,
            DDGSException,
            OSError,
            RuntimeError,
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
        finally:
            self._finish_operation(cancellation)

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
            output = CANCELED_ACTION_RESPONSE
            action_task_entry = self._record_canceled_action(action)
            _save_active_chat(state)
            self._send_json(
                {
                    "output": output,
                    "canceled": True,
                    "action_task": action_task_entry,
                    "tasks": [],
                    "transcript": state.transcript,
                }
            )
            return

        cancellation = self._begin_operation()
        output: str | None = None
        action_task_entry: dict[str, str] | None = None
        try:
            try:
                output = execute_approved_action(action, cancellation)
            except AgentCancelled:
                canceled_action_task = self._record_loop_break(action)
                self._send_json(
                    {
                        "output": CANCELED_ACTION_RESPONSE,
                        "canceled": True,
                        "action_task": canceled_action_task,
                        "transcript": state.transcript,
                        "chat_id": state.session["id"],
                    }
                )
                return
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
            ) as error:
                output, _ = wrap_untrusted_tool_output(
                    str(action.get("tool_name") or action.get("type") or "action"),
                    f"Action execution failed: {error}",
                )
            task = action_task(action, output)
            details = task["details"]
            state.messages.append(
                {
                    "role": "tool",
                    "tool_name": str(action["tool_name"]),
                    "content": output,
                    **tool_output_security_metadata(output),
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
            answer = get_assistant_reply(
                state.config,
                state.messages,
                active_chat_id=state.session["id"],
                cancellation=cancellation,
            )
            if not self._finish_operation(cancellation):
                raise AgentCancelled()
        except AgentCancelled:
            self._record_loop_break()
            self._send_json(
                {
                    "output": output or CANCELED_ACTION_RESPONSE,
                    "canceled": True,
                    "action_task": action_task_entry,
                    "transcript": state.transcript,
                    "chat_id": state.session["id"],
                }
            )
            return
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
        finally:
            self._finish_operation(cancellation)

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
    server: LoaHTTPServer | None = None
    state: WebChatState | None = None
    restore_sigterm = threading.current_thread() is threading.main_thread()
    previous_sigterm_handler = (
        signal.getsignal(signal.SIGTERM) if restore_sigterm else None
    )
    if restore_sigterm:
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
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
        state.start_runtime()

        def run_scheduled_job(job: dict[str, Any]) -> str:
            state.wait_for_runtime()
            return run_scheduled_prompt(state.config, job)

        cron_scheduler.start(run_scheduled_job)
        server.serve_forever()
    finally:
        try:
            cron_scheduler.stop(timeout=WEB_SHUTDOWN_TIMEOUT)
        finally:
            try:
                if server is not None:
                    server.server_close()
            finally:
                try:
                    if state is not None:
                        state.stop_runtime()
                    else:
                        mcp_manager.close(timeout=WEB_SHUTDOWN_TIMEOUT, permanent=True)
                finally:
                    if restore_sigterm:
                        signal.signal(signal.SIGTERM, previous_sigterm_handler)
