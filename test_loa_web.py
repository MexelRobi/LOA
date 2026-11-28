import base64
import signal
import http.client
import concurrent.futures
import json
import os
import select
import subprocess
import sys
import threading
import tempfile
import time
import unittest
import base64
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import loa_web
import loa_storage


class WebIconTests(unittest.TestCase):
    def test_icon_assets_are_served_without_authentication(self):
        state = loa_web.WebChatState(loa_web.default_config())
        server = loa_web.LoaHTTPServer(("127.0.0.1", 0), loa_web.LoaRequestHandler, state)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            for name, content_type in (
                ("favicon.ico", "image/x-icon"),
                ("apple-touch-icon.png", "image/png"),
            ):
                with self.subTest(name=name):
                    connection.request("GET", f"/{name}")
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.getheader("Content-Type"), content_type)
                    self.assertEqual(
                        response.read(), (loa_web.web_asset_directory() / name).read_bytes()
                    )
        finally:
            connection.close()
            server.shutdown()
            thread.join()
            server.server_close()


class ChatFileUploadTests(unittest.TestCase):
    def test_upload_preserves_general_file_bytes_and_private_permissions(self):
        data = b"print('hello')\n"
        handler = object.__new__(loa_web.LoaRequestHandler)
        handler.server = SimpleNamespace(state=SimpleNamespace())
        handler._send_json = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(loa_web, "chat_attachments_directory_path", return_value=Path(directory)):
                handler._upload_chat_file({
                    "filename": "example.py",
                    "data": base64.b64encode(data).decode("ascii"),
                })
                result = handler._send_json.call_args.args[0]
                saved = Path(directory) / result["id"]
                self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
                self.assertEqual(saved.read_bytes(), data)
                self.assertEqual(result["filename"], "example.py")
                with self.assertRaises(ValueError):
                    handler._upload_chat_file({"filename": "bad.txt", "data": "invalid!"})

    def test_upload_route_is_independent_of_image_generation_settings(self):
        file_bytes = b"%PDF-1.7\nfile contents"
        body = json.dumps({
            "filename": "report.pdf",
            "data": base64.b64encode(file_bytes).decode("ascii"),
        })
        config = loa_web.default_config()
        config["image_gen"].update(enabled=False)
        state = loa_web.WebChatState(config)
        server = loa_web.LoaHTTPServer(("127.0.0.1", 0), loa_web.LoaRequestHandler, state)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(loa_web, "chat_attachments_directory_path", return_value=Path(directory)),
                patch.object(loa_web.LoaRequestHandler, "_authorized", return_value=True) as authorized,
            ):
                connection.request("POST", "/api/files/upload", body, {"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                result = json.loads(response.read())
                saved = Path(directory) / result["id"]
                self.assertEqual(saved.read_bytes(), file_bytes)
                self.assertEqual(result["filename"], "report.pdf")
                authorized.return_value = False
                connection.request("POST", "/api/files/upload", body, {"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 401)
                response.read()
        finally:
            connection.close()
            server.shutdown()
            thread.join()
            server.server_close()


class ImageProgressTests(unittest.TestCase):
    def test_progress_endpoint_bypasses_the_chat_request_lock(self):
        state = loa_web.WebChatState(loa_web.default_config())
        state.set_image_progress(
            {
                "active": True,
                "step": 3,
                "step_count": 50,
                "filename": "a" * 32 + ".png",
                "status": "Denoising",
            }
        )
        server = loa_web.LoaHTTPServer(
            ("127.0.0.1", 0), loa_web.LoaRequestHandler, state
        )
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=2
        )
        try:
            with (
                patch.object(loa_web.LoaRequestHandler, "_authorized", return_value=True),
                state.lock,
            ):
                connection.request("GET", "/api/image-progress")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    json.loads(response.read()),
                    {
                        "active": True,
                        "step": 3,
                        "step_count": 50,
                        "filename": "a" * 32 + ".png",
                        "status": "Denoising",
                    },
                )
        finally:
            connection.close()
            server.shutdown()
            server_thread.join()
            server.server_close()


class WebServerShutdownTests(unittest.TestCase):
    def test_one_interrupt_exits_during_mcp_initialization(self):
        script = """
import asyncio
import threading
import loa_web

loa_web.has_web_password = lambda: True
loa_web.cron_scheduler.start = lambda executor: None
loa_web.cron_scheduler.stop = lambda timeout=3: None

def initialize(config):
    manager = loa_web.mcp_manager
    manager._ensure_loop()
    started = threading.Event()
    async def pending():
        started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(10)
    manager._tasks["test"] = asyncio.run_coroutine_threadsafe(pending(), manager._loop)
    if not started.wait(3):
        raise RuntimeError("Task did not start")
    print("READY", flush=True)
    threading.Event().wait(10)
    return []

loa_web.configure_runtime_tools = initialize
try:
    loa_web.run_web_server(None, host="127.0.0.1", port=0)
except KeyboardInterrupt:
    print("STOPPED", flush=True)
"""
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            output = b""
            deadline = time.monotonic() + 10
            while b"READY" not in output and time.monotonic() < deadline:
                ready, _, _ = select.select([process.stdout], [], [], 1)
                if ready:
                    chunk = os.read(process.stdout.fileno(), 4096)
                    if not chunk:
                        break
                    output += chunk
            self.assertIn(b"READY", output)
            started = time.monotonic()
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=2)
            self.assertLess(time.monotonic() - started, 1.5)
            self.assertEqual(process.returncode, 0, stderr.decode())
            self.assertIn(b"STOPPED", stdout)
            self.assertNotIn(b"Exception ignored in atexit", stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_interrupt_stops_owned_mcp_process_during_initialization(self):
        if os.name != "posix":
            self.skipTest("MCP process-group watchdog is POSIX-specific.")
        script = """
import sys
import loa
import loa_web

child_code = '''
import os
import signal
import sys
import time
from pathlib import Path
ready, stopped = map(Path, sys.argv[1:])
def stop(signum, frame):
    stopped.write_text("stopped")
signal.signal(signal.SIGTERM, stop)
ready.write_text(str(os.getpid()))
while True:
    time.sleep(1)
'''
server = {
    "name": "test", "enabled": True, "command": sys.executable,
    "args": ["-c", child_code, sys.argv[1], sys.argv[2]], "env": {},
}
loa_web.has_web_password = lambda: True
loa_web.cron_scheduler.start = lambda executor: None
loa_web.cron_scheduler.stop = lambda timeout=3: None
loa_web.configure_runtime_tools = lambda config: loa_web.mcp_manager.reload([server])
loa.load_config = lambda: None
sys.argv = ["loa.py", "--web", "--host", "127.0.0.1", "--port", "0"]
raise SystemExit(loa.main())
"""
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            stopped = Path(directory) / "stopped"
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", script, str(ready), str(stopped)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            child_pid = None
            completed = False
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and time.monotonic() < deadline:
                    self.assertIsNone(process.poll(), "Web server exited before MCP started.")
                    time.sleep(0.05)
                self.assertTrue(ready.exists(), "MCP subprocess did not start.")
                child_pid = int(ready.read_text())
                started = time.monotonic()
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=3)
                completed = True
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertEqual(process.returncode, 0, stderr.decode())
                self.assertIn(b"Web chat stopped.", stdout)
                self.assertNotIn(b"Exception ignored in atexit", stderr)
                deadline = time.monotonic() + 2
                while not stopped.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(stopped.exists(), "Owned MCP subprocess was not terminated.")
            finally:
                if process.poll() is None:
                    process.kill()
                if child_pid is not None and (not completed or not stopped.exists()):
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if not completed:
                    process.communicate(timeout=3)

    def test_sigterm_closes_http_and_mcp_resources(self):
        registered_handlers = {}

        class FakeServer:
            server_address = ("127.0.0.1", 8765)

            def serve_forever(self):
                registered_handlers[signal.SIGTERM](signal.SIGTERM, None)

            def server_close(self):
                pass

        server = FakeServer()
        original_handler = signal.SIG_DFL

        def set_handler(signum, handler):
            if handler is not signal.SIG_DFL:
                registered_handlers[signum] = handler
            return original_handler

        with (
            patch.object(loa_web, "has_web_password", return_value=True),
            patch.object(loa_web, "WebChatState") as state_class,
            patch.object(loa_web, "LoaHTTPServer", return_value=server),
            patch.object(loa_web.cron_scheduler, "start"),
            patch.object(loa_web.cron_scheduler, "stop") as stop_scheduler,
            patch.object(loa_web.mcp_manager, "close") as close_mcp,
            patch.object(loa_web.signal, "getsignal", return_value=original_handler),
            patch.object(loa_web.signal, "signal", side_effect=set_handler),
        ):
            state_class.return_value.stop_runtime.side_effect = close_mcp
            with self.assertRaises(KeyboardInterrupt):
                loa_web.run_web_server(None, host="127.0.0.1")

        stop_scheduler.assert_called_once_with(timeout=loa_web.WEB_SHUTDOWN_TIMEOUT)
        close_mcp.assert_called_once_with()


class WebStartupTests(unittest.TestCase):
    def test_shutdown_cancels_active_agent(self):
        state = loa_web.WebChatState(None)
        cancellation = loa_web.AgentCancellation()
        state.active_cancellation = cancellation
        with patch.object(loa_web.mcp_manager, "close") as close:
            state.stop_runtime()
        self.assertTrue(cancellation.event.is_set())
        close.assert_called_once_with(
            timeout=loa_web.WEB_SHUTDOWN_TIMEOUT, permanent=True
        )

    def test_http_server_serves_page_while_runtime_is_initializing(self):
        started = threading.Event()
        release = threading.Event()

        def initialize(_config):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test runtime initialization timed out.")
            return []

        with (
            patch.object(loa_web, "configure_runtime_tools", side_effect=initialize),
            patch.object(loa_web.mcp_manager, "close") as close_mcp,
        ):
            state = loa_web.WebChatState(None)
            server = loa_web.LoaHTTPServer(
                ("127.0.0.1", 0), loa_web.LoaRequestHandler, state
            )
            server_thread = threading.Thread(target=server.serve_forever)
            state.start_runtime()
            server_thread.start()
            try:
                self.assertTrue(started.wait(2))
                connection = http.client.HTTPConnection(
                    "127.0.0.1", server.server_address[1], timeout=2
                )
                connection.request("GET", "/")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                connection.close()
                self.assertFalse(state._runtime_ready.is_set())
            finally:
                release.set()
                state.wait_for_runtime()
                server.shutdown()
                server_thread.join()
                server.server_close()
                state.stop_runtime()
            close_mcp.assert_called_once_with(
                timeout=loa_web.WEB_SHUTDOWN_TIMEOUT, permanent=True
            )

    def test_shutdown_during_initialization_closes_mcp_immediately(self):
        started = threading.Event()
        release = threading.Event()

        def initialize(_config):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test runtime initialization timed out.")
            return []

        with (
            patch.object(loa_web, "configure_runtime_tools", side_effect=initialize),
            patch.object(loa_web.mcp_manager, "close") as close_mcp,
        ):
            state = loa_web.WebChatState(None)
            state.start_runtime()
            try:
                self.assertTrue(started.wait(2))
                state.stop_runtime()
                close_mcp.assert_called_once_with(
                    timeout=loa_web.WEB_SHUTDOWN_TIMEOUT, permanent=True
                )
            finally:
                release.set()
                self.assertTrue(state._runtime_ready.wait(5))
            close_mcp.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "server is stopping"):
                state.wait_for_runtime()

    def test_chat_request_waits_for_runtime(self):
        started = threading.Event()
        release = threading.Event()
        handled = threading.Event()

        def initialize(_config):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test runtime initialization timed out.")
            return []

        def send_message(handler, _payload):
            handled.set()
            handler._send_json({"ok": True})

        with (
            patch.object(loa_web, "configure_runtime_tools", side_effect=initialize),
            patch.object(loa_web.mcp_manager, "close"),
            patch.object(loa_web.LoaRequestHandler, "_authorized", return_value=True),
            patch.object(loa_web.LoaRequestHandler, "_send_message", send_message),
        ):
            state = loa_web.WebChatState(None)
            server = loa_web.LoaHTTPServer(
                ("127.0.0.1", 0), loa_web.LoaRequestHandler, state
            )
            server_thread = threading.Thread(target=server.serve_forever)
            state.start_runtime()
            server_thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            try:
                self.assertTrue(started.wait(2))
                connection.request("POST", "/api/chat", "{}", {"Content-Type": "application/json"})
                self.assertFalse(handled.wait(0.2))
                release.set()
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                self.assertTrue(handled.is_set())
            finally:
                release.set()
                state.wait_for_runtime()
                connection.close()
                server.shutdown()
                server_thread.join()
                server.server_close()
                state.stop_runtime()

    def test_runtime_failure_is_reported_to_dependent_requests(self):
        with (
            patch.object(
                loa_web, "configure_runtime_tools", side_effect=ValueError("broken tools")
            ),
            patch.object(loa_web.mcp_manager, "close"),
            self.assertLogs("loa_web", level="ERROR"),
        ):
            state = loa_web.WebChatState(None)
            state.start_runtime()
            try:
                with self.assertRaisesRegex(RuntimeError, "broken tools"):
                    state.wait_for_runtime()
            finally:
                state.stop_runtime()

    def test_waiting_for_runtime_does_not_block_other_requests(self):
        state = loa_web.WebChatState(None)
        started = threading.Event()
        release = threading.Event()

        def wait_for_runtime(timeout=None):
            started.set()
            if not release.wait(2):
                raise RuntimeError("Test did not release runtime wait.")
            raise loa_web.WebBusyError("LOA tools are still starting.")

        def save_settings():
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=3
            )
            try:
                connection.request("POST", "/api/settings", "{}")
                response = connection.getresponse()
                return response.status, response.read().decode()
            finally:
                connection.close()

        server = loa_web.LoaHTTPServer(
            ("127.0.0.1", 0), loa_web.LoaRequestHandler, state
        )
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        with (
            patch.object(loa_web.LoaRequestHandler, "_authorized", return_value=True),
            patch.object(state, "wait_for_runtime", side_effect=wait_for_runtime),
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
        ):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            try:
                pending = executor.submit(save_settings)
                self.assertTrue(started.wait(1))
                connection.request("GET", "/api/history")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                release.set()
                status, body = pending.result(timeout=2)
                self.assertEqual(status, 503)
                self.assertIn("still starting", body)
            finally:
                release.set()
                connection.close()
                server.shutdown()
                server_thread.join()
                server.server_close()

    def test_runtime_wait_has_a_timeout(self):
        state = loa_web.WebChatState(None)
        with self.assertRaisesRegex(loa_web.WebBusyError, "still starting"):
            state.wait_for_runtime(timeout=0.01)

    def test_image_readiness_check_does_not_block_history(self):
        state = loa_web.WebChatState(None)
        started = threading.Event()
        release = threading.Event()

        def check(_config):
            started.set()
            if not release.wait(2):
                raise RuntimeError("Test did not release image check.")
            return "Ready"

        server = loa_web.LoaHTTPServer(
            ("127.0.0.1", 0), loa_web.LoaRequestHandler, state
        )
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()

        def run_check():
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=3
            )
            try:
                connection.request("POST", "/api/image-gen/check", "{}")
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        with (
            patch.object(loa_web.LoaRequestHandler, "_authorized", return_value=True),
            patch.object(loa_web, "check_image_generation_readiness", side_effect=check),
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
        ):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            try:
                pending = executor.submit(run_check)
                self.assertTrue(started.wait(1))
                connection.request("GET", "/api/history")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                release.set()
                status, body = pending.result(timeout=2)
                self.assertEqual(status, 200)
                self.assertIn(b"Ready", body)
            finally:
                release.set()
                connection.close()
                server.shutdown()
                server_thread.join()
                server.server_close()

    def test_busy_request_returns_error_instead_of_waiting_for_chat(self):
        state = loa_web.WebChatState(None)
        state._runtime_ready.set()
        server = loa_web.LoaHTTPServer(
            ("127.0.0.1", 0), loa_web.LoaRequestHandler, state
        )
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        with (
            patch.object(loa_web.LoaRequestHandler, "_authorized", return_value=True),
            patch.object(loa_web, "WEB_LOCK_TIMEOUT", 0.1),
        ):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            try:
                with state.lock:
                    connection.request("GET", "/api/history")
                    response = connection.getresponse()
                    self.assertEqual(response.status, 503)
                    self.assertIn("busy", response.read().decode())
                    connection.request("POST", "/api/settings", "{}")
                    response = connection.getresponse()
                    self.assertEqual(response.status, 503)
                    self.assertIn("busy", response.read().decode())
            finally:
                connection.close()
                server.shutdown()
                server_thread.join()
                server.server_close()


class WebSettingsTests(unittest.TestCase):
    def test_existing_custom_identity_is_migrated_without_changing_other_settings(self):
        config = loa_web.default_config()
        config.update(name="Old assistant", prompt="Old prompt", model="custom-model")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with (
                patch.object(loa_storage, "config_file_path", return_value=path),
                patch.object(loa_storage, "ensure_skills_directory"),
            ):
                loaded = loa_storage.load_config()
                saved = json.loads(path.read_text())
                self.assertEqual(loaded["name"], loa_web.DEFAULT_NAME)
                self.assertEqual(loaded["prompt"], loa_web.DEFAULT_PROMPT)
                self.assertEqual(loaded["model"], "custom-model")
                self.assertEqual(saved, loaded)
                config["name"] = "Changed again"
                config["prompt"] = "Changed again"
                loa_storage.save_config(config)
                saved = json.loads(path.read_text())
                self.assertEqual(saved["name"], loa_web.DEFAULT_NAME)
                self.assertEqual(saved["prompt"], loa_web.DEFAULT_PROMPT)

    def test_saving_reuses_unchanged_mcp_connections_and_reloads_changes(self):
        for changed in (False, True):
            with self.subTest(mcp_changed=changed):
                state = loa_web.WebChatState(loa_web.default_config())
                payload = deepcopy(state.config)
                if changed:
                    payload["mcp_servers"][0]["enabled"] = not payload["mcp_servers"][0]["enabled"]
                handler = object.__new__(loa_web.LoaRequestHandler)
                handler.server = SimpleNamespace(state=state)
                handler._send_json = Mock()
                with (
                    patch.object(loa_web, "save_config") as save,
                    patch.object(loa_web, "reload_skill_modules"),
                    patch.object(loa_web, "configure_runtime_tools", return_value=[]) as configure,
                    patch.object(loa_web, "list_user_skills", return_value=[]),
                ):
                    handler._save_settings(payload)
                save.assert_called_once()
                configure.assert_called_once_with(
                    state.config,
                    force_mcp_reload=changed,
                    reuse_mcp_connections=not changed,
                )
                self.assertTrue(handler._send_json.call_args.args[0]["ok"])

    def test_settings_are_persisted_without_reconnecting_mcp(self):
        state = loa_web.WebChatState(loa_web.default_config())
        payload = deepcopy(state.config)
        payload.pop("name")
        payload.pop("prompt")
        payload["image_gen"]["model"] = "test-image-model"
        handler = object.__new__(loa_web.LoaRequestHandler)
        handler.server = SimpleNamespace(state=state)
        handler._send_json = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            with (
                patch.object(loa_storage, "config_file_path", return_value=path),
                patch.object(loa_web, "reload_skill_modules"),
                patch.object(loa_web, "configure_runtime_tools", return_value=[]),
                patch.object(loa_web, "list_user_skills", return_value=[]),
            ):
                handler._save_settings(payload)
            saved = json.loads(path.read_text())
        self.assertEqual(saved["name"], loa_web.DEFAULT_NAME)
        self.assertEqual(saved["prompt"], loa_web.DEFAULT_PROMPT)
        self.assertEqual(saved["image_gen"]["model"], "test-image-model")

    def test_settings_reject_changes_to_fixed_identity(self):
        state = loa_web.WebChatState(loa_web.default_config())
        handler = object.__new__(loa_web.LoaRequestHandler)
        handler.server = SimpleNamespace(state=state)
        for field in ("name", "prompt"):
            with self.subTest(field=field):
                payload = {"model": "another-model", field: "custom"}
                with self.assertRaisesRegex(ValueError, "cannot be changed"):
                    handler._save_settings(payload)
        self.assertEqual(state.config["name"], loa_web.DEFAULT_NAME)
        self.assertEqual(state.config["prompt"], loa_web.DEFAULT_PROMPT)
