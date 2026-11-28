import asyncio
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import loa_mcp
from loa_mcp import MCPManager


class MCPManagerShutdownTests(unittest.TestCase):
    def test_permanent_shutdown_does_not_wait_for_blocked_event_loop(self):
        manager = MCPManager()
        manager._ensure_loop()
        started = threading.Event()
        release = threading.Event()

        def blocked():
            started.set()
            release.wait(3)

        manager._loop.call_soon_threadsafe(blocked)
        try:
            self.assertTrue(started.wait(2))
            with self.assertLogs("loa_mcp", level="WARNING"):
                manager.close(timeout=0.1, permanent=True)
            with patch.object(manager, "_close_lock") as lock:
                manager.close()
                lock.acquire.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                manager.ensure_configured([])
        finally:
            release.set()
            manager._loop_thread.join(timeout=2)
            if not manager._loop.is_closed():
                manager._loop.run_until_complete(asyncio.sleep(0))
                manager._loop.close()

    def test_supervised_mcp_server_exchanges_messages_and_closes(self):
        if sys.platform == "win32":
            self.skipTest("MCP process-group watchdog is POSIX-specific.")
        script = """
import json
import sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "test", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": "read_echo", "description": "Echo",
            "inputSchema": {"type": "object", "properties": {}},
        }]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": sys.argv[1]}]}
    else:
        raise RuntimeError(method)
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""
        manager = MCPManager()
        server = {
            "name": "test",
            "enabled": True,
            "command": sys.executable,
            "args": ["-u", "-c", script, "argument with spaces"],
            "env": {},
        }
        try:
            statuses = manager.reload([server])
            self.assertEqual(statuses[0]["status"], "Connected", statuses)
            self.assertEqual(
                manager.call_tool("mcp_test_read_echo", {}),
                "argument with spaces",
            )
        finally:
            manager.close()

    def test_reload_stops_descendant_processes_before_reconnecting(self):
        if sys.platform == "win32":
            self.skipTest("MCP process-group watchdog is POSIX-specific.")
        with tempfile.TemporaryDirectory() as directory:
            pid_path = os.path.join(directory, "child.pid")
            script = """
import json
import signal
import subprocess
import sys
import time
child = subprocess.Popen(
    [
        sys.executable, "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)",
    ],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
with open(sys.argv[1], "w", encoding="ascii") as pid_file:
    pid_file.write(str(child.pid))
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    if request["method"] == "initialize":
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "test", "version": "1"},
        }
    elif request["method"] == "tools/list":
        result = {"tools": []}
    else:
        raise RuntimeError(request["method"])
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""
            manager = MCPManager()
            server = {
                "name": "test",
                "enabled": True,
                "command": sys.executable,
                "args": ["-u", "-c", script, pid_path],
                "env": {},
            }
            child_pid: int | None = None
            try:
                statuses = manager.reload([server])
                self.assertEqual(statuses[0]["status"], "Connected", statuses)
                deadline = time.monotonic() + 2
                while not os.path.exists(pid_path) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(os.path.exists(pid_path), "MCP child did not start.")
                with open(pid_path, encoding="ascii") as pid_file:
                    child_pid = int(pid_file.read())

                manager.reload([])

                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    self.fail(f"MCP descendant process {child_pid} survived reload.")
            finally:
                manager.close(permanent=True)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_single_interrupt_stops_process_with_pending_mcp_task(self):
        script = """
import asyncio
import signal
import threading
from loa_mcp import mcp_manager

manager = mcp_manager
manager._ensure_loop()
started = threading.Event()

async def pending():
    started.set()
    try:
        await asyncio.Future()
    finally:
        await asyncio.sleep(0.1)

manager._tasks["test"] = asyncio.run_coroutine_threadsafe(pending(), manager._loop)
if not started.wait(3):
    raise RuntimeError("MCP test task did not start")
try:
    print("READY", flush=True)
    signal.pause()
except KeyboardInterrupt:
    manager.close()
print("STOPPED", flush=True)
"""
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            ready, _, _ = select.select([process.stdout], [], [], 10)
            self.assertTrue(ready, "Child process did not become ready.")
            self.assertEqual(process.stdout.readline().strip(), "READY")
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn("STOPPED", stdout)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_close_finishes_pending_tasks_before_closing_loop(self):
        manager = MCPManager()
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(
            target=manager._run_loop,
            args=(loop,),
            daemon=True,
        )
        manager._loop = loop
        manager._loop_thread = loop_thread
        loop_thread.start()

        task_started = threading.Event()
        task_cleaned_up = threading.Event()

        async def pending_task():
            task_started.set()
            try:
                await asyncio.Future()
            finally:
                task_cleaned_up.set()

        asyncio.run_coroutine_threadsafe(pending_task(), loop)
        try:
            self.assertTrue(task_started.wait(timeout=2))
            manager.close()

            self.assertTrue(task_cleaned_up.is_set())
            self.assertFalse(loop.is_running())
            self.assertTrue(loop.is_closed())
            self.assertFalse(loop_thread.is_alive())
        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
                loop_thread.join(timeout=2)
            if not loop.is_closed():
                loop.close()

    def test_close_waits_for_asynchronous_cancellation_cleanup(self):
        manager = MCPManager()
        manager._ensure_loop()
        loop = manager._loop
        started = threading.Event()
        cleaned = threading.Event()

        async def pending_task():
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0.1)
                cleaned.set()

        future = asyncio.run_coroutine_threadsafe(pending_task(), loop)
        manager._tasks["test"] = future
        try:
            self.assertTrue(started.wait(2))
            manager.close()
            self.assertTrue(cleaned.is_set())
            self.assertTrue(loop.is_closed())
        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
                manager._loop_thread.join(timeout=2)
            if not loop.is_closed():
                loop.close()

    def test_close_returns_when_task_ignores_cancellation(self):
        manager = MCPManager()
        manager._ensure_loop()
        loop = manager._loop
        started = threading.Event()
        release = asyncio.Event()

        async def pending_task():
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                await release.wait()

        future = asyncio.run_coroutine_threadsafe(pending_task(), loop)
        manager._tasks["test"] = future
        real_close = loop.close
        try:
            self.assertTrue(started.wait(2))
            with (
                patch.object(loa_mcp, "_MCP_SHUTDOWN_TIMEOUT", 0.1),
                patch.object(loop, "close") as close,
                self.assertLogs("loa_mcp", level="WARNING"),
            ):
                manager.close()
                close.assert_called_once_with()
            self.assertFalse(loop.is_running())
            self.assertIsNone(manager._loop)
        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
                manager._loop_thread.join(timeout=2)
            if not loop.is_closed():
                release.set()
                loop.run_until_complete(asyncio.sleep(0))
                real_close()


if __name__ == "__main__":
    unittest.main()
