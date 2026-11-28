import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import loa_storage
import loa_ui
import loa_web


class HiddenChatTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patcher = patch.object(
            loa_storage, "config_file_path", return_value=Path(self.directory.name) / "config.json"
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class HiddenChatStorageTests(HiddenChatTestCase):
    def test_pin_and_hidden_chat_persist_without_recall_leaks(self):
        with self.assertRaises(ValueError):
            loa_storage.set_hidden_chat_pin("12")
        loa_storage.set_hidden_chat_pin("1234")
        self.assertFalse(loa_storage.verify_hidden_chat_pin("1235"))
        self.assertTrue(loa_storage.verify_hidden_chat_pin("1234"))
        with self.assertRaises(ValueError):
            loa_storage.set_hidden_chat_pin("5678")
        pin_path = Path(self.directory.name) / "hidden_chat_pin.json"
        self.assertNotIn("1234", pin_path.read_text())
        self.assertEqual(pin_path.stat().st_mode & 0o777, 0o600)

        chat = loa_storage.new_chat_session()
        chat["messages"] = [{"role": "user", "content": "Secret project details"}]
        chat["hidden"] = True
        loa_storage.save_chat_session(chat)
        self.assertEqual(loa_storage.list_chat_sessions(), [])
        self.assertEqual(loa_storage.list_chat_sessions(hidden=True)[0]["id"], chat["id"])
        self.assertNotIn("Secret", loa_storage.build_chat_catalog())
        self.assertNotIn("Secret", loa_storage.recall_past_chat(chat["id"]))
        self.assertNotIn("Secret", loa_storage.recall_past_chat("latest"))
        self.assertIsNone(loa_storage.find_saved_chat_id(chat["id"]))

        chat["hidden"] = False
        loa_storage.save_chat_session(chat)
        self.assertEqual(loa_storage.list_chat_sessions()[0]["id"], chat["id"])
        self.assertIn("Secret", loa_storage.recall_past_chat(chat["id"]))

    def test_old_chat_defaults_to_visible(self):
        chat = loa_storage.new_chat_session()
        loa_storage.save_chat_session(chat)
        path = loa_storage.chats_directory_path() / f"{chat['id']}.json"
        data = json.loads(path.read_text())
        del data["hidden"]
        path.write_text(json.dumps(data))
        self.assertFalse(loa_storage.list_chat_sessions()[0]["hidden"])


class HiddenChatWebTests(HiddenChatTestCase):
    def test_web_pin_and_session_isolation(self):
        state = loa_web.WebChatState(loa_web.default_config())
        state.sessions.update(first=time.monotonic() + 60, second=time.monotonic() + 60)
        server = loa_web.LoaHTTPServer(("127.0.0.1", 0), loa_web.LoaRequestHandler, state)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        self.addCleanup(connection.close)

        def request(path, payload=None, session="first"):
            headers = {"Cookie": f"loa_session={session}"}
            if payload is not None:
                headers["Content-Type"] = "application/json"
            connection.request("GET" if payload is None else "POST", path,
                               None if payload is None else json.dumps(payload), headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())

        chat = loa_storage.new_chat_session()
        chat["messages"] = [{"role": "user", "content": "Secret project details"}]
        loa_storage.save_chat_session(chat)
        chat_id = chat["id"]
        self.assertEqual(request("/api/chats", {"action": "set_pin", "pin": "1234",
                                               "confirm_pin": "9999"})[0], 400)
        self.assertEqual(request("/api/chats", {"action": "set_pin", "pin": "1234",
                                               "confirm_pin": "1234"})[0], 200)
        self.assertEqual(request("/api/chats", {"action": "hide", "chat_id": chat_id})[0], 200)
        self.assertEqual(request("/api/chats")[1]["chats"], [])
        self.assertEqual(request("/api/chats?hidden=1")[0], 403)
        self.assertEqual(request("/api/chats", {"action": "open", "chat_id": chat_id})[0], 400)
        self.assertEqual(request("/api/chats", {"action": "unlock", "pin": "0000"})[0], 401)
        self.assertEqual(request("/api/chats", {"action": "unlock", "pin": "1234"})[0], 200)
        self.assertEqual(request("/api/chats?hidden=1")[1]["chats"][0]["id"], chat_id)
        with patch.object(loa_web, "build_system_prompt", return_value="system"):
            self.assertEqual(request("/api/chats", {"action": "open", "chat_id": chat_id})[0], 200)
        self.assertIn("Secret", str(request("/api/history")[1]["transcript"]))
        self.assertEqual(request("/api/history", session="second")[1]["transcript"], [])
        self.assertEqual(request("/api/image-progress", session="second")[0], 403)
        self.assertEqual(request("/api/break-loop", {}, session="second")[0], 403)
        self.assertEqual(request("/api/chats?hidden=1", session="second")[0], 403)
        self.assertEqual(request("/api/chats", {"action": "rename", "chat_id": chat_id,
                                                "title": "Leaked"}, session="second")[0], 400)
        self.assertEqual(request("/api/chats", {"action": "home"})[0], 200)
        self.assertEqual(request("/api/chats?hidden=1")[0], 403)
        self.assertEqual(request("/api/chats", {"action": "unlock", "pin": "1234"})[0], 200)
        self.assertEqual(request("/api/chats", {"action": "unhide", "chat_id": chat_id})[0], 200)
        self.assertEqual(request("/api/chats")[1]["chats"][0]["id"], chat_id)
        for _ in range(5):
            self.assertEqual(request("/api/chats", {"action": "unlock", "pin": "0000"})[0], 401)
        self.assertEqual(request("/api/chats", {"action": "unlock", "pin": "1234"})[0], 429)


class HiddenChatTerminalTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_hide_unlock_and_open(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(loa_storage, "config_file_path", return_value=Path(directory) / "config.json"), \
             patch.object(loa_ui, "build_system_prompt", return_value="system"):
            chat = loa_storage.new_chat_session()
            loa_storage.save_chat_session(chat)
            other_chat = loa_storage.new_chat_session()
            loa_storage.save_chat_session(other_chat)
            app = loa_ui.LoaApp(loa_web.default_config())
            async with app.run_test(size=(160, 50)) as pilot:
                await pilot.click("#home_continue_chat")
                await pilot.click("#open_chats")
                hidden_id = app.screen.visible_sessions[0]["id"]
                await pilot.click("#toggle_hidden_chat")
                app.screen.query_one("#hidden_pin").value = "1234"
                app.screen.query_one("#hidden_pin_confirm").value = "1234"
                await pilot.click("#hidden_pin_submit")
                await pilot.pause()
                remaining = loa_storage.list_chat_sessions()
                self.assertEqual(len(remaining), 1)
                self.assertNotEqual(remaining[0]["id"], hidden_id)
                await pilot.click("#close_chat_manager")
                home = app.screen
                home.refresh_sessions()
                home.refresh_sessions()
                self.assertEqual(
                    sum(button.display for button in home.query(".home-chat-choice")), 1
                )
                await pilot.click("#home_chat_0")
                self.assertEqual(app.screen.session["id"], remaining[0]["id"])
                await pilot.click("#home")
                await pilot.click("#home_hidden_chats")
                app.screen.query_one("#hidden_pin").value = "1234"
                await pilot.click("#hidden_pin_submit")
                await pilot.pause()
                self.assertIsInstance(app.screen, loa_ui.ChatManagerScreen)
                self.assertEqual(app.screen.sessions[0]["id"], hidden_id)
                await pilot.click("#open_chat")
                self.assertIsInstance(app.screen, loa_ui.ChatScreen)
                self.assertEqual(app.screen.session["id"], hidden_id)


if __name__ == "__main__":
    unittest.main()
