"""Textual setup, chat, settings, and command-confirmation screens."""

from __future__ import annotations

import subprocess
from functools import partial

import httpx
import ollama
from ddgs.exceptions import DDGSException
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.theme import Theme
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Collapsible,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TextArea,
)
from textual.containers import VerticalScroll

from loa_agent import (
    AssistantReply,
    ChatMessage,
    action_task,
    action_details,
    execute_approved_action,
    format_ollama_error,
    get_assistant_reply,
    restore_pending_action,
    sanitize_tool_content,
)
from loa_config import (
    AssistantConfig,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_PROMPT,
    build_system_prompt,
    default_config,
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


LOA_THEME = Theme(
    name="loa",
    primary="#C8CDD2",
    secondary="#8E969F",
    accent="#C8CDD2",
    foreground="#E2E5E8",
    background="#17191C",
    surface="#202327",
    panel="#30343A",
    boost="#3A3F46",
    success="#AAB4AE",
    warning="#C5B99A",
    error="#C49A9A",
    dark=True,
)


class SettingsScreen(Screen):
    """Initial setup and editable assistant settings."""

    def __init__(self, config: AssistantConfig, initial_setup: bool = False) -> None:
        super().__init__()
        self.config = config
        self.initial_setup = initial_setup

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Label(
                    "Assistant setup" if self.initial_setup else "Assistant settings",
                    classes="title",
                ),
                Label("Choose the assistant name, Ollama model, and system prompt."),
                Label("Assistant name:"),
                Input(value=self.config["name"], id="ai_name"),
                Label("Ollama model:"),
                Input(value=self.config["model"], id="model_id"),
                Label("Ollama Host URL:"),
                Input(
                    value=self.config.get("ollama_host") or DEFAULT_OLLAMA_HOST,
                    placeholder=DEFAULT_OLLAMA_HOST,
                    id="ollama_host",
                ),
                Button("Reset Host", id="reset_host"),
                Label("System prompt:"),
                TextArea(self.config["prompt"], id="system_prompt"),
                Label("", id="settings_error"),
                Horizontal(
                    Button(
                        "Save and continue" if self.initial_setup else "Save settings",
                        variant="primary",
                        id="save_settings",
                    ),
                    Button("Reset Prompt", id="reset_prompt"),
                    Button("Cancel", id="cancel_settings"),
                    classes="buttons",
                ),
                id="form-container",
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel_settings":
            if self.initial_setup:
                self.app.exit()
            else:
                self.app.pop_screen()
            return
        if event.button.id == "reset_prompt":
            self.query_one("#system_prompt", TextArea).text = DEFAULT_PROMPT
            self.query_one("#settings_error", Label).update(
                "Default prompt restored. Save settings to keep it."
            )
            return
        if event.button.id == "reset_host":
            self.query_one("#ollama_host", Input).value = DEFAULT_OLLAMA_HOST
            self.query_one("#settings_error", Label).update(
                "Default host restored. Save settings to keep it."
            )
            return

        if event.button.id != "save_settings":
            return

        config: AssistantConfig = {
            "name": self.query_one("#ai_name", Input).value.strip(),
            "model": self.query_one("#model_id", Input).value.strip(),
            "prompt": self.query_one("#system_prompt", TextArea).text.strip(),
            "ollama_host": self.query_one("#ollama_host", Input).value.strip(),
        }
        if not all(config.values()):
            self.query_one("#settings_error", Label).update(
                "Name, model, Ollama host URL, and system prompt must not be empty."
            )
            return

        try:
            save_config(config)
        except RuntimeError as error:
            self.query_one("#settings_error", Label).update(str(error))
            return

        app = self.app
        if isinstance(app, LoaApp):
            app.apply_settings(config, self.initial_setup)


class ChatManagerScreen(ModalScreen[None]):
    """List saved chats and expose switching and maintenance actions."""

    def __init__(self) -> None:
        super().__init__()
        try:
            self.sessions = list_chat_sessions()
            self.load_error = ""
        except (OSError, RuntimeError, ValueError) as error:
            self.sessions = []
            self.load_error = str(error)

    def compose(self) -> ComposeResult:
        rows = [
            ListItem(
                Static(
                    f"{session['title']}  |  {session['updated_at'][:10]}\n"
                    f"{session['summary']}",
                    markup=False,
                )
            )
            for session in self.sessions
        ]
        yield Container(
            Vertical(
                Label("Chats", classes="title"),
                Label(self.load_error, id="chat_manager_error"),
                ListView(*rows, id="saved_chats"),
                Input(placeholder="Title for Rename", id="chat_title"),
                Horizontal(
                    Button("Open / Switch", variant="primary", id="open_chat"),
                    Button("New Chat", id="new_chat"),
                    Button("Rename", id="rename_chat"),
                    Button("Delete", variant="error", id="delete_chat"),
                    Button("Close", id="close_chat_manager"),
                    classes="buttons",
                ),
                id="chat-manager-container",
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self.app
        if not isinstance(app, LoaApp):
            return
        if event.button.id == "close_chat_manager":
            app.pop_screen()
            return
        if event.button.id == "new_chat":
            session = new_chat_session()
            save_chat_session(session)
            app.pop_screen()
            if isinstance(app.screen, ChatScreen):
                app.screen.load_session(session)
                app.screen._save_chat()
            return

        list_view = self.query_one("#saved_chats", ListView)
        index = list_view.index
        if index is None or index >= len(self.sessions):
            self.query_one("#chat_manager_error", Label).update(
                "Select a saved chat first."
            )
            return
        session = self.sessions[index]
        try:
            if event.button.id == "open_chat":
                loaded = load_chat_session(session["id"])
                app.pop_screen()
                if isinstance(app.screen, ChatScreen):
                    app.screen.load_session(loaded)
            elif event.button.id == "rename_chat":
                title = self.query_one("#chat_title", Input).value
                renamed = rename_chat_session(session["id"], title)
                app.pop_screen()
                if (
                    isinstance(app.screen, ChatScreen)
                    and app.screen.session["id"] == renamed["id"]
                ):
                    app.screen.session["title"] = renamed["title"]
                    app.screen._save_chat()
                app.push_screen(ChatManagerScreen())
            elif event.button.id == "delete_chat":
                delete_chat_session(session["id"])
                app.pop_screen()
                if isinstance(app.screen, ChatScreen):
                    if app.screen.session["id"] == session["id"]:
                        replacement = new_chat_session()
                        save_chat_session(replacement)
                        app.screen.load_session(replacement)
                app.push_screen(ChatManagerScreen())
        except (OSError, RuntimeError, ValueError) as error:
            self.query_one("#chat_manager_error", Label).update(str(error))

    @staticmethod
    def _refresh_manager(app: LoaApp) -> None:
        app.pop_screen()
        app.push_screen(ChatManagerScreen())


class ChatScreen(Screen):
    """Interactive chat interface with explicit shell and HTTP actions."""

    def __init__(
        self,
        config: AssistantConfig,
        session: ChatSession | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.session = session or new_chat_session()
        self.messages: list[ChatMessage] = [
            message.copy() for message in self.session["messages"]
        ]
        if not self.messages:
            self.messages.append(
                {"role": "system", "content": self._system_prompt()}
            )
        else:
            self._refresh_system_prompt()
        self.pending_action = restore_pending_action(self.messages)
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label(f"{self.config['name']} | {self.config['model']}", id="status"),
            VerticalScroll(id="conversation"),
            Input(
                placeholder="Message or /help",
                id="chat_input",
            ),
            Horizontal(
                Button("Send", variant="primary", id="send_message"),
                Button("Chats", id="open_chats"),
                Button("Settings", id="open_settings"),
                Button("Exit", id="exit_chat"),
                id="chat_controls",
            ),
            Horizontal(
                Button("Run command", variant="warning", id="confirm_command"),
                Button("Cancel command", id="decline_command"),
                id="command_controls",
            ),
            id="chat_layout",
        )

    def on_mount(self) -> None:
        self.query_one("#command_controls").display = False
        if len(self.messages) == 1:
            self._write("LOA is ready. Type /help to see available commands.")
            self._save_chat()
        else:
            self._render_history()
        if self.pending_action is not None:
            self._show_pending_action(self.pending_action)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat_input":
            self._handle_input(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_message":
            chat_input = self.query_one("#chat_input", Input)
            self._handle_input(chat_input.value)
        elif event.button.id == "open_settings":
            self._open_settings()
        elif event.button.id == "open_chats":
            self.app.push_screen(ChatManagerScreen())
        elif event.button.id == "exit_chat":
            self.app.exit()
        elif event.button.id == "confirm_command":
            self._confirm_action()
        elif event.button.id == "decline_command":
            self._finish_action("Action was canceled; no changes were made.")

    def _handle_input(self, raw_text: str) -> None:
        text = raw_text.strip()
        if not text or self.busy or self.pending_action is not None:
            return

        chat_input = self.query_one("#chat_input", Input)
        chat_input.value = ""
        self._write(f"You: {text}")

        if text.lower() in {"/exit", "/quit", "exit", "quit"}:
            self.app.exit()
            return
        if text.lower() == "/settings":
            self._open_settings()
            return
        if text.lower() == "/chats":
            self.app.push_screen(ChatManagerScreen())
            return
        if text.lower() == "/new":
            self.load_session(new_chat_session())
            self._save_chat()
            return
        if text.lower() == "/help":
            self._write(
                "Commands:\n"
                "/settings - edit and save assistant settings\n"
                "/chats - open the saved chat manager\n"
                "/new - start a new chat\n"
                "/exit - close LOA\n"
                "Ask LOA to search the web when you need online information."
            )
            return

        self.messages.append({"role": "user", "content": text})
        self._save_chat()
        self._set_busy(True)
        self._start_assistant_request()

    def _start_assistant_request(self) -> None:
        self._refresh_system_prompt()
        snapshot = [message.copy() for message in self.messages]
        self.app.run_worker(
            partial(self._request_assistant, snapshot, self.session["id"]),
            thread=True,
            name="assistant-request",
        )

    def _request_assistant(
        self,
        messages: list[ChatMessage],
        active_chat_id: str,
    ) -> None:
        try:
            reply = get_assistant_reply(
                self.config,
                messages,
                active_chat_id=active_chat_id,
            )
        except (
            ollama.ResponseError,
            httpx.HTTPError,
            DDGSException,
            OSError,
            ValueError,
        ) as error:
            self.app.call_from_thread(
                self._show_error,
                "Could not complete the request: "
                f"{format_ollama_error(error, self.config.get('ollama_host') or DEFAULT_OLLAMA_HOST)}",
            )
            return
        self.app.call_from_thread(self._show_reply, reply)

    def _show_reply(self, reply: AssistantReply) -> None:
        self._set_busy(False)
        answer = sanitize_tool_content(reply["text"])
        self.messages = reply["history"]
        for task in reply["tasks"]:
            self._add_collapsible(task["label"], task["details"])
        if reply["sources"]:
            assistant = next(
                (message for message in reversed(self.messages) if message.get("role") == "assistant"),
                None,
            )
            if assistant is not None:
                assistant.setdefault("ui", {})["sources"] = reply["sources"]
        if answer:
            self._write(f"{self.config['name']}: {answer}")

        action = reply["action"]
        if action is not None:
            self.pending_action = action
            self._show_pending_action(action)
        else:
            self.pending_action = None
            self.query_one("#command_controls").display = False
            self.query_one("#chat_controls").display = True
            self.query_one("#chat_input", Input).disabled = False
        self._save_chat()

    def _show_pending_action(self, action: ChatMessage) -> None:
        self._write(
            f"Pending {action['type']}:\n{action_details(action)}\n"
            "Review the details, then choose Run or Cancel."
        )
        self.query_one("#confirm_command", Button).label = "Run"
        self.query_one("#decline_command", Button).label = "Cancel"
        self.query_one("#command_controls").display = True
        self.query_one("#chat_input", Input).disabled = True
        self.query_one("#chat_controls").display = False

    def _show_error(self, message: str) -> None:
        self._set_busy(False)
        self._write(message)
        self.query_one("#command_controls").display = False
        self.query_one("#chat_controls").display = True
        self.query_one("#chat_input", Input).disabled = False
        self._save_chat()

    def _confirm_action(self) -> None:
        action = self.pending_action
        if action is None:
            return
        self.query_one("#command_controls").display = False
        self._set_busy(True)
        self.app.run_worker(
            partial(self._action_worker, action),
            thread=True,
            name="approved-action",
        )

    def _action_worker(self, action: ChatMessage) -> None:
        try:
            output = execute_approved_action(action)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            output = f"Action execution failed: {error}"
        self.app.call_from_thread(self._finish_action, output)

    def _finish_action(self, output: str) -> None:
        action = self.pending_action
        if action is None:
            return
        self.pending_action = None
        task = action_task(action, output)
        details = task["details"]
        self.messages.append(
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
        self._add_collapsible(task["label"], details)
        self.query_one("#command_controls").display = False
        self.query_one("#confirm_command", Button).label = "Run"
        self.query_one("#decline_command", Button).label = "Cancel"
        self._save_chat()
        self._set_busy(True)
        self._start_assistant_request()

    def load_session(self, session: ChatSession) -> None:
        """Switch this screen to a saved or newly created chat."""
        self.session = session
        self.messages = [message.copy() for message in session["messages"]]
        self.pending_action = restore_pending_action(self.messages)
        if not self.messages:
            self.messages.append(
                {"role": "system", "content": self._system_prompt()}
            )
        else:
            self._refresh_system_prompt()
        self.query_one("#conversation", VerticalScroll).remove_children()
        self._render_history()
        if len(self.messages) == 1:
            self._write("LOA is ready. Type /help to see available commands.")
        self._set_busy(False)
        if self.pending_action is not None:
            self._show_pending_action(self.pending_action)

    def _system_prompt(self) -> str:
        return build_system_prompt(self.config["prompt"], self.session["id"])

    def _refresh_system_prompt(self) -> None:
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = self._system_prompt()
        else:
            self.messages.insert(
                0,
                {"role": "system", "content": self._system_prompt()},
            )

    def _save_chat(self) -> None:
        self.session["messages"] = [message.copy() for message in self.messages]
        try:
            save_chat_session(self.session)
        except (OSError, RuntimeError, ValueError) as error:
            self._write(f"Could not save chat: {error}")

    def _render_history(self) -> None:
        for message in self.messages:
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                continue
            if role == "user":
                self._write(f"You: {message.get('content', '')}")
            elif role == "assistant":
                content = sanitize_tool_content(str(message.get("content") or ""))
                if content:
                    self._write(f"{self.config['name']}: {content}")
            elif role == "tool":
                ui = message.get("ui")
                if isinstance(ui, dict):
                    self._add_collapsible(
                        str(ui.get("summary", "Completed tool")),
                        str(ui.get("details", message.get("content", ""))),
                    )

    def _add_collapsible(self, summary: str, details: str) -> None:
        title = summary if summary.startswith("TASK:") else f"TASK: {summary}"
        self.query_one("#conversation", VerticalScroll).mount(
            Collapsible(
                Static(details, markup=False),
                title=title,
                collapsed=True,
            )
        )
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    def _open_settings(self) -> None:
        if isinstance(self.app, LoaApp):
            self.app.push_screen(SettingsScreen(self.config))

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.query_one("#chat_input", Input).disabled = busy
        self.query_one("#send_message", Button).disabled = busy

    def _write(self, text: str) -> None:
        self.query_one("#conversation", VerticalScroll).mount(
            Static(text, markup=False)
        )
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)


class LoaApp(App[None]):
    """Application that persists setup and hosts the chat interface."""

    ENABLE_COMMAND_PALETTE = False

    CSS = """
    #form-container {
        padding: 1 2;
        background: $background-lighten-1;
        border: solid $accent;
        margin: 1 2;
    }
    #chat-manager-container {
        width: 90%;
        height: 85%;
        padding: 1 2;
        background: $surface;
        border: solid $accent;
    }
    #saved_chats {
        height: 1fr;
        border: solid $panel;
    }
    #conversation Collapsible {
        margin-bottom: 1;
    }
    .title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    Label {
        margin-top: 1;
    }
    TextArea {
        height: 8;
        border: solid $panel;
    }
    .buttons {
        margin-top: 1;
        align: center middle;
    }
    Button {
        margin: 0 1;
    }
    #chat_layout {
        height: 1fr;
        padding: 0 1;
    }
    #status {
        height: 1;
        margin-bottom: 1;
    }
    #conversation {
        height: 1fr;
        border: round $panel;
        padding: 0 1;
    }
    #chat_input {
        margin-top: 1;
    }
    #chat_controls, #command_controls {
        height: 3;
        align: right middle;
    }
    """

    def __init__(self, config: AssistantConfig | None = None) -> None:
        super().__init__()
        self.config = config

    def on_mount(self) -> None:
        self.register_theme(LOA_THEME)
        self.theme = LOA_THEME.name
        if self.config is None:
            self.push_screen(SettingsScreen(default_config(), initial_setup=True))
        else:
            self.push_screen(ChatScreen(self.config))

    def apply_settings(self, config: AssistantConfig, initial_setup: bool) -> None:
        """Activate saved settings and transition to or update the chat screen."""
        self.config = config
        if not initial_setup and self.screen_stack:
            for screen in self.screen_stack:
                if isinstance(screen, ChatScreen):
                    screen.config = config
                    screen.messages[0] = {
                        "role": "system",
                        "content": build_system_prompt(
                            config["prompt"],
                            screen.session["id"],
                        ),
                    }
                    screen._save_chat()
                    screen.query_one("#status", Label).update(
                        f"{config['name']} | {config['model']}"
                    )
                    break

        self.pop_screen()
        if initial_setup:
            self.push_screen(ChatScreen(config))
