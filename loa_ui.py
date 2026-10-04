"""Textual setup, chat, settings, and command-confirmation screens."""

from __future__ import annotations

import subprocess
from functools import partial

import ollama
from ddgs.exceptions import DDGSException
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.theme import Theme
from textual.screen import Screen
from textual.widgets import Button, Header, Input, Label, RichLog, TextArea

from loa_agent import (
    AssistantReply,
    execute_shell_command,
    extract_bash_command,
    get_assistant_reply,
)
from loa_config import AssistantConfig, default_config
from loa_storage import save_config


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
                Label("System prompt:"),
                TextArea(self.config["prompt"], id="system_prompt"),
                Label("", id="settings_error"),
                Horizontal(
                    Button(
                        "Save and continue" if self.initial_setup else "Save settings",
                        variant="primary",
                        id="save_settings",
                    ),
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

        if event.button.id != "save_settings":
            return

        config: AssistantConfig = {
            "name": self.query_one("#ai_name", Input).value.strip(),
            "model": self.query_one("#model_id", Input).value.strip(),
            "prompt": self.query_one("#system_prompt", TextArea).text.strip(),
        }
        if not all(config.values()):
            self.query_one("#settings_error", Label).update(
                "Name, model, and system prompt must not be empty."
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


class ChatScreen(Screen):
    """Interactive chat interface with explicit shell and HTTP actions."""

    def __init__(self, config: AssistantConfig) -> None:
        super().__init__()
        self.config = config
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": config["prompt"]}
        ]
        self.pending_command: str | None = None
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label(f"{self.config['name']} | {self.config['model']}", id="status"),
            RichLog(id="conversation", wrap=True, markup=False, auto_scroll=True),
            Input(
                placeholder="Message or /help",
                id="chat_input",
            ),
            Horizontal(
                Button("Send", variant="primary", id="send_message"),
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
        self._write("LOA is ready. Type /help to see available commands.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat_input":
            self._handle_input(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_message":
            chat_input = self.query_one("#chat_input", Input)
            self._handle_input(chat_input.value)
        elif event.button.id == "open_settings":
            self._open_settings()
        elif event.button.id == "exit_chat":
            self.app.exit()
        elif event.button.id == "confirm_command":
            self._confirm_command()
        elif event.button.id == "decline_command":
            self._finish_command("Command was not run.")

    def _handle_input(self, raw_text: str) -> None:
        text = raw_text.strip()
        if not text or self.busy or self.pending_command is not None:
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
        if text.lower() == "/help":
            self._write(
                "Commands:\n"
                "/settings - edit and save assistant settings\n"
                "/exit - close LOA\n"
                "Ask LOA to search the web when you need online information."
            )
            return

        self.messages.append({"role": "user", "content": text})
        self._set_busy(True)
        snapshot = [message.copy() for message in self.messages]
        self.app.run_worker(
            partial(self._request_assistant, snapshot), thread=True, name="assistant-request"
        )

    def _request_assistant(self, messages: list[dict[str, str]]) -> None:
        try:
            reply = get_assistant_reply(self.config, messages)
        except (
            ollama.ResponseError,
            httpx.HTTPError,
            DDGSException,
            OSError,
            ValueError,
        ) as error:
            self.app.call_from_thread(
                self._show_error, f"Could not complete the request: {error}"
            )
            return
        self.app.call_from_thread(self._show_reply, reply)

    def _show_reply(self, reply: AssistantReply) -> None:
        self._set_busy(False)
        answer = reply["text"]
        self.messages = reply["history"]
        self._write(f"{self.config['name']}: {answer}")
        sources = reply["sources"]
        if sources:
            source_lines = ["Web sources:"]
            source_lines.extend(f"- {source['title']}: {source['url']}" for source in sources)
            self._write("\n".join(source_lines))

        command = extract_bash_command(answer)
        if command is not None:
            if not command:
                self._append_system("The assistant returned an empty command. Nothing was run.")
                return
            self.pending_command = command
            self._write(f"Proposed command:\n{command}\nReview it, then choose Run command or Cancel command.")
            self.query_one("#command_controls").display = True
            self.query_one("#chat_input", Input).disabled = True
            self.query_one("#chat_controls").display = False

    def _show_error(self, message: str) -> None:
        self._set_busy(False)
        self._write(message)
        self.messages.pop()

    def _confirm_command(self) -> None:
        command = self.pending_command
        if command is None:
            return
        self.query_one("#command_controls").display = False
        self._set_busy(True)
        self.app.run_worker(
            partial(self._command_worker, command),
            thread=True,
            name="shell-command",
        )

    def _command_worker(self, command: str) -> None:
        try:
            output = execute_shell_command(command)
        except (OSError, subprocess.SubprocessError) as error:
            output = f"Command execution failed: {error}"
        self.app.call_from_thread(self._finish_command, output)

    def _finish_command(self, output: str) -> None:
        self.pending_command = None
        self._set_busy(False)
        self._append_system(output)
        self.query_one("#command_controls").display = False
        self.query_one("#chat_controls").display = True
        self.query_one("#chat_input", Input).disabled = False
        self.query_one("#chat_input", Input).focus()

    def _append_system(self, content: str) -> None:
        self.messages.append({"role": "system", "content": content})
        self._write(content)

    def _open_settings(self) -> None:
        if isinstance(self.app, LoaApp):
            self.app.push_screen(SettingsScreen(self.config))

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.query_one("#chat_input", Input).disabled = busy
        self.query_one("#send_message", Button).disabled = busy

    def _write(self, text: str) -> None:
        self.query_one("#conversation", RichLog).write(text)


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
                    screen.messages[0] = {"role": "system", "content": config["prompt"]}
                    screen.query_one("#status", Label).update(
                        f"{config['name']} | {config['model']}"
                    )
                    break

        self.pop_screen()
        if initial_setup:
            self.push_screen(ChatScreen(config))
