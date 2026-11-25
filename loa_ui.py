"""Textual setup, chat, settings, and command-confirmation screens."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import httpx
import ollama
from ddgs.exceptions import DDGSException
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.theme import Theme
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TextArea,
    Select,
    TabbedContent,
    TabPane,
)
from textual.containers import VerticalScroll

from loa_agent import (
    AgentCancelled,
    AgentCancellation,
    AssistantReply,
    BREAK_LOOP_MESSAGE,
    CANCELED_ACTION_RESPONSE,
    ChatMessage,
    action_task,
    action_details,
    check_image_generation_readiness,
    configure_runtime_tools,
    execute_approved_action,
    format_ollama_error,
    get_assistant_reply,
    restore_pending_action,
    sanitize_tool_content,
    tool_output_security_metadata,
    wrap_untrusted_tool_output,
)
from loa_config import (
    AssistantConfig,
    DEFAULT_FLUX_MODEL,
    DEFAULT_JANUS_MODEL,
    DEFAULT_SANA_MODEL,
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
)
from loa_agent import run_scheduled_prompt
from loa_mcp import mcp_manager
from loa_skills import (
    activate_skill_command,
    create_markdown_skill,
    list_user_skills,
    reload_skill_modules,
    resolve_skill_command,
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
        mcp_manager.ensure_configured(config.get("mcp_servers", []))
        self.mcp_toggle_ids: dict[str, str] = {}
        self.skill_toggle_ids: dict[str, str] = {}
        self.brain_entries: list[dict[str, object]] = []
        self.cron_jobs: list[dict[str, object]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        skills = list_user_skills(self.config.get("skill_states", {}))
        mcp_servers = self.config.get("mcp_servers", [])
        profile = load_about_me()
        self.brain_entries = list_brain_entries()
        self.cron_jobs = list_cron_jobs()
        self.mcp_toggle_ids = {
            str(server["name"]): f"mcp-toggle-{index}"
            for index, server in enumerate(mcp_servers)
            if isinstance(server, dict) and isinstance(server.get("name"), str)
        }
        self.skill_toggle_ids = {
            str(skill["name"]): f"skill-toggle-{index}"
            for index, skill in enumerate(skills)
        }
        skill_summary = ", ".join(
            f"{skill['name']} ({skill['command']}, {'on' if skill['enabled'] else 'off'})"
            for skill in skills
        ) or "No user skills installed."
        with VerticalScroll(id="settings-scroll"):
            with Container(id="form-container"):
                with Vertical(id="settings-layout"):
                    yield Label(
                        "Assistant setup" if self.initial_setup else "Assistant settings",
                        classes="title",
                    )
                    with TabbedContent(id="settings-tabs"):
                        with TabPane("General", id="general-settings"):
                            with VerticalScroll(classes="settings-tab-scroll"):
                                yield Label("Assistant name:")
                                yield Input(value=self.config["name"], id="ai_name")
                                yield Label("Ollama model:")
                                yield Input(value=self.config["model"], id="model_id")
                                yield Label("Ollama Host URL:")
                                yield Input(
                                    value=self.config.get("ollama_host") or DEFAULT_OLLAMA_HOST,
                                    placeholder=DEFAULT_OLLAMA_HOST,
                                    id="ollama_host",
                                )
                                yield Button("Reset Host", id="reset_host")
                                yield Label("System prompt:")
                                yield TextArea(self.config["prompt"], id="system_prompt")
                                yield Button("Reset Prompt", id="reset_prompt")
                        with TabPane("About Me", id="about-me-settings"):
                            with VerticalScroll(classes="settings-tab-scroll"):
                                yield Label("Saved locally and included in LOA's context.")
                                yield Label("Name:")
                                yield Input(value=profile.get("name", ""), id="profile_name")
                                yield Label("Preferences:")
                                yield TextArea(profile.get("preferences", ""), id="profile_preferences")
                                yield Label("Hardware and environment:")
                                yield TextArea(profile.get("hardware", ""), id="profile_hardware")
                                yield Label("Routines:")
                                yield TextArea(profile.get("routines", ""), id="profile_routines")
                                yield Label("Goals:")
                                yield TextArea(profile.get("goals", ""), id="profile_goals")
                                yield Label("Custom notes:")
                                yield TextArea(profile.get("notes", ""), id="profile_notes")
                                yield Button("Save About Me", id="save_profile")
                        with TabPane("Brain / Memory", id="brain-settings"):
                            with VerticalScroll(classes="settings-tab-scroll"):
                                yield Label("Local token search only; entries stay on this device.")
                                yield Input(placeholder="Search Brain entries", id="brain_search")
                                yield ListView(
                                    *[
                                        ListItem(Static(
                                            f"[{entry['category']}] {entry['content']}",
                                            markup=False,
                                        ))
                                        for entry in self.brain_entries
                                    ],
                                    id="brain_entries",
                                )
                                yield Input(placeholder="Memory text", id="brain_content")
                                yield Input(value="fact", placeholder="Category", id="brain_category")
                                yield Horizontal(
                                    Button("Add memory", id="brain_add"),
                                    Button("Delete selected", id="brain_delete"),
                                )
                                yield Label("", id="brain_status")
                        with TabPane("Cronjobs", id="cron-settings"):
                            with VerticalScroll(classes="settings-tab-scroll"):
                                yield Label(
                                    "Approved jobs run automatically in the background while "
                                    "the LOA web server/service is running. Use five-field "
                                    "cron or @every 15m."
                                )
                                yield ListView(
                                    *[
                                        ListItem(Static(
                                            f"{job['name']}"
                                            f"{' | [ONE-TIME]' if job['one_time'] else ''} | "
                                            f"{'Completed' if job['completed'] else job['schedule']} | "
                                            f"{'enabled' if job['enabled'] else 'disabled'} | "
                                            f"Action: {job['action_type']} | "
                                            f"Pre-approved: {'Yes' if job['auto_approve'] else 'No'}\n"
                                            f"{job['payload']}",
                                            markup=False,
                                        ))
                                        for job in self.cron_jobs
                                    ],
                                    id="cron_jobs",
                                )
                                yield Input(placeholder="Job name", id="cron_name")
                                yield Input(placeholder="Cron expression or @every 15m", id="cron_schedule")
                                yield TextArea("", id="cron_payload")
                                yield Select(
                                    [("Prompt", "prompt"), ("Shell script (zsh)", "shell")],
                                    value="prompt",
                                    id="cron_action_type",
                                )
                                yield Checkbox(
                                    "Pre-approve actions in prompt jobs (run automatically)",
                                    value=True,
                                    id="cron_auto_approve",
                                )
                                yield Checkbox(
                                    "One-time (disable after first execution)",
                                    id="cron_one_time",
                                )
                                yield Horizontal(
                                    Button("Create job", id="cron_create"),
                                    Button("Toggle selected", id="cron_toggle"),
                                    Button("Delete selected", variant="error", id="cron_delete"),
                                    Button("Run now", id="cron_run"),
                                    Button("View history", id="cron_history_button"),
                                )
                                yield Label("", id="cron_status")
                        with TabPane("Skills", id="skills-settings"):
                            with VerticalScroll(classes="settings-tab-scroll"):
                                yield Label("Installed skill toggles:")
                                for skill in skills:
                                    yield Checkbox(
                                        f"{skill['name']} ({skill['kind']}) — {skill['command']}",
                                        value=skill["enabled"] is True,
                                        id=self.skill_toggle_ids[str(skill["name"])],
                                    )
                                yield Label("Additional or uninstalled skill states (JSON):")
                                yield TextArea(
                                    json.dumps(self.config.get("skill_states", {}), ensure_ascii=False, indent=2),
                                    id="skill_states",
                                )
                                yield Label(f"Installed skills: {skill_summary}", id="skills_summary")
                                yield Input(placeholder="New Markdown skill name", id="new_skill_name")
                                yield TextArea("", id="new_skill_content")
                                yield Horizontal(
                                    Button("Create Markdown skill", id="create_skill"),
                                    Button("Reload skills", id="reload_skills"),
                                )
                        with TabPane("MCP Servers", id="mcp-settings"):
                            with VerticalScroll(classes="settings-tab-scroll"):
                                yield Label("Toggle servers or edit the command, args, environment, and list.")
                                for server in mcp_servers:
                                    if (
                                        isinstance(server, dict)
                                        and isinstance(server.get("name"), str)
                                        and str(server["name"]) in self.mcp_toggle_ids
                                    ):
                                        yield Checkbox(
                                            str(server.get("name", "Unnamed server")),
                                            value=server.get("enabled") is True,
                                            id=self.mcp_toggle_ids[str(server["name"])],
                                        )
                                yield TextArea(
                                    json.dumps(self.config.get("mcp_servers", []), ensure_ascii=False, indent=2),
                                    id="mcp_servers",
                                )
                        with TabPane("Image Gen", id="image-gen-settings"):
                            with VerticalScroll(classes="settings-tab-scroll"):
                                image_gen = self.config.get("image_gen", {})
                                if not isinstance(image_gen, dict):
                                    image_gen = {}
                                yield Label(
                                    "Janus is autoregressive; FLUX uses flow matching "
                                    "and Sana uses diffusion."
                                )
                                yield Checkbox(
                                    "Enable image generation",
                                    value=image_gen.get("enabled", True) is True,
                                    id="image_gen_enabled",
                                )
                                yield Label("Provider:")
                                yield Select(
                                    [
                                        ("OpenAI (gpt-image-1)", "openai"),
                                        ("Google Gemini", "gemini"),
                                        ("Local Janus-Pro", "local_janus"),
                                        ("Local FLUX.2 Klein (MLX)", "local_flux"),
                                        ("Local Sana Sprint (PyTorch/MPS)", "local_sana"),
                                    ],
                                    value=image_gen.get("provider", "openai"),
                                    id="image_gen_provider",
                                )
                                yield Label("Model:")
                                yield Input(
                                    value=str(image_gen.get("model", "gpt-image-1")),
                                    id="image_gen_model",
                                )
                                yield Label(
                                    "On a 16 GB Mac use deepseek-ai/Janus-Pro-1B. "
                                    "LOA checks available memory before loading."
                                )
                                yield Label("API key (blank keeps saved key):")
                                yield Input(
                                    placeholder=(
                                        "Saved key configured"
                                        if image_gen.get("api_key")
                                        else "Optional; may use provider environment variable"
                                    ),
                                    password=True,
                                    id="image_gen_api_key",
                                )
                                yield Checkbox(
                                    "Clear saved API key",
                                    id="image_gen_clear_key",
                                )
                                yield Label("Default size:")
                                yield Select(
                                    [
                                        ("1024 × 1024", "1024x1024"),
                                        ("1536 × 1024", "1536x1024"),
                                        ("1024 × 1536", "1024x1536"),
                                    ],
                                    value=image_gen.get("size", "1024x1024"),
                                    id="image_gen_size",
                                )
                                yield Label("Default quality:")
                                yield Select(
                                    [("Low", "low"), ("Medium", "medium"), ("High", "high")],
                                    value=image_gen.get("quality", "high"),
                                    id="image_gen_quality",
                                )
                                yield Button(
                                    "Check connection / readiness",
                                    id="image_gen_check",
                                )
                                yield Label("", id="image_gen_status")
                    yield Label("", id="settings_error")
                    yield Horizontal(
                        Button(
                            "Save and continue" if self.initial_setup else "Save settings",
                            variant="primary",
                            id="save_settings",
                        ),
                        Button("Cancel", id="cancel_settings"),
                        classes="buttons",
                        id="settings-actions",
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
        if event.button.id == "create_skill":
            try:
                path = create_markdown_skill(
                    self.query_one("#new_skill_name", Input).value,
                    self.query_one("#new_skill_content", TextArea).text,
                )
                reload_skill_modules()
                skill_states = dict(self.config.get("skill_states", {}))
                skill_states.setdefault(path.name, False)
                self.query_one("#skill_states", TextArea).text = json.dumps(
                    skill_states,
                    ensure_ascii=False,
                    indent=2,
                )
                skills = list_user_skills(self.config.get("skill_states", {}))
                summary = ", ".join(
                    f"{skill['name']} {skill['command']} "
                    f"({'on' if skill['enabled'] else 'off'})"
                    for skill in skills
                )
                self.query_one("#skills_summary", Label).update(
                    f"Installed skills: {summary}"
                )
                self.query_one("#settings_error", Label).update(
                    f"Created {path.name} disabled. Reopen settings to use its toggle."
                )
            except (OSError, RuntimeError, ValueError) as error:
                self.query_one("#settings_error", Label).update(str(error))
            return
        if event.button.id == "reload_skills":
            try:
                reload_skill_modules()
                configure_runtime_tools(self.config)
                skills = list_user_skills(self.config.get("skill_states", {}))
                summary = ", ".join(
                    f"{skill['name']} {skill['command']} "
                    f"({'on' if skill['enabled'] else 'off'})"
                    for skill in skills
                ) or "No user skills installed."
                self.query_one("#skills_summary", Label).update(
                    f"Installed skills: {summary}"
                )
                self.query_one("#settings_error", Label).update("Skills reloaded.")
            except (OSError, RuntimeError, ValueError) as error:
                self.query_one("#settings_error", Label).update(str(error))
            return
        if event.button.id == "save_profile":
            profile = {
                "name": self.query_one("#profile_name", Input).value,
                "preferences": self.query_one("#profile_preferences", TextArea).text,
                "hardware": self.query_one("#profile_hardware", TextArea).text,
                "routines": self.query_one("#profile_routines", TextArea).text,
                "goals": self.query_one("#profile_goals", TextArea).text,
                "notes": self.query_one("#profile_notes", TextArea).text,
            }
            try:
                save_about_me(profile)
                if isinstance(self.app, LoaApp):
                    self.app.refresh_chat_context()
                self.query_one("#settings_error", Label).update(
                    "About Me profile saved locally and added to LOA's context."
                )
            except (OSError, RuntimeError, ValueError) as error:
                self.query_one("#settings_error", Label).update(str(error))
            return
        if event.button.id == "brain_add":
            try:
                add_brain_entry(
                    self.query_one("#brain_content", Input).value,
                    self.query_one("#brain_category", Input).value or "fact",
                )
                self.query_one("#brain_content", Input).value = ""
                self._refresh_brain_entries()
            except (OSError, RuntimeError, ValueError) as error:
                self.query_one("#brain_status", Label).update(str(error))
            return
        if event.button.id == "brain_delete":
            index = self.query_one("#brain_entries", ListView).index
            if index is None or index >= len(self.brain_entries):
                self.query_one("#brain_status", Label).update("Select a Brain entry first.")
                return
            try:
                delete_brain_entry(str(self.brain_entries[index]["id"]))
                self._refresh_brain_entries()
            except (OSError, RuntimeError, ValueError) as error:
                self.query_one("#brain_status", Label).update(str(error))
            return
        if event.button.id == "cron_create":
            name = self.query_one("#cron_name", Input).value.strip()
            schedule = self.query_one("#cron_schedule", Input).value.strip()
            payload = self.query_one("#cron_payload", TextArea).text.strip()
            one_time = self.query_one("#cron_one_time", Checkbox).value
            action_type = str(self.query_one("#cron_action_type", Select).value)
            auto_approve = self.query_one("#cron_auto_approve", Checkbox).value
            approval_label = (
                "Yes (runs automatically without further prompts)"
                if auto_approve else "No"
            )
            self.app.push_screen(
                ConfirmationScreen(
                    f"Create scheduled job '{name}'?\n\nSchedule: {schedule}\n"
                    f"Target action / prompt: {payload}\n"
                    f"Action type: {action_type}\n"
                    f"Pre-approved execution: {approval_label}\n"
                    f"One-time: {'[ONE-TIME] Yes; disable after first execution' if one_time else 'No'}\n\n"
                    "WARNING: If approved, it will run automatically in the background."
                ),
                lambda approved: self._create_cron_job(
                    name, schedule, payload, one_time, action_type, auto_approve
                )
                if approved else None,
            )
            return
        if event.button.id in {"cron_toggle", "cron_delete", "cron_run", "cron_history_button"}:
            index = self.query_one("#cron_jobs", ListView).index
            if index is None or index >= len(self.cron_jobs):
                self.query_one("#cron_status", Label).update("Select a scheduled job first.")
                return
            job = self.cron_jobs[index]
            job_id = str(job["id"])
            if (
                event.button.id == "cron_run"
                and job["one_time"]
                and job["completed"]
            ):
                self.query_one("#cron_status", Label).update(
                    "This one-time job has already completed."
                )
                return
            if event.button.id == "cron_history_button":
                try:
                    history = get_cron_job_history(job_id)
                    summary = "\n\n".join(
                        f"{item['started_at']} | {item['status']}\n{item['output']}"
                        for item in history
                    ) or "No execution history."
                    self.query_one("#cron_status", Label).update(summary[:4000])
                except (OSError, RuntimeError, ValueError) as error:
                    self.query_one("#cron_status", Label).update(str(error))
                return
            operation = event.button.id.removeprefix("cron_")
            self.app.push_screen(
                ConfirmationScreen(
                    f"{operation.capitalize()} scheduled job '{job['name']}'?\n\n"
                    f"Schedule: {job['schedule']}\nTarget action / prompt: {job['payload']}\n\n"
                    f"One-time: {'[ONE-TIME] Yes' if job['one_time'] else 'No'}\n"
                    "WARNING: An enabled job runs automatically in the background."
                ),
                lambda approved: self._apply_cron_action(operation, job_id, bool(job["enabled"]))
                if approved else None,
            )
            return

        if event.button.id == "image_gen_check":
            current_image_gen = self.config.get("image_gen", {})
            if not isinstance(current_image_gen, dict):
                current_image_gen = {}
            entered_key = self.query_one("#image_gen_api_key", Input).value.strip()
            image_gen = {
                "enabled": self.query_one("#image_gen_enabled", Checkbox).value,
                "provider": self.query_one("#image_gen_provider", Select).value,
                "model": self.query_one("#image_gen_model", Input).value.strip(),
                "api_key": (
                    ""
                    if self.query_one("#image_gen_clear_key", Checkbox).value
                    else entered_key or current_image_gen.get("api_key", "")
                ),
                "size": self.query_one("#image_gen_size", Select).value,
                "quality": self.query_one("#image_gen_quality", Select).value,
            }
            check_config = {**self.config, "image_gen": image_gen}
            try:
                status = check_image_generation_readiness(check_config)
                self.query_one("#image_gen_status", Label).update(status)
            except (
                httpx.HTTPError,
                OSError,
                RuntimeError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                self.query_one("#image_gen_status", Label).update(str(error))
            return

        if event.button.id != "save_settings":
            return

        try:
            mcp_servers = json.loads(self.query_one("#mcp_servers", TextArea).text)
            skill_states = json.loads(self.query_one("#skill_states", TextArea).text)
        except json.JSONDecodeError as error:
            self.query_one("#settings_error", Label).update(
                f"Invalid MCP server or skill-state JSON: {error}"
            )
            return
        if isinstance(mcp_servers, list):
            for server in mcp_servers:
                if not isinstance(server, dict):
                    continue
                toggle_id = self.mcp_toggle_ids.get(str(server.get("name", "")))
                if toggle_id:
                    server["enabled"] = self.query_one(
                        f"#{toggle_id}",
                        Checkbox,
                    ).value
        if isinstance(skill_states, dict):
            for name, toggle_id in self.skill_toggle_ids.items():
                skill_states[name] = self.query_one(
                    f"#{toggle_id}",
                    Checkbox,
                ).value
        current_image_gen = self.config.get("image_gen", {})
        if not isinstance(current_image_gen, dict):
            current_image_gen = {}
        entered_api_key = self.query_one("#image_gen_api_key", Input).value.strip()
        image_gen: dict[str, object] = {
            "enabled": self.query_one("#image_gen_enabled", Checkbox).value,
            "provider": self.query_one("#image_gen_provider", Select).value,
            "model": self.query_one("#image_gen_model", Input).value.strip(),
            "api_key": (
                ""
                if self.query_one("#image_gen_clear_key", Checkbox).value
                else entered_api_key or current_image_gen.get("api_key", "")
            ),
            "size": self.query_one("#image_gen_size", Select).value,
            "quality": self.query_one("#image_gen_quality", Select).value,
        }
        config: AssistantConfig = {
            "name": self.query_one("#ai_name", Input).value.strip(),
            "model": self.query_one("#model_id", Input).value.strip(),
            "prompt": self.query_one("#system_prompt", TextArea).text.strip(),
            "ollama_host": self.query_one("#ollama_host", Input).value.strip(),
            "image_gen": image_gen,
            "mcp_servers": mcp_servers,
            "skill_states": skill_states,
        }
        if not all(config[key] for key in ("name", "model", "prompt", "ollama_host")):
            self.query_one("#settings_error", Label).update(
                "Name, model, Ollama host URL, and system prompt must not be empty."
            )
            return

        try:
            save_config(config)
            reload_skill_modules()
            configure_runtime_tools(config, force_mcp_reload=True)
        except (RuntimeError, ValueError) as error:
            self.query_one("#settings_error", Label).update(str(error))
            return

        app = self.app
        if isinstance(app, LoaApp):
            app.apply_settings(config, self.initial_setup)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "brain_search":
            self._refresh_brain_entries(event.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "image_gen_provider" or not isinstance(event.value, str):
            return
        default_models = {
            "openai": "gpt-image-1",
            "gemini": "gemini-2.5-flash-image",
            "local_janus": DEFAULT_JANUS_MODEL,
            "local_flux": DEFAULT_FLUX_MODEL,
            "local_sana": DEFAULT_SANA_MODEL,
        }
        current_model = self.query_one("#image_gen_model", Input).value.strip()
        if current_model in default_models.values():
            self.query_one("#image_gen_model", Input).value = default_models[event.value]

    def _refresh_brain_entries(self, query: str = "") -> None:
        try:
            self.brain_entries = list_brain_entries(query)
            list_view = self.query_one("#brain_entries", ListView)
            list_view.clear()
            list_view.extend(
                ListItem(Static(
                    f"[{entry['category']}] {entry['content']}",
                    markup=False,
                ))
                for entry in self.brain_entries
            )
            self.query_one("#brain_status", Label).update(
                f"{len(self.brain_entries)} local Brain entries."
            )
        except (OSError, RuntimeError, ValueError) as error:
            self.query_one("#brain_status", Label).update(str(error))

    def _refresh_cron_jobs(self) -> None:
        self.cron_jobs = list_cron_jobs()
        list_view = self.query_one("#cron_jobs", ListView)
        list_view.clear()
        list_view.extend(
            ListItem(Static(
                f"{job['name']}"
                f"{' | [ONE-TIME]' if job['one_time'] else ''} | "
                f"{'Completed' if job['completed'] else job['schedule']} | "
                f"{'enabled' if job['enabled'] else 'disabled'} | "
                f"Action: {job['action_type']} | "
                f"Pre-approved: {'Yes' if job['auto_approve'] else 'No'}\n"
                f"{job['payload']}",
                markup=False,
            ))
            for job in self.cron_jobs
        )

    def _create_cron_job(
        self,
        name: str,
        schedule: str,
        payload: str,
        one_time: bool,
        action_type: str,
        auto_approve: bool,
    ) -> None:
        try:
            job = create_cron_job(
                name,
                schedule,
                payload,
                one_time=one_time,
                action_type=action_type,
                auto_approve=auto_approve,
            )
            self.cron_jobs.insert(0, job)
            self._refresh_cron_jobs()
            self.query_one("#cron_status", Label).update(
                "One-time job saved; it disables after its first execution."
                if one_time
                else "Scheduled job saved; it runs automatically while the LOA web service is running."
            )
        except (OSError, RuntimeError, ValueError) as error:
            self.query_one("#cron_status", Label).update(str(error))

    def _apply_cron_action(self, operation: str, job_id: str, enabled: bool) -> None:
        try:
            if operation == "toggle":
                toggle_cron_job(job_id, not enabled)
                self._refresh_cron_jobs()
            elif operation == "delete":
                delete_cron_job(job_id)
                self._refresh_cron_jobs()
            elif operation == "run":
                self.query_one("#cron_status", Label).update("Running approved job...")
                self.app.run_worker(
                    partial(self._run_cron_job, job_id),
                    thread=True,
                    name="cronjob-run",
                )
        except (OSError, RuntimeError, ValueError) as error:
            self.query_one("#cron_status", Label).update(str(error))

    def _run_cron_job(self, job_id: str) -> None:
        try:
            output = cron_scheduler.run_now(job_id)
        except (OSError, RuntimeError, ValueError) as error:
            output = str(error)
        self.app.call_from_thread(self._finish_cron_job, output[:4000])

    def _finish_cron_job(self, output: str) -> None:
        self.query_one("#cron_status", Label).update(output)

class ConfirmationScreen(ModalScreen[bool]):
    """Require an explicit confirmation before destructive chat maintenance."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        yield Container(
            Vertical(
                Label(self.message, id="confirmation_message"),
                Horizontal(
                    Button("Confirm", variant="error", id="confirm_yes"),
                    Button("Cancel", id="confirm_no"),
                    classes="buttons",
                ),
                id="confirmation-container",
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm_yes":
            self.dismiss(True)
        elif event.button.id == "confirm_no":
            self.dismiss(False)


def _relative_chat_time(timestamp: str) -> str:
    try:
        updated = datetime.fromisoformat(timestamp)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
    except ValueError:
        return timestamp[:10]
    elapsed = max(0, int((datetime.now(timezone.utc) - updated).total_seconds()))
    if elapsed < 60:
        return "just now"
    if elapsed < 3600:
        return f"{elapsed // 60}m ago"
    if elapsed < 86400:
        return f"{elapsed // 3600}h ago"
    if elapsed < 604800:
        return f"{elapsed // 86400}d ago"
    return updated.astimezone().strftime("%b %d")


class StartScreen(Screen):
    """Welcome screen for starting or resuming a saved conversation."""

    def __init__(self, config: AssistantConfig) -> None:
        super().__init__()
        self.config = config
        self.sessions: list[ChatSession] = []
        self.load_error = ""
        self.refresh_sessions()

    def _recent_rows(self) -> list[Button]:
        rows = []
        for session in self.sessions:
            message_count = sum(
                1
                for message in session["messages"]
                if message.get("role") in {"user", "assistant"}
                and str(message.get("content") or "").strip()
            )
            preview = " ".join(str(session["summary"]).split())
            rows.append(
                Button(
                    Text(
                        f"{session['title']}  |  {_relative_chat_time(session['updated_at'])}\n"
                        f"{message_count} messages  ·  {preview}",
                        overflow="fold",
                    ),
                    id=f"home_chat_{session['id']}",
                    classes="home-chat-choice",
                )
            )
        return rows

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Welcome to LOA", classes="title"),
            Label(
                "Start a new conversation or continue where you left off.",
                id="home_subtitle",
            ),
            Horizontal(
                Button("New Chat", variant="primary", id="home_new_chat"),
                Button(
                    "Continue Last Chat",
                    id="home_continue_chat",
                    disabled=not self.sessions,
                ),
                Button("Settings", id="home_settings"),
                Button("Exit", id="home_exit"),
                classes="buttons",
            ),
            Label("Recent chats", classes="title"),
            Label(
                "No saved conversations yet." if not self.sessions else "",
                id="home_empty",
            ),
            VerticalScroll(*self._recent_rows(), id="home_recent_chats"),
            Label(self.load_error, id="home_status"),
            id="home_layout",
        )

    def on_mount(self) -> None:
        self.refresh_sessions()

    def refresh_sessions(self) -> None:
        try:
            self.sessions = list_chat_sessions()[:5]
            self.load_error = ""
        except (OSError, RuntimeError, ValueError) as error:
            self.sessions = []
            self.load_error = str(error)
        if self.is_mounted:
            recent = self.query_one("#home_recent_chats", VerticalScroll)
            recent.remove_children()
            rows = self._recent_rows()
            if rows:
                recent.mount(*rows)
            self.query_one("#home_continue_chat", Button).disabled = not self.sessions
            self.query_one("#home_empty", Label).update(
                "No saved conversations yet." if not self.sessions else ""
            )
            self.query_one("#home_status", Label).update(self.load_error)

    def _open_session(self, session: ChatSession) -> None:
        self.app.push_screen(ChatScreen(self.config, session))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id.startswith("home_chat_"):
            chat_id = event.button.id.removeprefix("home_chat_")
            try:
                self._open_session(load_chat_session(chat_id))
            except (OSError, RuntimeError, ValueError) as error:
                self.query_one("#home_status", Label).update(str(error))
                self.refresh_sessions()
        elif event.button.id == "home_new_chat":
            session = new_chat_session()
            try:
                save_chat_session(session)
                self._open_session(session)
            except (OSError, RuntimeError, ValueError) as error:
                self.query_one("#home_status", Label).update(str(error))
        elif event.button.id == "home_continue_chat":
            if self.sessions:
                try:
                    self._open_session(
                        load_chat_session(self.sessions[0]["id"])
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    self.query_one("#home_status", Label).update(str(error))
                    self.refresh_sessions()
        elif event.button.id == "home_settings":
            self.app.push_screen(SettingsScreen(self.config))
        elif event.button.id == "home_exit":
            self.app.exit()


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
        self.visible_sessions = self.sessions

    def compose(self) -> ComposeResult:
        active_id = self._active_chat_id()
        rows = [
            ListItem(
                Static(
                    f"{'[ACTIVE] ' if session['id'] == active_id else ''}"
                    f"{session['title']}  |  {session['updated_at'][:10]}\n"
                    f"{session['summary']}",
                    markup=False,
                )
            )
            for session in self.visible_sessions
        ]
        yield Container(
            Vertical(
                Label("Chats", classes="title"),
                Label(self.load_error, id="chat_manager_error"),
                Input(placeholder="Search chats", id="chat_search"),
                ListView(*rows, id="saved_chats"),
                Input(placeholder="Title for Rename", id="chat_title"),
                Horizontal(
                    Button("Open / Switch", variant="primary", id="open_chat"),
                    Button("New Chat", id="new_chat"),
                    Button("Rename", id="rename_chat"),
                    Button("Vectorize and delete", id="vectorize_chat"),
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
        if index is None or index >= len(self.visible_sessions):
            self.query_one("#chat_manager_error", Label).update(
                "Select a saved chat first."
            )
            return
        session = self.visible_sessions[index]
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
                        app.pop_screen()
                if isinstance(app.screen, StartScreen):
                    app.screen.refresh_sessions()
                app.push_screen(ChatManagerScreen())
            elif event.button.id == "vectorize_chat":
                app.push_screen(
                    ConfirmationScreen(
                        f'Vectorize "{session["title"]}" into local Brain memories, '
                        "then permanently delete its raw chat history? Its key facts "
                        "and tokens will be saved first."
                    ),
                    lambda confirmed: self._vectorize(session["id"]) if confirmed else None,
                )
        except (OSError, RuntimeError, ValueError) as error:
            self.query_one("#chat_manager_error", Label).update(str(error))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "chat_search":
            query = event.value.casefold().strip()
            self.visible_sessions = [
                session for session in self.sessions
                if not query
                or query in session["title"].casefold()
                or query in session["summary"].casefold()
            ]
            list_view = self.query_one("#saved_chats", ListView)
            list_view.clear()
            list_view.extend(
                ListItem(Static(
                    f"{'[ACTIVE] ' if session['id'] == self._active_chat_id() else ''}"
                    f"{session['title']}  |  {session['updated_at'][:10]}\n"
                    f"{session['summary']}",
                    markup=False,
                ))
                for session in self.visible_sessions
            )

    def _active_chat_id(self) -> str | None:
        chat_screen = next(
            (
                screen for screen in reversed(self.app.screen_stack)
                if isinstance(screen, ChatScreen)
            ),
            None,
        )
        return chat_screen.session["id"] if chat_screen is not None else None

    def _vectorize(self, chat_id: str) -> None:
        app = self.app
        if not isinstance(app, LoaApp):
            return
        try:
            session = load_chat_session(chat_id)
            memories = extract_chat_memories(session["messages"])
            if not memories:
                self.query_one("#chat_manager_error", Label).update(
                    "No useful facts or tokens were found; the chat was kept."
                )
                return
            delete_chat_session(chat_id)
            app.pop_screen()
            if isinstance(app.screen, ChatScreen) and app.screen.session["id"] == chat_id:
                app.pop_screen()
            if isinstance(app.screen, StartScreen):
                app.screen.refresh_sessions()
            manager = ChatManagerScreen()
            manager.load_error = (
                f"Saved {len(memories)} local Brain entries; raw chat deleted."
            )
            app.push_screen(manager)
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
        session: ChatSession,
    ) -> None:
        super().__init__()
        self.config = config
        configure_runtime_tools(config)
        self.session = session
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
        self.current_cancellation: AgentCancellation | None = None
        self.operation_id = 0
        self.skill_commands = [
            str(skill["command"])
            + (
                " (enable Python skill first)"
                if skill["kind"] == "py" and not skill["enabled"]
                else ""
            )
            for skill in list_user_skills(config.get("skill_states", {}))
        ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label(f"{self.config['name']} | {self.config['model']}", id="status"),
            VerticalScroll(id="conversation"),
            Input(
                placeholder="Message, /<skill-command>, or /help",
                id="chat_input",
            ),
            Static("", id="skill-command-hints"),
            Horizontal(
                Button("Send", variant="primary", id="send_message"),
                Button("Break Loop", id="break_loop"),
                Button("Home", id="home"),
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
        cron_scheduler.set_executor(lambda job: run_scheduled_prompt(self.config, job))
        self.query_one("#command_controls").display = False
        self._update_break_loop_visibility()
        if len(self.messages) == 1:
            self._write("LOA is ready. Type /help to see available commands.")
            self._save_chat()
        else:
            self._render_history()
        self._update_skill_command_hints("")
        if self.pending_action is not None:
            self._show_pending_action(self.pending_action)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat_input":
            self._handle_input(event.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "chat_input":
            self._update_skill_command_hints(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_message":
            chat_input = self.query_one("#chat_input", Input)
            self._handle_input(chat_input.value)
        elif event.button.id == "open_settings":
            self._open_settings()
        elif event.button.id == "open_chats":
            self.app.push_screen(ChatManagerScreen())
        elif event.button.id == "home":
            self._go_home()
        elif event.button.id == "exit_chat":
            self.app.exit()
        elif event.button.id == "confirm_command":
            self._confirm_action()
        elif event.button.id == "decline_command":
            self._cancel_pending_action()
        elif event.button.id == "break_loop":
            self._break_loop()

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
        if text.lower() == "/home":
            self._go_home()
            return
        if text.lower() == "/chats":
            self.app.push_screen(ChatManagerScreen())
            return
        if text.lower() == "/mcp":
            self._write("Manage MCP servers in Settings > MCP Servers.")
            return
        if text.lower().startswith("/mcp "):
            self._handle_mcp_command(text)
            return
        if text.lower().startswith("/skills"):
            self._handle_skills_command(text)
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
                "/mcp [name on|off|reload] - show or manage MCP servers\n"
                "/skills [name on|off|reload] - list or manage user skills\n"
                "/new - start a new chat\n"
                "/home - return to the start screen\n"
                "/exit - close LOA\n"
                "Ask LOA to search the web when you need online information."
            )
            return

        try:
            skill = resolve_skill_command(text, self.config.get("skill_states", {}))
        except (OSError, RuntimeError, ValueError) as error:
            self._write(str(error))
            return
        if skill is not None:
            activate_skill_command(self.messages, skill)

        self.messages.append({"role": "user", "content": text})
        self._save_chat()
        self._set_busy(True)
        self._start_assistant_request()

    def _update_skill_command_hints(self, value: str) -> None:
        typed = value.strip()
        if not typed.startswith("/"):
            self.query_one("#skill-command-hints", Static).update("")
            return
        query = typed.split(maxsplit=1)[0].casefold()
        matches = [
            command for command in self.skill_commands
            if command.split(" ", 1)[0].casefold().startswith(query)
        ]
        if matches:
            hint = "Skill commands: " + ", ".join(matches)
        else:
            hint = "No matching skill commands."
        self.query_one("#skill-command-hints", Static).update(hint)

    def _start_assistant_request(self) -> None:
        self._refresh_system_prompt()
        snapshot = [message.copy() for message in self.messages]
        self.operation_id += 1
        operation_id = self.operation_id
        cancellation = AgentCancellation()
        self.current_cancellation = cancellation
        self.app.run_worker(
            partial(
                self._request_assistant,
                snapshot,
                self.session["id"],
                cancellation,
                operation_id,
            ),
            thread=True,
            name="assistant-request",
        )

    def _request_assistant(
        self,
        messages: list[ChatMessage],
        active_chat_id: str,
        cancellation: AgentCancellation,
        operation_id: int,
    ) -> None:
        try:
            reply = get_assistant_reply(
                self.config,
                messages,
                active_chat_id=active_chat_id,
                cancellation=cancellation,
            )
        except AgentCancelled:
            self.app.call_from_thread(self._assistant_canceled, operation_id)
            return
        except (
            ollama.ResponseError,
            httpx.HTTPError,
            DDGSException,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            self.app.call_from_thread(
                self._show_error,
                "Could not complete the request: "
                f"{format_ollama_error(error, self.config.get('ollama_host') or DEFAULT_OLLAMA_HOST)}",
                operation_id,
            )
            return
        self.app.call_from_thread(self._show_reply, reply, operation_id)

    def _show_reply(self, reply: AssistantReply, operation_id: int) -> None:
        if operation_id != self.operation_id:
            return
        self.current_cancellation = None
        self._set_busy(False)
        answer = sanitize_tool_content(reply["text"])
        self.messages = reply["history"]
        for task in reply["tasks"]:
            if task["label"] == "Generated image":
                self._write(f"Generated image\n{task['details']}")
            else:
                self._add_collapsible(task["label"], task["details"])
        if reply["sources"]:
            assistant = next(
                (message for message in reversed(self.messages) if message.get("role") == "assistant"),
                None,
            )
            if assistant is not None:
                assistant.setdefault("ui", {})["sources"] = reply["sources"]
        if answer:
            self._write_assistant(answer)

        action = reply["action"]
        if action is not None:
            self.pending_action = action
            self._show_pending_action(action)
        else:
            self.pending_action = None
            self.query_one("#command_controls").display = False
            self.query_one("#chat_controls").display = True
            self.query_one("#chat_input", Input).disabled = False
        self.query_one("#status", Label).update(
            "Request approval required." if action is not None else "Ready"
        )
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
        self._update_break_loop_visibility()

    def _show_error(self, message: str, operation_id: int) -> None:
        if operation_id != self.operation_id:
            return
        self.current_cancellation = None
        self._set_busy(False)
        self._write(message)
        self.query_one("#command_controls").display = False
        self.query_one("#chat_controls").display = True
        self.query_one("#chat_input", Input).disabled = False
        self._save_chat()

    def _assistant_canceled(self, operation_id: int) -> None:
        if operation_id != self.operation_id:
            return
        self.current_cancellation = None
        self._break_loop()

    def _confirm_action(self) -> None:
        action = self.pending_action
        if action is None:
            return
        self.query_one("#command_controls").display = False
        self._set_busy(True)
        self.operation_id += 1
        operation_id = self.operation_id
        cancellation = AgentCancellation()
        self.current_cancellation = cancellation
        self.query_one("#status", Label).update("Running approved action...")
        self.app.run_worker(
            partial(self._action_worker, action, cancellation, operation_id),
            thread=True,
            name="approved-action",
        )

    def _action_worker(
        self,
        action: ChatMessage,
        cancellation: AgentCancellation,
        operation_id: int,
    ) -> None:
        try:
            output = execute_approved_action(action, cancellation)
        except AgentCancelled:
            self.app.call_from_thread(self._assistant_canceled, operation_id)
            return
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            output, _ = wrap_untrusted_tool_output(
                str(action.get("tool_name") or action.get("type") or "action"),
                f"Action execution failed: {error}",
            )
        self.app.call_from_thread(self._finish_action, output, operation_id)

    def _finish_action(self, output: str, operation_id: int) -> None:
        if operation_id != self.operation_id:
            return
        self.current_cancellation = None
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
                **tool_output_security_metadata(output),
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

    def _cancel_pending_action(self) -> None:
        action = self.pending_action
        if action is None:
            return
        self._append_canceled_action(action)
        self.pending_action = None
        self.query_one("#command_controls").display = False
        self.query_one("#chat_controls").display = True
        self.query_one("#confirm_command", Button).label = "Run"
        self.query_one("#decline_command", Button).label = "Cancel"
        self.query_one("#chat_input", Input).disabled = False
        self.query_one("#status", Label).update("Request canceled by user")
        self._update_break_loop_visibility()
        self._save_chat()

    def _append_canceled_action(self, action: ChatMessage) -> None:
        task = action_task(action, CANCELED_ACTION_RESPONSE)
        self.messages.append(
            {
                "role": "tool",
                "tool_name": str(action.get("tool_name") or action.get("type") or "action"),
                "content": CANCELED_ACTION_RESPONSE,
                "ui": {
                    "type": "action",
                    "summary": "Request canceled by user",
                    "details": task["details"],
                },
            }
        )
        self._add_collapsible("Request canceled by user", task["details"])

    def _break_loop(self) -> None:
        if self.current_cancellation is not None:
            self.current_cancellation.cancel()
            self.current_cancellation = None
        self.operation_id += 1
        action = self.pending_action
        if action is not None:
            self._append_canceled_action(action)
            self.pending_action = None
        self.messages.append({"role": "user", "content": BREAK_LOOP_MESSAGE})
        self._write(f"You: {BREAK_LOOP_MESSAGE}")
        self.query_one("#command_controls").display = False
        self.query_one("#chat_controls").display = True
        self.query_one("#confirm_command", Button).label = "Run"
        self.query_one("#decline_command", Button).label = "Cancel"
        self.query_one("#chat_input", Input).disabled = False
        self._set_busy(False)
        self.query_one("#status", Label).update(
            "Loop stopped – waiting for user input"
        )
        self._save_chat()

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
        return build_system_prompt(
            self.config["prompt"],
            self.session["id"],
            self.config.get("skill_states", {}),
        )

    def _handle_skills_command(self, text: str) -> None:
        parts = text.split()
        try:
            if len(parts) == 1:
                skills = list_user_skills(self.config.get("skill_states", {}))
                summary = "\n".join(
                    f"{skill['name']} {skill['command']}: "
                    f"{'enabled' if skill['enabled'] else 'disabled'}"
                    for skill in skills
                ) or "No user skills installed."
                self._write(summary)
                return
            if len(parts) == 2 and parts[1].casefold() == "reload":
                reload_skill_modules()
                configure_runtime_tools(self.config)
                self._write("Skills reloaded.")
                return
            if len(parts) == 3 and parts[2].casefold() in {"on", "off"}:
                skill_name = parts[1]
                if not any(
                    skill["name"] == skill_name
                    for skill in list_user_skills()
                ):
                    self._write(f"Skill not found: {skill_name}")
                    return
                skill_states = dict(self.config.get("skill_states", {}))
                skill_states[skill_name] = parts[2].casefold() == "on"
                self.config["skill_states"] = skill_states
                save_config(self.config)
                reload_skill_modules()
                configure_runtime_tools(self.config)
                self._refresh_system_prompt()
                self._save_chat()
                self._write(f"{skill_name} {'enabled' if skill_states[skill_name] else 'disabled'}.")
                return
            self._write("Usage: /skills [name on|off|reload]")
        except (OSError, RuntimeError, ValueError) as error:
            self._write(f"Could not manage skills: {error}")

    def _handle_mcp_command(self, text: str) -> None:
        parts = text.split()
        try:
            if len(parts) == 2 and parts[1].casefold() == "reload":
                configure_runtime_tools(self.config, force_mcp_reload=True)
                self._write("MCP servers reloaded.")
                return
            if len(parts) == 3 and parts[2].casefold() in {"on", "off"}:
                server = next(
                    (
                        item
                        for item in self.config.get("mcp_servers", [])
                        if item.get("name") == parts[1]
                    ),
                    None,
                )
                if server is None:
                    self._write(f"MCP server not found: {parts[1]}")
                    return
                server["enabled"] = parts[2].casefold() == "on"
                save_config(self.config)
                configure_runtime_tools(
                    self.config,
                    force_mcp_reload=True,
                )
                self._write(f"MCP server {parts[1]} updated.")
                return
            self._write("Usage: /mcp [name on|off|reload]")
        except (OSError, RuntimeError, ValueError) as error:
            self._write(f"Could not manage MCP servers: {error}")

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
                    self._write_assistant(content)
            elif role == "tool":
                ui = message.get("ui")
                if isinstance(ui, dict):
                    if ui.get("type") == "image":
                        self._write_image_summary(ui)
                    else:
                        self._add_collapsible(
                            str(ui.get("summary", "Completed tool")),
                            str(ui.get("details", message.get("content", ""))),
                        )

    def _write_image_summary(self, ui: dict[str, object]) -> None:
        filename = str(ui.get("filename", ""))
        image_path = (
            Path.home() / ".loa" / "generated_images" / filename
            if re.fullmatch(r"[a-f0-9]{32}\.png", filename)
            else "unavailable"
        )
        self._write(
            "Generated image\n"
            f"Saved file: {image_path}\n"
            f"Provider: {ui.get('provider', 'unknown')}\n"
            f"Prompt: {ui.get('prompt', '')}"
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

    def _go_home(self) -> None:
        if self.busy or self.pending_action is not None:
            self._write("Finish or cancel the pending action before returning Home.")
            return
        self.app.pop_screen()
        if isinstance(self.app.screen, StartScreen):
            self.app.screen.refresh_sessions()

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.query_one("#chat_input", Input).disabled = busy
        self.query_one("#send_message", Button).disabled = busy
        self._update_break_loop_visibility()

    def _update_break_loop_visibility(self) -> None:
        button = self.query_one("#break_loop", Button)
        button.display = self.busy or self.pending_action is not None

    def _write(self, text: str) -> None:
        self.query_one("#conversation", VerticalScroll).mount(
            Static(text, markup=False)
        )
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    def _write_assistant(self, text: str) -> None:
        from rich.markdown import Markdown

        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.mount(
            Static(f"{self.config['name']}:", markup=False),
            Static(Markdown(text)),
        )
        conversation.scroll_end(animate=False)


class LoaApp(App[None]):
    """Application that persists setup and hosts the chat interface."""

    ENABLE_COMMAND_PALETTE = False

    CSS = """
    #settings-scroll {
        height: 1fr;
        min-height: 1;
    }
    #form-container {
        height: 1fr;
        min-height: 1;
        padding: 1 1;
        background: $background-lighten-1;
        border: solid $accent;
        margin: 0 1;
    }
    #settings-layout {
        height: 1fr;
        min-height: 1;
    }
    #settings-tabs {
        height: 1fr;
        min-height: 1;
        border: solid $panel;
        padding: 0 1;
    }
    TabPane {
        height: 1fr;
        padding: 0;
    }
    .settings-tab-scroll {
        height: 1fr;
        min-height: 1;
        padding: 1;
    }
    #brain_entries, #cron_jobs {
        height: 10;
        border: solid $panel;
    }
    #about-me-settings TextArea, #cron-settings TextArea {
        height: 5;
    }
    #confirmation-container {
        width: 80%;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: solid $warning;
    }
    #confirmation_message {
        height: auto;
        margin-bottom: 1;
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
        margin-top: 0;
        align: center middle;
    }
    #settings-actions {
        height: auto;
        min-height: 3;
    }
    #settings_error {
        height: auto;
        margin: 0;
    }
    Button {
        margin: 0 1;
    }
    #chat_layout {
        height: 1fr;
        padding: 0 1;
    }
    #home_layout {
        height: 1fr;
        padding: 1 2;
    }
    #home_subtitle {
        height: auto;
        margin-bottom: 1;
    }
    #home_recent_chats {
        height: 1fr;
        min-height: 3;
        border: solid $panel;
    }
    .home-chat-choice {
        width: 1fr;
        height: auto;
        content-align: left middle;
        margin: 0 0 1 0;
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
        cron_scheduler.set_executor(
            lambda job: run_scheduled_prompt(self.config or default_config(), job)
        )
        if self.config is None:
            self.push_screen(SettingsScreen(default_config(), initial_setup=True))
        else:
            self.push_screen(StartScreen(self.config))

    def apply_settings(self, config: AssistantConfig, initial_setup: bool) -> None:
        """Activate saved settings and transition to or update the chat screen."""
        self.config = config
        if not initial_setup and self.screen_stack:
            for screen in self.screen_stack:
                if isinstance(screen, ChatScreen):
                    screen.config = config
                    screen._refresh_system_prompt()
                    screen._save_chat()
                    screen.query_one("#status", Label).update(
                        f"{config['name']} | {config['model']}"
                    )
                elif isinstance(screen, StartScreen):
                    screen.config = config

        self.pop_screen()
        if initial_setup:
            self.push_screen(StartScreen(config))

    def refresh_chat_context(self) -> None:
        for screen in self.screen_stack:
            if isinstance(screen, ChatScreen):
                screen._refresh_system_prompt()
                screen._save_chat()
