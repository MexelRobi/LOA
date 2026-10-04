"""Load and save LOA settings in the user's application config directory."""

import json
import os
import sys
from pathlib import Path

from loa_config import AssistantConfig, LEGACY_DEFAULT_PROMPT, DEFAULT_PROMPT


def config_file_path() -> Path:
    """Return the per-user path used to persist assistant settings."""
    home = Path.home()
    if sys.platform == "darwin":
        config_dir = home / "Library" / "Application Support" / "LOA"
    elif os.name == "nt":
        config_dir = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "LOA"
    else:
        config_dir = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "loa"
    return config_dir / "config.json"


def load_config() -> AssistantConfig | None:
    """Load saved settings, returning None when the user has not set up LOA."""
    path = config_file_path()
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read LOA settings at {path}: {error}") from error

    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), str) for key in ("name", "model", "prompt")
    ):
        raise RuntimeError(
            f"LOA settings at {path} are invalid. Use /settings after startup to replace them."
        )
    config: AssistantConfig = {
        "name": data["name"],
        "model": data["model"],
        "prompt": data["prompt"],
    }
    if config["prompt"] == LEGACY_DEFAULT_PROMPT:
        config["prompt"] = DEFAULT_PROMPT
        save_config(config)
    return config


def save_config(config: AssistantConfig) -> None:
    """Persist settings, creating the per-user config directory if needed."""
    path = config_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise RuntimeError(f"Could not save LOA settings at {path}: {error}") from error
