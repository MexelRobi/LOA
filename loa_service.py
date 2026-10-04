"""Install and manage LOA's per-user macOS LaunchAgent for the web UI."""

import os
import getpass
import plistlib
import subprocess
import sys
from pathlib import Path

from loa_auth import MIN_PASSWORD_LENGTH, has_web_password, save_web_password
from loa_storage import config_file_path


SERVICE_LABEL = "com.loa.web"
SERVICE_FILENAME = f"{SERVICE_LABEL}.plist"


def launch_agent_path() -> Path:
    """Return the current user's LaunchAgents plist path."""
    return Path.home() / "Library" / "LaunchAgents" / SERVICE_FILENAME


def _launchctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["/bin/launchctl", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"launchctl {' '.join(arguments)} failed: {details}")
    return result


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _service_is_loaded() -> bool:
    result = _launchctl("print", f"{_domain()}/{SERVICE_LABEL}", check=False)
    return result.returncode == 0


def _stop_loaded_service() -> None:
    if _service_is_loaded():
        _launchctl("bootout", f"{_domain()}/{SERVICE_LABEL}")


def _program_arguments(
    executable: Path,
    host: str,
    port: int,
    certfile: str | None,
    keyfile: str | None,
) -> list[str]:
    if getattr(sys, "frozen", False):
        command = [str(executable)]
    else:
        command = [sys.executable, str(executable)]
    command.extend(["--web", "--host", host, "--port", str(port)])
    if certfile and keyfile:
        command.extend(["--certfile", str(Path(certfile).expanduser().resolve())])
        command.extend(["--keyfile", str(Path(keyfile).expanduser().resolve())])
    return command


def _ensure_web_password() -> None:
    """Prompt for and save a password before a first background-service install."""
    if has_web_password():
        return
    print(
        f"Create a web password with at least {MIN_PASSWORD_LENGTH} characters "
        "for the mobile chat."
    )
    while True:
        password = getpass.getpass("New web password: ")
        confirmation = getpass.getpass("Confirm web password: ")
        if password != confirmation:
            print("Passwords did not match. Try again.")
            continue
        try:
            save_web_password(password)
        except ValueError as error:
            print(error)
            continue
        except OSError as error:
            raise RuntimeError(f"Could not save the web password: {error}") from error
        return


def install_web_service(
    host: str = "0.0.0.0",
    port: int = 8765,
    certfile: str | None = None,
    keyfile: str | None = None,
) -> Path:
    """Register and start LOA web mode as a LaunchAgent for the current user."""
    if sys.platform != "darwin":
        raise RuntimeError("The LOA background web service is currently supported only on macOS.")
    _ensure_web_password()
    if bool(certfile) != bool(keyfile):
        raise ValueError("Both a TLS certificate and private key are required for HTTPS.")

    executable = Path(sys.executable).resolve()
    if not getattr(sys, "frozen", False):
        executable = Path(__file__).resolve().parent / "loa.py"

    support_dir = config_file_path().parent
    log_dir = support_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = launch_agent_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    plist = {
        "Label": SERVICE_LABEL,
        "ProgramArguments": _program_arguments(
            executable,
            host,
            port,
            certfile,
            keyfile,
        ),
        "WorkingDirectory": str(executable.parent),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "ProcessType": "Background",
        "StandardOutPath": str(log_dir / "web-service.log"),
        "StandardErrorPath": str(log_dir / "web-service-error.log"),
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        },
    }
    new_contents = plistlib.dumps(plist, fmt=plistlib.FMT_XML, sort_keys=True)
    previous_contents = path.read_bytes() if path.exists() else None
    previous_mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    was_loaded = _service_is_loaded()
    temporary_path = path.with_suffix(".plist.tmp")
    try:
        if was_loaded:
            _launchctl("bootout", f"{_domain()}/{SERVICE_LABEL}")
        temporary_path.write_bytes(new_contents)
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
        _launchctl("bootstrap", _domain(), str(path))
        _launchctl("kickstart", "-k", f"{_domain()}/{SERVICE_LABEL}")
    except (OSError, RuntimeError):
        if _service_is_loaded():
            _launchctl("bootout", f"{_domain()}/{SERVICE_LABEL}", check=False)
        if previous_contents is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(previous_contents)
            path.chmod(previous_mode)
            if was_loaded:
                _launchctl("bootstrap", _domain(), str(path), check=False)
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def uninstall_web_service() -> bool:
    """Stop and remove LOA's LaunchAgent, if one is installed."""
    if sys.platform != "darwin":
        raise RuntimeError("The LOA background web service is currently supported only on macOS.")
    path = launch_agent_path()
    if not path.exists():
        return False
    _stop_loaded_service()
    path.unlink()
    return True


def web_service_status() -> str:
    """Return the current LaunchAgent registration and runtime status."""
    if sys.platform != "darwin":
        raise RuntimeError("The LOA background web service is currently supported only on macOS.")
    path = launch_agent_path()
    if not path.exists():
        return "LOA web background service is not installed."
    result = _launchctl("print", f"{_domain()}/{SERVICE_LABEL}", check=False)
    if result.returncode != 0:
        return "LOA web background service is installed but not running."
    state_line = next(
        (line.strip() for line in result.stdout.splitlines() if "state =" in line),
        "LOA web background service is loaded.",
    )
    return f"LOA web background service is installed and {state_line}"
