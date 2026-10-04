"""Persistent password hashing for the mobile web interface."""

import hashlib
import hmac
import json
import secrets
from pathlib import Path

from loa_storage import config_file_path


PASSWORD_FILE = "web_password.json"
MIN_PASSWORD_LENGTH = 12
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1


def password_file_path() -> Path:
    """Return the per-user path where the web password verifier is stored."""
    return config_file_path().parent / PASSWORD_FILE


def has_web_password() -> bool:
    """Check whether a saved web password verifier exists."""
    return password_file_path().is_file()


def save_web_password(password: str) -> None:
    """Store a salted scrypt verifier without writing the plaintext password."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must contain at least {MIN_PASSWORD_LENGTH} characters.")
    salt = secrets.token_bytes(16)
    verifier = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=32,
    )
    path = password_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "salt": salt.hex(),
                "verifier": verifier.hex(),
                "n": SCRYPT_N,
                "r": SCRYPT_R,
                "p": SCRYPT_P,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def verify_web_password(password: str) -> bool:
    """Verify a password against the saved salted scrypt verifier."""
    path = password_file_path()
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        salt = bytes.fromhex(saved["salt"])
        expected = bytes.fromhex(saved["verifier"])
        n, r, p = (int(saved[key]) for key in ("n", "r", "p"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read the saved web password: {error}") from error

    actual = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=len(expected),
    )
    return hmac.compare_digest(actual, expected)
