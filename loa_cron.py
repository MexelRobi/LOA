"""Local persisted cron/interval jobs, scheduled within the LOA web process."""

from __future__ import annotations

import sqlite3
import logging
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx
import ollama
from ddgs.exceptions import DDGSException


_INTERVAL_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_SCHEDULED_ACTION = Callable[[dict[str, Any]], str]


def cron_database_path() -> Path:
    from loa_storage import config_file_path

    return config_file_path().parent / "cronjobs.sqlite3"


def _connect() -> sqlite3.Connection:
    path = cron_database_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS cron_jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                schedule TEXT NOT NULL,
                payload TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                created_at REAL NOT NULL,
                next_run REAL NOT NULL,
                one_time INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                action_type TEXT NOT NULL DEFAULT 'prompt',
                auto_approve INTEGER NOT NULL DEFAULT 1
            )"""
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(cron_jobs)")
        }
        if "one_time" not in columns:
            connection.execute(
                "ALTER TABLE cron_jobs ADD COLUMN one_time INTEGER NOT NULL DEFAULT 0"
            )
        if "completed" not in columns:
            connection.execute(
                "ALTER TABLE cron_jobs ADD COLUMN completed INTEGER NOT NULL DEFAULT 0"
            )
        if "action_type" not in columns:
            connection.execute(
                "ALTER TABLE cron_jobs ADD COLUMN action_type TEXT NOT NULL DEFAULT 'prompt'"
            )
        if "auto_approve" not in columns:
            connection.execute(
                "ALTER TABLE cron_jobs ADD COLUMN auto_approve INTEGER NOT NULL DEFAULT 1"
            )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS cron_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                started_at REAL NOT NULL,
                finished_at REAL,
                status TEXT NOT NULL,
                output TEXT NOT NULL DEFAULT ''
            )"""
        )
        path.chmod(0o600)
        return connection
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not open the local cron database: {error}") from error


def validate_schedule(schedule: str) -> str:
    if not isinstance(schedule, str):
        raise ValueError("Cron schedule must be text.")
    schedule = schedule.strip()
    if schedule.startswith("@every "):
        value = schedule[7:].strip().lower()
        if len(value) < 2 or not value[:-1].isdigit() or value[-1] not in _INTERVAL_UNITS:
            raise ValueError("Intervals use @every followed by a number and s, m, h, or d.")
        duration = int(value[:-1]) * _INTERVAL_UNITS[value[-1]]
        if not 5 <= duration <= 31_536_000:
            raise ValueError("Intervals must be between 5 seconds and 365 days.")
        return f"@every {value}"
    fields = schedule.split()
    if len(fields) != 5:
        raise ValueError("Use a five-field cron expression or an interval such as @every 15m.")
    ranges = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
    for field, bounds in zip(fields, ranges):
        _parse_cron_field(field, *bounds)
    return " ".join(fields)


def _parse_cron_field(field: str, minimum: int, maximum: int) -> set[int]:
    values: set[int] = set()
    for component in field.split(","):
        base, separator, step_text = component.partition("/")
        if separator:
            if not step_text.isdigit() or int(step_text) < 1:
                raise ValueError("Cron step values must be positive integers.")
            step = int(step_text)
        else:
            step = 1
        if base == "*":
            start, end = minimum, maximum
        elif "-" in base:
            pieces = base.split("-", 1)
            if not all(piece.isdigit() for piece in pieces):
                raise ValueError("Cron fields must use numeric values and ranges.")
            start, end = map(int, pieces)
            if start > end:
                raise ValueError("Cron ranges must be ascending.")
        elif base.isdigit():
            start = int(base)
            end = maximum if separator else start
        else:
            raise ValueError("Cron fields must use *, numbers, ranges, lists, and steps.")
        if start < minimum or end > maximum:
            raise ValueError(f"Cron value must be between {minimum} and {maximum}.")
        values.update(range(start, end + 1, step))
    if not values:
        raise ValueError("Cron field must match at least one value.")
    return values


def _cron_matches(schedule: str, moment: datetime) -> bool:
    fields = schedule.split()
    values = [
        _parse_cron_field(field, *bounds)
        for field, bounds in zip(fields, ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7)))
    ]
    weekday = (moment.weekday() + 1) % 7
    day_match = moment.day in values[2]
    weekday_match = weekday in values[4] or (weekday == 0 and 7 in values[4])
    if fields[2] == "*" or fields[4] == "*":
        calendar_match = day_match and weekday_match
    else:
        calendar_match = day_match or weekday_match
    return (
        moment.minute in values[0]
        and moment.hour in values[1]
        and moment.month in values[3]
        and calendar_match
    )


def _next_run(schedule: str, now: float | None = None) -> float:
    now = time.time() if now is None else now
    if schedule.startswith("@every "):
        value = schedule[7:]
        return now + int(value[:-1]) * _INTERVAL_UNITS[value[-1]]
    moment = datetime.fromtimestamp(now).replace(second=0, microsecond=0)
    for _ in range(527_040):
        moment = moment.replace(second=0, microsecond=0)
        moment = datetime.fromtimestamp(moment.timestamp() + 60)
        if _cron_matches(schedule, moment):
            return moment.timestamp()
    raise ValueError("Cron expression has no scheduled time within the next year.")


def _job(row: sqlite3.Row) -> dict[str, Any]:
    one_time = bool(row["one_time"])
    return {
        "id": row["id"],
        "name": row["name"],
        "schedule": row["schedule"],
        "payload": row["payload"],
        "enabled": bool(row["enabled"]),
        "created_at": datetime.fromtimestamp(row["created_at"]).astimezone().isoformat(timespec="seconds"),
        "next_run_at": datetime.fromtimestamp(row["next_run"]).astimezone().isoformat(timespec="seconds")
        if row["enabled"] else None,
        "one_time": one_time,
        "run_once": one_time,
        "completed": bool(row["completed"]),
        "action_type": row["action_type"],
        "auto_approve": bool(row["auto_approve"]),
    }


def list_cron_jobs() -> list[dict[str, Any]]:
    try:
        with _connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cron_jobs ORDER BY enabled DESC, next_run ASC, name COLLATE NOCASE"
            ).fetchall()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not list scheduled jobs: {error}") from error
    return [_job(row) for row in rows]


def _normalize_one_time(
    one_time: bool | None,
    run_once: bool | None,
) -> bool | None:
    if one_time is not None and not isinstance(one_time, bool):
        raise ValueError("Cronjob one_time must be a boolean.")
    if run_once is not None and not isinstance(run_once, bool):
        raise ValueError("Cronjob run_once must be a boolean.")
    if one_time is not None and run_once is not None and one_time != run_once:
        raise ValueError("Cronjob one_time and run_once values must agree.")
    return one_time if one_time is not None else run_once


def _validate_action_settings(action_type: str, auto_approve: bool) -> None:
    if not isinstance(action_type, str) or action_type not in {"prompt", "shell"}:
        raise ValueError("Cronjob action_type must be prompt or shell.")
    if not isinstance(auto_approve, bool):
        raise ValueError("Cronjob auto_approve must be a boolean.")


def create_cron_job(
    name: str,
    schedule: str,
    payload: str,
    one_time: bool | None = None,
    run_once: bool | None = None,
    action_type: str = "prompt",
    auto_approve: bool = True,
) -> dict[str, Any]:
    if not isinstance(name, str) or not isinstance(payload, str):
        raise ValueError("Cronjob name and action must be text.")
    name = name.strip()
    payload = payload.strip()
    schedule = validate_schedule(schedule)
    one_time = _normalize_one_time(one_time, run_once) or False
    _validate_action_settings(action_type, auto_approve)
    if not name or len(name) > 80:
        raise ValueError("Cronjob name must contain 1 to 80 characters.")
    if not payload or len(payload) > 4000:
        raise ValueError("Cronjob action must contain 1 to 4,000 characters.")
    now = time.time()
    job_id = uuid.uuid4().hex
    try:
        with _connect() as connection:
            connection.execute(
                """INSERT INTO cron_jobs
                (id, name, schedule, payload, enabled, created_at, next_run, one_time,
                 completed, action_type, auto_approve)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, 0, ?, ?)""",
                (
                    job_id,
                    name,
                    schedule,
                    payload,
                    now,
                    _next_run(schedule, now),
                    int(one_time),
                    action_type,
                    int(auto_approve),
                ),
            )
            row = connection.execute("SELECT * FROM cron_jobs WHERE id = ?", (job_id,)).fetchone()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not create scheduled job: {error}") from error
    return _job(row)


def update_cron_job(
    job_id: str,
    name: str | None = None,
    schedule: str | None = None,
    payload: str | None = None,
    one_time: bool | None = None,
    run_once: bool | None = None,
    action_type: str | None = None,
    auto_approve: bool | None = None,
) -> dict[str, Any]:
    if any(value is not None and not isinstance(value, str) for value in (name, schedule, payload)):
        raise ValueError("Cronjob name, schedule, and action must be text.")
    job = _get_job_row(job_id)
    new_name = (name if name is not None else job["name"]).strip()
    new_schedule = validate_schedule(schedule if schedule is not None else job["schedule"])
    new_payload = (payload if payload is not None else job["payload"]).strip()
    new_action_type = action_type if action_type is not None else job["action_type"]
    new_auto_approve = (
        auto_approve if auto_approve is not None else bool(job["auto_approve"])
    )
    _validate_action_settings(new_action_type, new_auto_approve)
    requested_one_time = _normalize_one_time(one_time, run_once)
    new_one_time = (
        bool(job["one_time"])
        if requested_one_time is None
        else requested_one_time
    )
    new_completed = bool(job["completed"]) if new_one_time else False
    if not new_name or len(new_name) > 80:
        raise ValueError("Cronjob name must contain 1 to 80 characters.")
    if not new_payload or len(new_payload) > 4000:
        raise ValueError("Cronjob action must contain 1 to 4,000 characters.")
    now = time.time()
    next_run = _next_run(new_schedule, now) if job["enabled"] else job["next_run"]
    try:
        with _connect() as connection:
            connection.execute(
                """UPDATE cron_jobs
                SET name=?, schedule=?, payload=?, next_run=?, one_time=?, completed=?,
                    action_type=?, auto_approve=?
                WHERE id=?""",
                (
                    new_name,
                    new_schedule,
                    new_payload,
                    next_run,
                    int(new_one_time),
                    int(new_completed),
                    new_action_type,
                    int(new_auto_approve),
                    job_id,
                ),
            )
            row = connection.execute("SELECT * FROM cron_jobs WHERE id=?", (job_id,)).fetchone()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not update scheduled job {job_id}: {error}") from error
    return _job(row)


def toggle_cron_job(job_id: str, enabled: bool) -> dict[str, Any]:
    if not isinstance(enabled, bool):
        raise ValueError("Cronjob enabled state must be a boolean.")
    job = _get_job_row(job_id)
    if enabled and bool(job["one_time"]) and bool(job["completed"]):
        raise ValueError("A completed one-time cronjob cannot be enabled again.")
    next_run = _next_run(job["schedule"]) if enabled else job["next_run"]
    try:
        with _connect() as connection:
            connection.execute(
                "UPDATE cron_jobs SET enabled=?, next_run=? WHERE id=?",
                (int(enabled), next_run, job_id),
            )
            row = connection.execute("SELECT * FROM cron_jobs WHERE id=?", (job_id,)).fetchone()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not toggle scheduled job {job_id}: {error}") from error
    return _job(row)


def delete_cron_job(job_id: str) -> None:
    _get_job_row(job_id)
    try:
        with _connect() as connection:
            connection.execute("DELETE FROM cron_jobs WHERE id=?", (job_id,))
            connection.execute("DELETE FROM cron_history WHERE job_id=?", (job_id,))
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not delete scheduled job {job_id}: {error}") from error


def get_cron_job_history(job_id: str, limit: int = 20) -> list[dict[str, Any]]:
    _get_job_row(job_id)
    try:
        with _connect() as connection:
            rows = connection.execute(
                """SELECT * FROM cron_history WHERE job_id=?
                ORDER BY started_at DESC LIMIT ?""",
                (job_id, min(max(limit, 1), 100)),
            ).fetchall()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not read cronjob history: {error}") from error
    return [
        {
            "started_at": datetime.fromtimestamp(row["started_at"]).astimezone().isoformat(timespec="seconds"),
            "finished_at": datetime.fromtimestamp(row["finished_at"]).astimezone().isoformat(timespec="seconds")
            if row["finished_at"] else None,
            "status": row["status"],
            "output": row["output"],
        }
        for row in rows
    ]


def _get_job_row(job_id: str) -> sqlite3.Row:
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("Invalid cronjob id.")
    try:
        with _connect() as connection:
            row = connection.execute("SELECT * FROM cron_jobs WHERE id=?", (job_id,)).fetchone()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not load cronjob {job_id}: {error}") from error
    if row is None:
        raise ValueError(f"Cronjob not found: {job_id}")
    return row


class CronScheduler:
    """Run persisted jobs in a daemon thread owned by one LOA server process."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._executor: _SCHEDULED_ACTION | None = None

    def start(self, executor: _SCHEDULED_ACTION) -> None:
        with self._lock:
            self._executor = executor
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="loa-cron", daemon=True)
            self._thread.start()

    def set_executor(self, executor: _SCHEDULED_ACTION) -> None:
        with self._lock:
            self._executor = executor

    def stop(self, timeout: float = 3) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def run_now(self, job_id: str) -> str:
        row = _get_job_row(job_id)
        job = _job(row)
        if job["one_time"] and job["completed"]:
            raise ValueError("This one-time cronjob has already completed.")
        with self._lock:
            if job_id in self._active:
                raise ValueError("This cronjob is already running.")
            self._active.add(job_id)
        try:
            return self._execute(job)
        finally:
            with self._lock:
                self._active.discard(job_id)

    def _run(self) -> None:
        while not self._stop.wait(1):
            now = time.time()
            try:
                with _connect() as connection:
                    due = connection.execute(
                        "SELECT * FROM cron_jobs WHERE enabled=1 AND next_run<=?",
                        (now,),
                    ).fetchall()
                    for row in due:
                        with self._lock:
                            if row["id"] in self._active:
                                continue
                            self._active.add(row["id"])
                        try:
                            connection.execute(
                                "UPDATE cron_jobs SET next_run=? WHERE id=?",
                                (_next_run(row["schedule"], now), row["id"]),
                            )
                        except (sqlite3.Error, RuntimeError, ValueError):
                            with self._lock:
                                self._active.discard(row["id"])
                            raise
                        job = _job(row)
                        threading.Thread(
                            target=self._execute_and_release,
                            args=(job,),
                            name=f"loa-cron-{row['id'][:8]}",
                            daemon=True,
                        ).start()
            except (OSError, RuntimeError, sqlite3.Error, ValueError):
                logging.getLogger(__name__).exception("Cron scheduler polling failed.")
                self._stop.wait(5)

    def _execute_and_release(self, job: dict[str, Any]) -> None:
        try:
            self._execute(job)
        finally:
            with self._lock:
                self._active.discard(job["id"])

    def _execute(self, job: dict[str, Any]) -> str:
        started = time.time()
        history_id: int | None = None
        unexpected_error: Exception | None = None
        try:
            with _connect() as connection:
                cursor = connection.execute(
                    "INSERT INTO cron_history(job_id, started_at, status) VALUES (?, ?, 'running')",
                    (job["id"], started),
                )
                history_id = cursor.lastrowid
            with self._lock:
                executor = self._executor
            if executor is None:
                output = "Scheduler has no executor configured; start the LOA web service."
                status = "failed"
            else:
                output = executor(job)
                status = "completed"
        except (
            DDGSException,
            httpx.HTTPError,
            ollama.ResponseError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            TimeoutError,
            ValueError,
            sqlite3.Error,
        ) as error:
            output = f"Cronjob failed: {error}"
            status = "failed"
            logging.getLogger(__name__).exception("Cronjob execution failed.")
        except Exception as error:
            output = f"Cronjob failed unexpectedly: {error}"
            status = "failed"
            unexpected_error = error
            logging.getLogger(__name__).exception("Cronjob execution failed unexpectedly.")
        if history_id is None:
            raise RuntimeError(output)
        try:
            with _connect() as connection:
                connection.execute(
                    "UPDATE cron_history SET finished_at=?, status=?, output=? WHERE id=?",
                    (time.time(), status, str(output)[:10000], history_id),
                )
                if job.get("one_time"):
                    connection.execute(
                        "UPDATE cron_jobs SET enabled=0, completed=1 WHERE id=?",
                        (job["id"],),
                    )
        except sqlite3.Error as error:
            raise RuntimeError(f"Could not save cronjob execution history: {error}") from error
        if unexpected_error is not None:
            raise RuntimeError(output) from unexpected_error
        return str(output)


cron_scheduler = CronScheduler()
