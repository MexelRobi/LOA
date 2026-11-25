import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from loa_cron import (
    CronScheduler,
    create_cron_job,
    get_cron_job_history,
    list_cron_jobs,
    toggle_cron_job,
    update_cron_job,
)


class CronOneTimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "cronjobs.sqlite3"
        self.path_patcher = patch(
            "loa_cron.cron_database_path",
            return_value=self.database_path,
        )
        self.path_patcher.start()

    def tearDown(self):
        self.path_patcher.stop()
        self.temp_dir.cleanup()

    def test_existing_database_migrates_with_recurring_defaults(self):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """CREATE TABLE cron_jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    next_run REAL NOT NULL
                )"""
            )
            connection.execute(
                "INSERT INTO cron_jobs VALUES (?, ?, ?, ?, 1, ?, ?)",
                ("a" * 32, "legacy", "@every 5s", "legacy action", time.time(), time.time() + 5),
            )

        job = list_cron_jobs()[0]

        self.assertFalse(job["one_time"])
        self.assertFalse(job["run_once"])
        self.assertFalse(job["completed"])
        self.assertEqual(job["action_type"], "prompt")
        self.assertTrue(job["auto_approve"])

    def test_create_and_update_persist_action_settings(self):
        job = create_cron_job(
            "shell task",
            "@every 5s",
            "echo ready",
            action_type="shell",
            auto_approve=False,
        )
        self.assertEqual(job["action_type"], "shell")
        self.assertFalse(job["auto_approve"])

        updated = update_cron_job(
            job["id"],
            action_type="prompt",
            auto_approve=True,
        )
        self.assertEqual(updated["action_type"], "prompt")
        self.assertTrue(updated["auto_approve"])

    def test_create_rejects_invalid_action_settings(self):
        with self.assertRaisesRegex(ValueError, "action_type"):
            create_cron_job(
                "invalid",
                "@every 5s",
                "echo ready",
                action_type="diffusion",
            )
        with self.assertRaisesRegex(ValueError, "auto_approve"):
            create_cron_job(
                "invalid",
                "@every 5s",
                "echo ready",
                auto_approve="yes",
            )

    def test_run_once_alias_is_disabled_and_retains_history_after_success(self):
        job = create_cron_job("single", "@every 5s", "do something", run_once=True)
        scheduler = CronScheduler()
        scheduler.set_executor(lambda _job: "done")

        self.assertEqual(scheduler.run_now(job["id"]), "done")

        saved_job = list_cron_jobs()[0]
        self.assertTrue(saved_job["one_time"])
        self.assertTrue(saved_job["run_once"])
        self.assertTrue(saved_job["completed"])
        self.assertFalse(saved_job["enabled"])
        self.assertEqual(get_cron_job_history(job["id"])[0]["status"], "completed")
        with self.assertRaises(ValueError):
            scheduler.run_now(job["id"])
        with self.assertRaises(ValueError):
            toggle_cron_job(job["id"], True)

    def test_one_time_job_is_consumed_after_failed_execution(self):
        job = create_cron_job("single failure", "@every 5s", "do something", one_time=True)
        scheduler = CronScheduler()

        def fail(_job):
            raise KeyError("unexpected failure")

        scheduler.set_executor(fail)
        with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
            scheduler.run_now(job["id"])

        saved_job = list_cron_jobs()[0]
        self.assertTrue(saved_job["completed"])
        self.assertFalse(saved_job["enabled"])
        self.assertEqual(get_cron_job_history(job["id"])[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
