import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import loa_skills


class SkillCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.skills_directory = Path(self.temporary_directory.name)
        self.directory_patcher = patch(
            "loa_skills.ensure_skills_directory",
            return_value=self.skills_directory,
        )
        self.directory_patcher.start()

    def tearDown(self) -> None:
        self.directory_patcher.stop()
        self.temporary_directory.cleanup()

    def test_explicit_and_default_commands_forward_arguments(self) -> None:
        (self.skills_directory / "hardware.md").write_text(
            '---\ncommand: "/hw" # shortcut\ndescription: Hardware helper\n'
            "---\nList hardware.",
            encoding="utf-8",
        )
        (self.skills_directory / "wasauchimmer.md").write_text(
            "Handle whatever is asked.",
            encoding="utf-8",
        )

        skills = loa_skills.list_user_skills({})
        self.assertEqual(
            [(skill["name"], skill["command"]) for skill in skills],
            [("hardware.md", "/hw"), ("wasauchimmer.md", "/wasauchimmer")],
        )
        resolved = loa_skills.resolve_skill_command("/hw list --all", {})
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["arguments"], "list --all")
        self.assertEqual(resolved["instructions"], "List hardware.")

        inferred = loa_skills.resolve_skill_command("/wasauchimmer now", {})
        self.assertEqual(inferred["arguments"], "now")

    def test_skill_activation_replaces_previous_command_context(self) -> None:
        messages = [
            {"role": "system", "content": "base prompt"},
            {
                "role": "system",
                "content": f"{loa_skills.SKILL_COMMAND_CONTEXT_MARKER}\nold",
            },
            {"role": "user", "content": "prior prompt"},
        ]
        skill = {
            "name": "hardware.md",
            "command": "/hw",
            "arguments": "list",
            "instructions": "List hardware.",
        }

        loa_skills.activate_skill_command(messages, skill)

        contexts = [
            message for message in messages
            if str(message.get("content", "")).startswith(
                loa_skills.SKILL_COMMAND_CONTEXT_MARKER
            )
        ]
        self.assertEqual(len(contexts), 1)
        self.assertIn("exact text after the command", contexts[0]["content"])
        self.assertEqual(messages[0]["content"], "base prompt")
        self.assertEqual(messages[1], contexts[0])

    def test_disabled_python_skill_requires_enabling(self) -> None:
        (self.skills_directory / "local_tool.py").write_text(
            '"""A local utility."""\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Enable the Python skill"):
            loa_skills.resolve_skill_command("/local_tool run", {})
        enabled = loa_skills.resolve_skill_command(
            "/local_tool run",
            {"local_tool.py": True},
        )
        self.assertEqual(enabled["instructions"], "A local utility.")


if __name__ == "__main__":
    unittest.main()
