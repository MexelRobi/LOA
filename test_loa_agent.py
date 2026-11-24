import unittest
from unittest.mock import patch

import loa_agent
import loa_config


class NonMcpToolCatalogTests(unittest.TestCase):
    def test_lists_integrated_tools_and_excludes_mcp_tools(self):
        extra_python_skill = {
            "type": "function",
            "function": {
                "name": "local_skill_action",
                "description": "Perform a local skill action.",
            },
        }
        mcp_tool = {
            "type": "function",
            "function": {
                "name": "remote_lookup",
                "description": "Call an MCP server.",
            },
        }
        with (
            patch.object(loa_agent, "TOOLS", [*loa_agent.BASE_TOOLS, extra_python_skill, mcp_tool]),
            patch.object(loa_agent, "MCP_TOOL_NAMES", {"remote_lookup"}),
        ):
            with (
                patch("loa_storage.build_chat_catalog", return_value=""),
                patch("loa_brain.about_me_context", return_value=""),
                patch("loa_skills.enabled_markdown_skill_instructions", return_value=""),
            ):
                prompt = loa_config.build_system_prompt("Base prompt")

        for tool_name in (
            "propose_shell_command",
            "web_search",
            "run_applescript",
            "read_file",
            "list_directory",
            "edit_or_write_file",
            "recall_past_chat",
            "recall_brain",
            "remember_brain",
            "cron_job",
            "local_skill_action",
        ):
            with self.subTest(tool_name=tool_name):
                self.assertIn(f"- {tool_name}:", prompt)
        self.assertNotIn("remote_lookup", prompt)


if __name__ == "__main__":
    unittest.main()
