import unittest
from unittest.mock import patch

import loa_config
import loa_ui


class FixedIdentitySettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_setup_saves_model_with_fixed_identity(self):
        with (
            patch.object(loa_ui.mcp_manager, "ensure_configured"),
            patch.object(loa_ui.cron_scheduler, "set_executor"),
            patch.object(loa_ui, "list_user_skills", return_value=[]),
            patch.object(loa_ui, "load_about_me", return_value={}),
            patch.object(loa_ui, "list_brain_entries", return_value=[]),
            patch.object(loa_ui, "list_cron_jobs", return_value=[]),
            patch.object(loa_ui, "list_chat_sessions", return_value=[]),
            patch.object(loa_ui, "reload_skill_modules"),
            patch.object(loa_ui, "configure_runtime_tools", return_value=[]),
            patch.object(loa_ui, "save_config") as save,
        ):
            app = loa_ui.LoaApp()
            async with app.run_test(size=(160, 50)) as pilot:
                self.assertEqual(len(app.screen.query("#ai_name")), 0)
                self.assertEqual(len(app.screen.query("#system_prompt")), 0)
                app.screen.query_one("#model_id").value = "custom-model"
                await pilot.click("#save_settings")
                saved = save.call_args.args[0]
                self.assertEqual(saved["name"], loa_config.DEFAULT_NAME)
                self.assertEqual(saved["prompt"], loa_config.DEFAULT_PROMPT)
                self.assertEqual(saved["model"], "custom-model")
                self.assertIsInstance(app.screen, loa_ui.StartScreen)
