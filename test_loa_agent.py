import base64
import io
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import loa_agent
import loa_config
import loa_storage


class ScheduledActionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.config = loa_config.default_config()

    def test_shell_job_executes_payload_directly(self):
        with patch(
            "loa_agent.execute_shell_command",
            return_value="Command exit status: 0\nready",
        ) as execute_shell:
            result = loa_agent.run_scheduled_prompt(
                self.config,
                {"action_type": "shell", "payload": "printf ready"},
            )

        self.assertIn("ready", result)
        execute_shell.assert_called_once_with("printf ready")

    def test_auto_approved_prompt_action_is_returned_to_assistant(self):
        action = {
            "type": "shell_command",
            "tool_name": "propose_shell_command",
            "command": "printf ready",
            "reason": "Complete the scheduled task.",
        }
        history = [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "Running the command."},
        ]
        replies = [
            {"action": action, "history": history, "text": ""},
            {"action": None, "history": history, "text": "Scheduled task complete."},
        ]
        wrapped_output = loa_agent.wrap_untrusted_tool_output(
            "propose_shell_command", "Command exit status: 0\nready"
        )[0]
        with (
            patch("loa_agent.build_system_prompt", return_value="system"),
            patch("loa_agent.get_assistant_reply", side_effect=replies) as get_reply,
            patch(
                "loa_agent.execute_approved_action",
                return_value=wrapped_output,
            ) as execute_action,
        ):
            result = loa_agent.run_scheduled_prompt(
                self.config,
                {"payload": "Run the scheduled command", "auto_approve": True},
            )

        self.assertEqual(result, "Scheduled task complete.")
        execute_action.assert_called_once_with(action)
        second_messages = get_reply.call_args_list[1].args[1]
        tool_result = second_messages[-1]
        self.assertEqual(tool_result["role"], "tool")
        self.assertTrue(tool_result["untrusted_wrapped"])
        self.assertEqual(tool_result["ui"]["type"], "action")

    def test_prompt_job_can_disable_automatic_action_execution(self):
        action = {
            "type": "shell_command",
            "tool_name": "propose_shell_command",
            "command": "printf ready",
            "reason": "Complete the scheduled task.",
        }
        with (
            patch("loa_agent.build_system_prompt", return_value="system"),
            patch(
                "loa_agent.get_assistant_reply",
                return_value={"action": action, "history": [], "text": ""},
            ),
            patch("loa_agent.execute_approved_action") as execute_action,
        ):
            result = loa_agent.run_scheduled_prompt(
                self.config,
                {
                    "payload": "Run the scheduled command",
                    "auto_approve": False,
                },
            )

        self.assertIn("requires explicit user approval", result)
        execute_action.assert_not_called()

    def test_cron_approval_details_show_execution_authorization(self):
        details = loa_agent.action_details({
            "type": "cron_job",
            "arguments": {
                "operation": "create",
                "action_type": "shell",
                "auto_approve": True,
            },
        })

        self.assertIn("Action type: shell", details)
        self.assertIn(
            "Pre-approved execution: Yes (runs automatically without further prompts)",
            details,
        )


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


class ImageGenerationTests(unittest.TestCase):
    def test_local_flux_edit_uses_reference_and_private_output(self):
        config = loa_config.default_config()
        config["image_gen"].update(provider="local_flux", model=loa_config.DEFAULT_FLUX_MODEL)
        cancellation = loa_agent.AgentCancellation()
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / ("a" * 32 + ".png")
            reference.write_bytes(b"reference")

            def fake_runner(arguments, model, cancellation):
                compile(arguments[2], "<flux-runner>", "exec")
                self.assertEqual(arguments[-1], str(reference))
                Path(arguments[-3]).write_bytes(b"\x89PNG\r\n\x1a\nimage")
                return subprocess.CompletedProcess(arguments, 0)

            with (
                patch.object(loa_agent, "generated_images_directory_path", return_value=Path(directory)),
                patch.object(loa_agent.os, "uname", return_value=type("Uname", (), {"machine": "arm64"})()),
                patch.object(loa_agent, "_run_local_image_process", side_effect=fake_runner) as runner,
            ):
                _, ui = loa_agent.generate_image(
                    config, "Edit this image", "1024x1024", "high",
                    cancellation, image_filename=reference.name,
                )
                self.assertEqual(stat.S_IMODE((Path(directory) / ui["filename"]).stat().st_mode), 0o600)
                self.assertIs(runner.call_args.kwargs["cancellation"], cancellation)

    def test_reference_image_rejects_traversal_and_non_flux_provider(self):
        config = loa_config.default_config()
        config["image_gen"].update(provider="local_flux", model=loa_config.DEFAULT_FLUX_MODEL)
        with self.assertRaisesRegex(ValueError, "Invalid reference"):
            loa_agent.generate_image(config, "edit", "1024x1024", "high", image_filename="../secret")
        config["image_gen"]["provider"] = "local_janus"
        with self.assertRaisesRegex(ValueError, "requires the local FLUX"):
            loa_agent.generate_image(config, "edit", "1024x1024", "high", image_filename="a" * 32 + ".png")

    def test_sana_runner_passes_requested_size_and_two_steps(self):
        config = loa_config.default_config()
        config["image_gen"].update(provider="local_sana", model=loa_config.DEFAULT_SANA_MODEL)

        def fake_runner(arguments, model, cancellation):
            compile(arguments[2], "<sana-runner>", "exec")
            self.assertEqual(arguments[-2], "1536x1024")
            self.assertIn("num_inference_steps=2", arguments[2])
            self.assertIn("SanaSprintPipeline.from_pretrained", arguments[2])
            Path(arguments[-3]).write_bytes(b"\x89PNG\r\n\x1a\nimage")
            return subprocess.CompletedProcess(arguments, 0)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(loa_agent, "generated_images_directory_path", return_value=Path(directory)),
            patch.object(loa_agent.os, "uname", return_value=type("Uname", (), {"machine": "arm64"})()),
            patch.object(loa_agent, "_run_local_image_process", side_effect=fake_runner),
        ):
            _, ui = loa_agent.generate_image(config, "a tree", "1536x1024", "high")
            self.assertEqual(ui["provider"], "local_sana")

    def test_coreml_stable_diffusion_generates_and_resizes_image(self):
        config = loa_config.default_config()
        config["image_gen"].update(
            provider="local_coreml_sd",
            model=loa_config.DEFAULT_COREML_SD_MODEL,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)

            def fake_runner(model, prompt, output_path, quality, cancellation):
                from PIL import Image

                self.assertEqual(model, loa_config.DEFAULT_COREML_SD_MODEL)
                self.assertEqual(prompt, "a small red apple")
                self.assertEqual(quality, "medium")
                Image.new("RGB", (512, 512), "red").save(output_path, format="PNG")
                return subprocess.CompletedProcess([], 0, "", "")

            with (
                patch.object(
                    loa_agent,
                    "generated_images_directory_path",
                    return_value=output_directory,
                ),
                patch.object(
                    loa_agent.os,
                    "uname",
                    return_value=type("Uname", (), {"machine": "arm64"})(),
                ),
                patch.object(
                    loa_agent,
                    "_run_coreml_sd_process",
                    side_effect=fake_runner,
                ) as runner,
            ):
                _, ui = loa_agent.generate_image(
                    config,
                    "a small red apple",
                    "1537x777",
                    "medium",
                )
            image_path = output_directory / ui["filename"]
            from PIL import Image

            with Image.open(image_path) as image:
                self.assertEqual(image.size, (1537, 777))
                self.assertEqual(image.format, "PNG")
            self.assertEqual(ui["provider"], "local_coreml_sd")
            runner.assert_called_once()

    def test_coreml_runner_reports_denoising_steps_and_keeps_rss_limit(self):
        steps = []
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "image.png"

            def fake_process(arguments, cancellation, timeout, process_memory_limit_bytes):
                self.assertEqual(arguments[-2], "100")
                self.assertEqual(timeout, 1800)
                self.assertEqual(process_memory_limit_bytes, int(1.9 * loa_agent._GIB))
                progress_path = Path(arguments[-1])
                progress_path.write_text("1/100", encoding="ascii")
                time.sleep(0.1)
                progress_path.write_text("2/100", encoding="ascii")
                time.sleep(0.1)
                return subprocess.CompletedProcess(arguments, 0, "", "")

            with (
                patch.object(loa_agent, "_check_local_image_memory"),
                patch.object(loa_agent, "_coreml_sd_runner_path", return_value=Path("/runner")),
                patch.object(loa_agent, "_coreml_sd_model_path", return_value=Path("/model")),
                patch.object(loa_agent, "_run_cancellable_process", side_effect=fake_process),
            ):
                loa_agent._run_coreml_sd_process(
                    loa_config.DEFAULT_COREML_SD_MODEL,
                    "A purple rabbit",
                    str(output_path),
                    "high",
                    None,
                    lambda step, total: steps.append((step, total)),
                )

        self.assertEqual(steps, [(1, 100), (2, 100)])
        self.assertFalse(Path(f"{output_path}.progress").exists())

    def test_coreml_stable_diffusion_readiness_uses_bundled_runner(self):
        config = loa_config.default_config()
        config["image_gen"].update(
            provider="local_coreml_sd",
            model=loa_config.DEFAULT_COREML_SD_MODEL,
        )
        memory = subprocess.CompletedProcess([], 0, stdout=str(16 * 1024**3))
        with (
            patch.object(loa_agent.sys, "platform", "darwin"),
            patch.object(
                loa_agent.os,
                "uname",
                return_value=type("Uname", (), {"machine": "arm64"})(),
            ),
            patch.object(loa_agent.subprocess, "run", return_value=memory),
            patch.object(loa_agent, "_mac_memory_available", return_value=3 * 1024**3),
            patch.object(loa_agent, "_coreml_sd_runner_path") as runner,
        ):
            message = loa_agent.check_image_generation_readiness(config)
        self.assertIn("4.3 GiB", message)
        runner.assert_called_once_with(None)

    def test_local_coreml_sd_defaults_to_apple_model(self):
        settings = loa_storage._normalize_image_gen_config(
            {"provider": "local_coreml_sd"}
        )
        self.assertEqual(settings["model"], loa_config.DEFAULT_COREML_SD_MODEL)
        config = loa_config.default_config()
        config["image_gen"] = settings
        loa_storage._validate_extension_config(config)

    def test_coreml_custom_dimensions_are_limited_to_4096_per_side(self):
        config = loa_config.default_config()
        config["image_gen"].update(
            provider="local_coreml_sd",
            size="4096x4096",
        )
        loa_storage._validate_extension_config(config)

        for size in ("4097x1024", "1024x4097", "0x512", "512x"):
            with self.subTest(size=size):
                config["image_gen"]["size"] = size
                with self.assertRaisesRegex(ValueError, "size"):
                    loa_storage._validate_extension_config(config)

        config["image_gen"].update(provider="openai", size="2048x1024")
        with self.assertRaisesRegex(ValueError, "size"):
            loa_storage._validate_extension_config(config)

    def test_verbose_local_process_does_not_deadlock(self):
        result = loa_agent._run_cancellable_process(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 200000); sys.stderr.write('y' * 200000)"],
            timeout=10,
        )
        self.assertEqual(len(result.stdout), 200000)
        self.assertEqual(len(result.stderr), 200000)

    def test_community_janus_selects_transformers_runner(self):
        config = loa_config.default_config()
        config["image_gen"].update(provider="local_janus", model="deepseek-community/Janus-Pro-1B")

        def fake_runner(arguments, model, cancellation):
            compile(arguments[2], "<community-janus-runner>", "exec")
            self.assertIn("JanusForConditionalGeneration", arguments[2])
            self.assertIn('generation_mode="image"', arguments[2])
            Path(arguments[-3]).write_bytes(b"\x89PNG\r\n\x1a\nimage")
            return subprocess.CompletedProcess(arguments, 0)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(loa_agent, "generated_images_directory_path", return_value=Path(directory)),
            patch.object(loa_agent, "_run_janus_process", side_effect=fake_runner),
        ):
            loa_agent.generate_image(config, "a tree", "1024x1024", "high")

    def test_janus_generation_passes_model_and_cancellation_to_guarded_runner(self):
        config = loa_config.default_config()
        config["image_gen"]["provider"] = "local_janus"
        config["image_gen"]["model"] = "deepseek-ai/Janus-Pro-1B"
        cancellation = loa_agent.AgentCancellation()

        def fake_runner(arguments, model, cancellation):
            Path(arguments[-3]).write_bytes(b"\x89PNG\r\n\x1a\nimage")
            return subprocess.CompletedProcess(arguments, 0)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(loa_agent, "generated_images_directory_path", return_value=Path(directory)),
                patch.object(loa_agent, "_run_janus_process", side_effect=fake_runner) as runner,
            ):
                loa_agent.generate_image(
                    config, "A purple elephant", "1024x1024", "high", cancellation
                )
            self.assertEqual(runner.call_args.kwargs["model"], "deepseek-ai/Janus-Pro-1B")
            self.assertIs(runner.call_args.kwargs["cancellation"], cancellation)

    def test_janus_readiness_uses_cancellable_import_check(self):
        config = loa_config.default_config()
        config["image_gen"]["provider"] = "local_janus"
        config["image_gen"]["model"] = "deepseek-ai/Janus-Pro-1B"
        cancellation = loa_agent.AgentCancellation()
        with (
            patch.object(loa_agent, "_check_janus_memory"),
            patch.object(
                loa_agent,
                "_run_cancellable_process",
                return_value=subprocess.CompletedProcess([], 0),
            ) as runner,
        ):
            result = loa_agent.check_image_generation_readiness(config, cancellation)
        self.assertIn("readiness check passed", result)
        self.assertIs(runner.call_args.kwargs["cancellation"], cancellation)
        self.assertEqual(runner.call_args.kwargs["timeout"], 60)

    def test_janus_runner_uses_image_token_head_and_initializes_model(self):
        stub = r'''
import sys, types, torch
from types import SimpleNamespace

torch.backends.mps.is_available = lambda: False
janus = types.ModuleType("janus")
models = types.ModuleType("janus.models")

class Tokenizer:
    def encode(self, prompt):
        assert prompt.endswith("<image>")
        return [1, 2, 3]

class Processor:
    tokenizer = Tokenizer()
    image_start_tag = "<image>"
    pad_id = 0
    sft_format = "test"

    @classmethod
    def from_pretrained(cls, model):
        return cls()

    def apply_sft_template_for_multi_turn_prompts(self, conversation, **kwargs):
        assert conversation[0]["role"] == "<|User|>"
        assert conversation[1]["role"] == "<|Assistant|>"
        return "prompt"

class Language:
    def get_input_embeddings(self):
        return lambda ids: torch.zeros((*ids.shape, 2))

    def model(self, inputs_embeds, past_key_values, use_cache):
        assert inputs_embeds.shape[0] == 2
        return SimpleNamespace(
            last_hidden_state=torch.zeros((2, 1, 2)),
            past_key_values=1,
        )

class Decoder:
    def decode_code(self, tokens, shape):
        assert tokens.shape == (1, 576)
        assert shape == [1, 8, 24, 24]
        assert (tokens == 0).all()
        return torch.zeros((1, 3, 384, 384))

class Model:
    def __init__(self, config):
        self.language_model = Language()
        self.gen_vision_model = Decoder()

    def get_expanded_tied_weights_keys(self):
        return {}

    def post_init(self):
        self.all_tied_weights_keys = {}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        instance = cls(None)
        assert instance.all_tied_weights_keys == {}
        return instance

    def to(self, **kwargs):
        return self

    def eval(self):
        return self

    def gen_head(self, hidden):
        return torch.tensor([[10.0, -100.0], [10.0, -100.0]])

    def prepare_gen_img_embeds(self, tokens):
        return torch.zeros((2, 2))

models.MultiModalityCausalLM = Model
models.VLChatProcessor = Processor
janus.models = models
sys.modules["janus"] = janus
sys.modules["janus.models"] = models
'''
        config = loa_config.default_config()
        config["image_gen"]["provider"] = "local_janus"
        config["image_gen"]["model"] = "fake-model"
        with tempfile.TemporaryDirectory() as directory:
            def run_fake_janus(arguments, cancellation, timeout, memory_guard=None):
                return subprocess.run(
                    [sys.executable, "-c", stub + arguments[2], *arguments[3:]],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            with (
                patch.object(
                    loa_agent,
                    "generated_images_directory_path",
                    return_value=Path(directory),
                ),
                patch.object(loa_agent, "_run_cancellable_process", side_effect=run_fake_janus),
                patch.object(loa_agent, "_check_janus_memory"),
            ):
                _, ui = loa_agent.generate_image(
                    config, "A purple elephant", "1024x1024", "high"
                )
            from PIL import Image

            with Image.open(Path(directory) / ui["filename"]) as image:
                self.assertEqual(image.size, (1024, 1024))
                self.assertEqual(image.getpixel((0, 0)), (127, 127, 127))

    def test_janus_rejects_7b_on_16gb_mac_before_start(self):
        memory = subprocess.CompletedProcess([], 0, stdout=str(16 * 1024 ** 3))
        with (
            patch.object(loa_agent.sys, "platform", "darwin"),
            patch.object(loa_agent.subprocess, "run", return_value=memory),
            patch.object(loa_agent, "_mac_memory_available") as available,
        ):
            with self.assertRaisesRegex(RuntimeError, "Janus-Pro-1B"):
                loa_agent._check_janus_memory("deepseek-ai/Janus-Pro-7B")
            available.assert_not_called()

    def test_janus_memory_estimate_counts_free_and_reclaimable_pages(self):
        stats = subprocess.CompletedProcess(
            [], 0, stdout=(
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free: 100.\nPages inactive: 80.\nFile-backed pages: 50.\n"
            )
        )
        with patch.object(loa_agent.subprocess, "run", return_value=stats):
            self.assertEqual(loa_agent._mac_memory_available(), 150 * 16384)

    def test_new_janus_config_defaults_to_1b(self):
        settings = loa_storage._normalize_image_gen_config({"provider": "local_janus"})
        self.assertEqual(settings["model"], "deepseek-ai/Janus-Pro-1B")

    def test_janus_rejects_low_available_memory(self):
        memory = subprocess.CompletedProcess([], 0, stdout=str(16 * 1024 ** 3))
        with (
            patch.object(loa_agent.sys, "platform", "darwin"),
            patch.object(loa_agent.subprocess, "run", return_value=memory),
            patch.object(loa_agent, "_mac_memory_available", return_value=4 * 1024 ** 3),
        ):
            with self.assertRaisesRegex(RuntimeError, "Close memory-intensive apps"):
                loa_agent._check_janus_memory("deepseek-ai/Janus-Pro-1B")

    def test_janus_allows_1b_with_sufficient_memory(self):
        memory = subprocess.CompletedProcess([], 0, stdout=str(16 * 1024 ** 3))
        with (
            patch.object(loa_agent.sys, "platform", "darwin"),
            patch.object(loa_agent.subprocess, "run", return_value=memory),
            patch.object(loa_agent, "_mac_memory_available", return_value=9 * 1024 ** 3),
        ):
            loa_agent._check_janus_memory("deepseek-ai/Janus-Pro-1B")

    def test_total_memory_check_uses_absolute_sysctl_path(self):
        memory = subprocess.CompletedProcess([], 0, stdout=str(16 * 1024 ** 3))
        with (
            patch.object(loa_agent.sys, "platform", "darwin"),
            patch.object(loa_agent.subprocess, "run", return_value=memory) as run,
            patch.object(loa_agent, "_mac_memory_available", return_value=9 * 1024 ** 3),
        ):
            loa_agent._check_local_image_memory(
                "test-model", 8 * 1024 ** 3, 2 * 1024 ** 3
            )
        self.assertEqual(run.call_args.args[0][0], "/usr/sbin/sysctl")

    def test_sana_memory_limit_applies_to_readiness_and_generation(self):
        config = loa_config.default_config()
        config["image_gen"].update(provider="local_sana", model=loa_config.DEFAULT_SANA_MODEL)
        memory = subprocess.CompletedProcess([], 0, stdout=str(16 * 1024 ** 3))
        with (
            patch.object(loa_agent.sys, "platform", "darwin"),
            patch.object(loa_agent.os, "uname", return_value=type("Uname", (), {"machine": "arm64"})()),
            patch.object(loa_agent.subprocess, "run", return_value=memory),
            patch.object(loa_agent, "_mac_memory_available", return_value=int(5.2 * 1024 ** 3)) as available,
            patch.object(loa_agent, "_run_cancellable_process", return_value=subprocess.CompletedProcess([], 0)),
        ):
            loa_agent.check_image_generation_readiness(config)
            loa_agent._run_local_image_process([], loa_config.DEFAULT_SANA_MODEL, None)
            available.return_value = 2 * 1024 ** 3 - 1
            with self.assertRaisesRegex(RuntimeError, "requires 2 GiB"):
                loa_agent.check_image_generation_readiness(config)
            with self.assertRaisesRegex(RuntimeError, "requires 2 GiB"):
                loa_agent._run_local_image_process([], loa_config.DEFAULT_SANA_MODEL, None)

    def test_flux_memory_limit_applies_to_readiness_and_generation(self):
        config = loa_config.default_config()
        config["image_gen"].update(provider="local_flux", model=loa_config.DEFAULT_FLUX_MODEL)
        memory = subprocess.CompletedProcess([], 0, stdout=str(16 * 1024 ** 3))
        with (
            patch.object(loa_agent.sys, "platform", "darwin"),
            patch.object(loa_agent.os, "uname", return_value=type("Uname", (), {"machine": "arm64"})()),
            patch.object(loa_agent.subprocess, "run", return_value=memory),
            patch.object(loa_agent, "_mac_memory_available", return_value=3 * 1024 ** 3) as available,
            patch.object(loa_agent, "_run_cancellable_process", return_value=subprocess.CompletedProcess([], 0)),
        ):
            loa_agent.check_image_generation_readiness(config)
            loa_agent._run_local_image_process([], loa_config.DEFAULT_FLUX_MODEL, None)
            available.return_value = 3 * 1024 ** 3 - 1
            with self.assertRaisesRegex(RuntimeError, "requires 3 GiB"):
                loa_agent.check_image_generation_readiness(config)
            with self.assertRaisesRegex(RuntimeError, "requires 3 GiB"):
                loa_agent._run_local_image_process([], loa_config.DEFAULT_FLUX_MODEL, None)

    def test_memory_guard_stops_running_process(self):
        def guard():
            raise RuntimeError("memory pressure")

        with self.assertRaisesRegex(RuntimeError, "memory pressure"):
            loa_agent._run_cancellable_process(
                [sys.executable, "-c", "import time; time.sleep(20)"],
                memory_guard=guard,
            )

    def test_process_memory_limit_stops_process_over_limit(self):
        with (
            patch.object(loa_agent, "_process_rss_bytes", return_value=2),
            self.assertRaisesRegex(RuntimeError, "exceeded the 0.0 GiB limit"),
        ):
            loa_agent._run_cancellable_process(
                [sys.executable, "-c", "import time; time.sleep(20)"],
                process_memory_limit_bytes=1,
            )

    def test_janus_rejects_concurrent_generation(self):
        loa_agent._LOCAL_IMAGE_GENERATION_LOCK.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                loa_agent._run_janus_process([], "deepseek-ai/Janus-Pro-1B", None)
        finally:
            loa_agent._LOCAL_IMAGE_GENERATION_LOCK.release()

    def test_janus_runner_honors_explicit_venv_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            interpreter = Path(directory) / "bin" / "python"
            interpreter.parent.mkdir()
            interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            interpreter.chmod(0o700)
            with patch.dict(
                os.environ,
                {"LOA_JANUS_PYTHON": str(interpreter), "VIRTUAL_ENV": ""},
            ):
                self.assertEqual(
                    loa_agent._janus_python_executable(),
                    str(interpreter),
                )

    def test_janus_runner_uses_project_venv_automatically(self):
        project_venv_python = (
            Path(loa_agent.__file__).resolve().parent / ".venv" / "bin" / "python"
        )
        if not project_venv_python.is_file():
            self.skipTest("Project virtual environment is not present.")
        with patch.dict(
            os.environ,
            {"LOA_JANUS_PYTHON": "", "VIRTUAL_ENV": ""},
        ):
            self.assertEqual(
                loa_agent._janus_python_executable(),
                str(project_venv_python),
            )

    def test_image_tool_is_available_only_when_enabled(self):
        original_tools = loa_agent.TOOLS
        original_supported = loa_agent.SUPPORTED_TOOL_NAMES
        config = loa_config.default_config()
        try:
            with (
                patch.object(loa_agent.mcp_manager, "ensure_configured"),
                patch.object(loa_agent.mcp_manager, "tool_schemas", return_value=[]),
                patch("loa_agent.load_python_skill_tools", return_value=([], {})),
            ):
                config["image_gen"]["enabled"] = False
                loa_agent.configure_runtime_tools(config)
                self.assertNotIn("generate_image", loa_agent.SUPPORTED_TOOL_NAMES)
                self.assertNotIn(
                    "generate_image",
                    {item["function"]["name"] for item in loa_agent.TOOLS},
                )

                config["image_gen"]["enabled"] = True
                loa_agent.configure_runtime_tools(config)
                self.assertIn("generate_image", loa_agent.SUPPORTED_TOOL_NAMES)
                self.assertIn(
                    "generate_image",
                    {item["function"]["name"] for item in loa_agent.TOOLS},
                )
                size_parameter = next(
                    item["function"]["parameters"]["properties"]["size"]
                    for item in loa_agent.TOOLS
                    if item["function"]["name"] == "generate_image"
                )
                self.assertEqual(
                    size_parameter["enum"],
                    ["1024x1024", "1536x1024", "1024x1536"],
                )
        finally:
            loa_agent.TOOLS = original_tools
            loa_agent.SUPPORTED_TOOL_NAMES = original_supported

    def test_coreml_image_tool_requests_english_prompt(self):
        original_tools = loa_agent.TOOLS
        original_supported = loa_agent.SUPPORTED_TOOL_NAMES
        config = loa_config.default_config()
        config["image_gen"]["provider"] = "local_coreml_sd"
        try:
            with (
                patch.object(loa_agent.mcp_manager, "ensure_configured"),
                patch.object(loa_agent.mcp_manager, "tool_schemas", return_value=[]),
                patch("loa_agent.load_python_skill_tools", return_value=([], {})),
            ):
                loa_agent.configure_runtime_tools(config)
                tool = next(
                    item for item in loa_agent.TOOLS
                    if item["function"]["name"] == "generate_image"
                )
                self.assertIn("in English", tool["function"]["description"])
                self.assertIn(
                    "write the prompt in English",
                    tool["function"]["parameters"]["properties"]["prompt"]["description"],
                )
                size_parameter = tool["function"]["parameters"]["properties"]["size"]
                self.assertNotIn("enum", size_parameter)
                self.assertIn("4096", size_parameter["description"])
                self.assertNotIn(
                    "in English",
                    next(
                        item["function"]["description"]
                        for item in loa_agent.BASE_TOOLS
                        if item["function"]["name"] == "generate_image"
                    ),
                )
        finally:
            loa_agent.TOOLS = original_tools
            loa_agent.SUPPORTED_TOOL_NAMES = original_supported

    def test_runtime_tools_can_refresh_without_reconnecting_mcp(self):
        config = loa_config.default_config()
        config["image_gen"]["enabled"] = False
        original_tools = loa_agent.TOOLS
        original_supported = loa_agent.SUPPORTED_TOOL_NAMES
        original_mcp_names = loa_agent.MCP_TOOL_NAMES
        original_functions = loa_agent.TOOL_FUNCTIONS
        try:
            with (
                patch.object(loa_agent.mcp_manager, "ensure_configured") as ensure,
                patch.object(loa_agent.mcp_manager, "reload") as reload,
                patch.object(loa_agent.mcp_manager, "tool_schemas", return_value=[]),
                patch.object(loa_agent.mcp_manager, "statuses", return_value=[]),
                patch("loa_agent.load_python_skill_tools", return_value=([], {})),
            ):
                loa_agent.configure_runtime_tools(config, reuse_mcp_connections=True)
            ensure.assert_not_called()
            reload.assert_not_called()
            self.assertNotIn("generate_image", loa_agent.SUPPORTED_TOOL_NAMES)
        finally:
            loa_agent.TOOLS = original_tools
            loa_agent.SUPPORTED_TOOL_NAMES = original_supported
            loa_agent.MCP_TOOL_NAMES = original_mcp_names
            loa_agent.TOOL_FUNCTIONS = original_functions

    def test_openai_image_is_saved_with_private_permissions_and_ui_metadata(self):
        png = b"\x89PNG\r\n\x1a\nimage"
        config = loa_config.default_config()
        config["image_gen"]["api_key"] = "test-key"

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    loa_agent,
                    "generated_images_directory_path",
                    return_value=Path(directory),
                ),
                patch.object(
                    loa_agent,
                    "_image_generation_http_json",
                    return_value={
                        "data": [{"b64_json": base64.b64encode(png).decode("ascii")}]
                    },
                ) as request,
            ):
                output, ui = loa_agent.generate_image(
                    config,
                    "A quiet mountain lake at dawn",
                    "1024x1024",
                    "high",
                )

            image_path = Path(directory) / ui["filename"]
            self.assertEqual(image_path.read_bytes(), png)
            self.assertEqual(stat.S_IMODE(image_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(Path(directory).stat().st_mode), 0o700)
            self.assertEqual(ui["type"], "image")
            self.assertEqual(ui["image_url"], f"/api/images/{ui['filename']}")
            self.assertEqual(ui["prompt"], "A quiet mountain lake at dawn")
            self.assertIn(str(image_path), output)
            request.assert_called_once_with(
                "POST",
                "https://api.openai.com/v1/images/generations",
                "openai",
                "test-key",
                None,
                {
                    "model": "gpt-image-1",
                    "prompt": "A quiet mountain lake at dawn",
                    "size": "1024x1024",
                    "quality": "high",
                },
            )

    def test_gemini_jpeg_response_is_normalized_to_private_png(self):
        from PIL import Image

        jpeg = io.BytesIO()
        Image.new("RGB", (24, 12), (200, 80, 20)).save(jpeg, format="JPEG")
        config = loa_config.default_config()
        config["image_gen"].update(
            provider="gemini",
            model="gemini-2.5-flash-image",
            api_key="test-key",
        )
        response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(jpeg.getvalue()).decode("ascii"),
                        }
                    }]
                }
            }]
        }

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    loa_agent,
                    "generated_images_directory_path",
                    return_value=Path(directory),
                ),
                patch.object(
                    loa_agent,
                    "_image_generation_http_json",
                    return_value=response,
                ) as request,
            ):
                loa_agent.generate_image(
                    config,
                    "A quiet mountain lake at dawn",
                    "1536x1024",
                    "medium",
                )

            image_path = next(Path(directory).glob("*.png"))
            self.assertEqual(stat.S_IMODE(image_path.stat().st_mode), 0o600)
            with Image.open(image_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (24, 12))

        self.assertEqual(
            request.call_args.args[:5],
            (
                "POST",
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash-image:generateContent",
                "gemini",
                "test-key",
                None,
            ),
        )
        payload = request.call_args.args[5]
        self.assertEqual(
            payload["generationConfig"]["imageConfig"]["aspectRatio"],
            "3:2",
        )

    def test_tool_result_keeps_image_metadata_in_chat_history(self):
        image_ui = {
            "type": "image",
            "filename": "a" * 32 + ".png",
            "image_url": "/api/images/" + "a" * 32 + ".png",
            "prompt": "A red fox in a snowy forest",
            "provider": "openai",
            "summary": "Generated image",
        }
        config = loa_config.default_config()
        tool_response = {
            "tool_calls": [
                {
                    "function": {
                        "name": "generate_image",
                        "arguments": {
                            "prompt": "A red fox in a snowy forest",
                            "size": "square",
                        },
                    }
                }
            ]
        }
        with (
            patch("loa_agent.configure_runtime_tools"),
            patch("loa_agent.ollama.Client", return_value=Mock()),
            patch(
                "loa_agent._request_assistant_message",
                side_effect=[(tool_response, True), ({"content": "Done."}, True)],
            ),
            patch(
                "loa_agent.generate_image",
                return_value=("Generated image saved.", image_ui),
            ) as generate,
        ):
            reply = loa_agent.get_assistant_reply(
                config,
                [{"role": "user", "content": "Make a fox image."}],
            )

        generate.assert_called_once_with(
            config,
            "A red fox in a snowy forest",
            "1024x1024",
            "high",
            None,
        )
        image_result = next(
            message
            for message in reply["history"]
            if message.get("tool_name") == "generate_image"
        )
        self.assertEqual(image_result["ui"], image_ui)
        self.assertEqual(image_result["role"], "tool")

    def test_image_tool_uses_custom_coreml_default_size(self):
        config = loa_config.default_config()
        config["image_gen"].update(
            provider="local_coreml_sd",
            size="2048x1024",
        )
        tool_response = {
            "tool_calls": [
                {
                    "function": {
                        "name": "generate_image",
                        "arguments": {"prompt": "A test image"},
                    }
                }
            ]
        }
        with (
            patch("loa_agent.configure_runtime_tools"),
            patch("loa_agent.ollama.Client", return_value=Mock()),
            patch(
                "loa_agent._request_assistant_message",
                side_effect=[(tool_response, True), ({"content": "Done."}, True)],
            ),
            patch(
                "loa_agent.generate_image",
                return_value=(
                    "Generated image saved.",
                    {
                        "type": "image",
                        "provider": "local_coreml_sd",
                        "filename": "b" * 32 + ".png",
                    },
                ),
            ) as generate,
        ):
            loa_agent.get_assistant_reply(
                config,
                [{"role": "user", "content": "Make an image."}],
            )

        generate.assert_called_once_with(
            config,
            "A test image",
            "2048x1024",
            "high",
            None,
        )

    def test_image_generation_configuration_is_validated(self):
        config = loa_config.default_config()
        config["image_gen"]["provider"] = "diffusion"
        with self.assertRaisesRegex(ValueError, "provider"):
            loa_storage.save_config(config)


if __name__ == "__main__":
    unittest.main()
