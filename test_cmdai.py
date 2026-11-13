import io
import json
import os
import struct
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from urllib.request import Request
from pathlib import Path

import cmdai
import cmdai_update
import install


class ChatCompletionTests(unittest.TestCase):
    def setUp(self):
        self.check_internet = cmdai.internet_available
        self.test_conversation_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_conversation_directory.cleanup)
        original_conversations_directory = cmdai.conversations_directory

        def conversations_directory_for_test():
            if os.environ.get("CMD_AI_CONVERSATIONS_DIR"):
                return original_conversations_directory()
            return self.test_conversation_directory.name

        conversation_directory_patch = patch(
            "cmdai.conversations_directory", side_effect=conversations_directory_for_test
        )
        conversation_directory_patch.start()
        self.addCleanup(conversation_directory_patch.stop)
        internet_check = patch("cmdai.internet_available", return_value=False)
        internet_check.start()
        self.addCleanup(internet_check.stop)

    def test_repository_icon_contains_an_extractable_png(self):
        icon_path = Path(install.ROOT) / "cmdAI.ico"
        icon_data = icon_path.read_bytes()
        png_data = install.png_data_from_icon(icon_path)
        with tempfile.TemporaryDirectory() as temporary_directory:
            logo_path = Path(temporary_directory) / "logo.png"
            install.write_logo_png(logo_path)
            extracted_logo = logo_path.read_bytes()

        self.assertEqual(icon_data[:6], struct.pack("<HHH", 0, 1, 1))
        self.assertEqual(icon_data[22:30], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(png_data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png_data[16:24]), (128, 128))
        self.assertEqual(extracted_logo, png_data)

    def test_windows_launcher_references_repository_icon(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            desktop = Path(temporary_directory)
            with patch("install.subprocess.run") as run:
                shortcut = install.install_windows(desktop)
            command = run.call_args.args[0][-1]

        self.assertEqual(shortcut.name, ".cmdAI.lnk")
        self.assertIn(str(Path(install.ROOT) / "cmdAI.ico"), command)

    def test_installed_icon_is_found_in_data_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "site-packages"
            data_directory = Path(temporary_directory) / "environment"
            packaged_icon = data_directory / "share" / "cmdai" / "cmdAI.ico"
            packaged_icon.parent.mkdir(parents=True)
            packaged_icon.write_bytes(b"icon")
            with patch.object(install, "ROOT", root):
                with patch("install.sysconfig.get_path", return_value=str(data_directory)):
                    self.assertEqual(install.icon_path(), packaged_icon)

    def test_linux_desktop_launcher_references_generated_logo(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "repository"
            desktop = Path(temporary_directory) / "Desktop"
            root.mkdir()
            desktop.mkdir()
            icon_path = Path(install.ROOT) / "cmdAI.ico"
            (root / "cmdAI.ico").write_bytes(icon_path.read_bytes())
            with patch.object(install, "ROOT", root):
                with patch.object(install, "SCRIPT", root / "cmdai.py"):
                    with patch.object(install.sys, "platform", "linux"):
                        shortcut = install.install_unix(desktop)

            contents = shortcut.read_text(encoding="utf-8")
            logo_path = root / "cmdAI.png"
            logo_data = logo_path.read_bytes()
            expected_logo = install.png_data_from_icon(root / "cmdAI.ico")

        escaped_logo_path = str(logo_path).replace("\\", "\\\\").replace('"', '\\"')
        self.assertIn('Icon="{}"'.format(escaped_logo_path), contents)
        self.assertEqual(logo_data, expected_logo)

    def test_ollama_posts_without_api_key_and_returns_assistant_text(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"message":{"content":"Hello from Ollama"}}'

        with patch("cmdai.urllib.request.urlopen", return_value=Response()) as urlopen:
            answer = cmdai.ollama_chat_completion(
                [{"role": "user", "content": "Hi"}], "http://localhost:11434", "test-model"
            )

        self.assertEqual(answer, "Hello from Ollama")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(json.loads(request.data), {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })

    def test_model_recommendation_uses_ram_and_nvidia_memory(self):
        self.assertEqual(cmdai.recommend_model(2), (
            "llama3.2:1b", "the smallest available option for a low-memory system"
        ))
        self.assertEqual(cmdai.recommend_model(8), ("llama3.2:3b", "your system memory"))
        self.assertEqual(cmdai.recommend_model(16), ("llama3.1:8b", "your system memory"))
        self.assertEqual(cmdai.recommend_model(8, 24), (
            "qwen2.5:32b", "your detected NVIDIA GPU memory"
        ))

    def test_setup_wizard_downloads_only_after_confirmation(self):
        output = io.StringIO()
        with patch("cmdai.system_memory_gb", return_value=16):
            with patch("cmdai.nvidia_memory_gb", return_value=None):
                with patch("cmdai.shutil.which", return_value="ollama"):
                    with patch("cmdai.subprocess.run", return_value=type("Result", (), {"returncode": 0})()) as run:
                        with redirect_stdout(output):
                            self.assertEqual(cmdai.run_setup_wizard(lambda prompt: "y"), 0)

        run.assert_called_once_with(["ollama", "pull", "llama3.1:8b"], check=False)
        self.assertIn("Recommended model: llama3.1:8b", output.getvalue())
        self.assertIn("Download complete.", output.getvalue())

    def test_setup_wizard_does_not_download_without_confirmation(self):
        output = io.StringIO()
        with patch("cmdai.system_memory_gb", return_value=16):
            with patch("cmdai.nvidia_memory_gb", return_value=None):
                with patch("cmdai.shutil.which", return_value="ollama"):
                    with patch("cmdai.subprocess.run") as run:
                        with redirect_stdout(output):
                            self.assertEqual(cmdai.run_setup_wizard(lambda prompt: "n"), 0)

        run.assert_not_called()
        self.assertIn("No model downloaded.", output.getvalue())

    def test_setup_wizard_without_ollama_only_suggests_model(self):
        output = io.StringIO()
        with patch("cmdai.system_memory_gb", return_value=8):
            with patch("cmdai.nvidia_memory_gb", return_value=None):
                with patch("cmdai.shutil.which", return_value=None):
                    with patch("cmdai.subprocess.run") as run:
                        with redirect_stdout(output):
                            self.assertEqual(cmdai.run_setup_wizard(), 0)

        run.assert_not_called()
        self.assertIn("Recommended model: llama3.2:3b", output.getvalue())
        self.assertIn("Install Ollama", output.getvalue())

    def test_setup_flag_runs_before_openai_api_key_prompt(self):
        with patch("cmdai.run_setup_wizard", return_value=0) as setup:
            with patch("cmdai.getpass.getpass") as getpass_prompt:
                self.assertEqual(cmdai.main(["--provider", "openai", "--setup"]), 0)

        setup.assert_called_once_with()
        getpass_prompt.assert_not_called()

    def test_ollama_model_list_returns_names_and_sizes(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"models":[{"name":"llama3.2:3b","size":3221225472}]}'

        with patch("cmdai.urllib.request.urlopen", return_value=Response()) as urlopen:
            models = cmdai.ollama_list_models("http://localhost:11434/")

        self.assertEqual(models, [("llama3.2:3b", 3221225472)])
        self.assertEqual(urlopen.call_args.args[0].full_url, "http://localhost:11434/api/tags")

    def test_ollama_model_pull_reports_streamed_status_and_success(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter([
                    b'{"status":"pulling model"}\n',
                    b'{"status":"success"}\n',
                ])

        updates = []
        with patch("cmdai.urllib.request.urlopen", return_value=Response()) as urlopen:
            cmdai.ollama_pull_model("llama3.2:3b", "http://localhost:11434", updates.append)

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/pull")
        self.assertEqual(json.loads(request.data), {"name": "llama3.2:3b", "stream": True})
        self.assertEqual([update["status"] for update in updates], ["pulling model", "success"])

    def test_ollama_model_pull_rejects_incomplete_response(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter([b'{"status":"pulling model"}\n'])

        with patch("cmdai.urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "without confirming success"):
                cmdai.ollama_pull_model("llama3.2:3b", "http://localhost:11434")

    def test_ollama_model_commands_list_switch_and_confirm_download(self):
        output = io.StringIO()
        prompts = [
            "/models",
            "/model llama3.2:3b",
            "/pull qwen2.5:7b",
            "n",
            "/pull qwen2.5:7b",
            "y",
            "Hi",
            "/exit",
        ]
        with patch("builtins.input", side_effect=prompts):
            with patch(
                "cmdai.ollama_list_models", return_value=[("llama3.1:8b", 8 * 1024 ** 3)]
            ):
                with patch("cmdai.ollama_pull_model") as pull_model:
                    with patch("cmdai.ollama_chat_completion", return_value="Hello") as completion:
                        with redirect_stdout(output):
                            self.assertEqual(cmdai.main(["--compact"]), 0)

        pull_model.assert_called_once()
        self.assertEqual(pull_model.call_args.args[:2], ("qwen2.5:7b", cmdai.DEFAULT_OLLAMA_BASE_URL))
        self.assertEqual(completion.call_args.args[2], "llama3.2:3b")
        self.assertIn("llama3.1:8b (current) - 8.0 GiB", output.getvalue())
        self.assertIn("Model download cancelled.", output.getvalue())
        self.assertIn("Model 'qwen2.5:7b' is ready.", output.getvalue())

    def test_posts_messages_and_returns_assistant_text(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"Hello"}}]}'

        with patch("cmdai.urllib.request.urlopen", return_value=Response()) as urlopen:
            answer = cmdai.chat_completion(
                [{"role": "user", "content": "Hi"}], "secret", "https://example.test/v1/", "test-model"
            )

        self.assertEqual(answer, "Hello")
        request = urlopen.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(json.loads(request.data), {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })

    def test_default_ollama_start_does_not_prompt_for_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.input", side_effect=["Hi", "/exit"]):
                with patch("cmdai.getpass.getpass") as getpass_prompt:
                    with patch("cmdai.ollama_chat_completion", return_value="Hello") as completion:
                        self.assertEqual(cmdai.main([]), 0)

        getpass_prompt.assert_not_called()
        self.assertEqual(completion.call_args.args[2], "llama3.1:8b")

    def test_completed_conversation_is_saved_and_listed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = io.StringIO()
            prompts = ["Hi", "/history", "/exit"]
            with patch.dict(os.environ, {"CMD_AI_CONVERSATIONS_DIR": temporary_directory}, clear=True):
                with patch("builtins.input", side_effect=prompts):
                    with patch("cmdai.ollama_chat_completion", return_value="Hello"):
                        with redirect_stdout(output):
                            self.assertEqual(cmdai.main(["--compact"]), 0)

            paths = cmdai.list_conversations(temporary_directory)
            self.assertEqual(len(paths), 1)
            saved_conversation = json.loads(Path(paths[0]).read_text(encoding="utf-8"))

        self.assertEqual(
            [(message["role"], message["content"]) for message in saved_conversation["messages"]],
            [("user", "Hi"), ("assistant", "Hello")],
        )
        self.assertNotIn("system", [message["role"] for message in saved_conversation["messages"]])
        self.assertIn("Conversation folder: {}".format(temporary_directory), output.getvalue())
        self.assertIn(Path(paths[0]).name, output.getvalue())

    def test_clear_starts_a_new_saved_conversation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            prompts = ["First", "/clear", "Second", "/exit"]
            with patch.dict(os.environ, {"CMD_AI_CONVERSATIONS_DIR": temporary_directory}, clear=True):
                with patch("builtins.input", side_effect=prompts):
                    with patch("cmdai.ollama_chat_completion", side_effect=["One", "Two"]):
                        self.assertEqual(cmdai.main(["--compact"]), 0)

            saved_conversations = [
                json.loads(Path(path).read_text(encoding="utf-8"))
                for path in cmdai.list_conversations(temporary_directory)
            ]

        self.assertEqual(len(saved_conversations), 2)
        actual_conversations = [
            [message["content"] for message in conversation["messages"]]
            for conversation in saved_conversations
        ]
        self.assertEqual(
            sorted(actual_conversations),
            sorted([["First", "One"], ["Second", "Two"]]),
        )

    def test_update_command_runs_pipx_upgrade(self):
        with patch("cmdai_update.shutil.which", return_value="pipx"):
            with patch("cmdai_update.subprocess.run", return_value=type("Result", (), {"returncode": 0})()) as run:
                self.assertEqual(cmdai_update.main(), 0)

        run.assert_called_once_with(["pipx", "upgrade", "cmdai-assistant"], check=False)

    def test_update_command_reports_missing_pipx(self):
        error_output = io.StringIO()
        with patch("cmdai_update.shutil.which", return_value=None):
            with redirect_stderr(error_output):
                self.assertEqual(cmdai_update.main(), 1)

        self.assertIn("pipx was not found on PATH", error_output.getvalue())

    def test_chat_update_command_invokes_updater(self):
        with patch("builtins.input", side_effect=["/update", "/exit"]):
            with patch("cmdai_update.main", return_value=0) as update:
                self.assertEqual(cmdai.main(["--compact"]), 0)

        update.assert_called_once_with()

    def test_response_status_reflects_internet_availability(self):
        for internet_connected, status in (
            (True, "internet connection detected"),
            (False, "offline; internet connection not detected"),
        ):
            with self.subTest(internet_connected=internet_connected):
                output = io.StringIO()
                with patch("cmdai.internet_available", return_value=internet_connected):
                    with patch("builtins.input", side_effect=["Hi", "/exit"]):
                        with patch("cmdai.ollama_chat_completion", return_value="Hello"):
                            with redirect_stdout(output):
                                self.assertEqual(cmdai.main(["--compact"]), 0)

                text = output.getvalue()
                self.assertIn(".cmdAI> Generating with llama3.1:8b ({})".format(status), text)
                answer_position = text.index(".cmdAI> Hello")
                self.assertLess(text.index(".cmdAI> Generating"), answer_position)

    def test_progress_indicator_updates_until_generation_finishes(self):
        output = io.StringIO()

        def delayed_answer(*args):
            time.sleep(0.25)
            return "Hello"

        with patch("builtins.input", side_effect=["Hi", "/exit"]):
            with patch("cmdai.ollama_chat_completion", side_effect=delayed_answer):
                with redirect_stdout(output):
                    self.assertEqual(cmdai.main(["--compact"]), 0)

        self.assertGreaterEqual(output.getvalue().count("Generating with llama3.1:8b"), 2)

    def test_internet_check_handles_unreachable_network(self):
        with patch("cmdai.socket.create_connection", side_effect=OSError("unreachable")) as connect:
            self.assertFalse(self.check_internet())

        connect.assert_called_once_with(("1.1.1.1", 443), timeout=1)

    def test_internet_check_closes_connection(self):
        connection = io.StringIO()
        with patch("cmdai.socket.create_connection", return_value=connection) as connect:
            self.assertTrue(self.check_internet())

        connect.assert_called_once_with(("1.1.1.1", 443), timeout=1)
        self.assertTrue(connection.closed)

    def test_terminal_commands_customize_chat_without_being_sent_as_prompts(self):
        prompts = [
            "/dev on",
            "/model custom-model",
            "/system Be concise.",
            "First question",
            "/clear",
            "/spacing off",
            "Second question",
            "/exit",
        ]
        with patch.dict(os.environ, {"CMD_AI_DEV_PASSWORD": "test-password"}, clear=True):
            with patch("builtins.input", side_effect=prompts):
                with patch("cmdai.getpass.getpass", return_value="test-password"):
                    with patch("cmdai.ollama_chat_completion", return_value="Answer") as completion:
                        self.assertEqual(cmdai.main([]), 0)

        self.assertEqual(completion.call_count, 2)
        first_messages = completion.call_args_list[0].args[0]
        self.assertEqual(completion.call_args_list[0].args[2], "custom-model")
        self.assertEqual(first_messages[0]["content"], "Be concise.")
        second_messages = completion.call_args_list[1].args[0]
        self.assertEqual([message["content"] for message in second_messages], [
            "Be concise.", "Second question", "Answer"
        ])

    def test_compact_flag_removes_extra_reply_spacing(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.input", side_effect=["Hi", "/exit"]) as user_input:
                with patch("cmdai.ollama_chat_completion", return_value="Hello"):
                    with redirect_stdout(output):
                        self.assertEqual(cmdai.main(["--compact"]), 0)

        self.assertIn(".cmdAI> Hello\n", output.getvalue())
        self.assertNotIn("\n\n.cmdAI> Hello", output.getvalue())
        self.assertEqual(user_input.call_args_list[0].args, ("you> ",))

    def test_default_spacing_adds_a_blank_line_before_each_prompt(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.input", side_effect=["Hi", "/exit"]) as user_input:
                with patch("cmdai.ollama_chat_completion", return_value="Hello"):
                    self.assertEqual(cmdai.main([]), 0)

        self.assertEqual(user_input.call_args_list[0].args, ("\nyou> ",))
        self.assertEqual(user_input.call_args_list[1].args, ("\nyou> ",))

    def test_trainmodel_saves_examples_and_uses_matching_example(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            training_path = Path(temporary_directory) / "examples.json"
            prompts = [
                "/dev on",
                "/trainmodel",
                "How should Python errors be explained?",
                "Explain the cause first, then show a small fix.",
                "How do I explain Python errors?",
                "/exit",
            ]
            with patch.dict(os.environ, {
                "CMD_AI_TRAINING_FILE": str(training_path),
                "CMD_AI_DEV_PASSWORD": "test-password",
            }, clear=True):
                with patch("builtins.input", side_effect=prompts):
                    with patch("cmdai.getpass.getpass", return_value="test-password"):
                        with patch("cmdai.ollama_chat_completion", return_value="Answer") as completion:
                            self.assertEqual(cmdai.main([]), 0)

            saved_examples = json.loads(training_path.read_text(encoding="utf-8"))

        self.assertEqual(saved_examples, [{
            "prompt": "How should Python errors be explained?",
            "response": "Explain the cause first, then show a small fix.",
        }])
        request_messages = completion.call_args.args[0]
        self.assertEqual(request_messages[1]["role"], "system")
        self.assertIn("Explain the cause first, then show a small fix.", request_messages[1]["content"])

    def test_irrelevant_training_examples_are_not_included(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "How do I bake bread?"},
        ]
        examples = [{"prompt": "Debug Python errors", "response": "Check the traceback."}]

        self.assertIs(cmdai.messages_with_training_examples(messages, examples), messages)

    def test_developer_tools_reject_incorrect_password(self):
        output = io.StringIO()
        prompts = ["/dev on", "/trainmodel", "/exit"]
        with patch.dict(os.environ, {"CMD_AI_DEV_PASSWORD": "correct"}, clear=True):
            with patch("builtins.input", side_effect=prompts):
                with patch("cmdai.getpass.getpass", return_value="incorrect"):
                    with patch("cmdai.ollama_chat_completion") as completion:
                        with redirect_stdout(output):
                            self.assertEqual(cmdai.main([]), 0)

        completion.assert_not_called()
        self.assertIn("Incorrect password.", output.getvalue())
        self.assertIn("Developer mode is off", output.getvalue())

    def test_developer_mode_unlocks_diagnostics_and_export(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            export_path = Path(temporary_directory) / "session.json"
            prompts = [
                "/dev on",
                "/debug",
                "/context",
                "/model dev-model",
                "A test question",
                "/export {}".format(export_path),
                "/dev off",
                "/model blocked-model",
                "/exit",
            ]
            output = io.StringIO()
            with patch.dict(os.environ, {"CMD_AI_DEV_PASSWORD": "test-password"}, clear=True):
                with patch("builtins.input", side_effect=prompts):
                    with patch("cmdai.getpass.getpass", return_value="test-password"):
                        with patch("cmdai.ollama_chat_completion", return_value="Answer") as completion:
                            with redirect_stdout(output):
                                self.assertEqual(cmdai.main([]), 0)

            exported = json.loads(export_path.read_text(encoding="utf-8"))

        self.assertEqual(completion.call_args.args[2], "dev-model")
        self.assertEqual(exported["model"], "dev-model")
        self.assertEqual([message["role"] for message in exported["messages"]], [
            "system", "user", "assistant"
        ])
        self.assertIn("Developer mode enabled.", output.getvalue())
        self.assertIn("Conversation context:", output.getvalue())
        self.assertIn("Developer mode is off.", output.getvalue())
        self.assertIn("Model: blocked-model", output.getvalue())


if __name__ == "__main__":
    unittest.main()