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

import nexus
import nexus_update
import nexus_install as install


class ConversationDirectoryTests(unittest.TestCase):
    def test_existing_conversations_stay_on_legacy_directory_until_new_one_exists(self):
        with tempfile.TemporaryDirectory() as home:
            legacy_directory = Path(home) / ".cmdai" / "conversations"
            legacy_directory.mkdir(parents=True)
            new_directory = Path(home) / ".nexus" / "conversations"

            with patch.dict(os.environ, {}, clear=True):
                with patch("nexus.os.path.expanduser", return_value=home):
                    self.assertEqual(nexus.conversations_directory(), str(legacy_directory))
                    new_directory.mkdir(parents=True)
                    self.assertEqual(nexus.conversations_directory(), str(new_directory))


class ChatCompletionTests(unittest.TestCase):
    def setUp(self):
        self.check_internet = nexus.internet_available
        self.test_conversation_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_conversation_directory.cleanup)
        original_conversations_directory = nexus.conversations_directory

        def conversations_directory_for_test():
            if os.environ.get("NEXUS_CONVERSATIONS_DIR"):
                return original_conversations_directory()
            return self.test_conversation_directory.name

        conversation_directory_patch = patch(
            "nexus.conversations_directory", side_effect=conversations_directory_for_test
        )
        conversation_directory_patch.start()
        self.addCleanup(conversation_directory_patch.stop)
        internet_check = patch("nexus.internet_available", return_value=False)
        internet_check.start()
        self.addCleanup(internet_check.stop)

    def test_repository_icon_contains_an_extractable_png(self):
        icon_path = Path(install.ROOT) / "nexus.ico"
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
            with patch("nexus_install.subprocess.run") as run:
                shortcut = install.install_windows(desktop)
            command = run.call_args.args[0][-1]

        self.assertEqual(shortcut.name, "Nexus Local AI Host.lnk")
        self.assertIn(str(Path(install.ROOT) / "nexus.ico"), command)

    def test_installed_icon_is_found_in_data_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "site-packages"
            data_directory = Path(temporary_directory) / "environment"
            packaged_icon = data_directory / "share" / "nexus" / "nexus.ico"
            packaged_icon.parent.mkdir(parents=True)
            packaged_icon.write_bytes(b"icon")
            with patch.object(install, "ROOT", root):
                with patch("nexus_install.sysconfig.get_path", return_value=str(data_directory)):
                    self.assertEqual(install.icon_path(), packaged_icon)

    def test_linux_desktop_launcher_references_generated_logo(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "repository"
            desktop = Path(temporary_directory) / "Desktop"
            root.mkdir()
            desktop.mkdir()
            icon_path = Path(install.ROOT) / "nexus.ico"
            (root / "nexus.ico").write_bytes(icon_path.read_bytes())
            with patch.object(install, "ROOT", root):
                with patch.object(install, "SCRIPT", root / "nexus.py"):
                    with patch.object(install.sys, "platform", "linux"):
                        shortcut = install.install_unix(desktop)

            contents = shortcut.read_text(encoding="utf-8")
            logo_path = root / "nexus.png"
            logo_data = logo_path.read_bytes()
            expected_logo = install.png_data_from_icon(root / "nexus.ico")

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

        with patch("nexus.urllib.request.urlopen", return_value=Response()) as urlopen:
            answer = nexus.ollama_chat_completion(
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
        self.assertEqual(nexus.recommend_model(2), (
            "llama3.2:1b", "the smallest available option for a low-memory system"
        ))
        self.assertEqual(nexus.recommend_model(8), ("llama3.2:3b", "your system memory"))
        self.assertEqual(nexus.recommend_model(16), ("llama3.1:8b", "your system memory"))
        self.assertEqual(nexus.recommend_model(8, 24), (
            "qwen2.5:32b", "your detected NVIDIA GPU memory"
        ))

    def test_coding_model_recommendation_uses_ram_and_nvidia_memory(self):
        self.assertEqual(nexus.recommend_coding_model(8), (
            "qwen2.5-coder:3b", "your system memory"
        ))
        self.assertEqual(nexus.recommend_coding_model(8, 24), (
            "qwen3-coder:30b", "your detected NVIDIA GPU memory"
        ))

    def test_setup_model_options_include_general_reasoning_and_coding_families(self):
        options = {
            category: model
            for category, model, _ in nexus.recommend_setup_models(8)
        }

        self.assertEqual(options, {
            "General": "llama3.2:3b",
            "Qwen 3": "qwen3:4b",
            "Gemma 3": "gemma3:4b",
            "DeepSeek R1": "deepseek-r1:1.5b",
            "Coding": "qwen2.5-coder:3b",
            "Granite 3.3": "granite3.3:2b",
        })

    def test_setup_model_options_include_large_models_only_when_hardware_fits(self):
        ordinary_options = {
            category: model
            for category, model, _ in nexus.recommend_setup_models(8)
        }
        high_memory_options = {
            category: model
            for category, model, _ in nexus.recommend_setup_models(96, 80)
        }

        self.assertNotIn("GPT-OSS", ordinary_options)
        self.assertNotIn("Mistral Small 3.2", ordinary_options)
        self.assertEqual(high_memory_options["GPT-OSS"], "gpt-oss:120b")
        self.assertEqual(high_memory_options["Phi-4"], "phi4")

    def test_setup_wizard_downloads_only_after_confirmation(self):
        output = io.StringIO()
        answers = iter(("1", "y"))
        with patch("nexus.system_memory_gb", return_value=16):
            with patch("nexus.nvidia_memory_gb", return_value=None):
                with patch("nexus.shutil.which", return_value="ollama"):
                    with patch("nexus.subprocess.run", return_value=type("Result", (), {"returncode": 0})()) as run:
                        with redirect_stdout(output):
                            self.assertEqual(nexus.run_setup_wizard(lambda prompt: next(answers)), 0)

        run.assert_called_once_with(["ollama", "pull", "llama3.1:8b"], check=False)
        self.assertIn("1. General: llama3.1:8b", output.getvalue())
        self.assertIn("Selected General model: llama3.1:8b.", output.getvalue())
        self.assertIn("Download complete.", output.getvalue())

    def test_setup_wizard_downloads_selected_deepseek_model(self):
        output = io.StringIO()
        answers = iter(("4", "y"))
        with patch("nexus.system_memory_gb", return_value=8):
            with patch("nexus.nvidia_memory_gb", return_value=None):
                with patch("nexus.shutil.which", return_value="ollama"):
                    with patch("nexus.subprocess.run", return_value=type("Result", (), {"returncode": 0})()) as run:
                        with redirect_stdout(output):
                            self.assertEqual(nexus.run_setup_wizard(lambda prompt: next(answers)), 0)

        run.assert_called_once_with(["ollama", "pull", "deepseek-r1:1.5b"], check=False)
        self.assertIn("Selected DeepSeek R1 model: deepseek-r1:1.5b.", output.getvalue())

    def test_setup_wizard_displays_devstral_ollama_version_requirement(self):
        output = io.StringIO()
        answers = iter(("10", "n"))
        with patch("nexus.system_memory_gb", return_value=32):
            with patch("nexus.nvidia_memory_gb", return_value=24):
                with patch("nexus.shutil.which", return_value="ollama"):
                    with patch("nexus.subprocess.run") as run:
                        with redirect_stdout(output):
                            self.assertEqual(nexus.run_setup_wizard(lambda prompt: next(answers)), 0)

        run.assert_not_called()
        self.assertIn("devstral-small-2:24b", output.getvalue())
        self.assertIn("Requires Ollama 0.13.3 or newer.", output.getvalue())

    def test_setup_wizard_does_not_download_without_confirmation(self):
        output = io.StringIO()
        answers = iter(("2", "n"))
        with patch("nexus.system_memory_gb", return_value=16):
            with patch("nexus.nvidia_memory_gb", return_value=None):
                with patch("nexus.shutil.which", return_value="ollama"):
                    with patch("nexus.subprocess.run") as run:
                        with redirect_stdout(output):
                            self.assertEqual(nexus.run_setup_wizard(lambda prompt: next(answers)), 0)

        run.assert_not_called()
        self.assertIn("Selected Qwen 3 model: qwen3:8b.", output.getvalue())
        self.assertIn("No model downloaded.", output.getvalue())

    def test_setup_wizard_without_ollama_only_suggests_model(self):
        output = io.StringIO()
        with patch("nexus.system_memory_gb", return_value=8):
            with patch("nexus.nvidia_memory_gb", return_value=None):
                with patch("nexus.shutil.which", return_value=None):
                    with patch("nexus.subprocess.run") as run:
                        with redirect_stdout(output):
                            self.assertEqual(nexus.run_setup_wizard(), 0)

        run.assert_not_called()
        self.assertIn("1. General: llama3.2:3b", output.getvalue())
        self.assertIn("Install Ollama", output.getvalue())

    def test_setup_wizard_rejects_invalid_model_selection(self):
        output = io.StringIO()
        errors = io.StringIO()
        with patch("nexus.system_memory_gb", return_value=8):
            with patch("nexus.nvidia_memory_gb", return_value=None):
                with redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(nexus.run_setup_wizard(lambda prompt: "7"), 1)

        self.assertIn("choose a model number", errors.getvalue())

    def test_setup_flag_runs_before_local_model_selection(self):
        with patch("nexus.run_setup_wizard", return_value=0) as setup:
            self.assertEqual(nexus.main(["--provider", "lm-studio", "--setup"]), 0)

        setup.assert_called_once_with()

    def test_setup_wizard_in_chat_recommends_session_model_command(self):
        output = io.StringIO()
        with patch("nexus.system_memory_gb", return_value=4):
            with patch("nexus.nvidia_memory_gb", return_value=None):
                with patch("nexus.shutil.which", return_value=None):
                    with redirect_stdout(output):
                        self.assertEqual(nexus.run_setup_wizard(in_chat=True), 0)

        self.assertIn("Use /model llama3.2:1b to switch models in this chat.", output.getvalue())

    def test_setup_chat_command_runs_wizard_only_with_ollama(self):
        prompts = ["/setup", "/exit"]
        with patch("builtins.input", side_effect=prompts):
            with patch("nexus.run_setup_wizard", return_value=0) as setup:
                self.assertEqual(nexus.main(["--compact"]), 0)

        setup.assert_called_once_with(in_chat=True)

    def test_setup_chat_command_rejects_local_api_provider(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"NEXUS_MODEL": "test-model"}, clear=True):
            with patch("builtins.input", side_effect=["/setup", "/exit"]):
                with patch("nexus.run_setup_wizard") as setup:
                    with redirect_stdout(output):
                        self.assertEqual(nexus.main(["--provider", "localai"]), 0)

        setup.assert_not_called()
        self.assertIn("Model setup is available only with --provider ollama.", output.getvalue())

    def test_selected_folder_rejects_paths_outside_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = nexus.validate_selected_folder(temporary_directory)
            self.assertEqual(nexus.resolve_selected_path(root, "notes.txt"), os.path.join(root, "notes.txt"))
            with self.assertRaisesRegex(ValueError, "inside the selected folder"):
                nexus.resolve_selected_path(root, os.path.join("..", "outside.txt"))

    def test_selected_text_file_requires_utf8_and_size_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            Path(temporary_directory, "notes.txt").write_text("private note", encoding="utf-8")
            _, contents = nexus.read_selected_text_file(temporary_directory, "notes.txt")
            self.assertEqual(contents, "private note")

            Path(temporary_directory, "binary.txt").write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                nexus.read_selected_text_file(temporary_directory, "binary.txt")

            Path(temporary_directory, "large.txt").write_bytes(
                b"x" * (nexus.MAX_SELECTED_TEXT_FILE_BYTES + 1)
            )
            with self.assertRaisesRegex(ValueError, "1 MiB"):
                nexus.read_selected_text_file(temporary_directory, "large.txt")

    def test_readfile_context_is_confirmed_one_request_and_not_saved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "notes.txt").write_text("private file contents", encoding="utf-8")
            output = io.StringIO()
            prompts = [
                "/folder {}".format(root),
                "/files",
                "/readfile notes.txt",
                "n",
                "/readfile notes.txt",
                "y",
                "Summarize this file.",
                "What did I ask you before?",
                "/exit",
            ]
            with patch("nexus.sys.platform", "win32"):
                with patch("builtins.input", side_effect=prompts):
                    with patch("nexus.ollama_chat_completion", side_effect=["Summary", "Follow-up"]) as completion:
                        with redirect_stdout(output):
                            self.assertEqual(nexus.main(["--compact"]), 0)

            first_request = completion.call_args_list[0].args[0]
            second_request = completion.call_args_list[1].args[0]
            saved_conversation = json.loads(
                Path(nexus.list_conversations(self.test_conversation_directory.name)[0]).read_text(
                    encoding="utf-8"
                )
            )

        self.assertIn("private file contents", first_request[1]["content"])
        self.assertNotIn("private file contents", json.dumps(second_request))
        self.assertNotIn("private file contents", json.dumps(saved_conversation))
        self.assertIn("notes.txt", output.getvalue())
        self.assertIn("File was not shared.", output.getvalue())
        self.assertIn("File loaded into the next request.", output.getvalue())

    def test_notepad_and_explorer_are_limited_to_selected_folder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = io.StringIO()
            prompts = [
                "/folder {}".format(root),
                "/notepad new.txt",
                "y",
                "/explorer",
                "/exit",
            ]
            with patch("nexus.sys.platform", "win32"):
                with patch("builtins.input", side_effect=prompts):
                    with patch("nexus.subprocess.Popen") as popen:
                        with redirect_stdout(output):
                            self.assertEqual(nexus.main(["--compact"]), 0)

            self.assertTrue((root / "new.txt").is_file())
            self.assertEqual(popen.call_args_list[0].args[0], ["notepad.exe", str(root / "new.txt")])
            self.assertEqual(popen.call_args_list[1].args[0], ["explorer.exe", str(root)])
            self.assertIn("Opened Notepad:", output.getvalue())
            self.assertIn("Opened File Explorer:", output.getvalue())

    def test_ollama_model_list_returns_names_and_sizes(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"models":[{"name":"llama3.2:3b","size":3221225472}]}'

        with patch("nexus.urllib.request.urlopen", return_value=Response()) as urlopen:
            models = nexus.ollama_list_models("http://localhost:11434/")

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
        with patch("nexus.urllib.request.urlopen", return_value=Response()) as urlopen:
            nexus.ollama_pull_model("llama3.2:3b", "http://localhost:11434", updates.append)

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

        with patch("nexus.urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "without confirming success"):
                nexus.ollama_pull_model("llama3.2:3b", "http://localhost:11434")

    def test_coding_model_wizard_recommends_installed_model(self):
        output = io.StringIO()
        with patch("nexus.system_memory_gb", return_value=16):
            with patch("nexus.nvidia_memory_gb", return_value=None):
                with patch(
                    "nexus.ollama_list_models",
                    return_value=[("qwen2.5-coder:7b", 7 * 1024 ** 3)],
                ):
                    with redirect_stdout(output):
                        model = nexus.run_coding_model_wizard("http://localhost:11434")

        self.assertEqual(model, "qwen2.5-coder:7b")
        self.assertIn("Recommended coding model: qwen2.5-coder:7b", output.getvalue())

    def test_coding_model_wizard_only_downloads_after_confirmation(self):
        with patch("nexus.system_memory_gb", return_value=16):
            with patch("nexus.nvidia_memory_gb", return_value=None):
                with patch("nexus.ollama_list_models", return_value=[]):
                    with patch("nexus.ollama_pull_model") as pull_model:
                        self.assertIsNone(nexus.run_coding_model_wizard(
                            "http://localhost:11434",
                            input_fn=lambda prompt: "n",
                        ))
        pull_model.assert_not_called()

        with patch("nexus.system_memory_gb", return_value=16):
            with patch("nexus.nvidia_memory_gb", return_value=None):
                with patch("nexus.ollama_list_models", return_value=[]):
                    with patch("nexus.ollama_pull_model") as pull_model:
                        model = nexus.run_coding_model_wizard(
                            "http://localhost:11434",
                            input_fn=lambda prompt: "yes",
                        )

        self.assertEqual(model, "qwen2.5-coder:7b")
        pull_model.assert_called_once_with(
            "qwen2.5-coder:7b", "http://localhost:11434", nexus.print_ollama_pull_progress
        )

    def test_web_search_parses_duckduckgo_results(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                return (
                    b'<a class="result__a" href="//duckduckgo.com/l/?uddg='
                    b'https%3A%2F%2Fexample.com%2Fguide">Example guide</a>'
                    b'<a class="result__snippet">A useful summary.</a>'
                )

        with patch("nexus.urllib.request.urlopen", return_value=Response()) as urlopen:
            results = nexus.web_search("python testing")

        self.assertEqual(results, [{
            "title": "Example guide",
            "url": "https://example.com/guide",
            "snippet": "A useful summary.",
        }])
        self.assertIn("python+testing", urlopen.call_args.args[0].full_url)

    def test_search_command_sends_search_results_to_model_once(self):
        output = io.StringIO()
        search_results = [{
            "title": "Python docs",
            "url": "https://docs.example.test/python",
            "snippet": "Official Python documentation.",
        }]
        with patch("builtins.input", side_effect=["/search Python docs", "/exit"]):
            with patch("nexus.web_search", return_value=search_results):
                with patch("nexus.ollama_chat_completion", return_value="See the docs.") as completion:
                    with redirect_stdout(output):
                        self.assertEqual(nexus.main(["--compact"]), 0)

        request_messages = completion.call_args.args[0]
        self.assertIn("Official Python documentation.", request_messages[1]["content"])
        self.assertEqual(request_messages[-1], {"role": "user", "content": "Python docs"})
        self.assertIn("https://docs.example.test/python", output.getvalue())

    def test_coding_mode_switches_model_and_restores_chat_settings(self):
        prompts = [
            "/coding on",
            "Write a test.",
            "/coding off",
            "Regular chat.",
            "/exit",
        ]
        requests = []

        def capture_completion(messages, base_url, model):
            requests.append((json.loads(json.dumps(messages)), model))
            return "Code answer" if model == "qwen2.5-coder:7b" else "Chat answer"

        with patch("builtins.input", side_effect=prompts):
            with patch("nexus.run_coding_model_wizard", return_value="qwen2.5-coder:7b") as wizard:
                with patch("nexus.ollama_chat_completion", side_effect=capture_completion):
                    self.assertEqual(nexus.main(["--system", "Be concise.", "--compact"]), 0)

        wizard.assert_called_once_with(nexus.DEFAULT_OLLAMA_BASE_URL, None)
        first_messages, first_model = requests[0]
        second_messages, second_model = requests[1]
        self.assertEqual(first_model, "qwen2.5-coder:7b")
        self.assertIn("Coding mode is enabled.", first_messages[0]["content"])
        self.assertEqual(second_model, "llama3.1:8b")
        self.assertEqual(second_messages[0]["content"], "Be concise.")

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
                "nexus.ollama_list_models", return_value=[("llama3.1:8b", 8 * 1024 ** 3)]
            ):
                with patch("nexus.ollama_pull_model") as pull_model:
                    with patch("nexus.ollama_chat_completion", return_value="Hello") as completion:
                        with redirect_stdout(output):
                            self.assertEqual(nexus.main(["--compact"]), 0)

        pull_model.assert_called_once()
        self.assertEqual(pull_model.call_args.args[:2], ("qwen2.5:7b", nexus.DEFAULT_OLLAMA_BASE_URL))
        self.assertEqual(completion.call_args.args[2], "llama3.2:3b")
        self.assertIn("llama3.1:8b (current) - 8.0 GiB", output.getvalue())
        self.assertIn("Model download cancelled.", output.getvalue())
        self.assertIn("Model 'qwen2.5:7b' is ready.", output.getvalue())

    def test_local_host_posts_messages_without_required_api_key(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"Hello"}}]}'

        with patch("nexus.urllib.request.urlopen", return_value=Response()) as urlopen:
            answer = nexus.local_api_chat_completion(
                [{"role": "user", "content": "Hi"}], None, "http://localhost:1234/v1/", "test-model"
            )

        self.assertEqual(answer, "Hello")
        request = urlopen.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertEqual(request.full_url, "http://localhost:1234/v1/chat/completions")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(json.loads(request.data), {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })

    def test_local_host_sends_api_key_only_when_configured(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"Hello"}}]}'

        with patch("nexus.urllib.request.urlopen", return_value=Response()) as urlopen:
            nexus.local_api_chat_completion(
                [{"role": "user", "content": "Hi"}],
                "local-token",
                "http://localhost:1234/v1",
                "test-model",
            )

        self.assertEqual(
            urlopen.call_args.args[0].get_header("Authorization"),
            "Bearer local-token",
        )

    def test_local_host_model_list_returns_ids(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"data":[{"id":"model-a"},{"id":"model-b"}]}'

        with patch("nexus.urllib.request.urlopen", return_value=Response()) as urlopen:
            models = nexus.local_api_list_models("http://localhost:1234/v1/")

        self.assertEqual(models, ["model-a", "model-b"])
        self.assertEqual(urlopen.call_args.args[0].full_url, "http://localhost:1234/v1/models")

    def test_local_host_starts_without_prompting_for_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {"NEXUS_MODEL": "test-model"}, clear=True):
                with patch("builtins.input", side_effect=["Hi", "/exit"]):
                    with patch("nexus.local_api_chat_completion", return_value="Hello") as completion:
                        self.assertEqual(nexus.main(["--provider", "lm-studio"]), 0)

        self.assertIsNone(completion.call_args.args[1])
        self.assertEqual(completion.call_args.args[3], "test-model")

    def test_supported_local_hosts_default_to_loopback_urls(self):
        for provider in ("lm-studio", "llama-cpp", "localai"):
            with self.subTest(provider=provider):
                with patch.dict(os.environ, {"NEXUS_MODEL": "test-model"}, clear=True):
                    with patch("builtins.input", side_effect=["Hi", "/exit"]):
                        with patch("nexus.local_api_chat_completion", return_value="Hello") as completion:
                            self.assertEqual(
                                nexus.main(["--provider", provider, "--compact"]), 0
                            )

                base_url = completion.call_args.args[2]
                self.assertEqual(base_url, nexus.PROVIDER_BASE_URLS[provider])
                self.assertTrue(base_url.startswith("http://localhost:"))

    def test_completed_conversation_is_saved_and_listed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = io.StringIO()
            prompts = ["Hi", "/history", "/exit"]
            with patch.dict(os.environ, {"NEXUS_CONVERSATIONS_DIR": temporary_directory}, clear=True):
                with patch("builtins.input", side_effect=prompts):
                    with patch("nexus.ollama_chat_completion", return_value="Hello"):
                        with redirect_stdout(output):
                            self.assertEqual(nexus.main(["--compact"]), 0)

            paths = nexus.list_conversations(temporary_directory)
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
            with patch.dict(os.environ, {"NEXUS_CONVERSATIONS_DIR": temporary_directory}, clear=True):
                with patch("builtins.input", side_effect=prompts):
                    with patch("nexus.ollama_chat_completion", side_effect=["One", "Two"]):
                        self.assertEqual(nexus.main(["--compact"]), 0)

            saved_conversations = [
                json.loads(Path(path).read_text(encoding="utf-8"))
                for path in nexus.list_conversations(temporary_directory)
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
        with patch("nexus_update.shutil.which", return_value="pipx"):
            with patch("nexus_update.subprocess.run", return_value=type("Result", (), {"returncode": 0})()) as run:
                self.assertEqual(nexus_update.main(), 0)

        run.assert_called_once_with(["pipx", "upgrade", "nexus-local-ai-host"], check=False)

    def test_update_command_reports_missing_pipx(self):
        error_output = io.StringIO()
        with patch("nexus_update.shutil.which", return_value=None):
            with redirect_stderr(error_output):
                self.assertEqual(nexus_update.main(), 1)

        self.assertIn("pipx was not found on PATH", error_output.getvalue())

    def test_chat_update_command_invokes_updater(self):
        with patch("builtins.input", side_effect=["/update", "/exit"]):
            with patch("nexus_update.main", return_value=0) as update:
                self.assertEqual(nexus.main(["--compact"]), 0)

        update.assert_called_once_with()

    def test_response_status_reflects_internet_availability(self):
        for internet_connected, status in (
            (True, "internet connection detected"),
            (False, "offline; internet connection not detected"),
        ):
            with self.subTest(internet_connected=internet_connected):
                output = io.StringIO()
                with patch("nexus.internet_available", return_value=internet_connected):
                    with patch("builtins.input", side_effect=["Hi", "/exit"]):
                        with patch("nexus.ollama_chat_completion", return_value="Hello"):
                            with redirect_stdout(output):
                                self.assertEqual(nexus.main(["--compact"]), 0)

                text = output.getvalue()
                self.assertIn("Nexus Client> llama3.1:8b ({})".format(status), text)
                self.assertIn("Nexus Client ready. Type /help for commands.", text)
                answer_position = text.index("Nexus Client> Hello")
                self.assertLess(text.index("Nexus Client> llama3.1:8b"), answer_position)

    def test_progress_indicator_updates_until_generation_finishes(self):
        output = io.StringIO()

        def delayed_answer(*args):
            time.sleep(0.25)
            return "Hello"

        with patch("builtins.input", side_effect=["Hi", "/exit"]):
            with patch("nexus.ollama_chat_completion", side_effect=delayed_answer):
                with redirect_stdout(output):
                    self.assertEqual(nexus.main(["--compact"]), 0)

        self.assertGreaterEqual(output.getvalue().count("Nexus Client> llama3.1:8b"), 2)

    def test_internet_check_handles_unreachable_network(self):
        with patch("nexus.socket.create_connection", side_effect=OSError("unreachable")) as connect:
            self.assertFalse(self.check_internet())

        connect.assert_called_once_with(("1.1.1.1", 443), timeout=1)

    def test_internet_check_closes_connection(self):
        connection = io.StringIO()
        with patch("nexus.socket.create_connection", return_value=connection) as connect:
            self.assertTrue(self.check_internet())

        connect.assert_called_once_with(("1.1.1.1", 443), timeout=1)
        self.assertTrue(connection.closed)

    def test_terminal_commands_customize_chat_without_being_sent_as_prompts(self):
        prompts = [
            "/model custom-model",
            "First question",
            "/clear",
            "/spacing off",
            "Second question",
            "/exit",
        ]
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.input", side_effect=prompts):
                with patch("nexus.ollama_chat_completion", return_value="Answer") as completion:
                    self.assertEqual(nexus.main([]), 0)

        self.assertEqual(completion.call_count, 2)
        first_messages = completion.call_args_list[0].args[0]
        self.assertEqual(completion.call_args_list[0].args[2], "custom-model")
        self.assertEqual(first_messages[0]["content"], nexus.DEFAULT_SYSTEM_PROMPT)
        second_messages = completion.call_args_list[1].args[0]
        self.assertEqual([message["content"] for message in second_messages], [
            nexus.DEFAULT_SYSTEM_PROMPT, "Second question", "Answer"
        ])

    def test_compact_flag_removes_extra_reply_spacing(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.input", side_effect=["Hi", "/exit"]) as user_input:
                with patch("nexus.ollama_chat_completion", return_value="Hello"):
                    with redirect_stdout(output):
                        self.assertEqual(nexus.main(["--compact"]), 0)

        self.assertIn("Nexus Client> Hello\n", output.getvalue())
        self.assertNotIn("\n\nNexus Client> Hello", output.getvalue())
        self.assertEqual(user_input.call_args_list[0].args, ("you> ",))

    def test_default_spacing_adds_a_blank_line_before_each_prompt(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.input", side_effect=["Hi", "/exit"]) as user_input:
                with patch("nexus.ollama_chat_completion", return_value="Hello"):
                    self.assertEqual(nexus.main([]), 0)

        self.assertEqual(user_input.call_args_list[0].args, ("\nyou> ",))
        self.assertEqual(user_input.call_args_list[1].args, ("\nyou> ",))

    def test_private_commands_are_absent_without_local_extension(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory:
            export_path = Path(temporary_directory) / "session.json"
            prompts = [
                "/dev on",
                "/system private instructions",
                "/trainmodel",
                "/debug",
                "/context",
                "/export {}".format(export_path),
                "/exit",
            ]
            with patch.dict(os.environ, {}, clear=True):
                with patch("nexus.load_private_developer", return_value=None):
                    with patch("builtins.input", side_effect=prompts):
                        with patch("nexus.ollama_chat_completion") as completion:
                            with redirect_stdout(output):
                                self.assertEqual(nexus.main([]), 0)

            self.assertFalse(export_path.exists())

        completion.assert_not_called()
        for command in ("/dev", "/system", "/trainmodel", "/debug", "/context", "/export"):
            self.assertIn("Unknown command: {}".format(command), output.getvalue())
        self.assertNotIn("Developer mode enabled.", output.getvalue())



if __name__ == "__main__":
    unittest.main()