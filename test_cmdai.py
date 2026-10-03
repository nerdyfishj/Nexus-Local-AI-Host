import io
import json
import os
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from urllib.request import Request
from pathlib import Path

import cmdai
import install


class ChatCompletionTests(unittest.TestCase):
    def test_windows_icon_is_a_valid_ico_with_png_image(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            icon_path = Path(temporary_directory) / "cmdAI.ico"
            install.write_windows_icon(icon_path)

            icon_data = icon_path.read_bytes()

        self.assertEqual(icon_data[:6], struct.pack("<HHH", 0, 1, 1))
        self.assertEqual(icon_data[22:30], b"\x89PNG\r\n\x1a\n")

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
            with patch("builtins.input", side_effect=["Hi", "/exit"]):
                with patch("cmdai.ollama_chat_completion", return_value="Hello"):
                    with redirect_stdout(output):
                        self.assertEqual(cmdai.main(["--compact"]), 0)

        self.assertIn(".cmdAI> Hello\n", output.getvalue())
        self.assertNotIn("\n\n.cmdAI> Hello", output.getvalue())

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
        self.assertIn("Developer mode is off. Use /dev on", output.getvalue())


if __name__ == "__main__":
    unittest.main()