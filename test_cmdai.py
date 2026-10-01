import json
import os
import struct
import tempfile
import unittest
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
            with patch("builtins.input", return_value="/exit"):
                with patch("cmdai.getpass.getpass") as getpass_prompt:
                    self.assertEqual(cmdai.main([]), 0)

        getpass_prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()