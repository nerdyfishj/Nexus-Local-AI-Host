"""A cross-platform terminal client for Ollama and OpenAI-compatible chat APIs."""

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "phi3:latest"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Keep answers clear and concise."


def ollama_chat_completion(messages, base_url, model):
    """Send a chat request to a local Ollama server and return its response."""
    url = base_url.rstrip("/") + "/api/chat"
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["message"]["content"]


def chat_completion(messages, api_key, base_url, model):
    """Send a chat completion request and return the assistant's text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "messages": messages}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chat with an AI from your terminal.")
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai"),
        default=os.environ.get("CMD_AI_PROVIDER", "ollama"),
        help="AI provider (default: ollama; or set CMD_AI_PROVIDER)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CMD_AI_BASE_URL"),
        help="provider API root (or set CMD_AI_BASE_URL)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CMD_AI_MODEL"),
        help="model name (or set CMD_AI_MODEL)",
    )
    parser.add_argument(
        "--system",
        default=os.environ.get("CMD_AI_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="system prompt (or set CMD_AI_SYSTEM_PROMPT)",
    )
    args = parser.parse_args(argv)
    if args.provider == "ollama":
        base_url = args.base_url or DEFAULT_OLLAMA_BASE_URL
        model = args.model or DEFAULT_OLLAMA_MODEL
        api_key = None
    else:
        base_url = args.base_url or DEFAULT_OPENAI_BASE_URL
        model = args.model or DEFAULT_OPENAI_MODEL
        api_key = os.environ.get("CMD_AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            try:
                api_key = getpass.getpass("API key (input hidden): ")
            except (EOFError, OSError):
                parser.error("set CMD_AI_API_KEY or OPENAI_API_KEY before starting")
        if not api_key:
            parser.error("an API key is required for the OpenAI provider")

    messages = [{"role": "system", "content": args.system}]
    print(".cmdAI ready. Type /exit to quit.")
    try:
        while True:
            try:
                prompt = input("you> ").strip()
            except EOFError:
                break
            if prompt.lower() in ("/exit", "/quit"):
                break
            if not prompt:
                continue
            messages.append({"role": "user", "content": prompt})
            try:
                if args.provider == "ollama":
                    answer = ollama_chat_completion(messages, base_url, model)
                else:
                    answer = chat_completion(messages, api_key, base_url, model)
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, ValueError) as error:
                messages.pop()
                print(".cmdAI error: {}".format(error), file=sys.stderr)
                if args.provider == "ollama" and isinstance(error, urllib.error.URLError):
                    print(
                        "Make sure Ollama is running and the '{}' model is installed.".format(model),
                        file=sys.stderr,
                    )
                continue
            messages.append({"role": "assistant", "content": answer})
            print(".cmdAI> {}".format(answer))
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())