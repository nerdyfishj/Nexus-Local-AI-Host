# .cmdAI

A small terminal chat assistant for Windows, Linux, and macOS. It uses only the Python standard library and defaults to Ollama running locally, so no API key is needed. An OpenAI-compatible API is also available as an optional provider.

## Requirements

- Python 3.9 or newer
- Ollama installed, with a model downloaded
- Python 3.9 or newer

## Setup

Clone or download this repository into a permanent folder, then run the installer once:

```text
python install.py
```

This creates a `.cmdAI` desktop launcher (`.cmdAI.lnk` on Windows, `cmdAI.desktop` on Linux, or a visible `cmdAI.app` on macOS with the display name `.cmdAI`). Repository downloads do not automatically execute installer code; running this command is the step that creates the shortcut.

Install Ollama from [ollama.com](https://ollama.com), start it, and download a model:

```text
ollama pull phi3
```

Then launch `.cmdAI` from the desktop. The default model is `phi3:latest`, served locally at `http://localhost:11434`. Choose another installed model with `CMD_AI_MODEL` or `--model`.

To use OpenAI instead, select the provider; it will prompt for an API key if needed:

```sh
python cmdai.py --provider openai
```

For the optional OpenAI-compatible provider, configure `CMD_AI_BASE_URL` and `CMD_AI_MODEL` as needed. The OpenAI defaults are `https://api.openai.com/v1` and `gpt-4o-mini`; set `CMD_AI_API_KEY` or `OPENAI_API_KEY` to avoid the key prompt.

Type `/exit` or `/quit` to end a chat. Conversation history is kept for the current run and is sent to the configured API with each request.

## Verify

```text
python -m unittest
```