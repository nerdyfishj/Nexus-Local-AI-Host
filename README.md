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
ollama pull llama3.1:8b
```

Then launch `.cmdAI` from the desktop. The default model is `llama3.1:8b`, served locally at `http://localhost:11434`. It is a larger, more capable model than the previous default and uses more disk space and memory. Choose another installed model with `CMD_AI_MODEL` or `--model`.

To use OpenAI instead, select the provider; it will prompt for an API key if needed:

```sh
python cmdai.py --provider openai
```

For the optional OpenAI-compatible provider, configure `CMD_AI_BASE_URL` and `CMD_AI_MODEL` as needed. The OpenAI defaults are `https://api.openai.com/v1` and `gpt-4o-mini`; set `CMD_AI_API_KEY` or `OPENAI_API_KEY` to avoid the key prompt.

Prompts and replies are spaced out by default; start with `--compact` or use `/spacing on|off` to change that during a session. Type `/help` for available commands:

- `/clear` starts a fresh conversation while keeping your system instructions.
- `/dev on|off|status` enables or disables password-protected developer tools.
- `/exit` or `/quit` ends the chat.

Training examples are not model-weight training. They are stored locally in `~/.cmdai/training_examples.json` and up to three keyword-matched examples are included with relevant requests. Set `CMD_AI_TRAINING_FILE` to choose another storage path. Conversation history is kept for the current run and is sent to the configured API with each request. You can also customize the startup model and system instructions with `--model` and `--system`, or with `CMD_AI_MODEL` and `CMD_AI_SYSTEM_PROMPT`.

## Developer Mode

Set `CMD_AI_DEV_PASSWORD` in the environment before launching `.cmdAI`, then use `/dev on` and enter the password when prompted. Password input is hidden and the application does not store the password. Do not commit it to the repository. This is a local convenience gate, not a security boundary against other programs or users with access to the same account.

While enabled, developer tools include:

- `/model [name]` and `/system [instructions]` to inspect or change session settings.
- `/trainmodel` to save a prompt and preferred response; `/trainmodel list` shows examples, and `/trainmodel clear` removes them.
- `/debug` for provider, model, context count, example count, and spacing status.
- `/context` for message sizes and a rough token estimate.
- `/export <new-file-path>` to save the current conversation as JSON without overwriting an existing file.

Use `/dev off` to relock the tools. Training examples and exported conversations may contain private text; keep those files in an appropriate location.

## Verify

```text
python -m unittest
```