# .cmdAI

A small terminal chat assistant for Windows, Linux, and macOS. It uses only the Python standard library and defaults to Ollama running locally, so no API key is needed. An OpenAI-compatible API is also available as an optional provider.

## Requirements

- Python 3.9 or newer
- Git and pipx (for installing directly from GitHub)
- Ollama with a model downloaded, unless you plan to use the OpenAI provider

## Setup

### Prepare your computer

Before installing from GitHub:

1. Install [Python 3.9 or newer](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads/).
2. Install [pipx](https://pipx.pypa.io/stable/installation/) for your operating system. Follow its instructions to make the `pipx` command available; on Windows, you may need to restart your terminal after setting up its PATH.
3. For the default local provider, install [Ollama](https://ollama.com), start it, and download the default model:

   ```sh
   ollama pull llama3.1:8b
   ```

   If you prefer OpenAI, you can skip Ollama and configure an API key instead.

### Install from GitHub

Run both commands in a terminal:

```sh
pipx install "git+https://github.com/nerdyfishj/.cmdAI.git"
cmdai-install
```

This installs `.cmdAI` in an isolated Python environment and creates a desktop launcher (`.cmdAI.lnk` on Windows, `cmdAI.desktop` on Linux, or `cmdAI.app` on macOS). Opening the launcher starts the chat in a terminal on each OS: Windows opens the Python console, Linux uses the desktop terminal, and macOS opens Terminal. To start it directly from any terminal instead, run `cmdai`.

After installing a version that includes the updater, update `.cmdAI` from inside a chat with `/update`, or from a terminal with:

```sh
cmdai-update
```

The update command upgrades the pipx-installed package; exit and restart `.cmdAI` to use the updated version. If you already have an older version without the updater, run `pipx upgrade cmdai-assistant` once to get it. The updater is for pipx installations; update a local Git clone with `git pull`. If you installed from a downloaded archive, download the newer archive and replace the old folder.

### Install from a local checkout

Clone or download this repository into a permanent folder, then run the installer once:

Open a terminal in the folder containing `install.py`, then run:

```sh
# Windows (PowerShell)
py install.py

# macOS or Linux
python3 install.py
```

This creates the platform-specific desktop launcher, which opens the chat in a terminal. Repository downloads do not automatically execute installer code; running this command is the step that creates the shortcut.

On Windows, the shortcut launches this folder's Python interpreter and `cmdai.py` directly; GitHub Desktop is not required. If you move the repository, run `python install.py` again from its new location to update the shortcut.

Then launch `.cmdAI` from the desktop. The default model is `llama3.1:8b`, served locally at `http://localhost:11434`. It is a larger, more capable model than the previous default and uses more disk space and memory. Choose another installed model with `CMD_AI_MODEL` or `--model`.

To use OpenAI instead, select the provider; it will prompt for an API key if needed:

```sh
python cmdai.py --provider openai
```

For the optional OpenAI-compatible provider, configure `CMD_AI_BASE_URL` and `CMD_AI_MODEL` as needed. The OpenAI defaults are `https://api.openai.com/v1` and `gpt-4o-mini`; set `CMD_AI_API_KEY` or `OPENAI_API_KEY` to avoid the key prompt.

Prompts and replies are spaced out by default; start with `--compact` or use `/spacing on|off` to change that during a session. Type `/help` for available commands:

While each response is generating, .cmdAI shows a live progress indicator with the model name and whether it detected an internet connection. This check does not send your prompt or API key.

- `/clear` starts a fresh conversation while keeping your system instructions.
- `/dev on|off|status` enables or disables password-protected developer tools.
- `/exit` or `/quit` ends the chat.

Training examples are not model-weight training. They are stored locally in `~/.cmdai/training_examples.json` and up to three keyword-matched examples are included with relevant requests. Set `CMD_AI_TRAINING_FILE` to choose another storage path. Each conversation is automatically saved as a JSON file in `~/.cmdai/conversations` (`%USERPROFILE%\.cmdai\conversations` on Windows). Use `/history` to see the folder and saved conversation filenames. Set `CMD_AI_CONVERSATIONS_DIR` to use a different folder. Files contain user and assistant messages, not system instructions or API keys; they may still include private text, so protect the folder accordingly. Conversation context is also sent to the configured API with each request. You can customize the startup model and system instructions with `--model` and `--system`, or with `CMD_AI_MODEL` and `CMD_AI_SYSTEM_PROMPT`.

## Developer Mode

Set `CMD_AI_DEV_PASSWORD` in the environment before launching `.cmdAI`, then use `/dev on` and enter the password when prompted. Password input is hidden and the application does not store the password. Do not commit it to the repository. This is a local convenience gate, not a security boundary against other programs or users with access to the same account.

While enabled, developer tools include:

- `/model [name]` and `/system [instructions]` to inspect or change session settings.
- `/update` to update a pipx installation from within the chat.
- `/history` to list locally saved conversations.
- `/trainmodel` to save a prompt and preferred response; `/trainmodel list` shows examples, and `/trainmodel clear` removes them.
- `/debug` for provider, model, context count, example count, and spacing status.
- `/context` for message sizes and a rough token estimate.
- `/export <new-file-path>` to save the current conversation as JSON without overwriting an existing file.

Use `/dev off` to relock the tools. Training examples and exported conversations may contain private text; keep those files in an appropriate location.

## Verify

```text
python -m unittest
```