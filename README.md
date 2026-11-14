# Nexus Local AI Host

**Your models. Your machine. One local hub.**

Nexus gives you one terminal chat for models running on your own computer. It supports Ollama, LM Studio, llama.cpp Server, and LocalAI. By default, requests go to a local service; no hosted AI account or API key is required.

## Requirements

- Python 3.9 or newer
- Ollama, LM Studio, llama.cpp Server, or LocalAI
- Git and pipx only if installing from the GitHub repository

## Install

### From GitHub

```sh
pipx install "git+https://github.com/nerdyfishj/nexus-local-ai-host.git"
nexus-install
```

The installer creates a desktop launcher. Start Nexus from a terminal with `nexus`.

### From a local checkout

```powershell
# Windows
py nexus_install.py
```

```sh
# macOS or Linux
python3 nexus_install.py
```

The desktop launcher points to this checkout. If you move the folder, run the installer again.

## Choose a local model host

Ollama is the default. Start Ollama and download a model:

```sh
ollama pull llama3.1:8b
nexus
```

Nexus also supports local chat services that expose a compatible chat API:

| Host | Provider | Default API root |
| --- | --- | --- |
| Ollama | `ollama` | `http://localhost:11434` |
| LM Studio | `lm-studio` | `http://localhost:1234/v1` |
| llama.cpp Server | `llama-cpp` | `http://localhost:8080/v1` |
| LocalAI | `localai` | `http://localhost:8080/v1` |

Start the host and load a model first. Select its model identifier with `--model`:

```sh
nexus --provider lm-studio --model <model-id>
nexus --provider llama-cpp --model <model-id>
nexus --provider localai --model <model-id>
```

`/models` lists models exposed by the selected host, and `/model <model-id>` switches models in the current chat. Use `--base-url` to override a host's default URL. The optional `NEXUS_API_KEY` environment variable is only used when a local service requires authentication; it is not needed for the default local setup.

You can set `NEXUS_PROVIDER`, `NEXUS_MODEL`, `NEXUS_BASE_URL`, and `NEXUS_SYSTEM_PROMPT` to customize startup. The default Ollama model is `llama3.1:8b`.

## Setup and updates

The hardware-based model recommendation wizard is for Ollama:

```sh
nexus --setup
```

Close Nexus before updating.

- For a pipx installation, run `nexus-update`.
- For a local Git checkout, pull the latest files and reopen the desktop shortcut.

## Chat commands

- `/clear` starts a fresh conversation.
- `/setup` recommends an Ollama model.
- `/models` lists models available from the active host.
- `/model [name]` shows or changes the active model.
- `/pull <name>` downloads an Ollama model after confirmation.
- `/coding on|off|status [model]` enables coding-focused instructions with Ollama.
- `/search <query>` searches DuckDuckGo and sends result snippets to the selected local model.
- `/folder`, `/files`, `/explorer`, `/notepad`, and `/readfile` provide selected-folder tools on Windows.
- `/update` runs the pipx updater.
- `/spacing on|off` changes reply spacing.
- `/exit` ends the chat.

Prompts and replies are saved locally in `%USERPROFILE%\.nexus\conversations` on Windows or `~/.nexus/conversations` on macOS and Linux. Protect that folder as it may contain private text. Set `NEXUS_CONVERSATIONS_DIR` to use another location. If an existing `.cmdai/conversations` folder is present and the new `.nexus` folder is not, Nexus continues using the existing folder so upgrading does not discard chat history.

## Verify

```text
python -m unittest
```
