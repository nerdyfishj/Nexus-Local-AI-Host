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

The Ollama model wizard uses available system RAM and detected NVIDIA GPU memory to suggest models that fit approximately. Choices include general, Qwen 3, Gemma 3, DeepSeek R1, coding, Granite 3.3, Phi-4, GPT-OSS, Mistral Small 3.2, and Devstral Small 2 models. High-memory choices are shown only when the detected hardware meets the wizard's rough thresholds.

```sh
nexus --setup
```

Choose a model number; the wizard asks before downloading it. To install or run a model manually, use its Ollama model ID:

```sh
ollama pull <model-id>
nexus --model <model-id>
```

Additional model IDs offered by the wizard:

| Family | Ollama model IDs | Notes |
| --- | --- | --- |
| Granite 3.3 | `granite3.3:2b`, `granite3.3:8b` | Lightweight general-purpose models. |
| Phi-4 | `phi4` | 14B reasoning/general model; needs more memory. |
| GPT-OSS | `gpt-oss:20b`, `gpt-oss:120b` | The 120B model is a very high-memory option (about 65 GB to download). |
| Mistral Small 3.2 | `mistral-small3.2:24b` | 24B general-purpose model. |
| Devstral Small 2 | `devstral-small-2:24b` | Coding model; requires Ollama 0.13.3 or newer. |

These are approximate hardware recommendations, not guarantees; actual memory use depends on the system, model, and context length. If the Ollama version is too old for Devstral Small 2, update Ollama before downloading it.

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

## Development tools

The optional private development commands require a source checkout. The `nexus_private_dev.py` module is intentionally excluded from GitHub and packaged installs, so copy it separately beside `nexus.py`. Set `NEXUS_DEV_PASSWORD` locally, then launch Nexus from that folder:

```powershell
# Windows PowerShell
$env:NEXUS_DEV_PASSWORD = "choose-a-local-password"
py nexus.py
```

```sh
# macOS or Linux
export NEXUS_DEV_PASSWORD="choose-a-local-password"
python3 nexus.py
```

At the Nexus prompt, run `/dev on` and enter the password, then `/dev test` to run the local test suite. `/dev preview on` also includes tests in the optional `experiments` folder; copy that folder if needed. `/dev modeltest [model]` requires a running local Ollama service. Keep the password local; do not put it on the USB stick or commit it.

## Verify

```text
python -m unittest
```
