# LOA

LOA is a local assistant that uses Ollama for chat and DuckDuckGo for requested web searches.

## Build and run

Build the standalone macOS executable:

```sh
./compile
./dist/loa
```

The first launch asks for an assistant name, an Ollama model, and a system prompt. Settings are saved in the user's application support directory. Later launches open the chat directly; use **Settings** or `/settings` to edit them.

## MCP servers and user skills

The **Settings** screen lets you enable MCP servers and edit their stdio command, arguments, and environment. GitHub, Playwright, and a configurable WhatsApp server are supplied as disabled presets; configure credentials or server paths before enabling one. Playwright MCP uses `~/.cache/loa/playwright-mcp` for its working and artifact output directory, outside synced project folders. Only MCP tools clearly named as read-only (`list`, `get`, `read`, `search`, `find`, `fetch`, `snapshot`, or `screenshot`) may run without approval; all other MCP calls require explicit approval showing the server, tool, and arguments. Custom-skill calls always wait for approval. MCP environment values are stored in the private LOA settings file.

The web server starts while MCP servers and skills initialize in the background. Chat and tool-related actions briefly wait for initialization, then ask you to retry if it is still running; failures are logged and reported. Requests that conflict with a running operation report that LOA is busy instead of waiting indefinitely. Saving settings reuses existing MCP connections unless their configuration changed.

Press Ctrl+C once to stop the web server. Shutdown cancels active chat work and signals LOA's MCP process groups; cleanup is time-limited, and LOA logs a warning if it cannot complete.

User skills live in `~/.loa/skills/`. Enabled `.md` files are appended to LOA's system prompt. A Markdown skill can set `command: /trigger` in YAML frontmatter; otherwise its filename maps to `/<skill-name>`. Start a chat message with that trigger to activate the skill instructions and pass the remaining text as the request. Available triggers appear as chat suggestions and on each skill card. Python skills map to their filename and must be enabled before use. They must export `register_tools()` returning `(tool_schemas, callables)`, where schemas use Ollama's `{"type":"function","function":{"name": ..., "description": ..., "parameters": ...}}` format and callables accept tool arguments as keyword arguments. Python skills are user code; enable only files you trust. Use `/mcp [name on|off|reload]` and `/skills [name on|off|reload]` in the terminal, or the corresponding controls in the web settings. The web settings also let you edit installed skill files and reload the skill state when saved.

## About Me and Brain memory

Settings are organized into General, About Me, Brain / Memory, Cronjobs, Skills, MCP Servers, and Image Gen tabs in both interfaces. Image Gen supports OpenAI `gpt-image-1`, Gemini native image generation, and local Janus-Pro, FLUX.2 Klein, and Sana Sprint. Janus is autoregressive; FLUX uses flow matching and Sana uses diffusion. OpenAI and Gemini keys can be saved in Settings or supplied through `OPENAI_API_KEY` and `GEMINI_API_KEY`. Generated PNGs are kept privately in `~/.loa/generated_images/` and appear inline in the WebUI.

Local models use the active or project `.venv`; `LOA_JANUS_PYTHON` can override the interpreter for all local image runners. FLUX uses `mlx-community/FLUX.2-Klein-4B-4bit` with MFLUX on Apple Silicon, at four steps. In the WebUI, select a PNG, JPEG, or WebP reference image (maximum 8 MB and 16 megapixels) and describe the edit; uploads require Image Gen to be enabled and the local FLUX provider selected. Sana uses `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` with Diffusers and PyTorch MPS at two steps, not the MLX checkpoint. The quality setting does not change the step count for FLUX or Sana. Check model licenses and any Hugging Face access conditions. Weights are downloaded to the Hugging Face cache on first use and require several gigabytes of free disk space; generation speed and memory usage depend on the machine. The readiness check verifies dependencies and memory, not whether weights are already cached.

Janus-Pro generates a native 384×384 image which LOA resizes and crops to the selected output size. New local Janus settings default to `deepseek-ai/Janus-Pro-1B`; `deepseek-community/Janus-Pro-1B` uses the Transformers implementation. On macOS, LOA checks memory before loading (at least 8 GiB estimated available for 1B, 3 GiB for FLUX, 2 GiB for Sana; 32 GiB physical and 24 GiB estimated available for Janus 7B) and stops local generation if estimated available memory drops below 2 GiB. These estimates are not a hard limit. LOA allows one local image generation at a time per process.

About Me profile details are saved locally and added to LOA's context. Brain entries are stored in a private local SQLite database with weighted tokens and FTS5 search; indexing, extracting chat facts, and retrieval use no cloud embedding or LLM calls. LOA can automatically remember explicit first-person preferences and searches relevant Brain entries before answering. Users can search, add, and delete entries in Settings.

The Chats manager can search, rename, and switch conversations. **Vectorize and delete** extracts facts and weighted keywords locally, saves them to Brain, then deletes that chat's raw history. It asks for confirmation first.

## Scheduled jobs

Create interval jobs such as `@every 15m` or five-field cron expressions in either Settings UI, or ask LOA to manage a job. Mark a job **One-time** (or use `one_time: true` / `run_once: true`) to disable and mark it completed after its first execution, including failed attempts; its execution history is retained. Jobs are persisted locally and run within the same process as LOA's web server; no system `crontab` is used. They execute while `--web` or the installed web service is running. LOA proposals to create, change, enable, delete, or run a job require the normal explicit approval step, which displays the schedule and target prompt/action and warns that enabled jobs run automatically in the background. Execution history is available in the Cronjobs tab. Any follow-on system action proposed by a scheduled prompt is logged but not executed without approval.

## Web search

Ask LOA to search for something online in ordinary language, for example, "Search the web for the latest Python release." LOA searches DuckDuckGo, uses the result snippets to prepare an answer, and shows the source links. Web pages, search results, files, MCP results, and other tool output are sandboxed as untrusted data. Suspected prompt injection triggers a security alert and requires approval for every later tool call in that turn. Credential and messaging database paths require approval before being read; shell commands always require approval.

## macOS shell actions

When a request needs a local action, LOA proposes a zsh command with a reason. Review it before choosing **Run command**. Canceling returns an explicit cancellation result and waits for your next instruction instead of retrying the action. Approved commands run in a login zsh with a 30-second timeout, and LOA uses the result to explain what happened or propose a next step. Use **Break Loop** at any time to cancel active work and return to ready state.

## Files and macOS apps

LOA can inspect text files with `read_file` and can propose file creation or edits with a preview. File writes and AppleScript app automation always wait for your approval. Completed actions and tool results are collapsed in the transcript; click a summary to inspect the details and output.

## Saved chats

Chats are saved as JSON files under `~/Library/Application Support/LOA/chats`. LOA opens to a Home screen with **New Chat**, **Continue Last Chat**, and the five most recent conversations; opening Home does not create an empty chat. A chat is created only when you choose **New Chat** or send your first message. Use **Home** or `/home` to return at any time, and use **Chats**, `/chats`, or `/new` to manage and switch conversations. LOA keeps a compact local index of summaries and keywords and can retrieve a short relevant excerpt from a previous chat when asked; it does not add full old transcripts to the prompt.

## Use the chat from a phone

Start LOA's mobile web chat on your Mac:

```sh
./dist/loa --web
```

The first time, LOA asks in the terminal for a web password. It is saved as a salted password hash in the user's application support directory; use `./dist/loa --web --reset-web-password` to replace it. The terminal prints the local address. Open it on your phone while it is on the same Wi-Fi network as the Mac, then sign in with the saved password. For convenience, append `#pass=<URL-encoded-password>` to the URL for automatic sign-in. The browser removes the fragment before login; LOA allows automatic login over HTTPS, localhost, or a request received from a Tailscale address. Tailscale encrypts its network traffic; ordinary unencrypted LAN connections are not allowed to use the automatic password link. Configure `--certfile` and `--keyfile` for HTTPS outside Tailscale. Assistant settings are saved separately, loaded automatically, and are not requested again on later visits.

The server uses an HTTP-only session cookie after password sign-in. Use it only on a trusted private network: HTTP does not encrypt the password or conversation in transit. Stop the server with `Ctrl+C`.

## Run the web UI as a background service

Register LOA as a macOS LaunchAgent:

```sh
./dist/loa --install-web-service
```

The first run asks you to create the web password if one has not been set yet. The service then starts immediately and automatically on your next sign-in. It does not open a browser. Check it or remove it with:

```sh
./dist/loa --web-service-status
./dist/loa --uninstall-web-service
```

Use `--host`, `--port`, `--certfile`, and `--keyfile` when installing to customize the service. Service logs are saved under the LOA Application Support directory.

## Model memory use

Ollama loads the selected model the first time LOA sends a chat request; starting the UI or web service does not load a model. LOA asks Ollama to keep it loaded for five minutes after the latest request. This keeps consecutive messages responsive while releasing its memory shortly after use.

The web chat has a flat transcript with user messages aligned right and LOA responses left; messages have no chat-bubble backgrounds. Use `./dist/loa --help` to see all web host, port, service, password, and TLS options. The terminal chat has no Textual footer or command-palette shortcut.
