# LOA

LOA is a local assistant that uses Ollama for chat and DuckDuckGo for requested web searches.

## Build and run

Build the standalone macOS executable:

```sh
./compile
./dist/loa
```

The first launch asks for an Ollama model and connection settings. The assistant name is always LOA, and its base system prompt is fixed and identifies MexelRobi as its creator. Existing custom names and prompts are replaced with these fixed values; other settings are retained. Settings are saved in the user's application support directory. Later launches open the chat directly; use **Settings** or `/settings` to edit the remaining settings.

## MCP servers and user skills

The **Settings** screen lets you enable MCP servers and edit their stdio command, arguments, and environment. GitHub, Playwright, and a configurable WhatsApp server are supplied as disabled presets; configure credentials or server paths before enabling one. Playwright MCP uses `~/.cache/loa/playwright-mcp` for its working and artifact output directory, outside synced project folders. Only MCP tools clearly named as read-only (`list`, `get`, `read`, `search`, `find`, `fetch`, `snapshot`, or `screenshot`) may run without approval; all other MCP calls require explicit approval showing the server, tool, and arguments. Custom-skill calls always wait for approval. MCP environment values are stored in the private LOA settings file.

The web server starts while MCP servers and skills initialize in the background. Chat and tool-related actions briefly wait for initialization, then ask you to retry if it is still running; failures are logged and reported. Requests that conflict with a running operation report that LOA is busy instead of waiting indefinitely. Saving settings reuses existing MCP connections unless their configuration changed.

Press Ctrl+C once to stop the web server. Shutdown cancels active chat work and signals LOA's MCP process groups; cleanup is time-limited, and LOA logs a warning if it cannot complete.

User skills live in `~/.loa/skills/`. Chat system prompts include the current catalog of available tools (including MCP tools), installed skill names and states, enabled Markdown skill instructions, About Me profile, and all Brain memories. LOA is instructed to use or follow them only when directly relevant to the user's request. A Markdown skill can set `command: /trigger` in YAML frontmatter; otherwise its filename maps to `/<skill-name>`. Start a chat message with that trigger to activate the skill instructions and pass the remaining text as the request. Available triggers appear as chat suggestions and on each skill card. Python skills map to their filename and must be enabled before use. They must export `register_tools()` returning `(tool_schemas, callables)`, where schemas use Ollama's `{"type":"function","function":{"name": ..., "description": ..., "parameters": ...}}` format and callables accept tool arguments as keyword arguments. Python skills are user code; enable only files you trust. Use `/mcp [name on|off|reload]` and `/skills [name on|off|reload]` in the terminal, or the corresponding controls in the web settings. The web settings also let you edit installed skill files and reload the skill state when saved.

## About Me and Brain memory

Settings are organized into General, About Me, Brain / Memory, Cronjobs, Skills, MCP Servers, and Image Gen tabs in both interfaces. Image Gen supports OpenAI `gpt-image-1`, Google Gemini image generation, and local Apple Core ML Stable Diffusion 2.1. OpenAI and Gemini keys can be saved in Settings or supplied through `OPENAI_API_KEY` and `GEMINI_API_KEY`. Generated PNGs are kept privately in `~/.loa/generated_images/` and appear inline in the WebUI.

The WebUI's **Add File** control is available regardless of the selected chat model or image-generation settings. It accepts images and files up to 8 MB; images are sent through Ollama's native image input, while text files, PDFs, and supported Office documents are provided to the model as extracted text. Apple Core ML Stable Diffusion uses the public `apple/coreml-stable-diffusion-2-1-base` model at 512×512 and resizes/crops the result to the selected output size; custom output width and height can each be set up to 4096 pixels, but this does not add native detail beyond the 512×512 generation. Its low, medium, and high quality settings use 20, 40, and 100 denoising steps, respectively; the WebUI shows intermediate snapshots and the current step while generating. High quality takes longer and may refine denoising, but it does not increase native resolution or guarantee sharper edges. Since SD 2.1 follows English much more reliably, LOA instructs the assistant to compose image prompts in English for this provider. It uses Apple's Core ML pipeline with reduced-memory loading and enforces a 1.9 GiB process-RSS limit. The first use downloads about 4.3 GiB of model data; the model is licensed under OpenRAIL++, so review its use restrictions. The standalone macOS build bundles the Swift runner and writes Apple's MIT license beside `dist/loa`; source/development runs build the runner on first readiness check. The readiness check verifies the runner and available memory, not whether weights are already cached.

Local Apple Stable Diffusion checks available memory before loading and limits its runner to 1.9 GiB of process RSS. LOA allows one local image generation at a time per process.

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

To hide a chat, select it in **Chats** and choose **Hide**. The first time, set and confirm a 4–12 digit PIN. Open **Hidden Chats** on Home and enter the PIN to view, open, or unhide those chats; returning Home locks the list again. Hidden chats are excluded from normal chat lists and past-chat recall. The PIN is stored as a salted verifier, but chat files and attachments are not encrypted; someone with access to the local files can still read them. Existing Brain memories are not removed when a chat is hidden. Keep the PIN safe: there is no PIN recovery in the app.

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

When `mpkgexec` installs an update, it uses `launchctl` to stop an existing LOA LaunchAgent and re-register it afterward using its existing plist, preserving its executable path and service settings. An installed but unloaded agent is started after the update; if no agent exists, none is created. If the install fails after stopping a running agent, the script attempts to restore it. Because `mpkg` copies its payload before running `mpkgexec`, this stop happens after that initial copy, but before `mpkgexec` installs the downloaded release.

## Model memory use

Ollama loads the selected model the first time LOA sends a chat request; starting the UI or web service does not load a model. LOA asks Ollama to keep it loaded for five minutes after the latest request. This keeps consecutive messages responsive while releasing its memory shortly after use.

The web chat has a flat transcript with user messages aligned right and LOA responses left; messages have no chat-bubble backgrounds. Use `./dist/loa --help` to see all web host, port, service, password, and TLS options. The terminal chat has no Textual footer or command-palette shortcut.
