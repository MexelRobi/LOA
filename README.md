# LOA

LOA is a local assistant that uses Ollama for chat and DuckDuckGo for requested web searches.

## Build and run

Build the standalone macOS executable:

```sh
./compile
./dist/loa
```

The first launch asks for an assistant name, an Ollama model, and a system prompt. Settings are saved in the user's application support directory. Later launches open the chat directly; use **Settings** or `/settings` to edit them.

## Web search

Ask LOA to search for something online in ordinary language, for example, "Search the web for the latest Python release." LOA searches DuckDuckGo, uses the result snippets to prepare an answer, and shows the source links. Search results are untrusted content; shell commands still require explicit approval.

## macOS shell actions

When a request needs a local action, LOA proposes a zsh command with a reason. Review it before choosing **Run command**; canceling also returns control to the assistant so it can suggest another approach. Approved commands run in a login zsh with a 30-second timeout, and LOA uses the result to explain what happened or propose a next step. For example, ask LOA to find a named application and open it: it can first propose a Spotlight `mdfind` query, then use an approved `open -a` command based on the result.

## Files and macOS apps

LOA can inspect text files with `read_file` and can propose file creation or edits with a preview. File writes and AppleScript app automation always wait for your approval. Completed actions and tool results are collapsed in the transcript; click a summary to inspect the details and output.

## Saved chats

Chats are saved as JSON files under `~/Library/Application Support/LOA/chats`. Use **Chats**, `/chats`, or `/new` to manage conversations, switch chats, rename or delete them, and start a new one. LOA keeps a compact local index of summaries and keywords and can retrieve a short relevant excerpt from a previous chat when asked; it does not add full old transcripts to the prompt.

## Use the chat from a phone

Start LOA's mobile web chat on your Mac:

```sh
./dist/loa --web
```

The first time, LOA asks in the terminal for a web password. It is saved as a salted password hash in the user's application support directory; use `./dist/loa --web --reset-web-password` to replace it. The terminal prints the local address. Open it on your phone while it is on the same Wi-Fi network as the Mac, then sign in with the saved password. Assistant settings are saved separately, loaded automatically, and are not requested again on later visits.

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

The web chat mirrors the terminal layout: centered app title, assistant/model status, a bordered transcript, and a separate message input and action row. Use `./dist/loa --help` to see all web host, port, service, password, and TLS options. The terminal chat has no Textual footer or command-palette shortcut.
