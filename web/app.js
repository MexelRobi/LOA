"use strict";

const elements = {
  appTitle: document.querySelector("#app-title"),
  chatPanel: document.querySelector("#chat-panel"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chat-form"),
  messageInput: document.querySelector("#message-input"),
  sendButton: document.querySelector("#send-button"),
  connectionStatus: document.querySelector("#connection-status"),
  settingsToggle: document.querySelector("#settings-toggle"),
  chatsToggle: document.querySelector("#chats-toggle"),
  logoutButton: document.querySelector("#logout-button"),
  settingsClose: document.querySelector("#settings-close"),
  settingsPanel: document.querySelector("#settings-panel"),
  chatsPanel: document.querySelector("#chats-panel"),
  chatsClose: document.querySelector("#chats-close"),
  savedChats: document.querySelector("#saved-chats"),
  chatTitle: document.querySelector("#chat-title"),
  chatsStatus: document.querySelector("#chats-status"),
  chatOpen: document.querySelector("#chat-open"),
  chatNew: document.querySelector("#chat-new"),
  chatRename: document.querySelector("#chat-rename"),
  chatDelete: document.querySelector("#chat-delete"),
  settingsHeading: document.querySelector("#settings-heading"),
  settingsForm: document.querySelector("#settings-form"),
  settingsStatus: document.querySelector("#settings-status"),
  assistantName: document.querySelector("#assistant-name"),
  modelName: document.querySelector("#model-name"),
  ollamaHost: document.querySelector("#ollama-host"),
  systemPrompt: document.querySelector("#system-prompt"),
  resetPrompt: document.querySelector("#reset-prompt"),
  resetHost: document.querySelector("#reset-host"),
  approveCommand: document.querySelector("#approve-command"),
  cancelCommand: document.querySelector("#cancel-command"),
  commandControls: document.querySelector("#command-controls"),
  chatControls: document.querySelector("#chat-controls"),
  statusLine: document.querySelector("#status-line"),
  loginPanel: document.querySelector("#login-panel"),
  loginForm: document.querySelector("#login-form"),
  webPassword: document.querySelector("#web-password"),
  loginStatus: document.querySelector("#login-status"),
};

let configured = false;
let defaultPrompt = "";
let defaultOllamaHost = "http://127.0.0.1:11434";
let commandPending = false;
let activeChatId = null;
let selectedChatId = null;
let renderedToolCount = 0;

function setStatus(element, message) {
  element.textContent = message;
}

async function api(path, payload) {
  const response = await fetch(path, {
    method: payload === undefined ? "GET" : "POST",
    headers: {
      ...(payload === undefined ? {} : { "Content-Type": "application/json" }),
    },
    ...(payload === undefined ? {} : { body: JSON.stringify(payload) }),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || `Request failed with status ${response.status}.`);
  }
  return result;
}

function appendMessage(message) {
  const item = document.createElement("article");
  item.className = `message ${message.role}`;
  if (message.role === "tool" && message.ui) {
    const block = document.createElement("details");
    const summary = document.createElement("summary");
    const details = document.createElement("pre");
    const label = message.ui.summary || "Completed tool";
    summary.textContent = label.startsWith("TASK:") ? label : `TASK: ${label}`;
    details.textContent = message.ui.details || message.content || "";
    block.append(summary, details);
    item.append(block);
    elements.messages.append(item);
    elements.messages.scrollTop = elements.messages.scrollHeight;
    return;
  }
  if (message.role === "tool") {
    const block = document.createElement("details");
    const summary = document.createElement("summary");
    const details = document.createElement("pre");
    summary.textContent = `TASK: ${message.tool_name || "Tool result"}`;
    details.textContent = message.content || "";
    block.append(summary, details);
    item.append(block);
    elements.messages.append(item);
    elements.messages.scrollTop = elements.messages.scrollHeight;
    return;
  }
  const author = {
    user: "You",
    assistant: elements.statusLine.textContent.split(" | ")[0] || "LOA",
  }[message.role];
  item.textContent = author ? `${author}: ${message.content}` : message.content;
  elements.messages.append(item);

  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function appendTask(task) {
  const item = document.createElement("article");
  item.className = "message tool";
  const block = document.createElement("details");
  const summary = document.createElement("summary");
  const details = document.createElement("pre");
  summary.textContent = `TASK: ${task.label || "Completed tool"}`;
  details.textContent = task.details || "";
  block.append(summary, details);
  item.append(block);
  elements.messages.append(item);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function renderReplyTasks(tasks, transcript) {
  for (const task of tasks || []) {
    appendTask(task);
  }
  renderedToolCount = (transcript || []).filter(
    (message) => message.role === "tool",
  ).length;
}

function pendingActionText(action) {
  const sections = [`Reason: ${action.reason || "Review this action before approving it."}`];
  if (action.type === "shell_command") {
    sections.push(`Command:\n${action.command || ""}`);
  } else if (action.type === "applescript") {
    sections.push(`AppleScript:\n${action.script || ""}`);
  } else if (action.type === "file_write") {
    sections.push(
      `File: ${action.path || ""}\nMode: ${action.mode || ""}\nPreview:\n${action.preview || ""}`,
    );
  }
  return sections.join("\n\n");
}

function showPendingAction(action) {
  commandPending = true;
  const item = document.createElement("article");
  item.className = "message system";
  const block = document.createElement("details");
  block.open = true;
  const summary = document.createElement("summary");
  summary.textContent = `Approval required: ${action.type || "action"}`;
  const details = document.createElement("pre");
  details.textContent = pendingActionText(action);
  block.append(summary, details);
  item.append(block);
  elements.messages.append(item);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  elements.chatControls.hidden = true;
  elements.commandControls.hidden = false;
  setBusy(false);
}

function setBusy(busy) {
  elements.sendButton.disabled = busy || !configured || commandPending;
  elements.messageInput.disabled = busy || !configured || commandPending;
  elements.settingsToggle.disabled = busy || commandPending;
  elements.chatsToggle.disabled = busy || commandPending;
  elements.logoutButton.disabled = busy || commandPending;
}

function showSettings(config, isSetup, openPanel = true) {
  if (typeof config.default_prompt === "string") {
    defaultPrompt = config.default_prompt;
  }
  if (typeof config.default_ollama_host === "string") {
    defaultOllamaHost = config.default_ollama_host;
  }
  if (openPanel) {
    elements.settingsPanel.hidden = false;
    elements.chatPanel.hidden = true;
  }
  elements.settingsHeading.textContent = isSetup ? "Set up your assistant" : "Assistant settings";
  elements.settingsClose.hidden = isSetup;
  elements.assistantName.value = config.name;
  elements.modelName.value = config.model;
  elements.ollamaHost.value = config.ollama_host || defaultOllamaHost;
  elements.ollamaHost.placeholder = defaultOllamaHost;
  elements.systemPrompt.value = config.prompt;
  if (isSetup) {
    elements.settingsPanel.hidden = false;
    elements.chatPanel.hidden = true;
    elements.settingsToggle.hidden = true;
    elements.chatForm.hidden = true;
    elements.logoutButton.hidden = true;
  }
}

async function loadAuthenticatedApplication() {
  try {
    const config = await api("/api/config");
    configured = config.configured;
    elements.statusLine.textContent = `${config.name} | ${config.model}`;
    showSettings(config, !configured, !configured);
    elements.logoutButton.hidden = false;
    if (configured) {
      elements.chatPanel.hidden = false;
      elements.settingsPanel.hidden = true;
      elements.chatsPanel.hidden = true;
      elements.settingsToggle.hidden = false;
      elements.chatsToggle.hidden = false;
      elements.chatForm.hidden = false;
    }
    if (configured) {
      appendMessage({ role: "system", content: "LOA is ready. Type /help to see available commands." });
      const history = await api("/api/history");
      activeChatId = history.active_chat_id;
      for (const message of history.transcript) {
        appendMessage(message);
      }
      renderedToolCount = history.transcript.filter(
        (message) => message.role === "tool",
      ).length;
      if (history.pending_action) {
        showPendingAction(history.pending_action);
      }
      setBusy(false);
    } else {
      setStatus(elements.settingsStatus, "Save these settings to start chatting.");
    }
  } catch (error) {
    setStatus(elements.connectionStatus, error.message);
    setBusy(true);
  }
}

async function refreshChatList() {
  const result = await api("/api/chats");
  activeChatId = result.active_chat_id;
  elements.savedChats.replaceChildren();
  for (const chat of result.chats) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chat-choice";
    button.dataset.chatId = chat.id;
    button.setAttribute("aria-pressed", String(chat.id === selectedChatId));
    const title = document.createElement("strong");
    title.textContent = chat.title;
    const metadata = document.createElement("small");
    metadata.textContent = `${chat.updated_at.slice(0, 10)} — ${chat.summary}`;
    button.append(title, metadata);
    elements.savedChats.append(button);
  }
  const hasSelection = result.chats.some((chat) => chat.id === selectedChatId);
  if (!hasSelection) {
    selectedChatId = activeChatId;
    for (const button of elements.savedChats.querySelectorAll("[data-chat-id]")) {
      button.setAttribute("aria-pressed", String(button.dataset.chatId === selectedChatId));
    }
  }
  const selected = result.chats.find((chat) => chat.id === selectedChatId);
  elements.chatTitle.value = selected?.title || "";
  elements.chatOpen.disabled = !selected;
  elements.chatRename.disabled = !selected;
  elements.chatDelete.disabled = !selected;
}

async function showChats() {
  try {
    await refreshChatList();
    elements.chatPanel.hidden = true;
    elements.settingsPanel.hidden = true;
    elements.chatsPanel.hidden = false;
    setStatus(elements.chatsStatus, "");
  } catch (error) {
    setStatus(elements.connectionStatus, error.message);
  }
}

function showChatTranscript(result) {
  activeChatId = result.active_chat_id;
  commandPending = false;
  renderedToolCount = 0;
  elements.messages.replaceChildren();
  for (const message of result.transcript || []) {
    appendMessage(message);
  }
  renderedToolCount = (result.transcript || []).filter(
    (message) => message.role === "tool",
  ).length;
  if (!result.transcript?.length) {
    appendMessage({ role: "system", content: "New chat started." });
  }
  elements.chatsPanel.hidden = true;
  elements.settingsPanel.hidden = true;
  elements.chatPanel.hidden = false;
  elements.chatControls.hidden = false;
  elements.commandControls.hidden = true;
  setBusy(false);
  if (result.pending_action) {
    showPendingAction(result.pending_action);
  }
  elements.messageInput.focus();
}

async function loadApplication() {
  try {
    const auth = await api("/auth/status");
    if (auth.password_configured) {
      const session = await api("/auth/session");
      if (session.authenticated) {
        await loadAuthenticatedApplication();
        return;
      }
    }
    elements.loginPanel.hidden = false;
    elements.chatPanel.hidden = true;
    elements.chatForm.hidden = true;
    elements.settingsPanel.hidden = true;
    elements.chatsPanel.hidden = true;
    setBusy(true);
    if (!auth.password_configured) {
      setStatus(elements.loginStatus, "Set a web password in the LOA terminal, then reload this page.");
      elements.webPassword.disabled = true;
      elements.loginForm.querySelector("button").disabled = true;
      return;
    }
    elements.webPassword.focus();
  } catch (error) {
    setStatus(elements.loginStatus, error.message);
  }
}

elements.loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const password = elements.webPassword.value;
  setStatus(elements.loginStatus, "Signing in...");
  try {
    await api("/auth/login", { password });
    elements.webPassword.value = "";
    elements.loginPanel.hidden = true;
    await loadAuthenticatedApplication();
  } catch (error) {
    setStatus(elements.loginStatus, error.message);
  }
});

elements.settingsToggle.addEventListener("click", async () => {
  try {
    showSettings(await api("/api/config"), false);
  } catch (error) {
    setStatus(elements.connectionStatus, error.message);
  }
});

elements.chatsToggle.addEventListener("click", showChats);

elements.chatsClose.addEventListener("click", () => {
  elements.chatsPanel.hidden = true;
  elements.chatPanel.hidden = !configured;
});

elements.savedChats.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-chat-id]");
  if (!button) {
    return;
  }
  selectedChatId = button.dataset.chatId;
  for (const choice of elements.savedChats.querySelectorAll("[data-chat-id]")) {
    choice.setAttribute("aria-pressed", String(choice === button));
  }
  const title = button.querySelector("strong");
  elements.chatTitle.value = title?.textContent || "";
});

elements.chatOpen.addEventListener("click", async () => {
  if (!selectedChatId) {
    return;
  }
  try {
    showChatTranscript(await api("/api/chats", {
      action: "open",
      chat_id: selectedChatId,
    }));
  } catch (error) {
    setStatus(elements.chatsStatus, error.message);
  }
});

elements.chatNew.addEventListener("click", async () => {
  try {
    showChatTranscript(await api("/api/chats", { action: "new" }));
    selectedChatId = activeChatId;
  } catch (error) {
    setStatus(elements.chatsStatus, error.message);
  }
});

elements.chatRename.addEventListener("click", async () => {
  if (!selectedChatId) {
    return;
  }
  try {
    const result = await api("/api/chats", {
      action: "rename",
      chat_id: selectedChatId,
      title: elements.chatTitle.value,
    });
    await refreshChatList();
    setStatus(elements.chatsStatus, "Chat renamed.");
    if (result.active_chat_id === selectedChatId) {
      activeChatId = result.active_chat_id;
    }
  } catch (error) {
    setStatus(elements.chatsStatus, error.message);
  }
});

elements.chatDelete.addEventListener("click", async () => {
  if (!selectedChatId) {
    return;
  }
  try {
    const result = await api("/api/chats", {
      action: "delete",
      chat_id: selectedChatId,
    });
    showChatTranscript(result);
    selectedChatId = result.active_chat_id;
  } catch (error) {
    setStatus(elements.chatsStatus, error.message);
  }
});

elements.resetPrompt.addEventListener("click", () => {
  elements.systemPrompt.value = defaultPrompt;
  setStatus(elements.settingsStatus, "Default prompt restored. Save settings to keep it.");
});

elements.resetHost.addEventListener("click", () => {
  elements.ollamaHost.value = defaultOllamaHost;
  setStatus(elements.settingsStatus, "Default host restored. Save settings to keep it.");
});

elements.settingsClose.addEventListener("click", () => {
  elements.settingsPanel.hidden = true;
  elements.chatPanel.hidden = !configured;
});

elements.settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(elements.settingsStatus, "Saving...");
  try {
    const config = await api("/api/settings", {
      name: elements.assistantName.value,
      model: elements.modelName.value,
      prompt: elements.systemPrompt.value,
      ollama_host: elements.ollamaHost.value,
    });
    configured = true;
    elements.statusLine.textContent = `${config.name} | ${elements.modelName.value.trim()}`;
    elements.settingsToggle.hidden = false;
    elements.chatsToggle.hidden = false;
    elements.settingsPanel.hidden = true;
    elements.chatsPanel.hidden = true;
    elements.chatPanel.hidden = false;
    elements.chatForm.hidden = false;
    setBusy(false);
    elements.messageInput.focus();
  } catch (error) {
    setStatus(elements.settingsStatus, error.message);
  }
});

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message || !configured || commandPending) {
    return;
  }
  elements.messageInput.value = "";
  appendMessage({ role: "user", content: message });
  const localCommand = message.toLowerCase();
  if (localCommand === "/help") {
    appendMessage({
      role: "system",
      content:
        "Commands:\n" +
        "/settings - edit and save assistant settings\n" +
        "/exit - close LOA\n" +
        "Ask LOA to search the web when you need online information.",
    });
    return;
  }
  if (localCommand === "/chats") {
    await showChats();
    return;
  }
  if (localCommand === "/new") {
    try {
      showChatTranscript(await api("/api/chats", { action: "new" }));
      selectedChatId = activeChatId;
    } catch (error) {
      appendMessage({ role: "system", content: error.message });
    }
    return;
  }
  if (localCommand === "/settings") {
    try {
      showSettings(await api("/api/config"), false);
    } catch (error) {
      appendMessage({ role: "system", content: error.message });
    }
    return;
  }
  if (["/exit", "/quit", "exit", "quit"].includes(localCommand)) {
    elements.logoutButton.click();
    return;
  }

  setBusy(true);
  setStatus(elements.connectionStatus, "Thinking...");
  try {
    const result = await api("/api/chat", { message });
    renderReplyTasks(result.tasks, result.transcript);
    if (result.reply) {
      appendMessage({
        role: "assistant",
        content: result.reply,
      });
    }
    if (result.action || result.command) {
      showPendingAction(
        result.action || {
          type: "shell_command",
          command: result.command,
          reason: result.command_reason,
        },
      );
    }
    setStatus(elements.connectionStatus, "");
  } catch (error) {
    appendMessage({ role: "system", content: error.message });
    setStatus(elements.connectionStatus, "");
  } finally {
    setBusy(false);
  }
});

async function respondToCommand(approved) {
  elements.commandControls.hidden = true;
  commandPending = false;
  setBusy(true);
  setStatus(elements.connectionStatus, approved ? "Running command..." : "Cancelling command...");
  try {
    const result = await api("/api/command", { approved });
    if (result.action_task) {
      appendTask(result.action_task);
    }
    renderReplyTasks(result.tasks, result.transcript);
    if (result.reply) {
      appendMessage({
        role: "assistant",
        content: result.reply,
      });
    }
    if (result.assistant_error) {
      appendMessage({ role: "system", content: result.assistant_error });
    }
    if (result.action || result.command) {
      showPendingAction(
        result.action || {
          type: "shell_command",
          command: result.command,
          reason: result.command_reason,
        },
      );
    }
    setStatus(elements.connectionStatus, "");
  } catch (error) {
    appendMessage({ role: "system", content: error.message });
    setStatus(elements.connectionStatus, "");
  } finally {
    elements.chatControls.hidden = commandPending;
    elements.commandControls.hidden = !commandPending;
    setBusy(false);
  }
}

elements.approveCommand.addEventListener("click", () => respondToCommand(true));
elements.cancelCommand.addEventListener("click", () => respondToCommand(false));

elements.chatForm.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

elements.logoutButton.addEventListener("click", async () => {
  try {
    await api("/auth/logout", {});
  } catch (error) {
    setStatus(elements.connectionStatus, `Sign out failed: ${error.message}`);
  } finally {
    configured = false;
    commandPending = false;
    elements.messages.replaceChildren();
    elements.loginPanel.hidden = false;
    elements.chatPanel.hidden = true;
    elements.settingsPanel.hidden = true;
    elements.chatsPanel.hidden = true;
    elements.chatForm.hidden = true;
    elements.settingsToggle.hidden = true;
    elements.chatsToggle.hidden = true;
    elements.logoutButton.hidden = true;
    setBusy(true);
    if (!elements.connectionStatus.textContent.startsWith("Sign out failed:")) {
      setStatus(elements.loginStatus, "You have signed out.");
    }
  }
});

loadApplication();
