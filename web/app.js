"use strict";

let fragmentPassword = null;
const fragmentParams = new URLSearchParams(window.location.hash.slice(1));
if (fragmentParams.has("pass")) {
  fragmentPassword = fragmentParams.get("pass") || null;
  history.replaceState(
    history.state,
    "",
    `${window.location.pathname}${window.location.search}`,
  );
}

const elements = {
  appTitle: document.querySelector("#app-title"),
  homePanel: document.querySelector("#home-panel"),
  homeNewChat: document.querySelector("#home-new-chat"),
  homeContinueChat: document.querySelector("#home-continue-chat"),
  homeHiddenChats: document.querySelector("#home-hidden-chats"),
  homeSettings: document.querySelector("#home-settings"),
  homeLogout: document.querySelector("#home-logout"),
  homeRecentChats: document.querySelector("#home-recent-chats"),
  homeStatus: document.querySelector("#home-status"),
  chatPanel: document.querySelector("#chat-panel"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chat-form"),
  messageInput: document.querySelector("#message-input"),
  chatFiles: document.querySelector("#chat-files"),
  addFileButton: document.querySelector("#add-file-button"),
  chatFileNames: document.querySelector("#chat-file-names"),
  clearChatFiles: document.querySelector("#clear-chat-files"),
  skillCommandHints: document.querySelector("#skill-command-hints"),
  sendButton: document.querySelector("#send-button"),
  breakLoopButton: document.querySelector("#break-loop-button"),
  homeToggle: document.querySelector("#home-toggle"),
  connectionStatus: document.querySelector("#connection-status"),
  settingsToggle: document.querySelector("#settings-toggle"),
  chatsToggle: document.querySelector("#chats-toggle"),
  logoutButton: document.querySelector("#logout-button"),
  settingsClose: document.querySelector("#settings-close"),
  settingsPanel: document.querySelector("#settings-panel"),
  chatsPanel: document.querySelector("#chats-panel"),
  chatsHeading: document.querySelector("#chats-heading"),
  chatsClose: document.querySelector("#chats-close"),
  savedChats: document.querySelector("#saved-chats"),
  chatTitle: document.querySelector("#chat-title"),
  chatSearch: document.querySelector("#chat-search"),
  chatsStatus: document.querySelector("#chats-status"),
  chatOpen: document.querySelector("#chat-open"),
  chatNew: document.querySelector("#chat-new"),
  chatRename: document.querySelector("#chat-rename"),
  chatHide: document.querySelector("#chat-hide"),
  hiddenPinDialog: document.querySelector("#hidden-pin-dialog"),
  hiddenPinForm: document.querySelector("#hidden-pin-form"),
  hiddenPinHeading: document.querySelector("#hidden-pin-heading"),
  hiddenPin: document.querySelector("#hidden-pin"),
  hiddenPinConfirm: document.querySelector("#hidden-pin-confirm"),
  hiddenPinConfirmGroup: document.querySelector("#hidden-pin-confirm-group"),
  hiddenPinStatus: document.querySelector("#hidden-pin-status"),
  hiddenPinCancel: document.querySelector("#hidden-pin-cancel"),
  chatDelete: document.querySelector("#chat-delete"),
  chatVectorize: document.querySelector("#chat-vectorize"),
  settingsHeading: document.querySelector("#settings-heading"),
  settingsForm: document.querySelector("#settings-form"),
  settingsStatus: document.querySelector("#settings-status"),
  modelName: document.querySelector("#model-name"),
  ollamaHost: document.querySelector("#ollama-host"),
  imageGenEnabled: document.querySelector("#image-gen-enabled"),
  imageGenProvider: document.querySelector("#image-gen-provider"),
  imageGenModel: document.querySelector("#image-gen-model"),
  imageGenApiKey: document.querySelector("#image-gen-api-key"),
  imageGenKeyStatus: document.querySelector("#image-gen-key-status"),
  imageGenClearKey: document.querySelector("#image-gen-clear-key"),
  imageGenSize: document.querySelector("#image-gen-size"),
  imageGenQuality: document.querySelector("#image-gen-quality"),
  imageGenCheck: document.querySelector("#image-gen-check"),
  imageGenStatus: document.querySelector("#image-gen-status"),
  imageLightbox: document.querySelector("#image-lightbox"),
  imageLightboxImage: document.querySelector("#image-lightbox-image"),
  imageLightboxClose: document.querySelector("#image-lightbox-close"),
  mcpServers: document.querySelector("#mcp-servers"),
  mcpControls: document.querySelector("#mcp-controls"),
  refreshMcpControls: document.querySelector("#refresh-mcp-controls"),
  skillsList: document.querySelector("#skills-list"),
  skillEditor: document.querySelector("#skill-editor"),
  skillEditorName: document.querySelector("#skill-editor-name"),
  skillEditorDescription: document.querySelector("#skill-editor-description"),
  skillEditorContent: document.querySelector("#skill-editor-content"),
  saveSkillEdit: document.querySelector("#save-skill-edit"),
  cancelSkillEdit: document.querySelector("#cancel-skill-edit"),
  skillsStatus: document.querySelector("#skills-status"),
  newSkillName: document.querySelector("#new-skill-name"),
  newSkillContent: document.querySelector("#new-skill-content"),
  createSkill: document.querySelector("#create-skill"),
  reloadSkills: document.querySelector("#reload-skills"),
  resetHost: document.querySelector("#reset-host"),
  settingsTabs: document.querySelector(".settings-tabs"),
  profileName: document.querySelector("#about-name"),
  profilePreferences: document.querySelector("#about-preferences"),
  profileHardware: document.querySelector("#about-hardware"),
  profileRoutines: document.querySelector("#about-routines"),
  profileGoals: document.querySelector("#about-goals"),
  profileNotes: document.querySelector("#about-notes"),
  saveProfile: document.querySelector("#save-profile"),
  profileStatus: document.querySelector("#profile-status"),
  brainSearch: document.querySelector("#brain-search"),
  brainEntries: document.querySelector("#brain-entries"),
  brainContent: document.querySelector("#brain-content"),
  brainCategory: document.querySelector("#brain-category"),
  brainAdd: document.querySelector("#brain-add"),
  brainStatus: document.querySelector("#brain-status"),
  cronjobsList: document.querySelector("#cronjobs-list"),
  cronName: document.querySelector("#cron-name"),
  cronSchedule: document.querySelector("#cron-schedule"),
  cronPayload: document.querySelector("#cron-payload"),
  cronActionType: document.querySelector("#cron-action-type"),
  cronAutoApprove: document.querySelector("#cron-auto-approve"),
  cronOneTime: document.querySelector("#cron-one-time"),
  cronSave: document.querySelector("#cron-save"),
  cronCancelEdit: document.querySelector("#cron-cancel-edit"),
  cronStatus: document.querySelector("#cron-status"),
  cronHistory: document.querySelector("#cron-history"),
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
let chatBusy = false;
let defaultOllamaHost = "http://127.0.0.1:11434";
let commandPending = false;
let loopStopped = false;
let activeChatId = null;
let selectedChatId = null;
let hiddenChatsMode = false;
let pinConfigured = false;
let renderedToolCount = 0;
let imageProgressCard = null;
let currentSkillStates = {};
let currentSkillCommands = [];
let editingSkillName = null;
let savedChatEntries = [];
let editingCronJobId = null;

function setStatus(element, message) {
  element.textContent = message;
}

function setBreakLoopVisible(visible) {
  elements.breakLoopButton.classList.toggle("is-visible", visible);
  elements.breakLoopButton.setAttribute("aria-hidden", String(!visible));
  elements.breakLoopButton.disabled = !visible;
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function highlightCode(code) {
  const keywords = /\b(?:and|as|async|await|break|case|catch|class|const|continue|def|del|elif|else|except|export|extends|false|finally|for|from|function|if|import|in|let|new|null|of|or|pass|print|raise|return|self|static|switch|this|throw|true|try|var|while|with|yield)\b/;
  const tokenPattern = /(\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|\b\d+(?:\.\d+)?\b|\b[A-Za-z_$][\w$]*\b/g;
  return code.replace(tokenPattern, (token, comment, string) => {
    let tokenClass = "";
    if (comment) {
      tokenClass = "token-comment";
    } else if (string) {
      tokenClass = "token-string";
    } else if (/^\d/.test(token)) {
      tokenClass = "token-number";
    } else if (keywords.test(token)) {
      tokenClass = "token-keyword";
    }
    return tokenClass ? `<span class="${tokenClass}">${token}</span>` : token;
  });
}

function renderInlineMarkdown(escapedText, tokens = []) {
  const protect = (html) => {
    const token = `<loa-markdown-token-${tokens.length}>`;
    tokens.push(html);
    return token;
  };
  let html = escapedText.replace(/`([^`\n]+)`/g, (_match, code) =>
    protect(`<code class="inline-code">${code}</code>`),
  );
  html = html.replace(/\[([^\]]+)\]\(([^()\s]+)\)/g, (match, label, href) => {
    if (!/^(https?:\/\/|mailto:)/i.test(href)) {
      return match;
    }
    return protect(
      `<a href="${href}" target="_blank" rel="noopener">${renderInlineMarkdown(label, tokens)}</a>`,
    );
  });
  html = html.replace(/\*\*(.+?)\*\*|__(.+?)__/g, (_match, a, b) =>
    `<strong>${a || b}</strong>`,
  );
  html = html.replace(/(^|[^\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
  html = html.replace(/(^|[^\w])_([^_\n]+)_(?!\w)/g, "$1<em>$2</em>");
  return tokens.reduce(
    (result, token, index) => result.replaceAll(`<loa-markdown-token-${index}>`, token),
    html,
  );
}

function isHorizontalRule(line) {
  return /^\s*(?:-{3,}|_{3,}|\*{3,})\s*$/.test(line);
}

function isListItem(line) {
  return /^\s*(?:[-+*]\s+|\d+[.)]\s+)/.test(line);
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function isTableDelimiter(line) {
  const cells = tableCells(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function isBlockStart(lines, index) {
  const line = lines[index];
  return (
    /^\s*```/.test(line)
    || /^#{1,4}\s+/.test(line)
    || isHorizontalRule(line)
    || /^\s*&gt;/.test(line)
    || isListItem(line)
    || (line.includes("|") && index + 1 < lines.length && isTableDelimiter(lines[index + 1]))
  );
}

function renderMarkdownBlocks(escapedText) {
  const lines = escapedText.split("\n");
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(/^\s*```([^\s`]*)/);
    if (fence) {
      index += 1;
      const codeLines = [];
      while (index < lines.length && !/^\s*```/.test(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      const language = /^[\w+-]{1,30}$/.test(fence[1]) ? fence[1] : "";
      blocks.push(
        `<div class="code-block"><div class="code-toolbar"><span>${language}</span><button class="copy-code" type="button">Copy</button></div><pre><code>${highlightCode(codeLines.join("\n"))}</code></pre></div>`,
      );
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const level = heading[1].length;
      blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (isHorizontalRule(line)) {
      blocks.push("<hr>");
      index += 1;
      continue;
    }

    if (/^\s*&gt;/.test(line)) {
      const quoteLines = [];
      while (index < lines.length && /^\s*&gt;/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^\s*&gt;\s?/, ""));
        index += 1;
      }
      blocks.push(`<blockquote>${renderMarkdownBlocks(quoteLines.join("\n"))}</blockquote>`);
      continue;
    }

    if (isListItem(line)) {
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const tag = ordered ? "ol" : "ul";
      const items = [];
      while (index < lines.length) {
        const match = lines[index].match(/^\s*(?:[-+*]\s+|\d+[.)]\s+)(.*)$/);
        if (!match || /^\s*\d+[.)]\s+/.test(lines[index]) !== ordered) {
          break;
        }
        items.push(`<li>${renderInlineMarkdown(match[1])}</li>`);
        index += 1;
      }
      blocks.push(`<${tag}>${items.join("")}</${tag}>`);
      continue;
    }

    if (line.includes("|") && index + 1 < lines.length && isTableDelimiter(lines[index + 1])) {
      const headers = tableCells(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      const head = headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("");
      const body = rows.map((row) =>
        `<tr>${headers.map((_header, column) => `<td>${renderInlineMarkdown(row[column] || "")}</td>`).join("")}</tr>`,
      ).join("");
      blocks.push(`<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
      continue;
    }

    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !isBlockStart(lines, index)) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push(`<p>${renderInlineMarkdown(paragraph.join("\n").replace(/\n/g, "<br>"))}</p>`);
  }
  return blocks.join("");
}

function renderMarkdown(source) {
  return renderMarkdownBlocks(escapeHTML(source));
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

function appendGeneratedImage(message) {
  const ui = message.ui || {};
  const filename = ui.filename || "";
  const imageUrl = ui.image_url || "";
  if (!/^[a-f0-9]{32}\.png$/.test(filename) || imageUrl !== `/api/images/${filename}`) {
    return false;
  }
  const item = document.createElement("article");
  item.className = "message assistant image-message";
  const card = document.createElement("section");
  card.className = "generated-image-card";
  const preview = document.createElement("button");
  preview.type = "button";
  preview.className = "image-preview";
  preview.setAttribute("aria-label", "Open generated image preview");
  const image = document.createElement("img");
  image.src = imageUrl;
  image.alt = ui.prompt || "Generated image";
  preview.append(image);
  const caption = document.createElement("p");
  caption.className = "image-caption";
  caption.textContent = ui.prompt || "Generated image";
  const metadata = document.createElement("small");
  metadata.className = "image-provider";
  metadata.textContent = `Generated with ${ui.provider || "image provider"}`;
  const download = document.createElement("a");
  download.className = "quiet-button image-download";
  download.href = imageUrl;
  download.download = filename;
  download.textContent = "Download PNG";
  card.append(preview, caption, metadata, download);
  item.append(card);
  elements.messages.append(item);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return true;
}

function clearImageGenerationProgress() {
  imageProgressCard?.article.remove();
  imageProgressCard = null;
}

function showImageGenerationProgress(progress) {
  const filename = progress.filename || "";
  if (!/^[a-f0-9]{32}\.png$/.test(filename)) {
    return;
  }
  if (!imageProgressCard || imageProgressCard.filename !== filename) {
    clearImageGenerationProgress();
    const article = document.createElement("article");
    article.className = "message assistant image-message";
    const card = document.createElement("section");
    card.className = "generated-image-card";
    const preview = document.createElement("button");
    preview.type = "button";
    preview.className = "image-preview";
    preview.setAttribute("aria-label", "Open live Stable Diffusion preview");
    const image = document.createElement("img");
    image.alt = "Live Stable Diffusion preview";
    preview.append(image);
    const caption = document.createElement("p");
    caption.className = "image-caption";
    caption.textContent = "Stable Diffusion live preview";
    const progressBar = document.createElement("progress");
    const status = document.createElement("small");
    status.className = "image-provider";
    card.append(preview, caption, progressBar, status);
    article.append(card);
    elements.messages.append(article);
    imageProgressCard = {
      article,
      filename,
      image,
      progressBar,
      status,
      lastUpdate: "",
    };
  }

  const step = Number.isInteger(progress.step) ? progress.step : 0;
  const stepCount = Number.isInteger(progress.step_count) && progress.step_count > 0
    ? progress.step_count
    : 1;
  const update = `${step}:${progress.status || ""}`;
  if (imageProgressCard.lastUpdate === update) {
    return;
  }
  imageProgressCard.lastUpdate = update;
  imageProgressCard.progressBar.max = stepCount;
  imageProgressCard.progressBar.value = Math.min(step, stepCount);
  imageProgressCard.status.textContent = step > 0
    ? `${progress.status || "Denoising"} — step ${step}/${stepCount}`
    : progress.status || "Preparing Core ML model";
  if (step > 0) {
    imageProgressCard.image.src = `/api/images/${filename}?step=${step}`;
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

async function pollImageGenerationProgress(shouldStop) {
  while (!shouldStop()) {
    try {
      const progress = await api(`/api/image-progress?tick=${Date.now()}`);
      if (progress.active) {
        showImageGenerationProgress(progress);
      }
    } catch (error) {
      if (!shouldStop()) {
        setStatus(elements.connectionStatus, `Image progress unavailable: ${error.message}`);
      }
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 120));
  }
}

function appendMessage(message) {
  const item = document.createElement("article");
  item.className = `message ${message.role}`;
  if (message.role === "tool" && message.ui?.type === "image") {
    if (appendGeneratedImage(message)) {
      return;
    }
  }
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
  if (message.role === "assistant") {
    item.innerHTML = `<span class="message-author">${escapeHTML(author)}:</span><div class="markdown-body">${renderMarkdown(message.content || "")}</div>`;
  } else {
    item.textContent = author ? `${author}: ${message.content}` : message.content;
  }
  elements.messages.append(item);

  elements.messages.scrollTop = elements.messages.scrollHeight;
}

elements.messages.addEventListener("click", async (event) => {
  const imagePreview = event.target.closest?.("button.image-preview");
  if (imagePreview) {
    const image = imagePreview.querySelector("img");
    if (image) {
      elements.imageLightboxImage.src = image.src;
      elements.imageLightboxImage.alt = image.alt;
      elements.imageLightbox.showModal();
    }
    return;
  }
  const button = event.target.closest?.("button.copy-code");
  if (!button) {
    return;
  }
  const code = button.nextElementSibling?.querySelector("code");
  if (!code) {
    return;
  }
  try {
    await navigator.clipboard.writeText(code.textContent || "");
    button.textContent = "Copied";
  } catch (error) {
    setStatus(elements.connectionStatus, `Could not copy code: ${error.message}`);
  }
});

elements.imageLightboxClose.addEventListener("click", () => {
  elements.imageLightbox.close();
});

elements.imageLightbox.addEventListener("click", (event) => {
  if (event.target === elements.imageLightbox) {
    elements.imageLightbox.close();
  }
});

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
  clearImageGenerationProgress();
  for (const task of tasks || []) {
    if (task.label !== "Generated image") {
      appendTask(task);
    }
  }
  const toolMessages = (transcript || []).filter(
    (message) => message.role === "tool",
  );
  for (const message of toolMessages.slice(renderedToolCount)) {
    if (message.ui?.type === "image") {
      appendMessage(message);
    }
  }
  renderedToolCount = toolMessages.length;
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
  } else if (action.type === "mcp_tool" || action.type === "skill_tool") {
    sections.push(
      `Tool: ${action.remote_tool_name || action.tool_name || ""}\nServer: ${action.server_name || "User skill"}\nArguments:\n${JSON.stringify(action.arguments || {}, null, 2)}`,
    );
  } else if (action.type === "tool_call") {
    sections.push(
      `Tool: ${action.tool_name || ""}\nArguments:\n${JSON.stringify(action.arguments || {}, null, 2)}`,
    );
  } else if (action.type === "cron_job") {
    const args = action.arguments || {};
    const enabledState = args.operation === "toggle"
      ? `\nProposed enabled state: ${args.enabled ? "enabled" : "disabled"}`
      : "";
    const preApproved = args.auto_approve !== false;
    sections.push(
      `Operation: ${args.operation || "scheduled job"}\nName: ${args.name || "existing job"}\nSchedule: ${args.schedule || "not specified"}\nTarget action / prompt:\n${args.payload || "Not specified"}\nAction type: ${args.action_type || "prompt"}\nPre-approved execution: ${
        preApproved ? "Yes (runs automatically without further prompts)" : "No"
      }\n${args.one_time || args.run_once ? "[ONE-TIME] Yes; disable after first execution." : "One-time: No"}${enabledState}\n\nWARNING: If approved, this job may run automatically in the background.`,
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
  elements.commandControls.hidden = false;
  setBusy(false);
}

function setBusy(busy) {
  chatBusy = busy;
  elements.sendButton.disabled = busy || !configured || commandPending;
  elements.messageInput.disabled = busy || !configured || commandPending;
  elements.chatFiles.disabled = busy || !configured || commandPending;
  elements.addFileButton.disabled = busy || !configured || commandPending;
  elements.clearChatFiles.disabled = busy || !configured || commandPending;
  elements.homeToggle.disabled = busy || commandPending;
  elements.settingsToggle.disabled = busy || commandPending;
  elements.chatsToggle.disabled = busy || commandPending;
  elements.logoutButton.disabled = busy || commandPending;
  setBreakLoopVisible(busy || commandPending);
}

function updateChatFileSelection() {
  const files = [...elements.chatFiles.files];
  elements.chatFileNames.textContent = files.map((file) => file.name).join(", ");
  elements.clearChatFiles.hidden = files.length === 0;
}

elements.addFileButton.addEventListener("click", () => elements.chatFiles.click());
elements.chatFiles.addEventListener("change", updateChatFileSelection);
elements.clearChatFiles.addEventListener("click", () => {
  elements.chatFiles.value = "";
  updateChatFileSelection();
});

function selectSettingsTab(name) {
  for (const button of elements.settingsTabs.querySelectorAll("[data-tab]")) {
    const active = button.dataset.tab === name;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  }
  for (const panel of elements.settingsPanel.querySelectorAll("[data-settings-panel]")) {
    panel.hidden = panel.dataset.settingsPanel !== name;
  }
}

function showSettings(config, isSetup, openPanel = true) {
  if (typeof config.default_ollama_host === "string") {
    defaultOllamaHost = config.default_ollama_host;
  }
  if (openPanel) {
    elements.settingsPanel.hidden = false;
    elements.homePanel.hidden = true;
    elements.chatsPanel.hidden = true;
    elements.chatPanel.hidden = true;
  }
  elements.settingsHeading.textContent = isSetup ? "Set up your assistant" : "Assistant settings";
  elements.settingsClose.hidden = isSetup;
  elements.modelName.value = config.model;
  elements.ollamaHost.value = config.ollama_host || defaultOllamaHost;
  elements.ollamaHost.placeholder = defaultOllamaHost;
  const imageGen = config.image_gen || {};
  elements.imageGenEnabled.checked = imageGen.enabled !== false;
  elements.imageGenProvider.value = imageGen.provider || "openai";
  elements.imageGenModel.value = imageGen.model || "gpt-image-1";
  elements.imageGenApiKey.value = "";
  elements.imageGenApiKey.disabled = false;
  elements.imageGenKeyStatus.textContent = imageGen.api_key_configured
    ? "An API key is configured. Leave blank to keep it."
    : "No saved API key. You can also use OPENAI_API_KEY or GEMINI_API_KEY.";
  elements.imageGenClearKey.checked = false;
  elements.imageGenSize.value = imageGen.size || "1024x1024";
  elements.imageGenQuality.value = imageGen.quality || "high";
  setStatus(elements.imageGenStatus, "");
  const profile = config.profile || {};
  elements.profileName.value = profile.name || "";
  elements.profilePreferences.value = profile.preferences || "";
  elements.profileHardware.value = profile.hardware || "";
  elements.profileRoutines.value = profile.routines || "";
  elements.profileGoals.value = profile.goals || "";
  elements.profileNotes.value = profile.notes || "";
  renderBrainEntries(config.brain_entries || []);
  renderCronJobs(config.cron_jobs || []);
  selectSettingsTab("general");
  elements.mcpServers.value = JSON.stringify(config.mcp_servers || [], null, 2);
  currentSkillStates = { ...(config.skill_states || {}) };
  renderMcpControls();
  renderSkillControls(config.skills || []);
  if (isSetup) {
    elements.settingsPanel.hidden = false;
    elements.homePanel.hidden = true;
    elements.chatPanel.hidden = true;
    elements.settingsToggle.hidden = true;
    elements.chatForm.hidden = true;
    elements.logoutButton.hidden = true;
  }
}

function relativeChatTime(timestamp) {
  const updatedAt = Date.parse(timestamp);
  if (!Number.isFinite(updatedAt)) {
    return timestamp.slice(0, 10);
  }
  const seconds = Math.max(0, Math.floor((Date.now() - updatedAt) / 1000));
  const units = [
    ["year", 31536000],
    ["month", 2592000],
    ["week", 604800],
    ["day", 86400],
    ["hour", 3600],
    ["minute", 60],
  ];
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, duration] of units) {
    if (seconds >= duration) {
      return formatter.format(-Math.floor(seconds / duration), unit);
    }
  }
  return "just now";
}

async function refreshHomeChats() {
  const result = await api("/api/chats");
  pinConfigured = result.pin_configured;
  savedChatEntries = result.chats || [];
  activeChatId = result.active_chat_id;
  elements.homeRecentChats.replaceChildren();
  elements.homeContinueChat.disabled = savedChatEntries.length === 0;
  if (!savedChatEntries.length) {
    const empty = document.createElement("p");
    empty.className = "home-copy";
    empty.textContent = "No saved conversations yet.";
    elements.homeRecentChats.append(empty);
    return;
  }
  for (const chat of savedChatEntries.slice(0, 5)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "home-chat-choice";
    button.dataset.chatId = chat.id;
    const title = document.createElement("strong");
    title.textContent = chat.title;
    const preview = document.createElement("small");
    preview.textContent =
      `${relativeChatTime(chat.updated_at)} · ${chat.message_count} messages · ${chat.summary}`;
    button.append(title, preview);
    elements.homeRecentChats.append(button);
  }
}

async function showHome() {
  if (hiddenChatsMode) {
    try {
      await api("/api/chats", { action: "home" });
    } catch (error) {
      setStatus(elements.connectionStatus, error.message);
      return;
    }
  }
  hiddenChatsMode = false;
  activeChatId = null;
  selectedChatId = null;
  commandPending = false;
  loopStopped = false;
  renderedToolCount = 0;
  elements.messages.replaceChildren();
  elements.homePanel.hidden = !configured;
  elements.settingsPanel.hidden = true;
  elements.chatsPanel.hidden = true;
  elements.chatPanel.hidden = true;
  elements.chatForm.hidden = true;
  setBusy(false);
  if (configured) {
    try {
      await refreshHomeChats();
      setStatus(elements.homeStatus, "");
    } catch (error) {
      setStatus(elements.homeStatus, error.message);
    }
  }
}

function renderBrainEntries(entries) {
  elements.brainEntries.replaceChildren();
  if (!entries.length) {
    elements.brainEntries.textContent = "No Brain entries yet.";
    return;
  }
  for (const entry of entries) {
    const row = document.createElement("article");
    row.className = "memory-entry";
    const text = document.createElement("div");
    const content = document.createElement("strong");
    content.textContent = entry.content;
    const meta = document.createElement("small");
    meta.textContent = `${entry.category} · ${entry.created_at.slice(0, 10)}`;
    text.append(content, meta);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "quiet-button";
    remove.textContent = "Delete";
    remove.dataset.deleteBrain = entry.id;
    row.append(text, remove);
    elements.brainEntries.append(row);
  }
}

function renderCronJobs(jobs) {
  elements.cronjobsList.replaceChildren();
  if (!jobs.length) {
    elements.cronjobsList.textContent = "No scheduled jobs.";
    return;
  }
  for (const job of jobs) {
    const row = document.createElement("article");
    row.className = "memory-entry cron-job";
    const details = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = job.name;
    const schedule = document.createElement("small");
    schedule.textContent = `${job.completed ? "Completed" : job.enabled ? "Enabled" : "Disabled"} · ${job.schedule}`;
    const execution = document.createElement("small");
    execution.textContent = `Action: ${job.action_type || "prompt"} · Pre-approved execution: ${
      job.auto_approve ? "Yes (runs automatically without further prompts)" : "No"
    }`;
    const oneTimeBadge = document.createElement("span");
    oneTimeBadge.className = "cron-one-time-badge";
    oneTimeBadge.textContent = "One-time";
    const action = document.createElement("p");
    action.textContent = job.payload;
    details.append(name, schedule, execution);
    if (job.one_time || job.run_once) {
      details.append(oneTimeBadge);
    }
    details.append(action);
    const controls = document.createElement("div");
    controls.className = "cron-actions";
    for (const [label, operation] of [
      [job.enabled ? "Disable" : "Enable", "toggle"],
      ["Edit", "edit"],
      ["Run now", "run"],
      ["Logs", "logs"],
      ["Delete", "delete"],
    ]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "quiet-button";
      button.textContent = label;
      button.dataset.cronAction = operation;
      button.dataset.jobId = job.id;
      if (operation === "run" && job.one_time && job.completed) {
        button.disabled = true;
      }
      controls.append(button);
    }
    row.append(details, controls);
    elements.cronjobsList.append(row);
  }
}

function applyChatSearch() {
  const query = elements.chatSearch.value.trim().toLocaleLowerCase();
  for (const button of elements.savedChats.querySelectorAll("[data-chat-id]")) {
    button.hidden = query.length > 0 && !button.textContent.toLocaleLowerCase().includes(query);
  }
}

function renderMcpControls() {
  let servers;
  try {
    servers = JSON.parse(elements.mcpServers.value || "[]");
  } catch {
    elements.mcpControls.textContent = "Fix the MCP server JSON to configure server toggles.";
    return;
  }
  if (!Array.isArray(servers)) {
    elements.mcpControls.textContent = "MCP server configuration must be a JSON array.";
    return;
  }
  elements.mcpControls.replaceChildren();
  for (const server of servers) {
    if (!server || typeof server.name !== "string") {
      continue;
    }
    const row = document.createElement("label");
    row.className = "extension-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = server.enabled === true;
    checkbox.dataset.serverName = server.name;
    const name = document.createElement("strong");
    name.textContent = server.name;
    row.append(checkbox, name);
    elements.mcpControls.append(row);
  }
}

function renderSkillControls(skills) {
  currentSkillCommands = skills;
  renderSkillCommandHints();
  elements.skillsList.replaceChildren();
  if (!skills.length) {
    elements.skillsList.textContent = "No user skills installed. Create a Markdown skill below.";
    return;
  }
  for (const skill of skills) {
    const row = document.createElement("div");
    row.className = "extension-toggle skill-row";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = currentSkillStates[skill.name] === true;
    checkbox.dataset.skillName = skill.name;
    checkbox.setAttribute("aria-label", `Enable ${skill.name}`);
    const name = document.createElement("strong");
    name.textContent = skill.name;
    const kind = document.createElement("small");
    kind.textContent = `${skill.kind} skill`;
    const command = document.createElement("small");
    command.className = "skill-command";
    command.textContent = `Command: ${skill.command || "unavailable"}${
      skill.kind === "py" && !skill.enabled ? " (enable in Settings first)" : ""
    }`;
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "quiet-button";
    edit.dataset.editSkill = skill.name;
    edit.textContent = "Edit";
    row.append(checkbox, name, kind, command, edit);
    elements.skillsList.append(row);
  }
}

function renderSkillCommandHints() {
  elements.skillCommandHints.replaceChildren();
  const typed = elements.messageInput.value.trim();
  if (!typed.startsWith("/")) {
    return;
  }
  const query = typed.split(/\s+/, 1)[0].toLocaleLowerCase();
  const matches = currentSkillCommands.filter((skill) =>
    skill.command.toLocaleLowerCase().startsWith(query),
  );
  if (!matches.length) {
    elements.skillCommandHints.textContent = "No matching skill commands.";
    return;
  }

  const label = document.createElement("span");
  label.className = "skill-command-hint-label";
  label.textContent = "Matching skill commands:";
  elements.skillCommandHints.append(label);
  for (const skill of matches) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "skill-command-hint";
    button.textContent = skill.command;
    button.title = skill.kind === "py" && !skill.enabled
      ? "Enable this Python skill in Settings before use"
      : `Insert ${skill.command}`;
    button.addEventListener("click", () => {
      elements.messageInput.value = `${skill.command} `;
      elements.messageInput.focus();
      renderSkillCommandHints();
    });
    elements.skillCommandHints.append(button);
  }
}

async function openSkillEditor(name) {
  setStatus(elements.skillsStatus, `Loading ${name}...`);
  try {
    const skill = await api("/api/skills", { action: "read", name });
    editingSkillName = skill.name;
    elements.skillEditorName.textContent = skill.name;
    elements.skillEditorDescription.textContent =
      skill.description || "No description provided.";
    elements.skillEditorContent.value = skill.content;
    elements.skillEditor.hidden = false;
    setStatus(elements.skillsStatus, "");
    elements.skillEditorContent.focus();
  } catch (error) {
    setStatus(elements.skillsStatus, error.message);
  }
}

function closeSkillEditor() {
  editingSkillName = null;
  elements.skillEditor.hidden = true;
  elements.skillEditorContent.value = "";
}

async function loadAuthenticatedApplication() {
  try {
    const config = await api("/api/config");
    configured = config.configured;
    elements.statusLine.textContent = `${config.name} | ${config.model}`;
    showSettings(config, !configured, !configured);
    elements.logoutButton.hidden = false;
    if (configured) {
      elements.settingsToggle.hidden = false;
      elements.chatsToggle.hidden = false;
      await api("/api/chats", { action: "home" });
      await showHome();
    } else {
      elements.homePanel.hidden = true;
      setStatus(elements.settingsStatus, "Save these settings to start chatting.");
    }
  } catch (error) {
    setStatus(elements.connectionStatus, error.message);
    setBusy(true);
  }
}

async function refreshChatList() {
  const result = await api(hiddenChatsMode ? "/api/chats?hidden=1" : "/api/chats");
  pinConfigured = result.pin_configured;
  savedChatEntries = result.chats;
  activeChatId = result.active_chat_id;
  elements.savedChats.replaceChildren();
  for (const chat of result.chats) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chat-choice";
    button.dataset.chatId = chat.id;
    button.setAttribute("aria-pressed", String(chat.id === selectedChatId));
    button.setAttribute("aria-current", chat.id === activeChatId ? "true" : "false");
    const title = document.createElement("strong");
    title.textContent = chat.title;
    const metadata = document.createElement("small");
    metadata.textContent = `${chat.id === activeChatId ? "Active · " : ""}${chat.updated_at.slice(0, 10)} — ${chat.summary}`;
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
  elements.chatHide.disabled = !selected;
  elements.chatHide.textContent = hiddenChatsMode ? "Unhide" : "Hide";
  elements.chatsHeading.textContent = hiddenChatsMode ? "Hidden Chats" : "Chats";
  elements.chatNew.hidden = hiddenChatsMode;
  elements.chatDelete.disabled = !selected;
  elements.chatVectorize.disabled = !selected;
  applyChatSearch();
}

function requestHiddenPin(setup) {
  return new Promise((resolve) => {
    elements.hiddenPinForm.reset();
    elements.hiddenPinHeading.textContent = setup ? "Set up hidden chats PIN" : "Unlock hidden chats";
    elements.hiddenPinConfirmGroup.hidden = !setup;
    elements.hiddenPinConfirm.required = setup;
    setStatus(elements.hiddenPinStatus, "");
    const cancel = () => elements.hiddenPinDialog.close();
    const close = () => {
      elements.hiddenPinForm.removeEventListener("submit", submit);
      elements.hiddenPinCancel.removeEventListener("click", cancel);
      resolve(elements.hiddenPinDialog.returnValue === "ok");
    };
    const submit = async (event) => {
      event.preventDefault();
      try {
        await api("/api/chats", setup
          ? { action: "set_pin", pin: elements.hiddenPin.value, confirm_pin: elements.hiddenPinConfirm.value }
          : { action: "unlock", pin: elements.hiddenPin.value });
        pinConfigured = true;
        elements.hiddenPinDialog.close("ok");
      } catch (error) {
        setStatus(elements.hiddenPinStatus, error.message);
      }
    };
    elements.hiddenPinDialog.returnValue = "";
    elements.hiddenPinForm.addEventListener("submit", submit);
    elements.hiddenPinCancel.addEventListener("click", cancel);
    elements.hiddenPinDialog.addEventListener("close", close, { once: true });
    elements.hiddenPinDialog.showModal();
    elements.hiddenPin.focus();
  });
}

async function showChats() {
  try {
    await refreshChatList();
    elements.chatPanel.hidden = true;
    elements.homePanel.hidden = true;
    elements.settingsPanel.hidden = true;
    elements.chatsPanel.hidden = false;
    setStatus(elements.chatsStatus, "");
  } catch (error) {
    setStatus(elements.connectionStatus, error.message);
  }
}

function showChatTranscript(result) {
  activeChatId = result.active_chat_id;
  if (!activeChatId) {
    void showHome();
    return;
  }
  commandPending = false;
  loopStopped = false;
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
  elements.homePanel.hidden = true;
  elements.chatPanel.hidden = false;
  elements.chatForm.hidden = false;
  elements.chatControls.hidden = false;
  elements.commandControls.hidden = true;
  setBusy(false);
  if (result.pending_action) {
    showPendingAction(result.pending_action);
  }
  elements.messageInput.focus();
}

async function loadApplication() {
  const linkPassword = fragmentPassword;
  fragmentPassword = null;
  try {
    const auth = await api("/auth/status");
    if (auth.password_configured) {
      const session = await api("/auth/session");
      if (session.authenticated) {
        await loadAuthenticatedApplication();
        return;
      }
      if (linkPassword !== null) {
        elements.loginPanel.hidden = false;
        elements.homePanel.hidden = true;
        elements.chatPanel.hidden = true;
        await signIn(linkPassword, true, auth.allow_http_auto_login === true);
        return;
      }
    }
    elements.loginPanel.hidden = false;
    elements.homePanel.hidden = true;
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

function canSendCredentialsSafely(allowHttpAutoLogin) {
  return window.location.protocol === "https:" || allowHttpAutoLogin;
}

async function signIn(password, fromFragment = false, allowHttpAutoLogin = false) {
  if (fromFragment && !canSendCredentialsSafely(allowHttpAutoLogin)) {
    setStatus(
      elements.loginStatus,
      "Auto-login was not sent: use HTTPS, localhost, or connect through Tailscale.",
    );
    return;
  }
  setStatus(elements.loginStatus, "Signing in...");
  try {
    await api("/auth/login", { password, auto_login: fromFragment });
    elements.webPassword.value = "";
    elements.loginPanel.hidden = true;
    await loadAuthenticatedApplication();
  } catch (error) {
    setStatus(elements.loginStatus, error.message);
  }
}

elements.loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await signIn(elements.webPassword.value);
});

elements.settingsToggle.addEventListener("click", async () => {
  try {
    showSettings(await api("/api/config"), false);
  } catch (error) {
    setStatus(elements.connectionStatus, error.message);
  }
});

elements.homeSettings.addEventListener("click", async () => {
  try {
    showSettings(await api("/api/config"), false);
  } catch (error) {
    setStatus(elements.homeStatus, error.message);
  }
});

elements.homeLogout.addEventListener("click", () => elements.logoutButton.click());

elements.homeHiddenChats.addEventListener("click", async () => {
  if (!pinConfigured) {
    setStatus(elements.homeStatus, "Hide a chat first to set up a PIN.");
    return;
  }
  if (await requestHiddenPin(false)) {
    hiddenChatsMode = true;
    selectedChatId = null;
    await showChats();
  }
});

elements.homeNewChat.addEventListener("click", async () => {
  try {
    showChatTranscript(await api("/api/chats", { action: "new" }));
    selectedChatId = activeChatId;
  } catch (error) {
    setStatus(elements.homeStatus, error.message);
  }
});

elements.homeContinueChat.addEventListener("click", async () => {
  if (!savedChatEntries.length) {
    return;
  }
  try {
    showChatTranscript(await api("/api/chats", {
      action: "open",
      chat_id: savedChatEntries[0].id,
    }));
  } catch (error) {
    setStatus(elements.homeStatus, error.message);
    await refreshHomeChats();
  }
});

elements.homeRecentChats.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-chat-id]");
  if (!button) {
    return;
  }
  try {
    showChatTranscript(await api("/api/chats", {
      action: "open",
      chat_id: button.dataset.chatId,
    }));
  } catch (error) {
    setStatus(elements.homeStatus, error.message);
    await refreshHomeChats();
  }
});

elements.chatsToggle.addEventListener("click", showChats);

elements.chatsClose.addEventListener("click", async () => {
  elements.chatsPanel.hidden = true;
  if (activeChatId) {
    elements.chatPanel.hidden = false;
  } else {
    if (hiddenChatsMode) {
      try {
        await api("/api/chats", { action: "home" });
      } catch (error) {
        setStatus(elements.chatsStatus, error.message);
        elements.chatsPanel.hidden = false;
        return;
      }
    }
    await showHome();
  }
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
  elements.chatOpen.disabled = false;
  elements.chatRename.disabled = false;
  elements.chatHide.disabled = false;
  elements.chatDelete.disabled = false;
  elements.chatVectorize.disabled = false;
});

elements.chatSearch.addEventListener("input", applyChatSearch);

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

elements.chatHide.addEventListener("click", async () => {
  if (!selectedChatId) return;
  if (!hiddenChatsMode && !pinConfigured && !await requestHiddenPin(true)) return;
  try {
    const result = await api("/api/chats", {
      action: hiddenChatsMode ? "unhide" : "hide",
      chat_id: selectedChatId,
    });
    activeChatId = result.active_chat_id;
    selectedChatId = null;
    await refreshChatList();
    setStatus(elements.chatsStatus, hiddenChatsMode ? "Chat unhidden." : "Chat hidden.");
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

elements.chatVectorize.addEventListener("click", async () => {
  if (!selectedChatId) {
    return;
  }
  const selected = savedChatEntries.find((chat) => chat.id === selectedChatId);
  const title = selected?.title || "this chat";
  if (!window.confirm(
    `Vectorize "${title}" into local Brain memories, then permanently delete the raw chat history? Its key facts and tokens will be saved first.`,
  )) {
    return;
  }
  try {
    const result = await api("/api/chats", {
      action: "vectorize",
      chat_id: selectedChatId,
    });
    selectedChatId = result.active_chat_id;
    if (!result.active_chat_id) {
      await showHome();
      setStatus(elements.homeStatus, `Saved ${result.vectorized_count} entries to Brain; raw chat deleted.`);
      return;
    }
    showChatTranscript(result);
    appendMessage({
      role: "system",
      content: `Saved ${result.vectorized_count} entries to local Brain memory; raw chat history deleted.`,
    });
  } catch (error) {
    setStatus(elements.chatsStatus, error.message);
  }
});

elements.homeToggle.addEventListener("click", async () => {
  try {
    await api("/api/chats", { action: "home" });
    await showHome();
  } catch (error) {
    setStatus(elements.connectionStatus, error.message);
  }
});

elements.settingsTabs.addEventListener("click", (event) => {
  const button = event.target.closest("[data-tab]");
  if (button) {
    selectSettingsTab(button.dataset.tab);
  }
});

elements.saveProfile.addEventListener("click", async () => {
  setStatus(elements.profileStatus, "Saving...");
  try {
    await api("/api/profile", {
      profile: {
        name: elements.profileName.value,
        preferences: elements.profilePreferences.value,
        hardware: elements.profileHardware.value,
        routines: elements.profileRoutines.value,
        goals: elements.profileGoals.value,
        notes: elements.profileNotes.value,
      },
    });
    setStatus(elements.profileStatus, "About Me profile saved locally and added to LOA's context.");
  } catch (error) {
    setStatus(elements.profileStatus, error.message);
  }
});

elements.brainSearch.addEventListener("input", async () => {
  try {
    const result = await api(`/api/brain?q=${encodeURIComponent(elements.brainSearch.value)}`);
    renderBrainEntries(result.entries || []);
  } catch (error) {
    setStatus(elements.brainStatus, error.message);
  }
});

elements.brainAdd.addEventListener("click", async () => {
  setStatus(elements.brainStatus, "Saving memory...");
  try {
    const result = await api("/api/brain", {
      action: "add",
      content: elements.brainContent.value,
      category: elements.brainCategory.value || "fact",
    });
    elements.brainContent.value = "";
    renderBrainEntries(result.entries || []);
    setStatus(elements.brainStatus, "Memory saved locally.");
  } catch (error) {
    setStatus(elements.brainStatus, error.message);
  }
});

elements.brainEntries.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-delete-brain]");
  if (!button) {
    return;
  }
  try {
    const result = await api("/api/brain", { action: "delete", id: button.dataset.deleteBrain });
    renderBrainEntries(result.entries || []);
    setStatus(elements.brainStatus, "Memory deleted.");
  } catch (error) {
    setStatus(elements.brainStatus, error.message);
  }
});

elements.cronSave.addEventListener("click", async () => {
  const editing = Boolean(editingCronJobId);
  const payload = {
    action: editing ? "update" : "create",
    name: elements.cronName.value,
    schedule: elements.cronSchedule.value,
    payload: elements.cronPayload.value,
    action_type: elements.cronActionType.value,
    auto_approve: elements.cronAutoApprove.checked,
    one_time: elements.cronOneTime.checked,
  };
  if (editing) {
    payload.job_id = editingCronJobId;
  }
  const oneTimeNotice = payload.one_time
    ? "\n[ONE-TIME] It will disable itself after its first execution."
    : "\nOne-time: No";
  const approvalNotice = payload.auto_approve
    ? "Yes (runs automatically without further prompts)"
    : "No";
  const warning = `This job will run automatically in the background when enabled.\n\nSchedule: ${payload.schedule}\nAction type: ${payload.action_type}\nPre-approved execution: ${approvalNotice}\nAction: ${payload.payload}${oneTimeNotice}\n\n${editing ? "Apply these changes?" : "Create this job?"}`;
  if (!window.confirm(warning)) {
    return;
  }
  try {
    const result = await api("/api/cronjobs", payload);
    renderCronJobs(result.jobs || []);
    editingCronJobId = null;
    elements.cronName.value = "";
    elements.cronSchedule.value = "";
    elements.cronPayload.value = "";
    elements.cronActionType.value = "prompt";
    elements.cronAutoApprove.checked = true;
    elements.cronOneTime.checked = false;
    elements.cronSave.textContent = "Create scheduled job";
    elements.cronCancelEdit.hidden = true;
    setStatus(elements.cronStatus, editing ? "Scheduled job updated." : "Scheduled job created.");
  } catch (error) {
    setStatus(elements.cronStatus, error.message);
  }
});

elements.cronCancelEdit.addEventListener("click", () => {
  editingCronJobId = null;
  elements.cronName.value = "";
  elements.cronSchedule.value = "";
  elements.cronPayload.value = "";
  elements.cronActionType.value = "prompt";
  elements.cronAutoApprove.checked = true;
  elements.cronOneTime.checked = false;
  elements.cronSave.textContent = "Create scheduled job";
  elements.cronCancelEdit.hidden = true;
});

elements.cronjobsList.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-cron-action]");
  if (!button) {
    return;
  }
  try {
    const job = (await api("/api/cronjobs")).jobs.find((item) => item.id === button.dataset.jobId);
    if (!job) {
      setStatus(elements.cronStatus, "Scheduled job no longer exists.");
      return;
    }
    const operation = button.dataset.cronAction;
    if (operation === "edit") {
      editingCronJobId = job.id;
      elements.cronName.value = job.name;
      elements.cronSchedule.value = job.schedule;
      elements.cronPayload.value = job.payload;
      elements.cronActionType.value = job.action_type || "prompt";
      elements.cronAutoApprove.checked = job.auto_approve !== false;
      elements.cronOneTime.checked = Boolean(job.one_time || job.run_once);
      elements.cronSave.textContent = "Save scheduled job changes";
      elements.cronCancelEdit.hidden = false;
    } else if (operation === "logs") {
      const result = await api(`/api/cronjobs/${encodeURIComponent(job.id)}/history`);
      elements.cronHistory.textContent = result.history.length
        ? result.history.map((item) => `${item.started_at} · ${item.status}\n${item.output}`).join("\n\n")
        : "No execution history.";
    } else if (operation === "toggle") {
      if (!window.confirm(
        `${job.enabled ? "Disable" : "Enable"} "${job.name}" (${job.schedule})? Enabled jobs run automatically in the background.\n\nAction type: ${job.action_type || "prompt"}\nPre-approved execution: ${
          job.auto_approve
            ? "Yes (runs automatically without further prompts)"
            : "No"
        }\nAction: ${job.payload}\n${job.one_time || job.run_once ? "[ONE-TIME] Yes; disable after first execution." : "One-time: No"}`,
      )) {
        return;
      }
      const result = await api("/api/cronjobs", {
        action: "toggle",
        job_id: job.id,
        enabled: !job.enabled,
      });
      renderCronJobs(result.jobs || []);
    } else if (operation === "run") {
      if (!window.confirm(
        `Run "${job.name}" now?\n\nSchedule: ${job.schedule}\nAction: ${job.payload}\n${job.one_time || job.run_once ? "[ONE-TIME] Yes; this first execution will complete and disable the job." : "One-time: No"}\n\nThis runs the prompt now; any proposed external action still needs approval.`,
      )) {
        return;
      }
      const result = await api("/api/cronjobs", { action: "run", job_id: job.id });
      setStatus(elements.cronStatus, result.output);
      elements.cronHistory.textContent = (result.history || [])
        .map((item) => `${item.started_at} · ${item.status}\n${item.output}`)
        .join("\n\n");
    } else if (operation === "delete") {
      if (!window.confirm(
        `Permanently delete "${job.name}" and its execution history?\n\nSchedule: ${job.schedule}\nAction: ${job.payload}\n${job.one_time || job.run_once ? "[ONE-TIME] Yes" : "One-time: No"}`,
      )) {
        return;
      }
      const result = await api("/api/cronjobs", { action: "delete", job_id: job.id });
      renderCronJobs(result.jobs || []);
      setStatus(elements.cronStatus, "Scheduled job deleted.");
    }
  } catch (error) {
    setStatus(elements.cronStatus, error.message);
  }
});

elements.resetHost.addEventListener("click", () => {
  elements.ollamaHost.value = defaultOllamaHost;
  setStatus(elements.settingsStatus, "Default host restored. Save settings to keep it.");
});

function imageGenSettingsPayload() {
  return {
    enabled: elements.imageGenEnabled.checked,
    provider: elements.imageGenProvider.value,
    model: elements.imageGenModel.value.trim(),
    api_key: elements.imageGenApiKey.value,
    clear_api_key: elements.imageGenClearKey.checked,
    size: elements.imageGenSize.value,
    quality: elements.imageGenQuality.value,
  };
}

elements.imageGenProvider.addEventListener("change", () => {
  const defaultModels = {
    openai: "gpt-image-1",
    gemini: "gemini-2.5-flash-image",
    local_coreml_sd: "apple/coreml-stable-diffusion-2-1-base",
  };
  if (Object.values(defaultModels).includes(elements.imageGenModel.value.trim())) {
    elements.imageGenModel.value = defaultModels[elements.imageGenProvider.value];
  }
});

elements.imageGenClearKey.addEventListener("change", () => {
  elements.imageGenApiKey.disabled = elements.imageGenClearKey.checked;
  if (elements.imageGenClearKey.checked) {
    elements.imageGenApiKey.value = "";
  }
});

elements.imageGenCheck.addEventListener("click", async () => {
  setStatus(elements.imageGenStatus, "Checking provider...");
  elements.imageGenCheck.disabled = true;
  try {
    const result = await api("/api/image-gen/check", {
      image_gen: imageGenSettingsPayload(),
    });
    setStatus(elements.imageGenStatus, result.message);
  } catch (error) {
    setStatus(elements.imageGenStatus, error.message);
  } finally {
    elements.imageGenCheck.disabled = false;
  }
});

elements.settingsClose.addEventListener("click", () => {
  elements.settingsPanel.hidden = true;
  if (activeChatId) {
    elements.chatPanel.hidden = false;
  } else {
    void showHome();
  }
});

elements.settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const saveButtons = elements.settingsForm.querySelectorAll('button[type="submit"]');
  for (const button of saveButtons) {
    button.disabled = true;
  }
  setStatus(elements.settingsStatus, "Saving...");
  try {
    const mcpServers = JSON.parse(elements.mcpServers.value || "[]");
    if (!Array.isArray(mcpServers)) {
      throw new Error("MCP server configuration must be a JSON array.");
    }
    for (const server of mcpServers) {
      const toggle = [...elements.mcpControls.querySelectorAll("[data-server-name]")]
        .find((checkbox) => checkbox.dataset.serverName === server.name);
      if (toggle) {
        server.enabled = toggle.checked;
      }
    }
    const skillStates = { ...currentSkillStates };
    for (const toggle of elements.skillsList.querySelectorAll("[data-skill-name]")) {
      skillStates[toggle.dataset.skillName] = toggle.checked;
    }
    const config = await api("/api/settings", {
      model: elements.modelName.value,
      ollama_host: elements.ollamaHost.value,
      image_gen: imageGenSettingsPayload(),
      mcp_servers: mcpServers,
      skill_states: skillStates,
    });
    currentSkillStates = skillStates;
    renderSkillControls(config.skills || []);
    configured = true;
    elements.statusLine.textContent = `${config.name} | ${elements.modelName.value.trim()}`;
    elements.settingsToggle.hidden = false;
    elements.chatsToggle.hidden = false;
    elements.settingsPanel.hidden = true;
    elements.chatsPanel.hidden = true;
    elements.chatForm.hidden = false;
    if (activeChatId) {
      elements.homePanel.hidden = true;
      elements.chatPanel.hidden = false;
      setBusy(false);
      elements.messageInput.focus();
    } else {
      await showHome();
    }
  } catch (error) {
    setStatus(elements.settingsStatus, error.message);
  } finally {
    for (const button of saveButtons) {
      button.disabled = false;
    }
  }
});

elements.mcpServers.addEventListener("input", () => renderMcpControls());
elements.refreshMcpControls.addEventListener("click", () => renderMcpControls());
elements.mcpControls.addEventListener("change", (event) => {
  const toggle = event.target.closest?.("[data-server-name]");
  if (!toggle) {
    return;
  }
  try {
    const servers = JSON.parse(elements.mcpServers.value || "[]");
    const server = servers.find((item) => item.name === toggle.dataset.serverName);
    if (server) {
      server.enabled = toggle.checked;
      elements.mcpServers.value = JSON.stringify(servers, null, 2);
    }
  } catch (error) {
    setStatus(elements.settingsStatus, `Invalid MCP server JSON: ${error.message}`);
  }
});
elements.skillsList.addEventListener("click", (event) => {
  const button = event.target.closest?.("[data-edit-skill]");
  if (button) {
    openSkillEditor(button.dataset.editSkill);
  }
});
elements.saveSkillEdit.addEventListener("click", async () => {
  if (!editingSkillName) {
    return;
  }
  elements.saveSkillEdit.disabled = true;
  setStatus(elements.skillsStatus, `Saving ${editingSkillName}...`);
  try {
    const result = await api("/api/skills", {
      action: "edit",
      name: editingSkillName,
      content: elements.skillEditorContent.value,
    });
    currentSkillStates = { ...result.skill_states };
    renderSkillControls(result.skills);
    closeSkillEditor();
    setStatus(elements.skillsStatus, "Skill saved and reloaded.");
  } catch (error) {
    setStatus(elements.skillsStatus, error.message);
  } finally {
    elements.saveSkillEdit.disabled = false;
  }
});
elements.cancelSkillEdit.addEventListener("click", () => {
  closeSkillEditor();
  setStatus(elements.skillsStatus, "Skill changes discarded.");
});
elements.createSkill.addEventListener("click", async () => {
  setStatus(elements.skillsStatus, "Creating skill...");
  try {
    const result = await api("/api/skills", {
      action: "create",
      name: elements.newSkillName.value,
      content: elements.newSkillContent.value,
    });
    currentSkillStates = { ...result.skill_states };
    renderSkillControls(result.skills);
    elements.newSkillName.value = "";
    elements.newSkillContent.value = "";
    setStatus(elements.skillsStatus, "Skill created and disabled. Enable it above, then save settings.");
  } catch (error) {
    setStatus(elements.skillsStatus, error.message);
  }
});
elements.reloadSkills.addEventListener("click", async () => {
  setStatus(elements.skillsStatus, "Reloading skills...");
  try {
    const result = await api("/api/skills", { action: "reload" });
    currentSkillStates = { ...result.skill_states };
    renderSkillControls(result.skills);
    setStatus(elements.skillsStatus, "Skills reloaded.");
  } catch (error) {
    setStatus(elements.skillsStatus, error.message);
  }
});

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message || !configured || commandPending) {
    return;
  }
  loopStopped = false;
  elements.messageInput.value = "";
  appendMessage({ role: "user", content: message });
  const localCommand = message.toLowerCase();
  if (localCommand === "/help") {
    appendMessage({
      role: "system",
      content:
        "Commands:\n" +
        "/settings - edit and save assistant settings\n" +
        "/mcp [name on|off] - show or toggle an MCP server\n" +
        "/skills [name on|off|reload] - list or manage user skills\n" +
        "/home - return to the start screen\n" +
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
  if (localCommand === "/home") {
    try {
      await api("/api/chats", { action: "home" });
      await showHome();
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
  if (localCommand === "/mcp" || localCommand.startsWith("/mcp ")) {
    try {
      const parts = message.trim().split(/\s+/);
      if (parts.length === 3 && ["on", "off"].includes(parts[2].toLowerCase())) {
        await api("/api/mcp", {
          name: parts[1],
          enabled: parts[2].toLowerCase() === "on",
        });
        appendMessage({ role: "system", content: `MCP server ${parts[1]} updated.` });
      } else if (parts.length === 1) {
        appendMessage({
          role: "system",
          content: "Manage MCP servers in Settings > MCP Servers.",
        });
      } else {
        appendMessage({ role: "system", content: "Usage: /mcp [name on|off]" });
      }
    } catch (error) {
      appendMessage({ role: "system", content: error.message });
    }
    return;
  }
  if (localCommand === "/skills" || localCommand.startsWith("/skills ")) {
    const parts = message.trim().split(/\s+/);
    try {
      let result;
      if (parts.length === 1) {
        result = await api("/api/skills", { action: "list" });
      } else if (parts.length === 2 && parts[1].toLowerCase() === "reload") {
        result = await api("/api/skills", { action: "reload" });
      } else if (parts.length === 3 && ["on", "off"].includes(parts[2].toLowerCase())) {
        result = await api("/api/skills", {
          action: "toggle",
          name: parts[1],
          enabled: parts[2].toLowerCase() === "on",
        });
      } else {
        appendMessage({ role: "system", content: "Usage: /skills [name on|off|reload]" });
        return;
      }
      const lines = (result.skills || []).map((skill) =>
        `${skill.name} ${skill.command}: ${skill.enabled ? "enabled" : "disabled"}`,
      );
      appendMessage({
        role: "system",
        content: lines.join("\n") || "No user skills installed.",
      });
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
  let stopImageProgressPolling = false;
  let imageProgressPolling;
  try {
    const attachments = [];
    if (elements.chatFiles.files.length > 12) {
      throw new Error("Attach no more than 12 files per message.");
    }
    for (const file of elements.chatFiles.files) {
      if (file.size > 8000000) {
        throw new Error(`${file.name} must be at most 8 MB.`);
      }
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
        reader.readAsDataURL(file);
      });
      const uploaded = await api("/api/files/upload", {
        filename: file.name,
        content_type: file.type,
        data: dataUrl.split(",")[1],
      });
      attachments.push({
        id: uploaded.id,
        filename: uploaded.filename,
        mime_type: uploaded.mime_type,
      });
    }
    imageProgressPolling = pollImageGenerationProgress(
      () => stopImageProgressPolling,
    );
    const result = await api("/api/chat", { message, attachments });
    elements.chatFiles.value = "";
    updateChatFileSelection();
    renderReplyTasks(result.tasks, result.transcript);
    if (result.canceled) {
      if (result.action_task) {
        appendTask(result.action_task);
      }
      setStatus(
        elements.connectionStatus,
        loopStopped
          ? "Loop stopped – waiting for user input"
          : "Request canceled by user",
      );
      return;
    }
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
    clearImageGenerationProgress();
    appendMessage({ role: "system", content: error.message });
    setStatus(elements.connectionStatus, "");
  } finally {
    stopImageProgressPolling = true;
    if (imageProgressPolling) {
      await imageProgressPolling;
    }
    clearImageGenerationProgress();
    setBusy(false);
  }
});

elements.messageInput.addEventListener("input", renderSkillCommandHints);

async function respondToCommand(approved) {
  elements.commandControls.hidden = true;
  commandPending = false;
  setBusy(true);
  setStatus(elements.connectionStatus, approved ? "Running command..." : "Cancelling command...");
  let stopImageProgressPolling = false;
  let imageProgressPolling;
  try {
    imageProgressPolling = pollImageGenerationProgress(
      () => stopImageProgressPolling,
    );
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
    setStatus(
      elements.connectionStatus,
      result.canceled
        ? loopStopped
          ? "Loop stopped – waiting for user input"
          : "Request canceled by user"
        : "",
    );
  } catch (error) {
    clearImageGenerationProgress();
    appendMessage({ role: "system", content: error.message });
    setStatus(elements.connectionStatus, "");
  } finally {
    stopImageProgressPolling = true;
    if (imageProgressPolling) {
      await imageProgressPolling;
    }
    clearImageGenerationProgress();
    elements.chatControls.hidden = commandPending;
    elements.commandControls.hidden = !commandPending;
    setBusy(false);
  }
}

elements.approveCommand.addEventListener("click", () => respondToCommand(true));
elements.cancelCommand.addEventListener("click", () => respondToCommand(false));

elements.breakLoopButton.addEventListener("click", async () => {
  elements.breakLoopButton.disabled = true;
  setStatus(elements.connectionStatus, "Stopping loop...");
  try {
    const result = await api("/api/break-loop", {});
    loopStopped = true;
    commandPending = false;
    elements.commandControls.hidden = true;
    elements.chatControls.hidden = false;
    if (result.action_task) {
      appendTask(result.action_task);
    }
    appendMessage({
      role: "user",
      content:
        "STOP. Immediately cancel all further tool and command requests, break out of the current loop, and wait for further user input.",
    });
    setBusy(false);
    setStatus(elements.connectionStatus, "Loop stopped – waiting for user input");
  } catch (error) {
    setStatus(elements.connectionStatus, error.message);
  } finally {
    setBreakLoopVisible(chatBusy || commandPending);
  }
});

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
    activeChatId = null;
    elements.messages.replaceChildren();
    elements.loginPanel.hidden = false;
    elements.homePanel.hidden = true;
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
