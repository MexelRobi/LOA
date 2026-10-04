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
  logoutButton: document.querySelector("#logout-button"),
  settingsClose: document.querySelector("#settings-close"),
  settingsPanel: document.querySelector("#settings-panel"),
  settingsHeading: document.querySelector("#settings-heading"),
  settingsForm: document.querySelector("#settings-form"),
  settingsStatus: document.querySelector("#settings-status"),
  assistantName: document.querySelector("#assistant-name"),
  modelName: document.querySelector("#model-name"),
  systemPrompt: document.querySelector("#system-prompt"),
  approveCommand: document.querySelector("#approve-command"),
  cancelCommand: document.querySelector("#cancel-command"),
  commandControls: document.querySelector("#command-controls"),
  chatControls: document.querySelector("#chat-controls"),
  statusLine: document.querySelector("#status-line"),
  loginPanel: document.querySelector("#login-panel"),
  loginForm: document.querySelector("#login-form"),
  webPassword: document.querySelector("#web-password"),
  loginStatus: document.querySelector("#login-status"),
  voiceButton: document.querySelector("#voice-button"),
};

let configured = false;
let commandPending = false;
let recognition = null;
let voiceText = "";
let voiceSupported = false;
let speakNextReply = false;
let speechGeneration = 0;

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

function responseLanguage(text) {
  const germanWords = /[äöüß]|\b(ich|nicht|und|für|bitte|kann|ihnen|dir|gerne|hallo|möchte|wäre|sind|können|danke)\b/i;
  return germanWords.test(text) ? "de-DE" : (navigator.language || "en-US");
}

function speechChunks(text, maxLength = 220) {
  const plainText = text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\[([^\]]+)\]\(https?:\/\/[^)]+\)/g, "$1")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/[*_#>`~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const sentences = plainText.match(/[^.!?]+[.!?]+["')\]]*|[^.!?]+$/g) || [];
  const chunks = [];
  let current = "";
  for (const rawSentence of sentences) {
    const sentence = rawSentence.trim();
    if (!sentence) {
      continue;
    }
    if (current && `${current} ${sentence}`.length > maxLength) {
      chunks.push(current);
      current = "";
    }
    if (sentence.length <= maxLength) {
      current = current ? `${current} ${sentence}` : sentence;
      continue;
    }
    const words = sentence.split(/\s+/);
    for (const word of words) {
      if (current && `${current} ${word}`.length > maxLength) {
        chunks.push(current);
        current = "";
      }
      current = current ? `${current} ${word}` : word;
    }
  }
  if (current) {
    chunks.push(current);
  }
  return chunks;
}

function speakReply(text) {
  if (!("speechSynthesis" in window)) {
    return;
  }

  const chunks = speechChunks(text);
  if (chunks.length === 0) {
    return;
  }
  const language = responseLanguage(text);
  const languagePrefix = language.split("-")[0].toLowerCase();
  const voices = window.speechSynthesis.getVoices();
  const matchingVoices = voices.filter(
    (voice) => voice.lang.toLowerCase().split("-")[0] === languagePrefix,
  );
  const preferredVoice = matchingVoices.sort((left, right) => {
    const score = (voice) =>
      (voice.localService ? 2 : 0) +
      (/natural|premium|enhanced|siri/i.test(voice.name) ? 4 : 0);
    return score(right) - score(left);
  })[0];

  const generation = ++speechGeneration;
  window.speechSynthesis.cancel();
  const speakChunk = (index) => {
    if (generation !== speechGeneration || index >= chunks.length) {
      return;
    }
    const utterance = new SpeechSynthesisUtterance(chunks[index]);
    utterance.lang = preferredVoice?.lang || language;
    utterance.rate = 0.88;
    utterance.pitch = 1;
    utterance.volume = 1;
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }
    utterance.onend = () => speakChunk(index + 1);
    utterance.onerror = (event) => {
      if (event.error !== "interrupted" && generation === speechGeneration) {
        setStatus(elements.connectionStatus, `Voice playback failed: ${event.error}.`);
      }
    };
    window.speechSynthesis.speak(utterance);
  };
  speakChunk(0);
}

function appendMessage(message) {
  const item = document.createElement("article");
  item.className = `message ${message.role}`;
  const speaker = {
    user: "You",
    assistant: elements.statusLine.textContent.split(" | ")[0] || "LOA",
  }[message.role];
  item.textContent = speaker ? `${speaker}: ${message.content}` : message.content;
  elements.messages.append(item);

  if (message.sources && message.sources.length > 0) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    for (const source of message.sources) {
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title || source.url;
      sourceList.append(link);
    }
    item.append(sourceList);
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function setBusy(busy) {
  elements.sendButton.disabled = busy || !configured || commandPending;
  elements.messageInput.disabled = busy || !configured || commandPending;
  elements.voiceButton.disabled = busy || !configured || commandPending || !voiceSupported;
  elements.settingsToggle.disabled = busy || commandPending;
  elements.logoutButton.disabled = busy || commandPending;
}

function showSettings(config, isSetup, openPanel = true) {
  if (openPanel) {
    elements.settingsPanel.hidden = false;
    elements.chatPanel.hidden = true;
  }
  elements.settingsHeading.textContent = isSetup ? "Set up your assistant" : "Assistant settings";
  elements.settingsClose.hidden = isSetup;
  elements.assistantName.value = config.name;
  elements.modelName.value = config.model;
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
      elements.settingsToggle.hidden = false;
      elements.chatForm.hidden = false;
    }
    if (!window.isSecureContext) {
      setStatus(
        elements.connectionStatus,
        "Voice input requires HTTPS on phones. Configure a trusted TLS certificate in the LOA web server.",
      );
    } else if (!SpeechRecognition) {
      setStatus(elements.connectionStatus, "Voice input is not supported by this browser.");
    }
    if (configured) {
      appendMessage({ role: "system", content: "LOA is ready. Type /help to see available commands." });
      const history = await api("/api/history");
      for (const message of history.transcript) {
        appendMessage(message);
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
    });
    configured = true;
    elements.statusLine.textContent = `${config.name} | ${elements.modelName.value.trim()}`;
    elements.settingsToggle.hidden = false;
    elements.settingsPanel.hidden = true;
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
    speakNextReply = false;
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
  if (localCommand === "/settings") {
    speakNextReply = false;
    try {
      showSettings(await api("/api/config"), false);
    } catch (error) {
      appendMessage({ role: "system", content: error.message });
    }
    return;
  }
  if (["/exit", "/quit", "exit", "quit"].includes(localCommand)) {
    speakNextReply = false;
    elements.logoutButton.click();
    return;
  }

  setBusy(true);
  setStatus(elements.connectionStatus, "Thinking...");
  try {
    const result = await api("/api/chat", { message });
    appendMessage({
      role: "assistant",
      content: result.reply,
      sources: result.sources,
    });
    if (result.command) {
      commandPending = true;
      appendMessage({
        role: "system",
        content: `Proposed command:\n${result.command}\nReview it, then choose Run command or Cancel command.`,
      });
      elements.chatControls.hidden = true;
      elements.commandControls.hidden = false;
    } else if (speakNextReply && "speechSynthesis" in window) {
      speakReply(result.reply);
    }
    speakNextReply = false;
    setStatus(elements.connectionStatus, "");
  } catch (error) {
    appendMessage({ role: "system", content: error.message });
    speakNextReply = false;
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
    appendMessage({ role: "system", content: result.output });
    setStatus(elements.connectionStatus, "");
  } catch (error) {
    appendMessage({ role: "system", content: error.message });
    setStatus(elements.connectionStatus, "");
  } finally {
    elements.chatControls.hidden = false;
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
  speechGeneration += 1;
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
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
    elements.chatForm.hidden = true;
    elements.settingsToggle.hidden = true;
    elements.logoutButton.hidden = true;
    setBusy(true);
    if (!elements.connectionStatus.textContent.startsWith("Sign out failed:")) {
      setStatus(elements.loginStatus, "You have signed out.");
    }
  }
});

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!window.isSecureContext) {
  elements.voiceButton.disabled = true;
  elements.voiceButton.title = "Voice input requires HTTPS on phones. Configure TLS in the LOA web server.";
} else if (!SpeechRecognition) {
  elements.voiceButton.disabled = true;
  elements.voiceButton.title = "Voice input is not supported by this browser.";
} else {
  voiceSupported = true;
  recognition = new SpeechRecognition();
  recognition.lang = navigator.language || "en-US";
  recognition.interimResults = true;
  recognition.continuous = false;

  recognition.addEventListener("start", () => {
    voiceText = "";
    elements.voiceButton.textContent = "Listening";
    elements.voiceButton.classList.add("is-listening");
    setStatus(elements.connectionStatus, "Listening. Speak now.");
  });

  recognition.addEventListener("result", (event) => {
    let interimText = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const transcript = event.results[index][0].transcript;
      if (event.results[index].isFinal) {
        voiceText += `${transcript} `;
      } else {
        interimText += transcript;
      }
    }
    elements.messageInput.value = `${voiceText}${interimText}`.trimStart();
  });

  recognition.addEventListener("error", (event) => {
    setStatus(
      elements.connectionStatus,
      event.error === "not-allowed"
        ? "Microphone access was denied. Allow it in your browser settings."
        : `Voice input failed: ${event.error}.`,
    );
  });

  recognition.addEventListener("end", () => {
    elements.voiceButton.textContent = "Speak";
    elements.voiceButton.classList.remove("is-listening");
    const spokenMessage = voiceText.trim();
    voiceText = "";
    if (spokenMessage && !commandPending && configured) {
      elements.messageInput.value = spokenMessage;
      speakNextReply = true;
      elements.chatForm.requestSubmit();
    } else {
      elements.messageInput.focus();
    }
  });

  elements.voiceButton.addEventListener("click", () => {
    if (elements.voiceButton.classList.contains("is-listening")) {
      recognition.stop();
      return;
    }
    try {
      recognition.start();
    } catch (error) {
      setStatus(elements.connectionStatus, `Could not start voice input: ${error.message}`);
    }
  });
}

loadApplication();
