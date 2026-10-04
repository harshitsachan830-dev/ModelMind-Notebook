import { ServerConnection } from '@jupyterlab/services';
import { requestAPI } from './request';

export type ChatProvider = 'ollama' | 'gemini';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  provider?: ChatProvider;
  timestamp: Date;
  isError?: boolean;
  isStreaming?: boolean;
  codeBlocks?: string[];
}

export interface ChatContext {
  activeCode?: string;
  errorContext?: {
    error_type: string;
    error_message: string;
    traceback: string;
  };
}

async function requestChatAPI(
  settings: ServerConnection.ISettings,
  payload: object
): Promise<{
  status: string;
  reply?: string;
  provider?: ChatProvider;
  model_id?: string;
  message?: string;
}> {
  return requestAPI<{
    status: string;
    reply?: string;
    provider?: ChatProvider;
    model_id?: string;
    message?: string;
  }>('api/ai/chat', settings, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

function parseCodeBlocks(text: string): { segments: Array<{ type: 'text' | 'code'; content: string; lang?: string }> } {
  const segments: Array<{ type: 'text' | 'code'; content: string; lang?: string }> = [];
  const codeBlockRegex = /```(\w*)\n?([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: 'text', content: text.slice(lastIndex, match.index) });
    }
    segments.push({ type: 'code', lang: match[1] || 'python', content: match[2].trim() });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    segments.push({ type: 'text', content: text.slice(lastIndex) });
  }

  return { segments };
}

function renderMessageContent(text: string): HTMLElement {
  const wrapper = document.createElement('div');
  wrapper.className = 'mm-chat-msg-content';

  const { segments } = parseCodeBlocks(text);
  segments.forEach(seg => {
    if (seg.type === 'code') {
      const codeWrap = document.createElement('div');
      codeWrap.className = 'mm-chat-code-block';

      const codeLang = document.createElement('div');
      codeLang.className = 'mm-chat-code-lang';
      const langLabel = document.createElement('span');
      langLabel.textContent = seg.lang || 'code';
      const copyBtn = document.createElement('button');
      copyBtn.className = 'mm-chat-copy-btn';
      copyBtn.textContent = 'Copy';
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(seg.content);
          copyBtn.textContent = '✓ Copied';
          setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
        } catch {
          copyBtn.textContent = 'Failed';
        }
      });
      codeLang.append(langLabel, copyBtn);

      const pre = document.createElement('pre');
      const code = document.createElement('code');
      code.textContent = seg.content;
      pre.appendChild(code);
      codeWrap.append(codeLang, pre);
      wrapper.appendChild(codeWrap);
    } else {
      // Render text with basic markdown (bold, inline code)
      const p = document.createElement('div');
      p.className = 'mm-chat-text-seg';
      // Convert **bold**, *italic*, `inline code`
      p.innerHTML = seg.content
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code class="mm-chat-inline-code">$1</code>')
        .replace(/\n/g, '<br>');
      wrapper.appendChild(p);
    }
  });

  return wrapper;
}

function createMessageBubble(msg: ChatMessage): HTMLElement {
  const wrapper = document.createElement('div');
  wrapper.className = `mm-chat-msg mm-chat-msg-${msg.role}`;

  const avatar = document.createElement('div');
  avatar.className = `mm-chat-avatar mm-chat-avatar-${msg.role}`;
  avatar.textContent = msg.role === 'user' ? '👤' : '🤖';

  const bubble = document.createElement('div');
  bubble.className = `mm-chat-bubble mm-chat-bubble-${msg.role}`;

  if (msg.isError) {
    bubble.classList.add('mm-chat-bubble-error');
  }

  if (msg.isStreaming) {
    bubble.classList.add('mm-chat-bubble-streaming');
    const dots = document.createElement('div');
    dots.className = 'mm-chat-typing-dots';
    dots.innerHTML = '<span></span><span></span><span></span>';
    bubble.appendChild(dots);
  } else {
    const content = renderMessageContent(msg.content);
    bubble.appendChild(content);

    if (msg.provider) {
      const badge = document.createElement('div');
      badge.className = `mm-chat-provider-badge mm-chat-badge-${msg.provider}`;
      badge.textContent = msg.provider === 'gemini' ? '✨ Gemini' : '🦙 Ollama';
      bubble.appendChild(badge);
    }
  }

  const time = document.createElement('div');
  time.className = 'mm-chat-time';
  time.textContent = msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  if (msg.role === 'user') {
    wrapper.append(bubble, avatar);
  } else {
    wrapper.append(avatar, bubble);
  }

  return wrapper;
}

function injectChatStyles(): void {
  if (document.getElementById('mm-chatbot-styles')) return;
  const style = document.createElement('style');
  style.id = 'mm-chatbot-styles';
  style.textContent = `
    /* ═══════════════════════════════════════════════
       AI CHATBOT PANEL — ModelMind
    ═══════════════════════════════════════════════ */
    .mm-chatbot-root {
      display: flex;
      flex-direction: column;
      height: 100%;
      background: #0a0f1e;
      color: #e2e8f0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      overflow: hidden;
      position: relative;
    }

    /* Animated background gradient */
    .mm-chatbot-root::before {
      content: '';
      position: absolute;
      inset: 0;
      background:
        radial-gradient(ellipse at 20% 10%, rgba(124, 58, 237, 0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 90%, rgba(37, 99, 235, 0.10) 0%, transparent 50%);
      pointer-events: none;
      z-index: 0;
    }

    /* Header */
    .mm-chatbot-header {
      flex-shrink: 0;
      padding: 12px 14px 10px;
      background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
      border-bottom: 1px solid rgba(167, 139, 250, 0.2);
      z-index: 1;
    }

    .mm-chatbot-title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }

    .mm-chatbot-title {
      font-size: 13px;
      font-weight: 800;
      display: flex;
      align-items: center;
      gap: 7px;
      background: linear-gradient(90deg, #a78bfa, #60a5fa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .mm-chatbot-title-icon {
      font-size: 18px;
      -webkit-text-fill-color: initial;
    }

    .mm-chatbot-provider-switch {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .mm-chatbot-provider-btn {
      padding: 3px 10px;
      border-radius: 20px;
      border: 1px solid rgba(255,255,255,0.15);
      background: transparent;
      color: #94a3b8;
      font-size: 10px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      letter-spacing: 0.3px;
    }

    .mm-chatbot-provider-btn.active-ollama {
      background: linear-gradient(135deg, #7c3aed, #4f46e5);
      border-color: transparent;
      color: #fff;
      box-shadow: 0 2px 8px rgba(124,58,237,0.4);
    }

    .mm-chatbot-provider-btn.active-gemini {
      background: linear-gradient(135deg, #0ea5e9, #2563eb);
      border-color: transparent;
      color: #fff;
      box-shadow: 0 2px 8px rgba(37,99,235,0.4);
    }

    .mm-chatbot-provider-btn:hover:not(.active-ollama):not(.active-gemini) {
      border-color: rgba(167,139,250,0.4);
      color: #a78bfa;
    }

    /* Status bar */
    .mm-chatbot-status-bar {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 10px;
      color: #64748b;
    }

    .mm-chatbot-status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 4px #10b981;
      animation: mm-chat-pulse 2s infinite;
    }

    .mm-chatbot-status-dot.offline {
      background: #ef4444;
      box-shadow: 0 0 4px #ef4444;
      animation: none;
    }

    @keyframes mm-chat-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    /* Context bar (shows current cell info) */
    .mm-chatbot-context-bar {
      flex-shrink: 0;
      padding: 5px 14px;
      background: rgba(167,139,250,0.06);
      border-bottom: 1px solid rgba(167,139,250,0.12);
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 10px;
      color: #94a3b8;
      z-index: 1;
    }

    .mm-chatbot-context-chip {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 8px;
      border-radius: 99px;
      background: rgba(167,139,250,0.15);
      border: 1px solid rgba(167,139,250,0.25);
      color: #a78bfa;
      font-size: 9px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }

    .mm-chatbot-context-chip:hover {
      background: rgba(167,139,250,0.25);
    }

    /* Messages area */
    .mm-chatbot-messages {
      flex: 1;
      overflow-y: auto;
      padding: 14px 10px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 1;
      scroll-behavior: smooth;
    }

    .mm-chatbot-messages::-webkit-scrollbar {
      width: 4px;
    }
    .mm-chatbot-messages::-webkit-scrollbar-track {
      background: transparent;
    }
    .mm-chatbot-messages::-webkit-scrollbar-thumb {
      background: rgba(167,139,250,0.3);
      border-radius: 99px;
    }

    /* Welcome state */
    .mm-chat-welcome {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 20px;
      text-align: center;
    }

    .mm-chat-welcome-icon {
      font-size: 40px;
      animation: mm-float 3s ease-in-out infinite;
    }

    @keyframes mm-float {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-6px); }
    }

    .mm-chat-welcome-title {
      font-size: 15px;
      font-weight: 700;
      background: linear-gradient(90deg, #a78bfa, #60a5fa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .mm-chat-welcome-sub {
      font-size: 11px;
      color: #64748b;
      line-height: 1.5;
      max-width: 220px;
    }

    .mm-chat-suggestions {
      display: flex;
      flex-direction: column;
      gap: 5px;
      width: 100%;
      margin-top: 4px;
    }

    .mm-chat-suggestion-chip {
      padding: 7px 12px;
      border-radius: 10px;
      border: 1px solid rgba(167,139,250,0.2);
      background: rgba(167,139,250,0.05);
      color: #94a3b8;
      font-size: 10px;
      cursor: pointer;
      text-align: left;
      transition: all 0.2s;
      line-height: 1.4;
    }

    .mm-chat-suggestion-chip:hover {
      border-color: rgba(167,139,250,0.5);
      background: rgba(167,139,250,0.12);
      color: #c4b5fd;
      transform: translateX(2px);
    }

    /* Message bubbles */
    .mm-chat-msg {
      display: flex;
      align-items: flex-end;
      gap: 7px;
      max-width: 100%;
    }

    .mm-chat-msg-user {
      flex-direction: row-reverse;
    }

    .mm-chat-avatar {
      width: 26px;
      height: 26px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      flex-shrink: 0;
    }

    .mm-chat-avatar-user {
      background: linear-gradient(135deg, #7c3aed, #4f46e5);
    }

    .mm-chat-avatar-assistant {
      background: linear-gradient(135deg, #0ea5e9, #2563eb);
    }

    .mm-chat-bubble {
      max-width: calc(100% - 40px);
      padding: 9px 12px;
      border-radius: 14px;
      font-size: 11.5px;
      line-height: 1.5;
      word-break: break-word;
    }

    .mm-chat-bubble-user {
      background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(79,70,229,0.25));
      border: 1px solid rgba(124,58,237,0.3);
      border-bottom-right-radius: 4px;
      color: #e2e8f0;
    }

    .mm-chat-bubble-assistant {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255,255,255,0.08);
      border-bottom-left-radius: 4px;
      color: #cbd5e1;
      backdrop-filter: blur(10px);
    }

    .mm-chat-bubble-error {
      background: rgba(239, 68, 68, 0.1);
      border-color: rgba(239, 68, 68, 0.3);
      color: #fca5a5;
    }

    /* Message content */
    .mm-chat-msg-content { display: flex; flex-direction: column; gap: 6px; }

    .mm-chat-text-seg { color: #cbd5e1; font-size: 11.5px; line-height: 1.6; }
    .mm-chat-text-seg strong { color: #e2e8f0; font-weight: 700; }
    .mm-chat-text-seg em { color: #a78bfa; }
    .mm-chat-inline-code {
      background: rgba(167,139,250,0.15);
      border: 1px solid rgba(167,139,250,0.2);
      border-radius: 4px;
      padding: 1px 5px;
      font-family: 'Fira Code', 'Cascadia Code', monospace;
      font-size: 10.5px;
      color: #a78bfa;
    }

    .mm-chat-code-block {
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.08);
      background: #050d1a;
    }

    .mm-chat-code-lang {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 5px 10px;
      background: rgba(255,255,255,0.04);
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .mm-chat-code-lang span {
      font-size: 9px;
      color: #64748b;
      font-weight: 600;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }

    .mm-chat-copy-btn {
      font-size: 9px;
      padding: 2px 8px;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.1);
      background: transparent;
      color: #64748b;
      cursor: pointer;
      transition: all 0.15s;
    }
    .mm-chat-copy-btn:hover {
      border-color: rgba(167,139,250,0.4);
      color: #a78bfa;
    }

    .mm-chat-code-block pre {
      margin: 0;
      padding: 10px 12px;
      overflow-x: auto;
    }
    .mm-chat-code-block code {
      font-family: 'Fira Code', 'Cascadia Code', 'Courier New', monospace;
      font-size: 10.5px;
      color: #7dd3fc;
      white-space: pre;
    }

    /* Provider badge */
    .mm-chat-provider-badge {
      margin-top: 5px;
      font-size: 9px;
      font-weight: 600;
      letter-spacing: 0.3px;
      opacity: 0.65;
    }
    .mm-chat-badge-ollama { color: #a78bfa; }
    .mm-chat-badge-gemini { color: #60a5fa; }

    /* Timestamp */
    .mm-chat-time {
      font-size: 9px;
      color: #475569;
      padding: 0 4px;
    }

    /* Typing animation */
    .mm-chat-bubble-streaming {
      padding: 12px 16px;
    }

    .mm-chat-typing-dots {
      display: flex;
      gap: 4px;
      align-items: center;
    }

    .mm-chat-typing-dots span {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #60a5fa;
      animation: mm-typing 1.2s infinite;
    }

    .mm-chat-typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .mm-chat-typing-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes mm-typing {
      0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
      30% { transform: translateY(-4px); opacity: 1; }
    }

    /* Input area */
    .mm-chatbot-input-area {
      flex-shrink: 0;
      padding: 10px 12px 12px;
      border-top: 1px solid rgba(255,255,255,0.06);
      background: rgba(10, 15, 30, 0.8);
      backdrop-filter: blur(10px);
      z-index: 1;
    }

    .mm-chatbot-input-row {
      display: flex;
      gap: 8px;
      align-items: flex-end;
    }

    .mm-chatbot-textarea {
      flex: 1;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 12px;
      color: #e2e8f0;
      font-size: 12px;
      padding: 9px 12px;
      resize: none;
      outline: none;
      font-family: inherit;
      line-height: 1.5;
      max-height: 120px;
      min-height: 38px;
      transition: border-color 0.2s, box-shadow 0.2s;
      scrollbar-width: thin;
    }

    .mm-chatbot-textarea:focus {
      border-color: rgba(167,139,250,0.5);
      box-shadow: 0 0 0 3px rgba(167,139,250,0.1);
    }

    .mm-chatbot-textarea::placeholder {
      color: #475569;
    }

    .mm-chatbot-send-btn {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      border: none;
      background: linear-gradient(135deg, #7c3aed, #4f46e5);
      color: #fff;
      font-size: 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
      flex-shrink: 0;
      box-shadow: 0 2px 10px rgba(124,58,237,0.4);
    }

    .mm-chatbot-send-btn:hover:not(:disabled) {
      transform: translateY(-1px) scale(1.05);
      box-shadow: 0 4px 14px rgba(124,58,237,0.5);
    }

    .mm-chatbot-send-btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
      transform: none;
    }

    .mm-chatbot-send-btn.gemini-mode {
      background: linear-gradient(135deg, #0ea5e9, #2563eb);
      box-shadow: 0 2px 10px rgba(37,99,235,0.4);
    }

    .mm-chatbot-send-btn.gemini-mode:hover:not(:disabled) {
      box-shadow: 0 4px 14px rgba(37,99,235,0.5);
    }

    /* Input helpers */
    .mm-chatbot-input-helpers {
      display: flex;
      gap: 6px;
      margin-bottom: 7px;
      flex-wrap: wrap;
    }

    .mm-chat-helper-btn {
      padding: 3px 9px;
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.08);
      background: transparent;
      color: #64748b;
      font-size: 9px;
      cursor: pointer;
      transition: all 0.15s;
    }

    .mm-chat-helper-btn:hover {
      border-color: rgba(167,139,250,0.3);
      color: #a78bfa;
      background: rgba(167,139,250,0.05);
    }

    /* Clear button */
    .mm-chatbot-clear-btn {
      padding: 3px 9px;
      border-radius: 6px;
      border: 1px solid rgba(239,68,68,0.2);
      background: transparent;
      color: #64748b;
      font-size: 9px;
      cursor: pointer;
      transition: all 0.15s;
      margin-left: auto;
    }
    .mm-chatbot-clear-btn:hover {
      border-color: rgba(239,68,68,0.4);
      color: #ef4444;
    }

    /* Scrollbar */
    .mm-chatbot-messages::-webkit-scrollbar { width: 3px; }
    .mm-chatbot-messages::-webkit-scrollbar-thumb {
      background: rgba(167,139,250,0.25);
      border-radius: 99px;
    }
  `;
  document.head.appendChild(style);
}

export function createAIChatbotPanel(
  settings: ServerConnection.ISettings,
  getActiveCode: () => string,
  getActiveError: () => { error_type: string; error_message: string; traceback: string; code: string } | null
): HTMLElement {
  injectChatStyles();

  const messages: ChatMessage[] = [];
  let currentProvider: ChatProvider = 'ollama';
  let isTyping = false;
  const conversationHistory: Array<{ role: 'user' | 'assistant'; content: string }> = [];

  // Root
  const root = document.createElement('div');
  root.className = 'mm-chatbot-root';

  // Header
  const header = document.createElement('div');
  header.className = 'mm-chatbot-header';

  const titleRow = document.createElement('div');
  titleRow.className = 'mm-chatbot-title-row';

  const title = document.createElement('div');
  title.className = 'mm-chatbot-title';
  title.innerHTML = '<span class="mm-chatbot-title-icon">💬</span> AI Chatbot';

  const providerSwitch = document.createElement('div');
  providerSwitch.className = 'mm-chatbot-provider-switch';

  const ollamaBtn = document.createElement('button');
  ollamaBtn.className = 'mm-chatbot-provider-btn active-ollama';
  ollamaBtn.title = 'Use local Ollama (offline, private)';
  ollamaBtn.textContent = '🦙 Ollama';

  const geminiBtn = document.createElement('button');
  geminiBtn.className = 'mm-chatbot-provider-btn';
  geminiBtn.title = 'Switch to Google Gemini (cloud)';
  geminiBtn.textContent = '✨ Gemini';

  providerSwitch.append(ollamaBtn, geminiBtn);
  titleRow.append(title, providerSwitch);

  const statusBar = document.createElement('div');
  statusBar.className = 'mm-chatbot-status-bar';
  const statusDot = document.createElement('div');
  statusDot.className = 'mm-chatbot-status-dot';
  const statusText = document.createElement('span');
  statusText.textContent = 'Ollama ready';
  statusBar.append(statusDot, statusText);

  header.append(titleRow, statusBar);

  // Context bar
  const contextBar = document.createElement('div');
  contextBar.className = 'mm-chatbot-context-bar';
  contextBar.innerHTML = '<span style="opacity:0.5">Context:</span>';

  const codeChip = document.createElement('button');
  codeChip.className = 'mm-chatbot-context-chip';
  codeChip.title = 'Paste active cell code into your message';
  codeChip.innerHTML = '📋 Active Cell';

  const errorChip = document.createElement('button');
  errorChip.className = 'mm-chatbot-context-chip';
  errorChip.style.display = 'none';
  errorChip.title = 'Paste last error into your message';
  errorChip.innerHTML = '🐛 Last Error';

  contextBar.append(codeChip, errorChip);

  // Messages area
  const messagesArea = document.createElement('div');
  messagesArea.className = 'mm-chatbot-messages';

  // Welcome screen
  const welcome = document.createElement('div');
  welcome.className = 'mm-chat-welcome';
  welcome.innerHTML = `
    <div class="mm-chat-welcome-icon">🤖</div>
    <div class="mm-chat-welcome-title">ModelMind AI Chatbot</div>
    <div class="mm-chat-welcome-sub">Ask me anything — code explanations, debugging help, ML concepts, or anything Python.</div>
  `;

  const suggestions = document.createElement('div');
  suggestions.className = 'mm-chat-suggestions';

  const SAMPLE_QUESTIONS = [
    '💡 Explain how gradient descent works',
    '🐛 Why does my model overfit?',
    '📊 How do I handle missing values in pandas?',
    '🔢 Explain the difference between L1 and L2 regularization',
    '📋 Explain my active cell code',
  ];

  SAMPLE_QUESTIONS.forEach(q => {
    const chip = document.createElement('button');
    chip.className = 'mm-chat-suggestion-chip';
    chip.textContent = q;
    chip.addEventListener('click', () => {
      let question = q.replace(/^[^\s]+\s/, ''); // remove emoji prefix
      if (q.includes('active cell')) {
        const code = getActiveCode();
        if (code) {
          question = `Explain this code:\n\`\`\`python\n${code}\n\`\`\``;
        }
      }
      textarea.value = question;
      textarea.dispatchEvent(new Event('input'));
      sendMessage();
    });
    suggestions.appendChild(chip);
  });

  welcome.appendChild(suggestions);
  messagesArea.appendChild(welcome);

  // Input area
  const inputArea = document.createElement('div');
  inputArea.className = 'mm-chatbot-input-area';

  const helpers = document.createElement('div');
  helpers.className = 'mm-chatbot-input-helpers';

  const helperBtns = [
    { icon: '📋', label: 'Paste Code', action: () => {
      const code = getActiveCode();
      if (code) {
        textarea.value += (textarea.value ? '\n' : '') + `\`\`\`python\n${code.slice(0, 3000)}\n\`\`\``;
        adjustTextareaHeight();
      }
    }},
    { icon: '🐛', label: 'Paste Error', action: () => {
      const err = getActiveError();
      if (err) {
        textarea.value += (textarea.value ? '\n' : '') +
          `Error: ${err.error_type}: ${err.error_message}\n` +
          (err.traceback ? `Traceback:\n\`\`\`\n${err.traceback.slice(0, 1000)}\n\`\`\`` : '');
        adjustTextareaHeight();
      }
    }},
    { icon: '📝', label: 'Explain Code', action: () => {
      const code = getActiveCode();
      if (code) {
        textarea.value = `Please explain this code in detail:\n\`\`\`python\n${code.slice(0, 3000)}\n\`\`\``;
        adjustTextareaHeight();
      }
    }},
  ];

  helperBtns.forEach(({ icon, label, action }) => {
    const btn = document.createElement('button');
    btn.className = 'mm-chat-helper-btn';
    btn.textContent = `${icon} ${label}`;
    btn.addEventListener('click', action);
    helpers.appendChild(btn);
  });

  const clearBtn = document.createElement('button');
  clearBtn.className = 'mm-chatbot-clear-btn';
  clearBtn.textContent = '🗑 Clear';
  clearBtn.addEventListener('click', () => {
    messages.length = 0;
    conversationHistory.length = 0;
    messagesArea.replaceChildren(welcome);
  });
  helpers.appendChild(clearBtn);

  const inputRow = document.createElement('div');
  inputRow.className = 'mm-chatbot-input-row';

  const textarea = document.createElement('textarea');
  textarea.className = 'mm-chatbot-textarea';
  textarea.placeholder = 'Ask anything about code, ML, or Python... (Shift+Enter for newline)';
  textarea.rows = 1;

  const sendBtn = document.createElement('button');
  sendBtn.className = 'mm-chatbot-send-btn';
  sendBtn.title = 'Send message';
  sendBtn.textContent = '▶';

  inputRow.append(textarea, sendBtn);
  inputArea.append(helpers, inputRow);

  function adjustTextareaHeight(): void {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
  }

  textarea.addEventListener('input', adjustTextareaHeight);

  // Provider switching
  function switchProvider(p: ChatProvider): void {
    currentProvider = p;
    if (p === 'ollama') {
      ollamaBtn.className = 'mm-chatbot-provider-btn active-ollama';
      geminiBtn.className = 'mm-chatbot-provider-btn';
      statusDot.className = 'mm-chatbot-status-dot';
      statusText.textContent = 'Ollama (local) active';
      sendBtn.className = 'mm-chatbot-send-btn';
    } else {
      geminiBtn.className = 'mm-chatbot-provider-btn active-gemini';
      ollamaBtn.className = 'mm-chatbot-provider-btn';
      statusDot.className = 'mm-chatbot-status-dot';
      statusText.textContent = 'Gemini (cloud) active';
      sendBtn.className = 'mm-chatbot-send-btn gemini-mode';
    }
  }

  ollamaBtn.addEventListener('click', () => switchProvider('ollama'));
  geminiBtn.addEventListener('click', () => switchProvider('gemini'));

  // Context chips
  codeChip.addEventListener('click', () => {
    const code = getActiveCode();
    if (code) {
      textarea.value += (textarea.value ? '\n' : '') + `\`\`\`python\n${code.slice(0, 3000)}\n\`\`\``;
      adjustTextareaHeight();
      textarea.focus();
    }
  });

  errorChip.addEventListener('click', () => {
    const err = getActiveError();
    if (err) {
      const errText = `Error: ${err.error_type}: ${err.error_message}\n\`\`\`\n${err.traceback.slice(0, 1000)}\n\`\`\``;
      textarea.value += (textarea.value ? '\n' : '') + errText;
      adjustTextareaHeight();
      textarea.focus();
    }
  });

  function appendMessage(msg: ChatMessage): HTMLElement {
    // Remove welcome screen on first message
    const welcomeEl = messagesArea.querySelector('.mm-chat-welcome');
    if (welcomeEl) {
      welcomeEl.remove();
    }

    const el = createMessageBubble(msg);
    messagesArea.appendChild(el);
    messagesArea.scrollTop = messagesArea.scrollHeight;
    return el;
  }

  async function sendMessage(): Promise<void> {
    const text = textarea.value.trim();
    if (!text || isTyping) return;

    isTyping = true;
    sendBtn.disabled = true;
    textarea.value = '';
    adjustTextareaHeight();

    // Add user message
    const userMsg: ChatMessage = { role: 'user', content: text, timestamp: new Date() };
    messages.push(userMsg);
    appendMessage(userMsg);

    // Update history
    conversationHistory.push({ role: 'user', content: text });

    // Show typing indicator
    const typingMsg: ChatMessage = { role: 'assistant', content: '', timestamp: new Date(), isStreaming: true };
    const typingEl = appendMessage(typingMsg);

    try {
      const result = await requestChatAPI(settings, {
        message: text,
        provider: currentProvider,
        history: conversationHistory.slice(-10), // last 5 exchanges
        context: {
          active_code: getActiveCode().slice(0, 3000),
        }
      });

      typingEl.remove();

      if (result.status === 'replied' && result.reply) {
        const replyMsg: ChatMessage = {
          role: 'assistant',
          content: result.reply,
          provider: result.provider || currentProvider,
          timestamp: new Date()
        };
        messages.push(replyMsg);
        appendMessage(replyMsg);
        conversationHistory.push({ role: 'assistant', content: result.reply });
      } else {
        let errText = 'I could not respond right now.';
        if (result.status === 'model_missing') {
          errText = '🦙 Ollama model not found. Please install it first via the AI Assistant.';
        } else if (result.status === 'ollama_unavailable') {
          errText = '🦙 Ollama is not running. Start Ollama, or switch to Gemini.';
        } else if (result.status === 'gemini_unconfigured') {
          errText = '✨ Gemini API key not configured. Set GEMINI_API_KEY environment variable.';
        } else if (result.status === 'gemini_unavailable') {
          errText = '✨ Gemini is temporarily unavailable. Try again or switch to Ollama.';
        } else if (result.message) {
          errText = result.message;
        }
        const errMsg: ChatMessage = {
          role: 'assistant',
          content: errText,
          timestamp: new Date(),
          isError: true
        };
        messages.push(errMsg);
        appendMessage(errMsg);
      }
    } catch (e) {
      typingEl.remove();
      const errMsg: ChatMessage = {
        role: 'assistant',
        content: '⚠️ Network error — could not reach the backend. Is the platform running?',
        timestamp: new Date(),
        isError: true
      };
      messages.push(errMsg);
      appendMessage(errMsg);
    } finally {
      isTyping = false;
      sendBtn.disabled = false;
      textarea.focus();
    }
  }

  sendBtn.addEventListener('click', sendMessage);

  textarea.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  });

  // Update error chip visibility based on active error
  setInterval(() => {
    const err = getActiveError();
    errorChip.style.display = err ? 'inline-flex' : 'none';
  }, 2000);

  root.append(header, contextBar, messagesArea, inputArea);
  return root;
}
