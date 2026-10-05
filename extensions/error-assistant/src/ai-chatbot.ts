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
  avatar.textContent = msg.role === 'user' ? '👤' : (msg.provider === 'gemini' ? '✨' : '🦙');

  const container = document.createElement('div');
  container.className = 'mm-chat-bubble-container';

  // Bubble meta header
  const metaHeader = document.createElement('div');
  metaHeader.className = `mm-chat-bubble-header mm-chat-bubble-header-${msg.role}`;

  const senderName = document.createElement('span');
  senderName.className = 'mm-chat-sender-name';
  senderName.textContent = msg.role === 'user' ? 'You' : 'ModelMind AI';

  metaHeader.appendChild(senderName);

  if (msg.role === 'assistant' && msg.provider) {
    const badge = document.createElement('span');
    badge.className = `mm-chat-provider-badge mm-chat-badge-${msg.provider}`;
    badge.textContent = msg.provider === 'gemini' ? 'Gemini Cloud' : 'Ollama Local';
    metaHeader.appendChild(badge);
  }

  const time = document.createElement('span');
  time.className = 'mm-chat-time';
  time.textContent = msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  metaHeader.appendChild(time);

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
  }

  container.append(metaHeader, bubble);

  if (msg.role === 'user') {
    wrapper.append(container, avatar);
  } else {
    wrapper.append(avatar, container);
  }

  return wrapper;
}

function injectChatStyles(): void {
  if (document.getElementById('mm-chatbot-styles')) {
    const existing = document.getElementById('mm-chatbot-styles');
    if (existing) existing.remove();
  }
  const style = document.createElement('style');
  style.id = 'mm-chatbot-styles';
  style.textContent = `
    /* ═══════════════════════════════════════════════
       AI CHATBOT PANEL — AI Assistant Matching Theme
    ═══════════════════════════════════════════════ */
    .mm-chatbot-root {
      display: flex;
      flex-direction: column;
      height: 100%;
      background: var(--ml-assistant-panel-bg, #f5f6f8);
      color: var(--ml-assistant-text, #253041);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
      position: relative;
      border-left: 1px solid var(--ml-assistant-border, #dfe3ea);
    }

    /* Header matching AI Assistant */
    .mm-chatbot-header {
      flex-shrink: 0;
      padding: 14px 14px 10px;
      background: var(--ml-assistant-card-bg, #ffffff);
      border-bottom: 1px solid var(--ml-assistant-border, #dfe3ea);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .mm-chatbot-title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .mm-chatbot-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--ml-assistant-text, #253041);
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0;
    }

    .mm-chatbot-title-icon {
      font-size: 18px;
      line-height: 1;
    }

    /* Provider Switch styled like AI Assistant tabs / buttons */
    .mm-chatbot-provider-switch {
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--ml-assistant-panel-bg, #f5f6f8);
      padding: 3px;
      border-radius: 8px;
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
    }

    .mm-chatbot-provider-btn {
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid transparent;
      background: transparent;
      color: var(--ml-assistant-muted, #5d6b80);
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .mm-chatbot-provider-btn:hover:not(.active-ollama):not(.active-gemini) {
      color: var(--ml-assistant-text, #253041);
      background: rgba(255, 255, 255, 0.7);
    }

    .mm-chatbot-provider-btn.active-ollama {
      background: var(--ml-assistant-card-bg, #ffffff);
      border-color: var(--ml-assistant-border, #dfe3ea);
      color: #6b21a8;
      box-shadow: 0 1px 3px var(--ml-assistant-shadow, rgba(31, 41, 55, 0.08));
    }

    .mm-chatbot-provider-btn.active-gemini {
      background: var(--ml-assistant-card-bg, #ffffff);
      border-color: var(--ml-assistant-border, #dfe3ea);
      color: #173ea5;
      box-shadow: 0 1px 3px var(--ml-assistant-shadow, rgba(31, 41, 55, 0.08));
    }

    /* Status row matching AI Assistant */
    .mm-chatbot-status-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .mm-chatbot-status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 600;
      background: var(--ml-assistant-success-bg, #dff5eb);
      color: var(--ml-assistant-success-text, #1d7b54);
      border: 1px solid transparent;
    }

    .mm-chatbot-status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #10b981;
    }

    .mm-chatbot-status-pill.offline {
      background: var(--ml-assistant-warn-bg, #fdf0d5);
      color: var(--ml-assistant-warn-text, #9c6b11);
    }

    .mm-chatbot-status-pill.offline .mm-chatbot-status-dot {
      background: #f59e0b;
    }

    /* Context row matching AI Assistant secondary controls */
    .mm-chatbot-context-bar {
      flex-shrink: 0;
      padding: 7px 14px;
      background: var(--ml-assistant-panel-bg, #f5f6f8);
      border-bottom: 1px solid var(--ml-assistant-border, #dfe3ea);
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      color: var(--ml-assistant-muted, #5d6b80);
      font-weight: 600;
    }

    .mm-chatbot-context-chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 10px;
      border-radius: 999px;
      background: var(--ml-assistant-card-bg, #ffffff);
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      color: var(--ml-assistant-text, #253041);
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }

    .mm-chatbot-context-chip:hover {
      background: var(--ml-assistant-accent-soft, #eef1ff);
      border-color: var(--ml-assistant-accent, #6d7cff);
      color: var(--ml-assistant-accent, #6d7cff);
    }

    /* Messages thread */
    .mm-chatbot-messages {
      flex: 1;
      overflow-y: auto;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: var(--ml-assistant-panel-bg, #f5f6f8);
      scroll-behavior: smooth;
    }

    .mm-chatbot-messages::-webkit-scrollbar {
      width: 5px;
    }
    .mm-chatbot-messages::-webkit-scrollbar-track {
      background: transparent;
    }
    .mm-chatbot-messages::-webkit-scrollbar-thumb {
      background: var(--ml-assistant-border, #dfe3ea);
      border-radius: 99px;
    }
    .mm-chatbot-messages::-webkit-scrollbar-thumb:hover {
      background: #cbd5e1;
    }

    /* Welcome / Empty State matching AI Assistant */
    .mm-chat-welcome {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      padding: 24px 16px;
      text-align: center;
      background: var(--ml-assistant-card-bg, #ffffff);
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      border-radius: 12px;
      box-shadow: 0 1px 3px var(--ml-assistant-shadow, rgba(31, 41, 55, 0.05));
      margin: auto 0;
    }

    .mm-chat-welcome-icon {
      font-size: 36px;
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: var(--ml-assistant-accent-soft, #eef1ff);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--ml-assistant-accent, #6d7cff);
    }

    .mm-chat-welcome-title {
      font-size: 15px;
      font-weight: 700;
      color: var(--ml-assistant-text, #253041);
      margin: 0;
    }

    .mm-chat-welcome-sub {
      font-size: 12px;
      color: var(--ml-assistant-muted, #5d6b80);
      line-height: 1.5;
      max-width: 260px;
      margin: 0;
    }

    .mm-chat-suggestions {
      display: flex;
      flex-direction: column;
      gap: 6px;
      width: 100%;
      margin-top: 6px;
    }

    .mm-chat-suggestion-chip {
      padding: 9px 12px;
      border-radius: 8px;
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      background: var(--ml-assistant-panel-bg, #f8fafc);
      color: var(--ml-assistant-text, #253041);
      font-size: 11.5px;
      font-weight: 500;
      cursor: pointer;
      text-align: left;
      transition: all 0.15s ease;
      line-height: 1.4;
    }

    .mm-chat-suggestion-chip:hover {
      border-color: var(--ml-assistant-accent, #6d7cff);
      background: var(--ml-assistant-accent-soft, #eef1ff);
      color: #312e81;
      transform: translateX(2px);
    }

    /* Message Bubbles */
    .mm-chat-msg {
      display: flex;
      gap: 9px;
      max-width: 100%;
    }

    .mm-chat-msg-user {
      flex-direction: row-reverse;
      align-self: flex-end;
    }

    .mm-chat-msg-assistant {
      align-self: flex-start;
    }

    .mm-chat-avatar {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      flex-shrink: 0;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .mm-chat-avatar-user {
      background: #e0e7ff;
      border: 1px solid #c7d2fe;
      color: #4338ca;
    }

    .mm-chat-avatar-assistant {
      background: var(--ml-assistant-accent-soft, #eef1ff);
      border: 1px solid rgba(109, 124, 255, 0.3);
      color: var(--ml-assistant-accent, #6d7cff);
    }

    .mm-chat-bubble-container {
      display: flex;
      flex-direction: column;
      gap: 3px;
      max-width: calc(100% - 38px);
    }

    .mm-chat-bubble-header {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 10.5px;
      color: var(--ml-assistant-muted, #5d6b80);
      padding: 0 3px;
    }

    .mm-chat-bubble-header-user {
      justify-content: flex-end;
    }

    .mm-chat-sender-name {
      font-weight: 700;
      color: var(--ml-assistant-text, #253041);
    }

    .mm-chat-provider-badge {
      display: inline-flex;
      align-items: center;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 9.5px;
      font-weight: 600;
    }

    .mm-chat-badge-gemini {
      background: #edf4ff;
      border: 1px solid #d3e3ff;
      color: #173ea5;
    }

    .mm-chat-badge-ollama {
      background: #f5f3ff;
      border: 1px solid #ddd6fe;
      color: #6b21a8;
    }

    .mm-chat-time {
      font-size: 10px;
      color: #94a3b8;
    }

    /* Bubble bodies matching AI Assistant Cards */
    .mm-chat-bubble {
      padding: 11px 13px;
      border-radius: 10px;
      font-size: 12px;
      line-height: 1.55;
      word-break: break-word;
    }

    .mm-chat-bubble-user {
      background: var(--ml-assistant-accent-soft, #eef1ff);
      border: 1px solid #d3e3ff;
      border-top-right-radius: 2px;
      color: #1e1b4b;
      box-shadow: 0 1px 2px rgba(109, 124, 255, 0.06);
    }

    .mm-chat-bubble-assistant {
      background: var(--ml-assistant-card-bg, #ffffff);
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      border-top-left-radius: 2px;
      color: var(--ml-assistant-text, #253041);
      box-shadow: 0 1px 3px var(--ml-assistant-shadow, rgba(31, 41, 55, 0.06));
    }

    .mm-chat-bubble-error {
      background: var(--ml-assistant-error-bg, #fff1f1) !important;
      border: 1px solid rgb(163 61 77 / 20%) !important;
      color: var(--ml-assistant-error-text, #a33d4d) !important;
    }

    /* Message content */
    .mm-chat-msg-content {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .mm-chat-text-seg {
      color: inherit;
      font-size: 12px;
      line-height: 1.6;
    }

    .mm-chat-text-seg strong {
      color: #0f172a;
      font-weight: 700;
    }

    .mm-chat-text-seg em {
      color: #4338ca;
    }

    .mm-chat-inline-code {
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 1px 5px;
      font-family: 'Fira Code', Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
      color: #0f172a;
    }

    /* Code block matching AI Assistant style */
    .mm-chat-code-block {
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #cbd5e1;
      background: #1e293b;
      margin: 4px 0;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .mm-chat-code-lang {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 5px 10px;
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
    }

    .mm-chat-code-lang span {
      font-size: 10px;
      color: #475569;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }

    .mm-chat-copy-btn {
      font-size: 10px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 4px;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #475569;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .mm-chat-copy-btn:hover {
      background: #f1f5f9;
      border-color: #94a3b8;
      color: #1e293b;
    }

    .mm-chat-code-block pre {
      margin: 0;
      padding: 11px 13px;
      overflow-x: auto;
      background: #1e293b;
    }

    .mm-chat-code-block code {
      font-family: 'Fira Code', Menlo, Monaco, Consolas, monospace;
      font-size: 11.5px;
      line-height: 1.55;
      color: #f8fafc;
      white-space: pre;
    }

    /* Typing animation */
    .mm-chat-bubble-streaming {
      padding: 12px 16px;
    }

    .mm-chat-typing-dots {
      display: flex;
      gap: 5px;
      align-items: center;
    }

    .mm-chat-typing-dots span {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--ml-assistant-accent, #6d7cff);
      animation: mm-typing 1.2s infinite;
    }

    .mm-chat-typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .mm-chat-typing-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes mm-typing {
      0%, 60%, 100% { transform: translateY(0); opacity: 0.35; }
      30% { transform: translateY(-5px); opacity: 1; }
    }

    /* Input Area matching AI Assistant form inputs */
    .mm-chatbot-input-area {
      flex-shrink: 0;
      padding: 10px 14px 14px;
      border-top: 1px solid var(--ml-assistant-border, #dfe3ea);
      background: var(--ml-assistant-card-bg, #ffffff);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .mm-chatbot-input-helpers {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
    }

    .mm-chat-helper-btn {
      padding: 4px 9px;
      border-radius: 6px;
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      background: var(--ml-assistant-card-bg, #ffffff);
      color: var(--ml-assistant-muted, #5d6b80);
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .mm-chat-helper-btn:hover {
      border-color: #cbd5e1;
      color: var(--ml-assistant-text, #253041);
      background: var(--ml-assistant-panel-bg, #f8fafc);
    }

    .mm-chatbot-clear-btn {
      padding: 4px 9px;
      border-radius: 6px;
      border: 1px solid rgba(163, 61, 77, 0.2);
      background: #ffffff;
      color: var(--ml-assistant-error-text, #a33d4d);
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      margin-left: auto;
    }

    .mm-chatbot-clear-btn:hover {
      background: #fff1f1;
      border-color: rgba(163, 61, 77, 0.4);
    }

    .mm-chatbot-input-row {
      display: flex;
      gap: 8px;
      align-items: flex-end;
    }

    .mm-chatbot-textarea {
      flex: 1;
      background: #ffffff;
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      border-radius: 8px;
      color: var(--ml-assistant-text, #253041);
      font-size: 12px;
      padding: 9px 12px;
      resize: none;
      outline: none;
      font-family: inherit;
      line-height: 1.45;
      max-height: 120px;
      min-height: 38px;
      box-sizing: border-box;
      transition: border-color 0.15s, box-shadow 0.15s;
    }

    .mm-chatbot-textarea:focus {
      border-color: var(--ml-assistant-accent, #6d7cff);
      box-shadow: 0 0 0 3px rgba(109, 124, 255, 0.16);
    }

    .mm-chatbot-textarea::placeholder {
      color: #94a3b8;
    }

    /* Send button matching .ml-assistant-cta-button */
    .mm-chatbot-send-btn {
      width: 38px;
      height: 38px;
      border-radius: 8px;
      border: 1px solid var(--ml-assistant-accent, #6d7cff);
      background: var(--ml-assistant-accent, #6d7cff);
      color: #ffffff;
      font-size: 14px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s ease;
      flex-shrink: 0;
      box-shadow: 0 1px 2px rgba(109, 124, 255, 0.2);
    }

    .mm-chatbot-send-btn:hover:not(:disabled) {
      background: #5b6aff;
      border-color: #5b6aff;
      box-shadow: 0 2px 4px rgba(109, 124, 255, 0.3);
    }

    .mm-chatbot-send-btn:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      box-shadow: none;
    }

    .mm-chatbot-send-btn.gemini-mode {
      background: #2563eb;
      border-color: #2563eb;
    }

    .mm-chatbot-send-btn.gemini-mode:hover:not(:disabled) {
      background: #1d4ed8;
      border-color: #1d4ed8;
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

  const localPill = document.createElement('span');
  localPill.className = 'mm-chatbot-status-pill';
  localPill.innerHTML = '<span class="mm-chatbot-status-dot"></span> Ollama (Local AI)';

  const cloudPill = document.createElement('span');
  cloudPill.className = 'mm-chatbot-status-pill';
  cloudPill.innerHTML = '<span class="mm-chatbot-status-dot"></span> Gemini (Cloud AI)';

  statusBar.append(localPill, cloudPill);
  header.append(titleRow, statusBar);

  // Context bar
  const contextBar = document.createElement('div');
  contextBar.className = 'mm-chatbot-context-bar';
  contextBar.innerHTML = '<span>Context:</span>';

  const codeChip = document.createElement('button');
  codeChip.className = 'mm-chatbot-context-chip';
  codeChip.title = 'Paste active cell code into your message';
  codeChip.innerHTML = '📋 Active Cell';

  const errorChip = document.createElement('button');
  errorChip.className = 'mm-chatbot-context-chip';
  errorChip.style.display = 'none';
  errorChip.title = 'Paste last error into your message';
  errorChip.innerHTML = '🐞 Last Error';

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
    '💡 Explain active cell code',
    '🐞 How to fix the latest error?',
    '📊 How to visualize data with matplotlib?',
    '⚡ Optimize my code performance',
    '🤖 Recommend best ML model for my data',
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
      } else if (q.includes('latest error')) {
        const err = getActiveError();
        if (err) {
          question = `How do I fix this error?\n${err.error_type}: ${err.error_message}\n\`\`\`python\n${err.code || ''}\n\`\`\``;
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
        textarea.focus();
      }
    }},
    { icon: '🐞', label: 'Paste Error', action: () => {
      const err = getActiveError();
      if (err) {
        textarea.value += (textarea.value ? '\n' : '') +
          `Error: ${err.error_type}: ${err.error_message}\n` +
          (err.traceback ? `Traceback:\n\`\`\`\n${err.traceback.slice(0, 1000)}\n\`\`\`` : '');
        adjustTextareaHeight();
        textarea.focus();
      }
    }},
    { icon: '📝', label: 'Explain Code', action: () => {
      const code = getActiveCode();
      if (code) {
        textarea.value = `Please explain this code in detail:\n\`\`\`python\n${code.slice(0, 3000)}\n\`\`\``;
        adjustTextareaHeight();
        textarea.focus();
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
  sendBtn.innerHTML = '➤';

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
      localPill.style.borderColor = 'rgba(107, 33, 168, 0.4)';
      localPill.style.background = '#f3e8ff';
      localPill.style.color = '#6b21a8';
      cloudPill.style.borderColor = 'transparent';
      cloudPill.style.background = 'var(--ml-assistant-success-bg, #dff5eb)';
      cloudPill.style.color = 'var(--ml-assistant-success-text, #1d7b54)';
      sendBtn.className = 'mm-chatbot-send-btn';
    } else {
      geminiBtn.className = 'mm-chatbot-provider-btn active-gemini';
      ollamaBtn.className = 'mm-chatbot-provider-btn';
      cloudPill.style.borderColor = 'rgba(30, 64, 175, 0.4)';
      cloudPill.style.background = '#eff6ff';
      cloudPill.style.color = '#1e40af';
      localPill.style.borderColor = 'transparent';
      localPill.style.background = 'var(--ml-assistant-success-bg, #dff5eb)';
      localPill.style.color = 'var(--ml-assistant-success-text, #1d7b54)';
      sendBtn.className = 'mm-chatbot-send-btn gemini-mode';
    }
  }

  // Refresh status from server
  async function refreshPlatformStatus(): Promise<void> {
    try {
      const health = await requestAPI<{
        status: string;
        cloud_fallback: string;
        gemini_fallback_available: boolean;
      }>('api/health', settings);

      if (health.gemini_fallback_available) {
        cloudPill.classList.remove('offline');
        cloudPill.innerHTML = '<span class="mm-chatbot-status-dot"></span> Gemini (Cloud AI)';
      } else {
        cloudPill.classList.add('offline');
        cloudPill.innerHTML = '<span class="mm-chatbot-status-dot"></span> Gemini unconfigured';
      }
    } catch {
      cloudPill.classList.add('offline');
      cloudPill.innerHTML = '<span class="mm-chatbot-status-dot"></span> Gemini unavailable';
    }

    try {
      const ollama = await requestAPI<{
        status: string;
        model_status: 'available' | 'missing' | 'unavailable' | 'not_checked';
      }>('api/local-ai/status', settings);

      if (ollama.status === 'connected' && ollama.model_status === 'available') {
        localPill.classList.remove('offline');
        localPill.innerHTML = '<span class="mm-chatbot-status-dot"></span> Ollama (Local AI)';
      } else {
        localPill.classList.add('offline');
        localPill.innerHTML = '<span class="mm-chatbot-status-dot"></span> Ollama offline';
      }
    } catch {
      localPill.classList.add('offline');
      localPill.innerHTML = '<span class="mm-chatbot-status-dot"></span> Ollama offline';
    }
    // Re-apply highlight to currently selected provider
    switchProvider(currentProvider);
  }

  void refreshPlatformStatus();

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
        conversationHistory.pop(); // Remove un-replied user message on error
        let errText = 'I could not respond right now.';
        if (result.status === 'model_missing') {
          errText = '🦙 Ollama model not found. Please install it first via the AI Assistant.';
        } else if (result.status === 'ollama_unavailable') {
          errText = '🦙 Ollama is not running. Start Ollama, or switch to Gemini.';
        } else if (result.status === 'gemini_unconfigured') {
          errText = '✨ Gemini API key not configured. Set GEMINI_API_KEY environment variable.';
        } else if (result.status === 'gemini_unavailable') {
          errText = result.message ? `✨ ${result.message}` : '✨ Gemini is temporarily unavailable. Try again or switch to Ollama.';
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
      conversationHistory.pop(); // Remove un-replied user message on error
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
