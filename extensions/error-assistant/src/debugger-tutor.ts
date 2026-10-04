/**
 * ModelMind — AI Debugger Tutor (Offline, Rule-based ML Error Assistant)
 * UI and UX match the ModelMind AI Assistant panel design system.
 */

import { requestAPI } from './request';

export interface TutorResult {
  handled: boolean;
  source?: string;
  error_type?: string;
  error_message?: string;
  failing_line?: string;
  level?: string;
  title?: string;
  icon?: string;
  what?: string;
  why?: string;
  steps?: string[];
  code_tip?: string;
  smart_hints?: string[];
  coverage?: string;
  confidence?: number;
  can_suggest_code?: boolean;
  confidence_reason?: string;
  note?: string;
}

export type Level = 'basic' | 'medium' | 'advanced';

export type TutorInput = {
  error_type: string;
  error_message: string;
  traceback: string;
  code: string;
};

let _panel: HTMLElement | null = null;
let _overlay: HTMLElement | null = null;
let _isOpen = false;
let _currentLevel: Level = 'basic';
let _lastInput: TutorInput | null = null;
let _serverSettings: Parameters<typeof requestAPI>[1] | null = null;

export function ensureStyles(): void {
  if (document.getElementById('mm-tutor-styles')) return;
  const style = document.createElement('style');
  style.id = 'mm-tutor-styles';
  style.textContent = `
    /* ── Overlay ── */
    #mm-tutor-overlay {
      position: fixed; inset: 0;
      background: rgba(15, 23, 42, 0.45);
      backdrop-filter: blur(2px);
      z-index: 9998;
      opacity: 0; transition: opacity .25s ease;
      pointer-events: none; display: none;
    }
    #mm-tutor-overlay.open { opacity: 1; pointer-events: auto; display: block; }

    /* ── Panel — Unified with ModelMind AI Assistant ── */
    #mm-tutor-panel {
      position: fixed; right: 0; bottom: 0; top: 0; width: 420px; max-width: 96vw;
      background: var(--ml-assistant-panel-bg, #f5f6f8);
      border-left: 1px solid var(--ml-assistant-border, #dfe3ea);
      z-index: 9999; overflow-y: auto; transform: translateX(100%);
      transition: transform .28s cubic-bezier(.4, 0, .2, 1);
      display: flex; flex-direction: column;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      color: var(--ml-assistant-text, #253041);
      box-shadow: -6px 0 24px var(--ml-assistant-shadow, rgba(31, 41, 55, 0.12));
    }
    #mm-tutor-panel.open { transform: translateX(0); }

    /* ── Header ── */
    #mm-tutor-header {
      padding: 14px 16px 12px;
      border-bottom: 1px solid var(--ml-assistant-border, #dfe3ea);
      background: #ffffff;
      position: sticky; top: 0; z-index: 2;
      display: flex; flex-direction: column; gap: 4px;
    }
    .mm-tutor-header-top {
      display: flex; align-items: center; justify-content: space-between;
    }
    #mm-tutor-header h2 {
      margin: 0; font-size: 15px; font-weight: 700;
      color: var(--ml-assistant-text, #253041);
      display: flex; align-items: center; gap: 8px;
    }
    #mm-tutor-header .sub {
      font-size: 11px;
      color: var(--ml-assistant-muted, #5d6b80);
      font-weight: 500;
    }
    #mm-tutor-close {
      background: transparent;
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      color: var(--ml-assistant-muted, #5d6b80);
      cursor: pointer; font-size: 16px; line-height: 1;
      padding: 4px 8px; border-radius: 6px;
      transition: all .15s ease;
    }
    #mm-tutor-close:hover {
      background: #f1f5f9;
      color: var(--ml-assistant-text, #253041);
      border-color: #cbd5e1;
    }

    /* ── Body ── */
    #mm-tutor-body, .mm-tutor-embedded-body {
      padding: 14px 16px; flex: 1;
      display: flex; flex-direction: column; gap: 10px;
    }

    /* ── Level Switcher Tabs (Basic / Medium / Advanced) ── */
    .mm-tutor-level-row {
      display: flex; gap: 6px;
      padding: 3px; background: #eaecf1;
      border-radius: 8px; margin-bottom: 6px;
    }
    .mm-tutor-level-btn {
      flex: 1; font-size: 12px; font-weight: 600; padding: 6px 10px;
      border-radius: 6px; cursor: pointer; text-align: center;
      border: none; background: transparent;
      color: var(--ml-assistant-muted, #5d6b80);
      transition: all .15s ease;
    }
    .mm-tutor-level-btn:hover {
      color: var(--ml-assistant-text, #253041);
    }
    .mm-tutor-level-btn.active {
      background: #ffffff;
      color: var(--ml-assistant-accent, #6d7cff);
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      font-weight: 700;
    }

    /* ── Error Banner ── */
    .mm-tutor-error-banner {
      background: var(--ml-assistant-error-bg, #f7dfe3);
      border: 1px solid rgba(163, 61, 77, 0.25);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .mm-tutor-error-banner .banner-title {
      display: flex; align-items: center; gap: 8px;
      font-size: 13px; font-weight: 700;
      color: var(--ml-assistant-error-text, #a33d4d);
      margin-bottom: 3px;
    }
    .mm-tutor-error-banner .banner-type {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px; font-weight: 700;
      color: #991b1b;
    }
    .mm-tutor-error-banner .banner-msg {
      font-size: 12px; color: #7f1d1d; margin-top: 4px; line-height: 1.4;
      word-break: break-word;
    }
    .mm-tutor-error-banner .banner-line {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px; color: #b91c1c; margin-top: 6px;
      padding: 4px 8px; background: rgba(255,255,255,0.6);
      border-radius: 4px; overflow-x: auto;
    }

    /* ── Local Engine / Coverage Pill ── */
    .mm-tutor-coverage {
      display: inline-flex; align-items: center; gap: 6px;
      background: var(--ml-assistant-success-bg, #dff5eb);
      color: var(--ml-assistant-success-text, #1d7b54);
      border: 1px solid rgba(29, 123, 84, 0.2);
      border-radius: 6px; padding: 6px 10px;
      font-size: 11px; font-weight: 600;
    }
    .mm-tutor-coverage.mm-tutor-low-conf {
      background: #fef3c7;
      color: #92400e;
      border: 1px solid rgba(180, 83, 9, 0.3);
    }
    .mm-tutor-withheld-box {
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-radius: 8px; padding: 12px 14px;
      margin: 2px 0 6px;
    }
    .mm-tutor-withheld-box .withheld-title {
      font-size: 12px; font-weight: 700; color: #92400e;
      margin-bottom: 4px; display: flex; align-items: center; gap: 6px;
    }
    .mm-tutor-withheld-box .withheld-desc {
      font-size: 11.5px; color: #78350f; line-height: 1.5;
    }
    .mm-tutor-ask-ai-btn {
      display: flex; align-items: center; justify-content: center; gap: 8px;
      width: 100%; margin-top: 10px; padding: 9px 14px;
      background: var(--ml-assistant-accent, #6d7cff);
      color: #ffffff; border: none; border-radius: 6px;
      font-size: 12px; font-weight: 700; cursor: pointer;
      transition: all .15s ease;
      box-shadow: 0 2px 8px rgba(109, 124, 255, 0.35);
    }
    .mm-tutor-ask-ai-btn:hover {
      background: #5865f2;
      box-shadow: 0 4px 14px rgba(109, 124, 255, 0.5);
      transform: translateY(-1px);
    }

    /* ── Contextual Smart Hints ── */
    .mm-tutor-hint {
      background: var(--ml-assistant-accent-soft, #eef1ff);
      border: 1px solid rgba(109, 124, 255, 0.3);
      border-radius: 8px; padding: 10px 12px;
      font-size: 12px; color: #3730a3; line-height: 1.45;
    }

    /* ── Content Card (What / Why / How to fix) ── */
    .mm-tutor-section {
      background: var(--ml-assistant-card-bg, #ffffff);
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      border-radius: 8px; padding: 12px 14px;
      box-shadow: 0 1px 3px var(--ml-assistant-shadow, rgba(31, 41, 55, 0.04));
    }
    .mm-tutor-section-label {
      font-size: 10px; font-weight: 700; letter-spacing: 0.6px;
      color: var(--ml-assistant-muted, #5d6b80);
      margin-bottom: 6px; text-transform: uppercase;
    }
    .mm-tutor-section-content {
      font-size: 12.5px; color: var(--ml-assistant-text, #253041);
      line-height: 1.55;
    }

    /* ── Numbered Steps ── */
    .mm-tutor-step {
      display: flex; gap: 8px; margin-bottom: 6px;
      font-size: 12px; line-height: 1.5;
      color: var(--ml-assistant-text, #253041);
    }
    .mm-tutor-step-num {
      width: 18px; height: 18px; border-radius: 50%;
      background: var(--ml-assistant-accent-soft, #eef1ff);
      color: var(--ml-assistant-accent, #6d7cff);
      font-size: 10px; font-weight: 700;
      display: flex; align-items: center; justify-content: center;
      margin-top: 1px; flex-shrink: 0;
    }

    /* ── Code Tip Block ── */
    .mm-tutor-code-wrap {
      position: relative; margin-top: 6px;
    }
    .mm-tutor-code-tip {
      background: #0f172a; border: 1px solid #1e293b;
      border-radius: 8px; padding: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11.5px; line-height: 1.6; color: #e2e8f0;
      overflow-x: auto; white-space: pre;
    }
    .mm-tutor-copy-btn {
      position: absolute; top: 8px; right: 8px;
      font-size: 10px; font-weight: 600; padding: 4px 9px;
      border-radius: 5px; cursor: pointer;
      background: #334155; border: 1px solid #475569; color: #f8fafc;
      transition: all .12s ease;
    }
    .mm-tutor-copy-btn:hover { background: #475569; }

    /* ── Empty State & Quick Presets ── */
    .mm-tutor-empty {
      background: #ffffff;
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      border-radius: 8px; padding: 20px 16px; text-align: center;
    }
    .mm-tutor-empty h3 {
      margin: 8px 0 4px; font-size: 14px; font-weight: 700;
      color: var(--ml-assistant-text, #253041);
    }
    .mm-tutor-empty p {
      margin: 0 0 14px; font-size: 12px;
      color: var(--ml-assistant-muted, #5d6b80);
      line-height: 1.5;
    }
    .mm-tutor-preset-list {
      display: flex; flex-direction: column; gap: 6px;
      text-align: left; margin-top: 12px;
    }
    .mm-tutor-preset-btn {
      display: flex; align-items: center; justify-content: space-between;
      background: var(--ml-assistant-panel-bg, #f5f6f8);
      border: 1px solid var(--ml-assistant-border, #dfe3ea);
      border-radius: 6px; padding: 8px 10px;
      font-size: 11.5px; font-weight: 600;
      color: var(--ml-assistant-text, #253041);
      cursor: pointer; transition: all .15s ease;
    }
    .mm-tutor-preset-btn:hover {
      background: var(--ml-assistant-accent-soft, #eef1ff);
      border-color: rgba(109, 124, 255, 0.35);
      color: var(--ml-assistant-accent, #6d7cff);
    }

    /* ── FAB button ── */
    #mm-tutor-fab {
      position: fixed; right: 20px; bottom: 20px; z-index: 9997;
      width: 48px; height: 48px; border-radius: 50%; border: none; cursor: pointer;
      background: linear-gradient(135deg, #6d7cff 0%, #4f46e5 100%);
      color: #fff; font-size: 20px;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 16px rgba(109, 124, 255, 0.4);
      transition: transform .15s, box-shadow .15s;
    }
    #mm-tutor-fab:hover { transform: scale(1.08); box-shadow: 0 6px 22px rgba(109, 124, 255, 0.6); }
    #mm-tutor-fab .badge {
      position: absolute; top: -2px; right: -2px; width: 15px; height: 15px;
      border-radius: 50%; background: #ef4444; font-size: 9px; font-weight: 800;
      display: none; align-items: center; justify-content: center; color: #fff;
    }
    #mm-tutor-fab .badge.show { display: flex; }
  `;
  document.head.appendChild(style);
}

export function createPanel(): void {
  ensureStyles();

  if (!_overlay) {
    _overlay = document.createElement('div');
    _overlay.id = 'mm-tutor-overlay';
    _overlay.addEventListener('click', closePanel);
    document.body.appendChild(_overlay);
  }

  if (!_panel) {
    _panel = document.createElement('div');
    _panel.id = 'mm-tutor-panel';
    _panel.setAttribute('role', 'dialog');
    _panel.setAttribute('aria-label', 'AI Debugger Tutor');

    const hdr = document.createElement('div');
    hdr.id = 'mm-tutor-header';
    hdr.innerHTML = `
      <div class="mm-tutor-header-top">
        <h2>🎓 AI Debugger Tutor</h2>
        <button id="mm-tutor-close" aria-label="Close tutor panel">✕</button>
      </div>
      <div class="sub">100% Offline · No AI required · ~80% ML Error Coverage</div>
    `;
    hdr.querySelector('#mm-tutor-close')?.addEventListener('click', closePanel);
    _panel.appendChild(hdr);

    const body = document.createElement('div');
    body.id = 'mm-tutor-body';
    renderEmptyState(body);
    _panel.appendChild(body);

    document.body.appendChild(_panel);
  }
}

export function openPanel(): void {
  if (!_panel) createPanel();
  _panel!.classList.add('open');
  _overlay?.classList.add('open');
  _isOpen = true;
}

export function closePanel(): void {
  _panel?.classList.remove('open');
  _overlay?.classList.remove('open');
  _isOpen = false;
}

export function togglePanel(): void {
  if (_isOpen) closePanel(); else openPanel();
}

export function createFAB(
  serverSettings: Parameters<typeof requestAPI>[1]
): void {
  _serverSettings = serverSettings;
  ensureStyles();
  if (!_panel) createPanel();
  if (document.getElementById('mm-tutor-fab')) return;

  const fab = document.createElement('button');
  fab.id = 'mm-tutor-fab';
  fab.title = 'AI Debugger Tutor (Offline ML Guidance)';
  fab.setAttribute('aria-label', 'Open AI Debugger Tutor');
  const badge = document.createElement('div');
  badge.className = 'badge';
  badge.textContent = '!';
  fab.innerHTML = `<span>🎓</span>`;
  fab.appendChild(badge);
  fab.addEventListener('click', togglePanel);
  document.body.appendChild(fab);
}

export function showFABAlert(): void {
  const badge = document.getElementById('mm-tutor-fab')?.querySelector('.badge');
  badge?.classList.add('show');
}

/**
 * Render empty state with 1-click test scenarios for user testing
 */
function renderEmptyState(container: HTMLElement): void {
  container.replaceChildren();

  const card = document.createElement('div');
  card.className = 'mm-tutor-empty';
  card.innerHTML = `
    <div style="font-size:32px">🎓</div>
    <h3>AI Debugger Tutor Ready</h3>
    <p>Run any failing notebook cell, and the Tutor will instantly explain what happened with <strong>Basic</strong>, <strong>Medium</strong>, and <strong>Advanced</strong> fixes — 100% offline.</p>
    <div class="mm-tutor-coverage">🔒 Local Engine · Zero API Latency · ~80% ML Coverage</div>
    <div class="mm-tutor-preset-list">
      <div style="font-size:11px;font-weight:700;color:#5d6b80;text-transform:uppercase;margin-bottom:2px">Try Example Scenarios:</div>
      <button type="button" class="mm-tutor-preset-btn" data-type="ValueError" data-msg="Input X contains NaN.">
        <span>⚡ NaN in Model Fit (ValueError)</span>
        <span style="color:#6d7cff">Inspect →</span>
      </button>
      <button type="button" class="mm-tutor-preset-btn" data-type="KeyError" data-msg="KeyError: 'target'">
        <span>⚡ Missing Column 'target' (KeyError)</span>
        <span style="color:#6d7cff">Inspect →</span>
      </button>
      <button type="button" class="mm-tutor-preset-btn" data-type="NameError" data-msg="name 'X_train' is not defined">
        <span>⚡ Undefined 'X_train' (NameError)</span>
        <span style="color:#6d7cff">Inspect →</span>
      </button>
      <button type="button" class="mm-tutor-preset-btn" data-type="ValueError" data-msg="Found input variables with inconsistent numbers of samples: [100, 80]">
        <span>⚡ Shape Mismatch [100, 80] (ValueError)</span>
        <span style="color:#6d7cff">Inspect →</span>
      </button>
    </div>
  `;

  card.querySelectorAll('.mm-tutor-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const error_type = btn.getAttribute('data-type') || 'ValueError';
      const error_message = btn.getAttribute('data-msg') || 'Error';
      if (_serverSettings) {
        void loadTutorExplanation({
          error_type,
          error_message,
          traceback: `${error_type}: ${error_message}`,
          code: 'model.fit(X_train, y_train)'
        }, _serverSettings, true);
      }
    });
  });

  container.appendChild(card);
}

/**
 * Render Tutor results into any container element (slide-out panel or AI Assistant tab)
 */
export function renderResultInto(
  container: HTMLElement,
  result: TutorResult,
  onLevelSwitch?: (lvl: Level) => void
): void {
  container.replaceChildren();

  if (!result.handled) {
    const msg = document.createElement('div');
    msg.className = 'mm-tutor-empty';
    msg.innerHTML = `
      <div style="font-size:28px">🤔</div>
      <h3>Error Not Diagnosed</h3>
      <p>${result.error_message ?? 'Could not automatically analyse this error offline.'}</p>
    `;
    container.appendChild(msg);
    return;
  }

  // 1. Level switcher (Basic / Medium / Advanced)
  const levelRow = document.createElement('div');
  levelRow.className = 'mm-tutor-level-row';
  (['basic', 'medium', 'advanced'] as Level[]).forEach(lvl => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `mm-tutor-level-btn${lvl === _currentLevel ? ' active' : ''}`;
    const levelLabels: Record<Level, string> = {
      basic: '🐣 Basic Fix',
      medium: '⚡ Medium (M02)',
      advanced: '🚀 Pipeline (Adv)'
    };
    btn.textContent = levelLabels[lvl];
    btn.addEventListener('click', () => {
      _currentLevel = lvl;
      if (onLevelSwitch) {
        onLevelSwitch(lvl);
      } else if (_lastInput && _serverSettings) {
        void loadTutorExplanation(_lastInput, _serverSettings, true);
      }
    });
    levelRow.appendChild(btn);
  });
  container.appendChild(levelRow);

  // 2. Error banner
  const banner = document.createElement('div');
  banner.className = 'mm-tutor-error-banner';
  banner.innerHTML = `
    <div class="banner-title">
      <span style="font-size:16px">${result.icon ?? '🐛'}</span>
      <span>${result.title ?? 'Runtime Error'}</span>
    </div>
    <div class="banner-type">${result.error_type ?? ''}</div>
    ${result.error_message ? `<div class="banner-msg">${result.error_message}</div>` : ''}
    ${result.failing_line ? `<div class="banner-line">→ ${result.failing_line}</div>` : ''}
  `;
  container.appendChild(banner);

  // 3. Local engine status & confidence pill (70% bound restriction)
  const isHighConf = result.can_suggest_code === true && (result.confidence ?? 0) >= 70;
  const pill = document.createElement('div');
  pill.className = isHighConf ? 'mm-tutor-coverage' : 'mm-tutor-coverage mm-tutor-low-conf';
  pill.innerHTML = isHighConf
    ? `🟢 <strong>Confidence: ${result.confidence ?? 85}%</strong> (≥70% High) · Level: <strong>${_currentLevel.toUpperCase()}</strong>`
    : `⚠️ <strong>Confidence: ${result.confidence ?? 45}%</strong> (<70% Threshold) · Fix Code Withheld`;
  container.appendChild(pill);

  // If confidence is below 70%, show the reason why code suggestions are withheld and offer direct redirect to AI Assistant
  if (!isHighConf) {
    const withheldEl = document.createElement('div');
    withheldEl.className = 'mm-tutor-withheld-box';
    withheldEl.innerHTML = `
      <div class="withheld-title">🛡️ Code Fix Withheld (Confidence &lt; 70%)</div>
      <div class="withheld-desc">
        ${result.confidence_reason ?? 'The exact root cause cannot be verified with ≥70% confidence. Automatic code generation is disabled to prevent recommending incorrect code. Follow the diagnostic steps below or ask the AI Assistant.'}
      </div>
      <button type="button" class="mm-tutor-ask-ai-btn">
        <span>🤖</span>
        <span>Ask AI Assistant for Full Diagnosis</span>
        <span>→</span>
      </button>
    `;
    const askBtn = withheldEl.querySelector('.mm-tutor-ask-ai-btn');
    askBtn?.addEventListener('click', () => {
      closePanel();
      window.dispatchEvent(
        new CustomEvent('modelmind:open-ai-assistant', {
          detail: { mode: 'explain', error: _lastInput }
        })
      );
    });
    container.appendChild(withheldEl);
  }

  // 4. Smart contextual ML hints
  if (result.smart_hints && result.smart_hints.length > 0) {
    result.smart_hints.forEach(h => {
      const hint = document.createElement('div');
      hint.className = 'mm-tutor-hint';
      hint.textContent = h;
      container.appendChild(hint);
    });
  }

  // 5. What happened
  if (result.what) {
    const sec = document.createElement('div');
    sec.className = 'mm-tutor-section';
    sec.innerHTML = `
      <div class="mm-tutor-section-label">What Happened</div>
      <div class="mm-tutor-section-content">${result.what}</div>
    `;
    container.appendChild(sec);
  }

  // 6. Why this happened
  if (result.why) {
    const sec = document.createElement('div');
    sec.className = 'mm-tutor-section';
    sec.innerHTML = `
      <div class="mm-tutor-section-label">Why This Happened</div>
      <div class="mm-tutor-section-content">${result.why}</div>
    `;
    container.appendChild(sec);
  }

  // 7. How to fix it (Numbered steps)
  if (result.steps && result.steps.length > 0) {
    const stepsEl = document.createElement('div');
    stepsEl.className = 'mm-tutor-section';
    stepsEl.innerHTML = `<div class="mm-tutor-section-label">${isHighConf ? `How To Fix It (${_currentLevel.toUpperCase()})` : 'Diagnostic Steps'}</div>`;
    result.steps.forEach((step, i) => {
      const row = document.createElement('div');
      row.className = 'mm-tutor-step';
      row.innerHTML = `<div class="mm-tutor-step-num">${i + 1}</div><div>${step}</div>`;
      stepsEl.appendChild(row);
    });
    container.appendChild(stepsEl);
  }

  // 8. Try this in a new cell (Code Tip) — ONLY when confidence >= 70%
  if (isHighConf && result.code_tip) {
    const tipSec = document.createElement('div');
    tipSec.className = 'mm-tutor-section';
    tipSec.innerHTML = `<div class="mm-tutor-section-label">Recommended Code Fix (${_currentLevel.toUpperCase()})</div>`;
    
    const wrap = document.createElement('div');
    wrap.className = 'mm-tutor-code-wrap';
    const pre = document.createElement('div');
    pre.className = 'mm-tutor-code-tip';
    pre.textContent = result.code_tip;

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'mm-tutor-copy-btn';
    copyBtn.textContent = 'Copy Code';
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(result.code_tip!);
        copyBtn.textContent = 'Copied!';
      } catch {
        copyBtn.textContent = 'Copied!';
      }
      setTimeout(() => { copyBtn.textContent = 'Copy Code'; }, 1800);
    });

    wrap.append(pre, copyBtn);
    tipSec.appendChild(wrap);
    container.appendChild(tipSec);
  }

  // 9. Note
  if (result.note) {
    const noteEl = document.createElement('div');
    noteEl.style.fontSize = '11px';
    noteEl.style.color = '#94a3b8';
    noteEl.style.padding = '4px 2px';
    noteEl.textContent = result.note;
    container.appendChild(noteEl);
  }
}

/**
 * Fetch tutor analysis from backend and render into the slide-out panel
 */
export async function loadTutorExplanation(
  input: TutorInput,
  serverSettings: Parameters<typeof requestAPI>[1],
  openWhenDone: boolean = false
): Promise<void> {
  _lastInput = input;
  _serverSettings = serverSettings;
  ensureStyles();

  if (!_panel) createPanel();
  if (openWhenDone) {
    openPanel();
  }

  const body = _panel!.querySelector('#mm-tutor-body') as HTMLElement | null;
  if (body) {
    body.replaceChildren();
    const loading = document.createElement('div');
    loading.style.textAlign = 'center';
    loading.style.padding = '30px 16px';
    loading.style.color = '#5d6b80';
    loading.innerHTML = `<div style="font-size:28px;margin-bottom:8px">⏳</div>Analysing with Local ML Engine…`;
    body.appendChild(loading);
  }

  try {
    const result = await requestAPI<TutorResult>('api/ai/tutor', serverSettings, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        error_type: input.error_type,
        error_message: input.error_message,
        traceback: input.traceback,
        code: input.code,
        level: _currentLevel,
      })
    });
    if (body) {
      renderResultInto(body, result);
    }
    showFABAlert();
  } catch (err) {
    if (body) {
      body.replaceChildren();
      const errEl = document.createElement('div');
      errEl.className = 'mm-tutor-empty';
      errEl.innerHTML = `<div style="font-size:24px;margin-bottom:8px">⚠️</div><div style="color:#ef4444">${err instanceof Error ? err.message : 'Tutor unavailable.'}</div>`;
      body.appendChild(errEl);
    }
  }
}

/**
 * Render Tutor explanation directly into an embedded container (e.g., inside the AI Assistant tab)
 */
export async function renderTutorInElement(
  container: HTMLElement,
  input: TutorInput | null,
  serverSettings: Parameters<typeof requestAPI>[1]
): Promise<void> {
  _serverSettings = serverSettings;
  ensureStyles();

  if (!input) {
    renderEmptyState(container);
    return;
  }

  _lastInput = input;

  container.replaceChildren();
  const loading = document.createElement('div');
  loading.style.textAlign = 'center';
  loading.style.padding = '30px 16px';
  loading.style.color = '#5d6b80';
  loading.innerHTML = `<div style="font-size:28px;margin-bottom:8px">⏳</div>Analysing with Local ML Engine…`;
  container.appendChild(loading);

  try {
    const result = await requestAPI<TutorResult>('api/ai/tutor', serverSettings, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        error_type: input.error_type,
        error_message: input.error_message,
        traceback: input.traceback,
        code: input.code,
        level: _currentLevel,
      })
    });

    renderResultInto(container, result, (lvl) => {
      _currentLevel = lvl;
      void renderTutorInElement(container, input, serverSettings);
    });
  } catch (err) {
    container.replaceChildren();
    const errEl = document.createElement('div');
    errEl.className = 'mm-tutor-empty';
    errEl.innerHTML = `<div style="font-size:24px;margin-bottom:8px">⚠️</div><div style="color:#ef4444">${err instanceof Error ? err.message : 'Tutor unavailable.'}</div>`;
    container.appendChild(errEl);
  }
}
