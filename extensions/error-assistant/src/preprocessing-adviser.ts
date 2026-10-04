/**
 * ModelMind — Smart Preprocessing Adviser 2.0 Panel
 * Displays intelligent, level-aware preprocessing recommendations per column.
 */

import { requestAPI } from './request';

export interface PreprocessingRec {
  feature: string;
  feature_type: 'numerical' | 'categorical';
  missing_count: number;
  missing_pct: number;
  missing_severity: 'none' | 'low' | 'moderate' | 'high' | 'very_high';
  unique_count: number;
  skewness?: number;
  outlier_count?: number;
  outlier_pct?: number;
  imputation_strategy: string;
  scaling_strategy?: string;
  recommended_action: string;
  explanation: string;
  example_code: string;
  warnings: string[];
}

export interface PreprocessingResult {
  handled: boolean;
  error?: string;
  source?: string;
  engine?: string;
  level?: string;
  filename?: string;
  rows?: number;
  columns?: number;
  numerical_features?: number;
  categorical_features?: number;
  total_missing_values?: number;
  duplicate_rows?: number;
  feature_warning_count?: number;
  dataset_warnings?: string[];
  recommendations?: PreprocessingRec[];
  safety?: { message: string };
}

type Level = 'basic' | 'medium' | 'advanced';

const SEVERITY_COLOR: Record<string, string> = {
  none: '#34d399',
  low: '#fbbf24',
  moderate: '#f59e0b',
  high: '#ef4444',
  very_high: '#dc2626',
};

function copyBtn(code: string): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.style.cssText = `
    font-size:10px; padding:3px 8px; border-radius:5px; cursor:pointer;
    background:rgba(167,139,250,.15); border:1px solid rgba(167,139,250,.3);
    color:#a78bfa; white-space:nowrap;
  `;
  btn.textContent = 'Copy';
  btn.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(code); btn.textContent = 'Copied!'; }
    catch { btn.textContent = 'Error'; }
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });
  return btn;
}

function renderRecommendation(rec: PreprocessingRec): HTMLDivElement {
  const card = document.createElement('div');
  card.style.cssText = `
    border:1px solid rgba(255,255,255,0.08); border-radius:12px;
    padding:14px; margin-bottom:10px; background:rgba(255,255,255,0.03);
  `;

  // Header
  const hdr = document.createElement('div');
  hdr.style.cssText = 'display:flex; align-items:flex-start; gap:8px; margin-bottom:10px;';

  const nameEl = document.createElement('span');
  nameEl.style.cssText = 'font-size:13px; font-weight:700; flex:1;';
  nameEl.textContent = rec.feature;

  const typeBadge = document.createElement('span');
  typeBadge.style.cssText = `
    font-size:9px; font-weight:700; padding:2px 7px; border-radius:99px;
    background:${rec.feature_type === 'numerical' ? '#34d39922' : '#a78bfa22'};
    color:${rec.feature_type === 'numerical' ? '#34d399' : '#a78bfa'};
    border:1px solid ${rec.feature_type === 'numerical' ? '#34d39944' : '#a78bfa44'};
  `;
  typeBadge.textContent = rec.feature_type.toUpperCase();

  const missBadge = document.createElement('span');
  const col = SEVERITY_COLOR[rec.missing_severity] ?? '#94a3b8';
  missBadge.style.cssText = `
    font-size:9px; font-weight:700; padding:2px 7px; border-radius:99px;
    background:${col}22; color:${col}; border:1px solid ${col}44;
  `;
  missBadge.textContent = `${rec.missing_pct}% missing`;
  hdr.append(nameEl, typeBadge, missBadge);
  card.appendChild(hdr);

  // Warnings
  if (rec.warnings.length > 0) {
    rec.warnings.forEach(w => {
      const wEl = document.createElement('div');
      wEl.style.cssText = `
        padding:7px 10px; border-radius:7px; font-size:11px; margin-bottom:6px;
        background:#f59e0b15; border:1px solid #f59e0b44; color:#f59e0b;
      `;
      wEl.innerHTML = `⚠️ ${w}`;
      card.appendChild(wEl);
    });
  }

  // Recommended action
  const actionEl = document.createElement('div');
  actionEl.style.cssText = `
    font-size:12px; padding:10px 12px; border-radius:8px; margin-bottom:8px;
    background:rgba(52,211,153,.07); border:1px solid rgba(52,211,153,.2);
    color:#a7f3d0;
  `;
  actionEl.innerHTML = `<strong>✅ Action:</strong> ${rec.recommended_action}`;
  card.appendChild(actionEl);

  // Explanation
  const explEl = document.createElement('div');
  explEl.style.cssText = 'font-size:11px; opacity:.65; margin-bottom:10px; line-height:1.5;';
  explEl.textContent = rec.explanation;
  card.appendChild(explEl);

  // Code block
  if (rec.example_code) {
    const codeWrap = document.createElement('div');
    codeWrap.style.cssText = 'position:relative;';
    const pre = document.createElement('pre');
    pre.style.cssText = `
      background:#0f172a; border:1px solid rgba(255,255,255,.08); border-radius:8px;
      padding:10px 12px; font-size:11px; overflow-x:auto; margin:0;
      font-family:ui-monospace,SFMono-Regular,monospace; line-height:1.6;
    `;
    pre.textContent = rec.example_code;
    const copyWrap = document.createElement('div');
    copyWrap.style.cssText = 'position:absolute; top:6px; right:6px;';
    copyWrap.appendChild(copyBtn(rec.example_code));
    codeWrap.append(pre, copyWrap);
    card.appendChild(codeWrap);
  }

  // Stats row
  const statsRow = document.createElement('div');
  statsRow.style.cssText = 'display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; font-size:10px; opacity:.55;';
  const addStat = (label: string, value: unknown) => {
    if (value === undefined || value === null) return;
    const s = document.createElement('span');
    s.textContent = `${label}: ${value}`;
    statsRow.appendChild(s);
  };
  addStat('Unique', rec.unique_count);
  addStat('Strategy', rec.imputation_strategy);
  if (rec.skewness !== undefined) addStat('Skewness', Number(rec.skewness).toFixed(3));
  if (rec.outlier_count !== undefined) addStat('Outliers', `${rec.outlier_count} (${rec.outlier_pct?.toFixed(1)}%)`);
  if (rec.scaling_strategy) addStat('Scaling', rec.scaling_strategy);
  card.appendChild(statsRow);

  return card;
}

export function renderPreprocessingAdviser(
  container: HTMLElement,
  result: PreprocessingResult,
  currentLevel: Level,
  onLevelChange: (level: Level) => void
): void {
  container.replaceChildren();

  if (!result.handled) {
    const err = document.createElement('div');
    err.style.cssText = 'color:#f87171; font-size:13px; padding:12px;';
    err.textContent = result.error ?? 'Preprocessing analysis failed.';
    container.appendChild(err);
    return;
  }

  // ── Header + Level Switcher ──
  const headerEl = document.createElement('div');
  headerEl.style.cssText = 'margin-bottom:14px;';
  const titleEl = document.createElement('div');
  titleEl.style.cssText = 'font-size:15px; font-weight:700; display:flex; align-items:center; gap:8px; margin-bottom:8px;';
  titleEl.innerHTML = `<span>🧠</span> Smart Preprocessing Adviser 2.0`;
  headerEl.appendChild(titleEl);

  const levelRow = document.createElement('div');
  levelRow.style.cssText = 'display:flex; gap:6px; margin-bottom:8px;';
  (['basic', 'medium', 'advanced'] as Level[]).forEach(lvl => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.style.cssText = `
      font-size:11px; padding:4px 12px; border-radius:99px; cursor:pointer;
      border:1px solid ${lvl === currentLevel ? '#a78bfa' : 'rgba(255,255,255,.12)'};
      background:${lvl === currentLevel ? '#a78bfa22' : 'transparent'};
      color:${lvl === currentLevel ? '#a78bfa' : 'inherit'};
    `;
    btn.textContent = lvl.charAt(0).toUpperCase() + lvl.slice(1);
    btn.addEventListener('click', () => onLevelChange(lvl));
    levelRow.appendChild(btn);
  });
  headerEl.appendChild(levelRow);

  const sub = document.createElement('div');
  sub.style.cssText = 'font-size:11px; opacity:.5;';
  sub.textContent = `${result.filename} · ${result.rows} rows · ${result.columns} columns`;
  headerEl.appendChild(sub);
  container.appendChild(headerEl);

  // ── Summary ──
  const summaryGrid = document.createElement('div');
  summaryGrid.style.cssText = 'display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:14px;';
  const addMetric = (label: string, value: number | string, warn = false) => {
    const card = document.createElement('div');
    card.style.cssText = `
      padding:10px 12px; border-radius:9px; font-size:11px;
      background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    `;
    card.innerHTML = `
      <div style="font-size:18px; font-weight:700; color:${warn ? '#f87171' : '#a78bfa'}">${value}</div>
      <div style="opacity:.55">${label}</div>
    `;
    summaryGrid.appendChild(card);
  };
  addMetric('Numerical', result.numerical_features ?? 0);
  addMetric('Categorical', result.categorical_features ?? 0);
  addMetric('Missing Values', result.total_missing_values ?? 0, (result.total_missing_values ?? 0) > 0);
  addMetric('Warnings', result.feature_warning_count ?? 0, (result.feature_warning_count ?? 0) > 0);
  container.appendChild(summaryGrid);

  // ── Dataset-level warnings ──
  result.dataset_warnings?.forEach(w => {
    const wEl = document.createElement('div');
    wEl.style.cssText = `
      padding:9px 12px; border-radius:8px; font-size:12px; margin-bottom:8px;
      background:#f59e0b15; border:1px solid #f59e0b44; color:#fbbf24;
    `;
    wEl.innerHTML = `⚠️ ${w}`;
    container.appendChild(wEl);
  });

  // ── Safety note ──
  if (result.safety?.message) {
    const note = document.createElement('div');
    note.style.cssText = `
      padding:9px 12px; border-radius:8px; font-size:11px; margin-bottom:12px;
      background:rgba(52,211,153,.06); border:1px solid rgba(52,211,153,.2); opacity:.7;
    `;
    note.innerHTML = `🔒 ${result.safety.message}`;
    container.appendChild(note);
  }

  // ── Column recommendations ──
  if (result.recommendations?.length) {
    const secHeader = document.createElement('div');
    secHeader.style.cssText = 'font-size:13px; font-weight:700; margin-bottom:10px;';
    secHeader.textContent = `Column Recommendations (${result.recommendations.length})`;
    container.appendChild(secHeader);
    result.recommendations.forEach(rec => {
      container.appendChild(renderRecommendation(rec));
    });
  }
}

export async function loadAndRenderPreprocessingAdviser(
  container: HTMLElement,
  filePath: string,
  level: Level,
  serverSettings: Parameters<typeof requestAPI>[1],
  onLevelChange: (lvl: Level) => void
): Promise<void> {
  container.replaceChildren();
  const loading = document.createElement('div');
  loading.style.cssText = 'opacity:.6; font-size:13px; padding:12px;';
  loading.textContent = `🧠 Running Smart Preprocessing Adviser (${level})…`;
  container.appendChild(loading);

  try {
    const result = await requestAPI<PreprocessingResult>('api/dataset/preprocessing', serverSettings, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_path: filePath, level })
    });
    renderPreprocessingAdviser(container, result, level, onLevelChange);
  } catch (err) {
    container.replaceChildren();
    const errEl = document.createElement('div');
    errEl.style.cssText = 'color:#f87171; font-size:13px; padding:12px;';
    errEl.textContent = err instanceof Error ? err.message : 'Preprocessing Adviser failed.';
    container.appendChild(errEl);
  }
}
