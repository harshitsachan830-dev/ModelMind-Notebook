/**
 * ModelMind — Dataset X-Ray Panel
 * Displays a rich dataset profile (preview, column stats, health metrics).
 */

import { requestAPI } from './request';

export interface DatasetXRayResult {
  supported: boolean;
  message?: string;
  filename?: string;
  rows?: number;
  columns?: number;
  column_names?: string[];
  numeric_columns?: string[];
  categorical_columns?: string[];
  missing_values?: Record<string, number>;
  total_missing_values?: number;
  duplicate_rows?: number;
  likely_id_columns?: string[];
  dtypes?: Record<string, string>;
  preview?: Record<string, unknown>[];
  column_stats?: {
    name: string;
    dtype: string;
    is_numeric: boolean;
    missing_count: number;
    missing_pct: number;
    unique_count: number;
    min?: number;
    max?: number;
    mean?: number;
    median?: number;
    std?: number;
    skewness?: number;
  }[];
}

function badge(text: string, color: string): HTMLSpanElement {
  const b = document.createElement('span');
  b.style.cssText = `
    display:inline-block; padding:2px 8px; border-radius:99px;
    font-size:10px; font-weight:700; letter-spacing:.5px;
    background:${color}22; color:${color}; border:1px solid ${color}44;
  `;
  b.textContent = text;
  return b;
}

function metricCard(label: string, value: string | number, warn = false): HTMLDivElement {
  const card = document.createElement('div');
  card.style.cssText = `
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09);
    border-radius:10px; padding:12px 14px; display:flex; flex-direction:column;
    gap:4px;
  `;
  const vEl = document.createElement('div');
  vEl.style.cssText = `font-size:20px; font-weight:700; color:${warn ? '#f87171' : '#a78bfa'}`;
  vEl.textContent = String(value);
  const lEl = document.createElement('div');
  lEl.style.cssText = 'font-size:11px; opacity:.6';
  lEl.textContent = label;
  card.append(vEl, lEl);
  return card;
}

function sectionHeader(title: string, emoji: string): HTMLDivElement {
  const h = document.createElement('div');
  h.style.cssText = 'font-size:13px; font-weight:700; margin-bottom:10px; display:flex; align-items:center; gap:6px;';
  h.innerHTML = `<span>${emoji}</span> <span>${title}</span>`;
  return h;
}

export function renderDatasetXRay(
  container: HTMLElement,
  result: DatasetXRayResult
): void {
  container.replaceChildren();

  if (!result.supported) {
    const err = document.createElement('div');
    err.style.cssText = 'color:#f87171; padding:12px; font-size:13px;';
    err.textContent = result.message ?? 'Dataset format not supported.';
    container.appendChild(err);
    return;
  }

  // ── Header ──
  const header = document.createElement('div');
  header.style.cssText = 'margin-bottom:16px;';
  const title = document.createElement('div');
  title.style.cssText = 'font-size:15px; font-weight:700; display:flex; align-items:center; gap:8px;';
  title.innerHTML = `<span>🔬</span> Dataset X-Ray`;
  const sub = document.createElement('div');
  sub.style.cssText = 'font-size:11px; opacity:.55; margin-top:3px;';
  sub.textContent = result.filename ?? '';
  header.append(title, sub);
  container.appendChild(header);

  // ── Health Metrics ──
  const metricsSection = document.createElement('div');
  metricsSection.style.cssText = 'margin-bottom:16px;';
  metricsSection.appendChild(sectionHeader('Dataset Health', '📊'));
  const metricsGrid = document.createElement('div');
  metricsGrid.style.cssText = 'display:grid; grid-template-columns:1fr 1fr; gap:8px;';
  metricsGrid.append(
    metricCard('Rows', result.rows ?? 0),
    metricCard('Columns', result.columns ?? 0),
    metricCard('Missing Values', result.total_missing_values ?? 0, (result.total_missing_values ?? 0) > 0),
    metricCard('Duplicate Rows', result.duplicate_rows ?? 0, (result.duplicate_rows ?? 0) > 0)
  );
  metricsSection.appendChild(metricsGrid);
  container.appendChild(metricsSection);

  // ── Alerts ──
  if (
    (result.likely_id_columns?.length ?? 0) > 0 ||
    (result.total_missing_values ?? 0) > 0 ||
    (result.duplicate_rows ?? 0) > 0
  ) {
    const alertSection = document.createElement('div');
    alertSection.style.cssText = 'margin-bottom:16px;';
    alertSection.appendChild(sectionHeader('Alerts', '⚠️'));

    const alerts: { msg: string; color: string }[] = [];
    if ((result.total_missing_values ?? 0) > 0)
      alerts.push({ msg: `${result.total_missing_values} missing values detected across the dataset.`, color: '#f59e0b' });
    if ((result.duplicate_rows ?? 0) > 0)
      alerts.push({ msg: `${result.duplicate_rows} duplicate rows found.`, color: '#f59e0b' });
    if ((result.likely_id_columns?.length ?? 0) > 0)
      alerts.push({ msg: `Possible ID columns: ${result.likely_id_columns!.join(', ')} — review before training.`, color: '#818cf8' });

    alerts.forEach(({ msg, color }) => {
      const a = document.createElement('div');
      a.style.cssText = `
        padding:10px 12px; border-radius:8px; font-size:12px; margin-bottom:6px;
        background:${color}15; border:1px solid ${color}44; color:${color};
      `;
      a.textContent = msg;
      alertSection.appendChild(a);
    });
    container.appendChild(alertSection);
  }

  // ── Column Stats ──
  if (result.column_stats?.length) {
    const colSection = document.createElement('div');
    colSection.style.cssText = 'margin-bottom:16px;';
    colSection.appendChild(sectionHeader('Column Analysis', '🧮'));

    result.column_stats.forEach(col => {
      const card = document.createElement('div');
      card.style.cssText = `
        background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
        border-radius:10px; padding:12px; margin-bottom:8px;
      `;

      const nameRow = document.createElement('div');
      nameRow.style.cssText = 'display:flex; align-items:center; gap:6px; margin-bottom:8px;';
      const nameEl = document.createElement('span');
      nameEl.style.cssText = 'font-size:13px; font-weight:600;';
      nameEl.textContent = col.name;
      nameRow.append(nameEl);
      nameRow.append(badge(col.is_numeric ? 'NUMERIC' : 'CATEGORICAL', col.is_numeric ? '#34d399' : '#a78bfa'));
      nameRow.append(badge(col.dtype, '#94a3b8'));
      if (col.missing_count > 0) nameRow.append(badge(`${col.missing_pct}% missing`, '#f59e0b'));
      card.appendChild(nameRow);

      const statsGrid = document.createElement('div');
      statsGrid.style.cssText = 'display:grid; grid-template-columns:repeat(3,1fr); gap:4px;';
      const addStat = (label: string, value: string | number | undefined) => {
        if (value === undefined || value === null) return;
        const s = document.createElement('div');
        s.style.cssText = 'font-size:11px;';
        s.innerHTML = `<span style="opacity:.5">${label}:</span> <span style="font-weight:600">${value}</span>`;
        statsGrid.appendChild(s);
      };
      addStat('Unique', col.unique_count);
      addStat('Missing', col.missing_count);
      if (col.is_numeric) {
        addStat('Min', col.min !== undefined ? Number(col.min).toFixed(2) : undefined);
        addStat('Max', col.max !== undefined ? Number(col.max).toFixed(2) : undefined);
        addStat('Mean', col.mean !== undefined ? Number(col.mean).toFixed(3) : undefined);
        addStat('Skew', col.skewness !== undefined ? Number(col.skewness).toFixed(3) : undefined);
      }
      card.appendChild(statsGrid);
      colSection.appendChild(card);
    });
    container.appendChild(colSection);
  }

  // ── Data Preview ──
  if (result.preview?.length) {
    const previewSection = document.createElement('div');
    previewSection.style.cssText = 'margin-bottom:16px;';
    previewSection.appendChild(sectionHeader('Data Preview (first 10 rows)', '📄'));

    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'overflow-x:auto;';

    const table = document.createElement('table');
    table.style.cssText = `
      width:100%; border-collapse:collapse; font-size:11px;
      border:1px solid rgba(255,255,255,0.08);
    `;

    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    (result.column_names ?? Object.keys(result.preview[0])).forEach(col => {
      const th = document.createElement('th');
      th.style.cssText = `
        padding:8px 10px; text-align:left; font-weight:600; white-space:nowrap;
        background:rgba(255,255,255,0.06); border-bottom:1px solid rgba(255,255,255,0.08);
      `;
      th.textContent = col;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    result.preview.forEach((row, i) => {
      const tr = document.createElement('tr');
      tr.style.background = i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)';
      (result.column_names ?? Object.keys(row)).forEach(col => {
        const td = document.createElement('td');
        td.style.cssText = `
          padding:6px 10px; border-bottom:1px solid rgba(255,255,255,0.05); white-space:nowrap;
        `;
        const val = row[col];
        td.textContent = val === null || val === undefined ? '—' : String(val);
        if (val === null || val === undefined) td.style.opacity = '0.35';
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    previewSection.appendChild(wrapper);
    container.appendChild(previewSection);
  }
}

export async function loadAndRenderDatasetXRay(
  container: HTMLElement,
  filePath: string,
  serverSettings: Parameters<typeof requestAPI>[1]
): Promise<DatasetXRayResult | null> {
  container.replaceChildren();
  const loading = document.createElement('div');
  loading.style.cssText = 'opacity:.6; font-size:13px; padding:12px;';
  loading.textContent = '🔬 Analysing dataset…';
  container.appendChild(loading);

  try {
    const result = await requestAPI<DatasetXRayResult>('api/dataset/xray', serverSettings, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_path: filePath })
    });
    renderDatasetXRay(container, result);
    return result;
  } catch (err) {
    container.replaceChildren();
    const errEl = document.createElement('div');
    errEl.style.cssText = 'color:#f87171; font-size:13px; padding:12px;';
    errEl.textContent = err instanceof Error ? err.message : 'Dataset X-Ray failed.';
    container.appendChild(errEl);
    return null;
  }
}
