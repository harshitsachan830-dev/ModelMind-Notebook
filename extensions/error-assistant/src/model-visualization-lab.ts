/**
 * ModelMind — Model Visualization Lab
 * An interactive, self-contained visualization for Gradient Descent.
 * Rendered in a JupyterLab modal/tab with no external dependencies.
 */

// ── HTML + CSS for the lab ──

export function createModelLabHTML(): string {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ModelMind — Model Visualization Lab</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0a0f1e; color: #e2e8f0; min-height: 100vh;
  }
  header {
    padding: 16px 24px; border-bottom: 1px solid rgba(167,139,250,.2);
    background: linear-gradient(135deg, #1e1b4b 0%, #0a0f1e 100%);
    display: flex; align-items: center; justify-content: space-between;
  }
  header h1 {
    font-size: 18px; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #818cf8, #6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: flex; align-items: center; gap: 10px;
  }
  header .sub { font-size: 11px; opacity: .5; margin-top: 2px; }

  .lab-body {
    display: grid; grid-template-columns: 280px 1fr;
    height: calc(100vh - 65px); overflow: hidden;
  }

  /* Sidebar */
  .sidebar {
    border-right: 1px solid rgba(255,255,255,.07);
    overflow-y: auto; padding: 16px;
    background: rgba(255,255,255,.02);
  }
  .model-card {
    border: 1px solid rgba(255,255,255,.08); border-radius: 11px;
    padding: 13px; margin-bottom: 8px; cursor: pointer;
    transition: all .2s; background: rgba(255,255,255,.03);
  }
  .model-card:hover { border-color: rgba(167,139,250,.4); background: rgba(167,139,250,.06); }
  .model-card.active { border-color: #a78bfa; background: rgba(167,139,250,.12); }
  .model-card .name { font-size: 13px; font-weight: 700; margin-bottom: 3px; }
  .model-card .cat { font-size: 10px; opacity: .5; }
  .model-card .desc { font-size: 11px; opacity: .7; margin-top: 6px; line-height: 1.4; }
  .model-card.soon { opacity: .45; cursor: default; }
  .soon-badge {
    display: inline-block; font-size: 9px; padding: 2px 7px; border-radius: 99px;
    background: rgba(100,116,139,.2); border: 1px solid rgba(100,116,139,.3);
    color: #94a3b8; margin-left: 6px;
  }

  /* Main area */
  .main { overflow-y: auto; padding: 20px; }

  /* Controls */
  .controls-panel {
    background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
    border-radius: 14px; padding: 16px; margin-bottom: 16px;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 12px;
  }
  .ctrl { display: flex; flex-direction: column; gap: 5px; }
  .ctrl label { font-size: 11px; font-weight: 600; opacity: .65; }
  .ctrl input[type=range] { width: 100%; accent-color: #a78bfa; }
  .ctrl .val { font-size: 13px; font-weight: 700; color: #a78bfa; }
  .ctrl select {
    background: #1e293b; border: 1px solid rgba(255,255,255,.12);
    color: inherit; padding: 5px 8px; border-radius: 6px; font-size: 12px;
  }

  /* Action bar */
  .action-bar { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .btn {
    padding: 8px 16px; border-radius: 8px; border: none; cursor: pointer;
    font-size: 12px; font-weight: 700; transition: all .15s;
  }
  .btn-primary { background: #7c3aed; color: #fff; }
  .btn-primary:hover { background: #6d28d9; transform: translateY(-1px); }
  .btn-secondary {
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
    color: inherit;
  }
  .btn-secondary:hover { background: rgba(255,255,255,.1); }
  .btn-danger { background: rgba(239,68,68,.15); border: 1px solid rgba(239,68,68,.3); color: #f87171; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }

  /* Metrics */
  .metrics-row { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }
  .metric {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 10px; padding: 10px 14px; min-width: 110px;
  }
  .metric .mval { font-size: 20px; font-weight: 700; color: #a78bfa; }
  .metric .mlbl { font-size: 10px; opacity: .5; margin-top: 2px; }

  /* Charts */
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .chart-card {
    background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 14px; overflow: hidden;
  }
  .chart-card h3 { font-size: 12px; font-weight: 700; margin-bottom: 10px; opacity: .75; }
  canvas { border-radius: 6px; display: block; width: 100%; }

  /* Explanation */
  .explanation {
    background: rgba(167,139,250,.06); border: 1px solid rgba(167,139,250,.2);
    border-radius: 12px; padding: 14px; margin-top: 16px; font-size: 12px; line-height: 1.6;
  }
  .explanation h3 { font-size: 13px; font-weight: 700; margin-bottom: 8px; color: #a78bfa; }
  .explanation p { opacity: .8; }

  .progress-bar-wrap {
    height: 4px; background: rgba(255,255,255,.08); border-radius: 99px; margin: 10px 0;
    overflow: hidden;
  }
  .progress-bar { height: 100%; background: linear-gradient(90deg,#7c3aed,#818cf8); transition: width .3s; }

  @media (max-width: 700px) {
    .lab-body { grid-template-columns: 1fr; }
    .sidebar { display: none; }
    .charts-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div>
    <h1>⚗️ Model Visualization Lab</h1>
    <div class="sub">ModelMind · Interactive ML Education</div>
  </div>
</header>

<div class="lab-body">
  <!-- Model Selector -->
  <div class="sidebar">
    <div style="font-size:11px;font-weight:700;opacity:.4;letter-spacing:.8px;margin-bottom:10px">MODELS</div>
    <div class="model-card active" data-model="gradient-descent" onclick="selectModel('gradient-descent')">
      <div class="name">⬇️ Gradient Descent</div>
      <div class="cat">OPTIMIZATION</div>
      <div class="desc">Watch a Linear Regression model learn step by step by following the negative gradient.</div>
    </div>
    <div class="model-card soon" data-model="decision-tree">
      <div class="name">🌳 Decision Tree <span class="soon-badge">COMING SOON</span></div>
      <div class="cat">CLASSIFICATION</div>
      <div class="desc">Visualize how a tree splits data at each node to separate classes.</div>
    </div>
    <div class="model-card soon" data-model="knn">
      <div class="name">🔵 k-Nearest Neighbours <span class="soon-badge">COMING SOON</span></div>
      <div class="cat">CLASSIFICATION</div>
      <div class="desc">See how the vote of nearby points determines a prediction.</div>
    </div>
    <div class="model-card soon" data-model="kmeans">
      <div class="name">🔮 K-Means Clustering <span class="soon-badge">COMING SOON</span></div>
      <div class="cat">CLUSTERING</div>
      <div class="desc">Watch centroids converge to cluster centres over iterations.</div>
    </div>
    <div class="model-card soon" data-model="pca">
      <div class="name">📐 PCA <span class="soon-badge">COMING SOON</span></div>
      <div class="cat">DIMENSIONALITY REDUCTION</div>
      <div class="desc">Project high-dimensional data onto principal components.</div>
    </div>
  </div>

  <!-- Main visualization area -->
  <div class="main" id="main-area">
    <!-- Injected by JS -->
  </div>
</div>

<script>
// ================================================================
// GRADIENT DESCENT VISUALIZER
// ================================================================

const DATA_PRESETS = {
  'house-prices': {
    label: 'House Prices',
    points: [
      {x:1,y:2.1},{x:2,y:4.0},{x:3,y:5.8},{x:4,y:7.9},{x:5,y:10.1},
      {x:6,y:12.2},{x:7,y:13.8},{x:8,y:16.1},{x:9,y:17.9},{x:10,y:20.0},
    ],
  },
  'salary-exp': {
    label: 'Salary vs Experience',
    points: [
      {x:1,y:38},{x:2,y:42},{x:3,y:50},{x:4,y:55},{x:5,y:63},
      {x:6,y:68},{x:7,y:75},{x:8,y:80},{x:9,y:90},{x:10,y:98},
    ],
  },
  'noisy': {
    label: 'Noisy Data',
    points: [
      {x:1,y:3.5},{x:2,y:3.0},{x:3,y:7.0},{x:4,y:5.5},{x:5,y:9.0},
      {x:6,y:8.0},{x:7,y:12.5},{x:8,y:14.0},{x:9,y:13.0},{x:10,y:17.0},
    ],
  },
};

let state = {
  dataset: 'house-prices',
  lr: 0.01,
  iterations: 200,
  w: 0,
  b: 0,
  history: [],
  running: false,
  stepIndex: 0,
  animFrame: null,
};

function selectModel(id) {
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
  document.querySelector('[data-model="' + id + '"]').classList.add('active');
  if (id === 'gradient-descent') renderGD();
}

function renderGD() {
  const main = document.getElementById('main-area');
  main.innerHTML = \`
    <div class="controls-panel">
      <div class="ctrl">
        <label>Learning Rate</label>
        <input type="range" id="lr" min="0.001" max="0.05" step="0.001" value="0.01">
        <div class="val" id="lr-val">0.01</div>
      </div>
      <div class="ctrl">
        <label>Iterations</label>
        <input type="range" id="iters" min="10" max="500" step="10" value="200">
        <div class="val" id="iters-val">200</div>
      </div>
      <div class="ctrl">
        <label>Initial Weight (w₀)</label>
        <input type="range" id="init-w" min="-2" max="2" step="0.1" value="0">
        <div class="val" id="init-w-val">0</div>
      </div>
      <div class="ctrl">
        <label>Initial Bias (b₀)</label>
        <input type="range" id="init-b" min="-5" max="5" step="0.5" value="0">
        <div class="val" id="init-b-val">0</div>
      </div>
      <div class="ctrl">
        <label>Dataset</label>
        <select id="dataset">
          <option value="house-prices">House Prices</option>
          <option value="salary-exp">Salary vs Experience</option>
          <option value="noisy">Noisy Data</option>
        </select>
      </div>
    </div>

    <div class="action-bar">
      <button class="btn btn-primary" id="btn-run">▶ Run</button>
      <button class="btn btn-secondary" id="btn-step">⏭ Step</button>
      <button class="btn btn-danger" id="btn-reset">↺ Reset</button>
    </div>

    <div class="progress-bar-wrap"><div class="progress-bar" id="prog" style="width:0%"></div></div>

    <div class="metrics-row">
      <div class="metric"><div class="mval" id="m-loss">—</div><div class="mlbl">MSE Loss</div></div>
      <div class="metric"><div class="mval" id="m-w">0</div><div class="mlbl">Weight (w)</div></div>
      <div class="metric"><div class="mval" id="m-b">0</div><div class="mlbl">Bias (b)</div></div>
      <div class="metric"><div class="mval" id="m-step">0</div><div class="mlbl">Step</div></div>
    </div>

    <div class="charts-grid">
      <div class="chart-card">
        <h3>📉 Regression Fit</h3>
        <canvas id="c-fit" height="220"></canvas>
      </div>
      <div class="chart-card">
        <h3>📈 Loss Curve</h3>
        <canvas id="c-loss" height="220"></canvas>
      </div>
    </div>

    <div class="explanation">
      <h3>How Gradient Descent Works</h3>
      <p>
        Gradient Descent minimises the Mean Squared Error (MSE) loss by repeatedly
        computing the gradient of the loss with respect to the weight <strong>w</strong>
        and bias <strong>b</strong>, then moving in the <em>opposite</em> direction
        (negative gradient) by a step size called the <strong>learning rate α</strong>.
      </p>
      <p style="margin-top:8px">
        <code style="background:#1e293b;padding:3px 7px;border-radius:4px;font-size:11px">
          w ← w − α · ∂L/∂w &nbsp;&nbsp; b ← b − α · ∂L/∂b
        </code>
      </p>
      <p style="margin-top:8px">
        A <strong>small learning rate</strong> converges slowly but safely.
        A <strong>large learning rate</strong> may overshoot and diverge.
      </p>
    </div>
  \`;

  // Wire controls
  const connect = (id, key, display) => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => {
      state[key] = parseFloat(el.value);
      document.getElementById(display).textContent = el.value;
    });
  };
  connect('lr', 'lr', 'lr-val');
  connect('iters', 'iterations', 'iters-val');
  connect('init-w', 'initW', 'init-w-val');
  connect('init-b', 'initB', 'init-b-val');
  state.initW = 0; state.initB = 0;

  document.getElementById('dataset').addEventListener('change', (e) => {
    state.dataset = e.target.value;
    resetGD();
  });

  document.getElementById('btn-run').addEventListener('click', () => {
    if (state.running) {
      state.running = false;
      document.getElementById('btn-run').textContent = '▶ Run';
    } else {
      state.running = true;
      document.getElementById('btn-run').textContent = '⏸ Pause';
      runAnimation();
    }
  });

  document.getElementById('btn-step').addEventListener('click', stepGD);
  document.getElementById('btn-reset').addEventListener('click', resetGD);

  resetGD();
  drawFit();
  drawLoss();
}

function resetGD() {
  cancelAnimationFrame(state.animFrame);
  state.running = false;
  state.history = [];
  state.stepIndex = 0;
  state.w = state.initW ?? 0;
  state.b = state.initB ?? 0;
  if (document.getElementById('btn-run')) document.getElementById('btn-run').textContent = '▶ Run';
  updateMetrics();
  updateProgress();
  drawFit();
  drawLoss();
}

function stepGD() {
  if (state.stepIndex >= state.iterations) return;
  const data = DATA_PRESETS[state.dataset].points;
  const { gw, gb, loss } = mseGradients(data, state.w, state.b);
  state.w -= state.lr * gw;
  state.b -= state.lr * gb;
  state.stepIndex++;
  state.history.push({ w: state.w, b: state.b, loss });
  updateMetrics(loss);
  updateProgress();
  drawFit();
  drawLoss();
}

function runAnimation() {
  if (!state.running || state.stepIndex >= state.iterations) {
    state.running = false;
    if (document.getElementById('btn-run')) document.getElementById('btn-run').textContent = '▶ Run';
    return;
  }
  for (let i = 0; i < 3; i++) stepGD();
  state.animFrame = requestAnimationFrame(runAnimation);
}

function mseGradients(data, w, b) {
  let gw = 0, gb = 0, loss = 0;
  const n = data.length;
  for (const { x, y } of data) {
    const diff = (w * x + b) - y;
    gw += diff * x;
    gb += diff;
    loss += diff * diff;
  }
  return { gw: gw / n, gb: gb / n, loss: loss / n };
}

function updateMetrics(loss) {
  const fmt = v => (typeof v === 'number' && isFinite(v) ? v.toFixed(3) : '—');
  const el = id => document.getElementById(id);
  if (el('m-loss')) el('m-loss').textContent = loss !== undefined ? fmt(loss) : '—';
  if (el('m-w')) el('m-w').textContent = fmt(state.w);
  if (el('m-b')) el('m-b').textContent = fmt(state.b);
  if (el('m-step')) el('m-step').textContent = state.stepIndex;
}

function updateProgress() {
  const bar = document.getElementById('prog');
  if (bar) bar.style.width = (state.stepIndex / state.iterations * 100) + '%';
}

function drawFit() {
  const canvas = document.getElementById('c-fit');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 400;
  const H = 220;
  canvas.width = W; canvas.height = H;
  const pad = 30;
  const data = DATA_PRESETS[state.dataset].points;

  // Scale
  const xs = data.map(p => p.x), ys = data.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys) - 2, yMax = Math.max(...ys) + 2;
  const sx = x => pad + (x - xMin) / (xMax - xMin) * (W - 2 * pad);
  const sy = y => H - pad - (y - yMin) / (yMax - yMin) * (H - 2 * pad);

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0f1e';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad + i * (H - 2 * pad) / 5;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
  }

  // Data points
  data.forEach(({ x, y }) => {
    ctx.beginPath();
    ctx.arc(sx(x), sy(y), 5, 0, Math.PI * 2);
    ctx.fillStyle = '#a78bfa';
    ctx.fill();
  });

  // Regression line
  if (isFinite(state.w) && isFinite(state.b)) {
    const x1 = xMin, x2 = xMax;
    const y1 = state.w * x1 + state.b, y2 = state.w * x2 + state.b;
    ctx.beginPath();
    ctx.moveTo(sx(x1), sy(y1));
    ctx.lineTo(sx(x2), sy(y2));
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  // Axes labels
  ctx.fillStyle = 'rgba(255,255,255,.3)';
  ctx.font = '10px monospace';
  ctx.fillText('x', W - pad + 4, H / 2);
  ctx.fillText('y', pad / 2, pad / 2);
}

function drawLoss() {
  const canvas = document.getElementById('c-loss');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 400;
  const H = 220;
  canvas.width = W; canvas.height = H;
  const pad = 30;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0f1e';
  ctx.fillRect(0, 0, W, H);

  if (state.history.length < 2) {
    ctx.fillStyle = 'rgba(255,255,255,.2)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Run the model to see the loss curve', W / 2, H / 2);
    ctx.textAlign = 'left';
    return;
  }

  const losses = state.history.map(h => h.loss).filter(isFinite);
  const lMin = Math.min(...losses), lMax = Math.max(...losses);
  const sx = i => pad + (i / (state.history.length - 1)) * (W - 2 * pad);
  const sy = l => H - pad - (l - lMin) / ((lMax - lMin) || 1) * (H - 2 * pad);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad + i * (H - 2 * pad) / 5;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
  }

  // Loss curve
  ctx.beginPath();
  state.history.forEach((h, i) => {
    if (!isFinite(h.loss)) return;
    if (i === 0) ctx.moveTo(sx(i), sy(h.loss));
    else ctx.lineTo(sx(i), sy(h.loss));
  });
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Fill under
  ctx.lineTo(sx(state.history.length - 1), H - pad);
  ctx.lineTo(sx(0), H - pad);
  ctx.closePath();
  ctx.fillStyle = 'rgba(245,158,11,.08)';
  ctx.fill();

  // Final loss label
  const last = losses[losses.length - 1];
  ctx.fillStyle = '#f59e0b';
  ctx.font = '11px monospace';
  ctx.fillText('Loss: ' + last.toFixed(3), W - 90, pad + 10);
}

// Init
renderGD();
</script>
</body>
</html>`;
}

// ── JupyterLab panel opener ──

export function openModelVisualizationLab(): void {
  // Check if already open
  const existingId = 'mm-model-lab-window';
  const existing = document.getElementById(existingId);
  if (existing) {
    existing.style.display = 'flex';
    return;
  }

  const overlay = document.createElement('div');
  overlay.id = existingId;
  overlay.style.cssText = `
    position: fixed; inset: 0; z-index: 10000;
    background: rgba(0,0,0,.7); display: flex;
    align-items: center; justify-content: center;
  `;

  const modal = document.createElement('div');
  modal.style.cssText = `
    width: 95vw; height: 90vh; border-radius: 16px;
    overflow: hidden; border: 1px solid rgba(167,139,250,.3);
    box-shadow: 0 25px 80px rgba(0,0,0,.7);
    display: flex; flex-direction: column;
    position: relative;
  `;

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕ Close Lab';
  closeBtn.style.cssText = `
    position: absolute; top: 12px; right: 16px; z-index: 1;
    background: rgba(0,0,0,.5); border: 1px solid rgba(255,255,255,.2);
    color: #fff; padding: 5px 12px; border-radius: 8px;
    cursor: pointer; font-size: 12px; font-weight: 600;
  `;
  closeBtn.addEventListener('click', () => { overlay.style.display = 'none'; });

  const iframe = document.createElement('iframe');
  iframe.style.cssText = 'flex: 1; border: none; background: #0a0f1e;';
  iframe.srcdoc = createModelLabHTML();
  iframe.title = 'ModelMind Model Visualization Lab';

  modal.append(closeBtn, iframe);
  overlay.appendChild(modal);

  // Close on backdrop click
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.style.display = 'none';
  });

  document.body.appendChild(overlay);
}
