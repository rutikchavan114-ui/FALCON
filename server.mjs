import http from 'node:http';
import fs from 'node:fs/promises';
import crypto from 'node:crypto';

const port = Number(process.env.PORT || 8787);
const clients = new Set();
const telemetry = [];
const componentState = new Map();
const sessions = new Map();
const MAX = 2400;
const key = String(process.env.GEMINI_API_KEY || '').trim().replace(/^['"]|['"]$/g, '');
const requestedModel = String(process.env.GEMINI_MODEL || '').trim();
const model = ['gemini-2.5-flash','models/gemini-2.5-flash','gemini-3-flash-preview'].includes(requestedModel) || !requestedModel ? 'gemini-3.8-flash' : requestedModel;
const logoPath = new URL('./public/falcon-logo.png', import.meta.url);

const credentials = {
  Admin: {
    username: process.env.FALCON_ADMIN_USER || 'admin',
    password: process.env.FALCON_ADMIN_PASSWORD || 'FALCON@2026'
  },
  Engineer: {
    username: process.env.FALCON_ENGINEER_USER || 'engineer',
    password: process.env.FALCON_ENGINEER_PASSWORD || 'ENGINEER@2026'
  }
};

let demoTimer = null;
let demoTick = 0;
let demoRunning = false;
let demoScenario = 'healthy';

function json(res, status, payload) {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
  });
  res.end(JSON.stringify(payload));
}

async function body(req) {
  let b = '';
  for await (const c of req) b += c;
  if (!b) return {};
  return JSON.parse(b);
}

function broadcast(data) {
  const msg = `data: ${JSON.stringify({ type: 'telemetry', data })}\n\n`;
  for (const res of clients) {
    try { res.write(msg); } catch { clients.delete(res); }
  }
}

function broadcastStatus() {
  const msg = `data: ${JSON.stringify({ type: 'status', demoRunning, demoScenario })}\n\n`;
  for (const res of clients) {
    try { res.write(msg); } catch { clients.delete(res); }
  }
}

function validNumber(x) { return typeof x === 'number' && Number.isFinite(x); }
function clamp(v, a = 0, b = 1) { return Math.max(a, Math.min(b, v)); }
function safe(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c])); }

function sessionFrom(req) {
  const raw = req.headers.authorization || '';
  const token = raw.startsWith('Bearer ') ? raw.slice(7) : '';
  return token && sessions.get(token);
}

function requireAuth(req, res) {
  const session = sessionFrom(req);
  if (!session) {
    json(res, 401, { error: 'Authentication required' });
    return null;
  }
  return session;
}

function analytics({ voltage, current, temperature, componentId, source }) {
  const state = componentState.get(componentId) || { last: null, equivalentAgeSeconds: 0, samples: 0 };
  const prev = state.last;
  const now = Date.now();
  const dt = prev ? Math.max(0, (now - prev.time) / 1000) : 0;
  const tempSlope = prev && dt > 0 ? (temperature - prev.temperature) / dt : 0;
  const tempDelta = prev ? Math.abs(temperature - prev.temperature) : 0;
  const voltageDrift = prev ? Math.abs(voltage - prev.voltage) : 0;
  const currentDrift = prev ? Math.abs(current - prev.current) : 0;

  const TrefK = 333.15;
  const TtestK = temperature + 273.15;
  const EaEV = 0.70;
  const kEV = 8.617333262e-5;
  const arrheniusAF = Math.exp((EaEV / kEV) * (1 / TrefK - 1 / TtestK));
  state.equivalentAgeSeconds += dt * arrheniusAF;
  state.last = { time: now, temperature, voltage, current };
  state.samples += 1;
  componentState.set(componentId, state);

  // Transparent demo evidence engine. This is NOT presented as a trained production ML model.
  const tempEvidence = clamp((temperature - 78) / 25);
  const driftEvidence = clamp(tempDelta / 8.0);
  const electricalEvidence = clamp((voltageDrift * 3 + currentDrift * 1.5));
  const anomalyScore = clamp(0.55 * tempEvidence + 0.30 * driftEvidence + 0.15 * electricalEvidence);
  const riskScore = clamp(0.60 * anomalyScore + 0.25 * clamp((arrheniusAF - 1) / 8) + 0.15 * driftEvidence);
  const decision = riskScore >= 0.72 ? 'ENGINEER REVIEW' : riskScore >= 0.48 ? 'MONITOR' : 'PASS / CONTINUE';

  return {
    componentId,
    timestamp: new Date().toISOString(),
    voltage,
    current,
    temperature,
    power: voltage * current,
    driftScore: driftEvidence,
    arrheniusAF,
    equivalentAgeSeconds: state.equivalentAgeSeconds,
    anomalyScore,
    riskScore,
    decision,
    sampleCount: state.samples,
    modelStatus: source === 'DEMO_SIMULATION' ? 'DEMO_EVIDENCE_ENGINE' : 'CALIBRATION_REQUIRED',
    source,
    physics: { referenceTemperatureK: TrefK, activationEnergyEV: EaEV },
    provenance: source === 'DEMO_SIMULATION' ? 'Synthetic demo stream; not a hardware measurement.' : 'Received through /api/telemetry.'
  };
}

function processTelemetry(x, source = 'LIVE_HARDWARE_TELEMETRY') {
  const voltage = Number(x.voltage), current = Number(x.current), temperature = Number(x.temperature);
  if (![voltage, current, temperature].every(Number.isFinite)) {
    throw new Error('voltage, current and temperature must be finite numbers');
  }
  const componentId = String(x.componentId || 'DUT-UNASSIGNED');
  const p = analytics({ voltage, current, temperature, componentId, source });
  if (x.timestamp && !Number.isNaN(Date.parse(x.timestamp))) p.timestamp = new Date(x.timestamp).toISOString();
  telemetry.push(p);
  if (telemetry.length > MAX) telemetry.shift();
  broadcast(p);
  return p;
}

function makeDemoPoint() {
  demoTick += 1;
  const t = demoTick;
  const cycle = Math.sin(t / 7);
  const slow = Math.sin(t / 29);
  const stress = demoScenario === 'stress' ? clamp((t % 110) / 90) : 0;
  const fault = demoScenario === 'fault' ? clamp((t % 90) / 55) : 0;
  const temperature = 58 + 3 * cycle + 2 * slow + 24 * stress + 29 * fault;
  const voltage = 12.05 - 0.06 * cycle - 0.35 * fault + 0.02 * Math.sin(t / 3);
  const current = 0.42 + 0.025 * Math.sin(t / 5) + 0.12 * stress + 0.25 * fault;
  const componentId = ['MOS-01', 'MOS-07', 'MOS-13', 'MOS-19'][Math.floor(t / 35) % 4];
  return processTelemetry({ componentId, voltage, current, temperature }, 'DEMO_SIMULATION');
}

function startDemo(scenario = 'healthy') {
  demoScenario = ['healthy', 'stress', 'fault'].includes(scenario) ? scenario : 'healthy';
  componentState.clear();
  if (demoTimer) clearInterval(demoTimer);
  demoRunning = true;
  demoTimer = setInterval(makeDemoPoint, 900);
  makeDemoPoint();
  broadcastStatus();
}

function stopDemo() {
  if (demoTimer) clearInterval(demoTimer);
  demoTimer = null;
  demoRunning = false;
  broadcastStatus();
}

async function llm(input) {
  if (!key) return { error: 'Gemini is not configured. Put your Gemini API key in .env as GEMINI_API_KEY=... and restart the backend.', model };

  const system = `You are FALCON, an engineering AI copilot for SIH26170 component burn-in and screening. Be evidence-bound. Never invent measurements, component specifications, model accuracy, training results, failure rates, or industrial validation. Distinguish measured telemetry, deterministic calculations, demo simulation evidence, model outputs, assumptions and missing evidence. A component part number is identity only, not telemetry. If evidence is insufficient, explicitly state what is missing. Keep final engineering decisions human-controlled. Use concise engineering language and structure answers with Facts, Interpretation, Limitations, Next action when appropriate.`;
  const context = {
    mode: input.mode || 'copilot', question: input.question || '', page: input.page || '',
    component: input.component || null, telemetry: (input.telemetry || []).slice(-60),
    prompt: input.prompt || '', demoMode: Boolean(input.demoMode)
  };

  // Gemini 3.8 Flash is served through the current Interactions API.
  // Keep this server-side so the browser never receives the API key.
  const url = 'https://generativelanguage.googleapis.com/v1beta/interactions';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
      signal: controller.signal,
      body: JSON.stringify({
        model,
        input: JSON.stringify(context, null, 2),
        system_instruction: system,
        store: false,
        generation_config: { max_output_tokens: input.healthCheck ? 32 : 1400, thinking_level: input.healthCheck ? 'minimal' : 'low' }
      })
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      const message = d?.error?.message || `Gemini HTTP ${r.status}`;
      return { error: `${message} [model: ${model}]`, model, provider: 'Google Gemini Interactions API' };
    }

    const text = d?.output_text ||
      d?.steps?.filter(step => step?.type === 'model_output')
        ?.flatMap(step => Array.isArray(step.content) ? step.content : [])
        ?.filter(part => part?.type === 'text' && part?.text)
        ?.map(part => part.text).join('') ||
      d?.outputs?.flatMap(o => Array.isArray(o?.content) ? o.content : [])
        ?.filter(part => part?.type === 'text' && part?.text)
        ?.map(part => part.text).join('') || '';

    return { text: String(text).trim() || 'Gemini completed without a text response.', model, provider: 'Google Gemini Interactions API' };
  } catch (e) {
    return { error: e?.name === 'AbortError' ? `Gemini request timed out after 30s [model: ${model}]` : `Gemini request failed: ${e?.message || e} [model: ${model}]`, model, provider: 'Google Gemini Interactions API' };
  } finally {
    clearTimeout(timeout);
  }
}

async function logoDataUri() {
  try {
    const b = await fs.readFile(logoPath);
    return `data:image/png;base64,${b.toString('base64')}`;
  } catch { return ''; }
}

function sparkSvg(rows, field, stroke) {
  const pts = rows.slice(-100).map(r => Number(r[field])).filter(Number.isFinite);
  if (pts.length < 2) return '<div class="empty">No telemetry points available.</div>';
  const min = Math.min(...pts), max = Math.max(...pts), span = max - min || 1;
  const coords = pts.map((v, i) => `${(i / (pts.length - 1)) * 720},${185 - ((v - min) / span) * 150}`).join(' ');
  return `<svg viewBox="0 0 720 200" preserveAspectRatio="none"><defs><linearGradient id="fill-${field}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${stroke}" stop-opacity=".30"/><stop offset="1" stop-color="${stroke}" stop-opacity="0"/></linearGradient></defs><polyline points="0,200 ${coords} 720,200" fill="url(#fill-${field})"/><polyline points="${coords}" fill="none" stroke="${stroke}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

async function reportHTML({ components = [], telemetry: rows = [], user = null }) {
  const logo = await logoDataUri();
  const latest = rows.at(-1);
  const generated = new Date().toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
  const decisions = rows.length ? { pass: rows.filter(r => r.decision === 'PASS / CONTINUE').length, monitor: rows.filter(r => r.decision === 'MONITOR').length, review: rows.filter(r => r.decision === 'ENGINEER REVIEW').length } : { pass:0, monitor:0, review:0 };
  const componentRows = components.map(c => {
    const rs = rows.filter(r => r.componentId === c.id);
    const l = rs.at(-1);
    return `<tr><td><b>${safe(c.part)}</b><small>${safe(c.id)}</small></td><td>${safe(c.category)}</td><td>${rs.length ? 'RECEIVED' : 'NOT RECEIVED'}</td><td>${l ? Number(l.anomalyScore).toFixed(2) : '—'}</td><td>${l ? Number(l.riskScore).toFixed(2) : '—'}</td><td class="${l?.decision === 'ENGINEER REVIEW' ? 'bad' : l?.decision === 'MONITOR' ? 'warn' : 'good'}">${l?.decision || 'NO DECISION'}</td></tr>`;
  }).join('');
  const samples = rows.slice(-120).map(r => `<tr><td>${new Date(r.timestamp).toLocaleTimeString('en-IN')}</td><td>${safe(r.componentId)}</td><td>${Number(r.voltage).toFixed(3)}</td><td>${Number(r.current).toFixed(3)}</td><td>${Number(r.temperature).toFixed(2)}</td><td>${Number(r.power).toFixed(3)}</td><td>${Number(r.arrheniusAF).toFixed(3)}×</td><td>${safe(r.source === 'DEMO_SIMULATION' ? 'SIMULATION' : 'HARDWARE')}</td></tr>`).join('');
  return `<!doctype html><html><head><meta charset="utf-8"><title>FALCON Engineering Report</title><style>
  @page{size:A4;margin:12mm}*{box-sizing:border-box}body{font-family:Inter,Arial,sans-serif;color:#122038;background:#fff;margin:0}.cover{min-height:260px;border-radius:26px;padding:30px;color:white;background:linear-gradient(135deg,#07111f,#102a54 48%,#5b21b6);position:relative;overflow:hidden}.cover:after{content:"";position:absolute;width:500px;height:500px;border-radius:50%;right:-220px;top:-280px;border:1px solid #7dd3fc55;box-shadow:0 0 90px #38bdf855}.logo{width:250px;height:125px;object-fit:contain;object-position:left center;filter:drop-shadow(0 8px 24px #22d3ee55);background:#fff;border-radius:14px;padding:8px;margin-bottom:18px}.kicker{font-size:10px;letter-spacing:.18em;color:#8ee8ff;font-weight:800}.cover h1{font-size:31px;margin:6px 0}.cover p{max-width:680px;color:#c9dcf5;line-height:1.55}.meta{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:20px}.meta div{background:#ffffff10;border:1px solid #ffffff18;padding:11px;border-radius:12px}.meta span{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.12em;color:#9fb8d7}.meta b{display:block;margin-top:4px}.section{margin-top:24px}.section h2{font-size:17px;margin:0 0 9px}.section p{color:#66748a;line-height:1.5}.notice{padding:12px 14px;border-radius:12px;background:#f3f7ff;border-left:4px solid #4f7cff;font-size:10px;line-height:1.55}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.card{border:1px solid #dfe6f0;border-radius:14px;padding:13px;background:linear-gradient(145deg,#fff,#f7f9fd)}.card .n{font-size:9px;text-transform:uppercase;color:#7787a1;letter-spacing:.12em}.card b{font-size:20px;display:block;margin-top:5px}.chart{border:1px solid #dfe6f0;border-radius:16px;padding:12px;margin-top:10px}.chart h3{margin:0 0 8px;font-size:12px}.chart svg{width:100%;height:150px;display:block}.chart .empty{height:150px;display:grid;place-items:center;color:#8995a7;font-size:11px}.table{width:100%;border-collapse:collapse;font-size:9px}.table th{text-align:left;background:#edf3fb;color:#52617a;font-size:8px;text-transform:uppercase;letter-spacing:.08em}.table td,.table th{padding:7px;border-bottom:1px solid #e5eaf2}.table small{display:block;color:#8995a7;margin-top:2px}.good{color:#087f5b}.warn{color:#ad6800}.bad{color:#c02b52}.footer{margin-top:24px;padding-top:12px;border-top:1px solid #e1e6ef;font-size:8px;color:#7b8799;line-height:1.5}@media print{.no-print{display:none}}</style></head><body>
  <div class="cover">${logo ? `<img class="logo" src="${logo}"/>` : ''}<div class="kicker">FALCON · SIH26170 · BURNSIGHT</div><h1>Consolidated Burn-In Engineering Report</h1><p>Evidence-led review of the current FALCON screening population, telemetry stream, deterministic physics calculations, demo/hardware provenance and decision state.</p><div class="meta"><div><span>Components</span><b>${components.length}</b></div><div><span>Telemetry points</span><b>${rows.length}</b></div><div><span>Source</span><b>${rows.length ? (rows.at(-1).source === 'DEMO_SIMULATION' ? 'SIMULATION' : 'HARDWARE') : 'NONE'}</b></div><div><span>Generated</span><b>${generated}</b></div></div></div>
  <div class="section"><div class="notice"><b>Engineering boundary:</b> Demo-mode telemetry is synthetic and clearly marked. It is suitable for UI/backend demonstration only. Hardware measurements, calibrated healthy populations and validated ML artifacts are required before production screening claims.</div></div>
  <div class="section"><h2>Executive evidence snapshot</h2><div class="cards"><div class="card"><span class="n">Latest temperature</span><b>${latest ? Number(latest.temperature).toFixed(2) + ' °C' : '—'}</b></div><div class="card"><span class="n">Arrhenius acceleration</span><b>${latest ? Number(latest.arrheniusAF).toFixed(2) + '×' : '—'}</b></div><div class="card"><span class="n">Latest disposition</span><b>${latest ? safe(latest.decision) : 'NO DECISION'}</b></div></div></div>
  <div class="section"><h2>Signal behaviour</h2><div class="chart"><h3>Temperature trajectory</h3>${sparkSvg(rows,'temperature','#4f7cff')}</div><div class="chart"><h3>Power trajectory</h3>${sparkSvg(rows,'power','#8b5cf6')}</div></div>
  <div class="section"><h2>Disposition matrix</h2><table class="table"><thead><tr><th>Part</th><th>Category</th><th>Telemetry</th><th>Anomaly</th><th>Risk</th><th>Disposition</th></tr></thead><tbody>${componentRows || '<tr><td colspan="6">No components supplied.</td></tr>'}</tbody></table></div>
  <div class="section"><h2>Decision distribution</h2><div class="cards"><div class="card"><span class="n">Pass / Continue samples</span><b>${decisions.pass}</b></div><div class="card"><span class="n">Monitor samples</span><b>${decisions.monitor}</b></div><div class="card"><span class="n">Engineer review samples</span><b>${decisions.review}</b></div></div></div>
  <div class="section"><h2>Recent telemetry evidence</h2><table class="table"><thead><tr><th>Time</th><th>DUT</th><th>V</th><th>I</th><th>T</th><th>P</th><th>AF</th><th>Source</th></tr></thead><tbody>${samples || '<tr><td colspan="8">No telemetry received.</td></tr>'}</tbody></table></div>
  <div class="footer">FALCON generated report · User: ${safe(user?.name || 'Engineering demo')} · Role: ${safe(user?.role || 'Engineer')} · Backend provenance is recorded per telemetry point. No industrial validation or model accuracy is claimed by this report.</div><script>window.onload=()=>setTimeout(()=>window.print(),350)</script></body></html>`;
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') return json(res, 204, {});
  try {
    if (req.method === 'POST' && req.url === '/api/auth/login') {
      const x = await body(req);
      const role = x.role === 'Admin' ? 'Admin' : 'Engineer';
      const c = credentials[role];
      if (String(x.username || '') !== c.username || String(x.password || '') !== c.password) return json(res, 401, { error: 'Invalid role, username or password.' });
      const token = crypto.randomBytes(24).toString('hex');
      sessions.set(token, { role, username: c.username, createdAt: Date.now() });
      return json(res, 200, { ok: true, token, user: { name: c.username, role } });
    }
    if (req.method === 'POST' && req.url === '/api/auth/logout') {
      const raw = req.headers.authorization || ''; const token = raw.startsWith('Bearer ') ? raw.slice(7) : ''; sessions.delete(token); return json(res, 200, { ok: true });
    }
    if (req.method === 'GET' && req.url === '/api/health') return json(res, 200, {
      ok: true, uptime: process.uptime(), liveClients: clients.size, telemetrySamples: telemetry.length,
      demoRunning, demoScenario, geminiConfigured: Boolean(key), geminiModel: model,
      memory: process.memoryUsage().rss, capabilities: { telemetry: true, sse: true, demoSimulation: true, arrhenius: true, reports: true, llm: Boolean(key), auth: true, mlCalibrationRequired: true }
    });
    if (req.method === 'GET' && req.url === '/api/stream') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive', 'Access-Control-Allow-Origin': '*' });
      res.write(`data: ${JSON.stringify({ type: 'hello', samples: telemetry.length, demoRunning, demoScenario })}\n\n`);
      clients.add(res); req.on('close', () => clients.delete(res)); return;
    }
    if (req.method === 'GET' && req.url === '/api/telemetry/latest') return json(res, 200, { data: telemetry.at(-1) || null, samples: telemetry.length });
    if (req.method === 'GET' && req.url === '/api/demo/status') return json(res, 200, { running: demoRunning, scenario: demoScenario, samples: telemetry.length });

    if (req.method === 'POST' && req.url === '/api/demo/start') {
      const session = requireAuth(req, res); if (!session) return;
      const x = await body(req); startDemo(x.scenario); return json(res, 200, { ok: true, running: true, scenario: demoScenario, startedBy: session.role });
    }
    if (req.method === 'POST' && req.url === '/api/demo/stop') {
      const session = requireAuth(req, res); if (!session) return;
      stopDemo(); return json(res, 200, { ok: true, running: false, stoppedBy: session.role });
    }
    if (req.method === 'POST' && req.url === '/api/telemetry') {
      const session = sessionFrom(req);
      if (!session) return json(res, 401, { error: 'Authentication required' });
      const p = processTelemetry(await body(req), 'LIVE_HARDWARE_TELEMETRY');
      return json(res, 202, { accepted: true, data: p });
    }
    if (req.method === 'POST' && req.url === '/api/llm') {
      const session = requireAuth(req, res); if (!session) return;
      return json(res, 200, await llm(await body(req)));
    }
    if (req.method === 'POST' && req.url === '/api/ai/ping') {
      const session = requireAuth(req, res); if (!session) return;
      if (!key) return json(res, 200, { ok: false, configured: false, model, error: 'GEMINI_API_KEY is missing.' });
      const result = await llm({ healthCheck: true, mode: 'health', question: 'Respond with exactly: FALCON AI ONLINE' });
      return json(res, 200, { ok: Boolean(result.text), configured: true, model, provider: result.provider, error: result.error || null, text: result.text || null });
    }
    if (req.method === 'POST' && req.url === '/api/report') {
      const session = requireAuth(req, res); if (!session) return;
      return json(res, 200, { html: await reportHTML({ ...(await body(req)), user: session }) });
    }
    return json(res, 404, { error: 'Not found' });
  } catch (e) {
    return json(res, 400, { error: e?.message || 'Bad request' });
  }
});

server.listen(port, () => console.log(`FALCON backend listening on http://localhost:${port}`));
process.on('SIGINT', () => { stopDemo(); process.exit(0); });
process.on('SIGTERM', () => { stopDemo(); process.exit(0); });
