'use strict';
const $ = id => document.getElementById(id);
const finite = v => typeof v === 'number' && Number.isFinite(v);
const num = (v, digits = 2) => finite(v) ? v.toLocaleString('en-IN', {minimumFractionDigits: digits, maximumFractionDigits: digits}) : '—';
const money = v => finite(v) ? `₹${num(v)}` : '—';
const pct = v => finite(v) ? `${num(v * 100, 1)}%` : '—';
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sign = v => finite(v) ? (v >= 0 ? 'positive' : 'negative') : 'muted';
const age = value => { const t = Date.parse(value); return Number.isFinite(t) ? Math.max(0, (Date.now() - t) / 1000) : Infinity; };
const ageText = value => { const a = age(value); return !Number.isFinite(a) ? 'timestamp unavailable' : a < 60 ? `${Math.floor(a)}s ago` : a < 3600 ? `${Math.floor(a / 60)}m ago` : `${Math.floor(a / 3600)}h ago`; };
let data = {}, lastBody = '', connected = false, shortRange = false, selectedPosition = '', chartPoints = [], cursor = -1;
const values = new Map(), animations = new Map();
let paused = matchMedia('(prefers-reduced-motion: reduce)').matches;

function metric(id, value, format = money) {
  const el = $(id), old = values.get(id);
  values.set(id, value);
  if (animations.has(id)) cancelAnimationFrame(animations.get(id));
  if (!finite(value) || !finite(old) || old === value || paused) { el.textContent = format(value); return; }
  el.classList.remove('flash-up', 'flash-down');
  void el.offsetWidth;
  el.classList.add(value >= old ? 'flash-up' : 'flash-down');
  const start = performance.now();
  function frame(now) {
    const progress = Math.min(1, (now - start) / 650);
    el.textContent = format(old + (value - old) * (1 - (1 - progress) ** 3));
    if (progress < 1 && !paused) animations.set(id, requestAnimationFrame(frame));
    else el.textContent = format(value);
  }
  animations.set(id, requestAnimationFrame(frame));
}

function freshness() {
  $('clock').textContent = new Date().toLocaleTimeString('en-GB', {timeZone:'Asia/Kolkata'}) + ' IST';
  const book = data.live_book || {}, timestamp = book.fetched_at;
  $('connection').textContent = !connected ? 'DISCONNECTED' : age(timestamp) > 120 ? 'CONNECTED / STALE BOOK' : 'CONNECTED / RECENT BOOK';
  $('connection').style.color = connected && age(timestamp) <= 120 ? 'var(--green)' : 'var(--amber)';
  $('book-age').textContent = `Book observation: ${timestamp || 'unavailable'} · ${ageText(timestamp)}. Prices are last recorded values, not guaranteed executable quotes.`;
  $('tick-age').textContent = `Ticks: ${ageText(data.ticks?.updated_at)}`;
  const warnings = [];
  if (!connected) warnings.push('Connection lost. Keeping the last successfully loaded snapshot.');
  if (age(timestamp) > 120) warnings.push('Book data is stale or unavailable. Animated charts do not imply a live market feed.');
  if (data.regime_now?.available && !data.regime_now.confident) warnings.push('Model confidence is below its configured threshold.');
  warnings.push('Paper-book scripts bypass the production allocation/risk flow; displayed exposure is actual book exposure, not an approved target.');
  $('notice').textContent = warnings.join(' ');
}

function render() {
  const b = data.live_book?.available ? data.live_book : {}, r = data.regime_now?.available ? data.regime_now : {};
  metric('equity', b.equity); metric('pnl', b.pnl); metric('realized', data.realized?.available ? data.realized.net_pnl : undefined);
  metric('exposure', finite(b.market_value) && b.equity > 0 ? b.market_value / b.equity : undefined, pct);
  $('pnl').className = sign(b.pnl); $('realized').className = sign(data.realized?.net_pnl);
  $('cash').textContent = `Cash ${money(b.cash)}`;
  $('equity-sub').textContent = b.hypothetical ? 'Hypothetical sizing — not a paper ledger' : `Paper account · return ${pct(b.equity_pct)}`;
  $('closed-count').textContent = `${data.realized?.trades ?? '—'} closed paper trades · after costs`;
  $('regime').textContent = r.label ? r.label.charAt(0).toUpperCase() + r.label.slice(1) : 'Model unavailable';
  $('regime-date').textContent = r.as_of || '';
  metric('confidence', r.confidence, pct);
  $('gauge').style.strokeDasharray = `${Math.max(0, Math.min(100, (r.confidence || 0) * 100))} 100`;
  $('gauge').style.stroke = r.confident ? 'var(--green)' : 'var(--amber)';
  $('conviction').textContent = finite(r.min_confidence) ? `${r.confident ? 'Above' : 'Below'} the ${pct(r.min_confidence)} confidence threshold` : 'No current inference';
  const states = data.hmm?.states || [];
  $('probabilities').innerHTML = (r.probabilities || []).map((p, i) => `<div class="probability"><span>${esc(states[i]?.label || `State ${i}`)}</span><div class="track"><div class="fill" data-width="${finite(p) ? Math.max(0, Math.min(100, p * 100)) : 0}"></div></div><span>${pct(p)}</span></div>`).join('');
  requestAnimationFrame(() => document.querySelectorAll('[data-width]').forEach(el => { el.style.width = el.dataset.width + '%'; }));
  const symbolSet = new Set();
  for (const p of data.ticks?.points || []) for (const key of Object.keys(p.px || {})) symbolSet.add(key);
  for (const p of b.positions || []) symbolSet.add(`NSE:${p.symbol}`);
  const symbols = [...symbolSet].sort(), current = $('symbol').value;
  $('symbol').innerHTML = symbols.map(s => `<option value="${esc(s)}">${esc(s.replace('NSE:', ''))}</option>`).join('');
  if (symbols.includes(current)) $('symbol').value = current;
  else if (symbols.includes('NSE:LAURUSLABS')) $('symbol').value = 'NSE:LAURUSLABS';
  renderChart(); renderPositions(); renderActivity(); renderResearch();
  $('snapshot-time').textContent = `SNAPSHOT ${data.generated_at || 'unavailable'} · polling every 2s`;
  freshness();
}

function renderChart() {
  $('chart').onpointermove = null;
  $('chart').onpointerleave = null;
  $('chart').onkeydown = null;
  const symbol = $('symbol').value;
  chartPoints = (data.ticks?.points || []).filter(p => finite(p.px?.[symbol])).map(p => ({time:p.hhmm || p.t, value:p.px[symbol]}));
  if (shortRange) chartPoints = chartPoints.slice(-30);
  $('chart-title').textContent = symbol ? symbol.replace('NSE:', '') : 'Intraday observations';
  cursor = chartPoints.length - 1;
  if (chartPoints.length < 2) {
    $('chart').innerHTML = '<div class="empty">No intraday series for this instrument yet.<br>Recorded observations will appear when the quote recorder supplies them.</div>';
    $('chart-price').textContent = chartPoints.length ? money(chartPoints[0].value) : '—';
    $('chart-detail').textContent = 'Insufficient recorded observations'; return;
  }
  const prices = chartPoints.map(p => p.value), min = Math.min(...prices), max = Math.max(...prices), pad = Math.max((max - min) * .15, max * .0001);
  const low = min - pad, high = max + pad, x = i => 55 + i / (prices.length - 1) * 635, y = v => 205 - (v - low) / (high - low) * 185;
  let svg = '<svg viewBox="0 0 710 240" role="img" aria-label="Recorded price in rupees by time in IST">';
  for (let i = 0; i < 4; i++) { const v = low + (high - low) * i / 3; svg += `<line class="chart-grid" x1="55" x2="690" y1="${y(v)}" y2="${y(v)}"/><text class="chart-text" text-anchor="end" x="47" y="${y(v)+3}">${num(v,1)}</text>`; }
  const points = prices.map((v,i) => `${x(i)},${y(v)}`).join(' ');
  svg += `<polygon points="55,205 ${points} 690,205" fill="var(--green)" opacity=".045"/><polyline class="chart-line" pathLength="1000" stroke-dasharray="1000" points="${points}"/><line id="crosshair" x1="690" x2="690" y1="20" y2="205" stroke="var(--muted)" stroke-dasharray="3 4"/><circle id="cursor-dot" cx="690" cy="${y(prices.at(-1))}" r="4" fill="var(--green)"/>`;
  for (const i of [0, Math.floor((prices.length - 1) / 2), prices.length - 1]) svg += `<text class="chart-text" x="${x(i)}" y="230" text-anchor="${i === 0 ? 'start' : i === prices.length - 1 ? 'end' : 'middle'}">${esc(chartPoints[i].time)}</text>`;
  $('chart').innerHTML = svg + '</svg>';
  $('chart').onpointermove = event => { const rect = $('chart').querySelector('svg').getBoundingClientRect(); inspectPoint(Math.round(((event.clientX - rect.left) / rect.width * 710 - 55) / 635 * (prices.length - 1))); };
  $('chart').onpointerleave = () => inspectPoint(prices.length - 1);
  $('chart').onkeydown = event => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); inspectPoint(cursor + (event.key === 'ArrowLeft' ? -1 : 1)); } };
  function inspectPoint(index) {
    cursor = Math.max(0, Math.min(prices.length - 1, index)); const p = chartPoints[cursor];
    $('chart-price').textContent = money(p.value);
    $('chart-detail').textContent = `${p.time} IST · ${pct(p.value / prices[0] - 1)} from first shown sample`;
    $('crosshair').setAttribute('x1', x(cursor)); $('crosshair').setAttribute('x2', x(cursor));
    $('cursor-dot').setAttribute('cx', x(cursor)); $('cursor-dot').setAttribute('cy', y(p.value));
  }
  inspectPoint(cursor);
}

function renderPositions() {
  const all = data.live_book?.available ? data.live_book.positions || [] : [];
  const query = $('search').value.toUpperCase();
  const positions = all.filter(p => String(p.symbol).toUpperCase().includes(query)).slice();
  positions.sort((a,b) => $('sort').value === 'symbol' ? a.symbol.localeCompare(b.symbol) : $('sort').value === 'risk' ? (a.stop?.hard_distance_pct ?? Infinity) - (b.stop?.hard_distance_pct ?? Infinity) : (b.pnl || 0) - (a.pnl || 0));
  $('position-count').textContent = ` / ${all.length}`;
  $('holdings').innerHTML = positions.map(p => `<tr><td>${esc(p.symbol)}</td><td>${num(p.shares,0)}</td><td>${money(p.last)}</td><td class="${sign(p.day_pct)}">${pct(p.day_pct)}</td><td class="${sign(p.pnl)}">${money(p.pnl)}</td><td><span class="risk-meter"><i style="width:${Math.max(0,Math.min(100,(p.stop?.hard_distance_pct || 0)*1000))}%"></i></span>${pct(p.stop?.hard_distance_pct)}</td><td><button data-symbol="${esc(p.symbol)}" aria-label="Inspect ${esc(p.symbol)}">Inspect ↗</button></td></tr>`).join('') || '<tr><td colspan="7" class="empty">No matching positions in this snapshot.</td></tr>';
  $('holdings').querySelectorAll('button').forEach(button => button.onclick = () => { selectedPosition = selectedPosition === button.dataset.symbol ? '' : button.dataset.symbol; renderDetail(); });
  renderDetail();
}

function renderDetail() {
  const p = (data.live_book?.positions || []).find(p => p.symbol === selectedPosition);
  $('position-detail').hidden = !p;
  if (p) $('position-detail').innerHTML = `<strong>${esc(p.symbol)}</strong> · entered ${esc(p.entry_date)} at ${money(p.entry)}<br>Market value ${money(p.value)} · hard stop ${money(p.stop?.hard_level)} · trailing stop ${money(p.stop?.trail_level)}<br>Reported stop state: <strong>${esc(p.stop?.state || 'unavailable')}</strong>. Stop levels are monitoring information; this dashboard submits no orders.`;
}

function renderActivity() {
  const history = data.realized?.available ? data.realized.history || [] : [];
  $('activity').innerHTML = history.slice(-6).reverse().map(t => `<div class="event"><div>${esc(t.symbol)}<small>${esc(t.exit_reason?.replaceAll('_',' '))} · ${esc(t.exit_date)} · ${num(t.shares,0)} shares</small></div><div class="${sign(t.net_pnl)}">${money(t.net_pnl)}<small>net of ${money(t.costs)} costs</small></div></div>`).join('') || '<p class="empty">No closed trades recorded.</p>';
}

function renderResearch() {
  const reports = data.strategies?.reports || {};
  $('results').innerHTML = Object.entries(reports).map(([name, report]) => `<div class="result"><div>${esc(name.replaceAll('_',' '))}<small>Sharpe ${num(report.sharpe_ratio ?? report.sharpe)} · max drawdown ${pct(report.max_drawdown)}</small></div><strong class="${sign(report.cagr)}">${pct(report.cagr)}<small>CAGR after costs</small></strong></div>`).join('') || '<p class="empty">No completed strategy reports in this snapshot.</p>';
}

async function poll() {
  if (!document.hidden) {
    try {
      const response = await fetch('/api/snapshot', {cache:'no-store', signal:AbortSignal.timeout(5000)});
      if (!response.ok) throw new Error('Snapshot unavailable');
      const body = await response.text(), parsed = JSON.parse(body);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Invalid snapshot');
      connected = true;
      if (body !== lastBody) { data = parsed; lastBody = body; render(); }
    } catch { connected = false; }
    freshness();
  }
  setTimeout(poll, 2000);
}
$('symbol').onchange = renderChart;
$('range').onclick = () => { shortRange = !shortRange; $('range').setAttribute('aria-pressed', shortRange); $('range').textContent = shortRange ? 'Full session' : 'Last 30'; renderChart(); };
$('search').oninput = renderPositions; $('sort').onchange = renderPositions;
function motionState() { document.documentElement.classList.toggle('motion-off', paused); $('motion').setAttribute('aria-pressed', paused); $('motion').textContent = paused ? 'Enable motion' : 'Pause motion'; }
$('motion').onclick = () => { paused = !paused; motionState(); };
motionState(); setInterval(freshness, 1000); poll();
