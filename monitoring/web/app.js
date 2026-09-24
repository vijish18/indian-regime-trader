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
  renderChart(); renderPositions(); renderActivity(); renderResearch(); renderHMMHistory();
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

const researchNames = {hmm:'HMM regime strategy', buy_and_hold:'Always-invested selection', rolling_volatility:'Rolling volatility', moving_average_trend:'Moving-average trend', shuffled_regime_control:'Shuffled regime control'};
let researchPoints = [], researchIndex = -1;
function renderResearch() {
  const reports = data.strategies?.reports || {}, run = data.backtest_run || {};
  const completed = Object.keys(reports);
  $('research-status').textContent = `UNVALIDATED \u00b7 ${completed.length} / 5 RUNS COMPLETE`;
  $('research-meta').textContent = run.run_id ? `Corporate-action accounting has unresolved errors. These figures cannot establish strategy profitability. ${run.run_id} \u00b7 Data through ${run.end} \u00b7 ${run.folds_total} test windows \u00b7 Starting capital ${money(run.initial_equity)}` : 'No current Azure research snapshot.';
  $('results').innerHTML = Object.entries(researchNames).map(([name,label]) => {
    const report = reports[name], progress = run.progress?.[name];
    if (!report) return `<article class="research-card pending"><span class="eyebrow">${esc(label)}</span><strong>Pending</strong><p>${progress ? `${progress.completed} / ${progress.total} windows saved` : 'Awaiting report'}</p><div class="track"><div class="fill" data-width="${progress?.total ? 100*progress.completed/progress.total : 0}"></div></div><small>No final performance reported yet</small></article>`;
    return `<article class="research-card"><span class="eyebrow">${esc(label)}</span><strong class="${sign(report.return_from_capital)}">${money(report.ending_equity)}</strong><small>Ending equity \u00b7 ${pct(report.return_from_capital)} total return</small><dl><div><dt>Annualised return</dt><dd class="${sign(report.cagr)}">${pct(report.cagr)}</dd></div><div><dt>Max drawdown</dt><dd class="negative">${pct(report.max_drawdown)}</dd></div><div><dt>Sharpe</dt><dd>${num(report.sharpe)}</dd></div><div><dt>Fills / total costs</dt><dd>${num(report.trade_count,0)} / ${money(report.total_costs)}</dd></div></dl><small>COMPLETE \u00b7 ${esc(report.last_execution)}</small></article>`;
  }).join('');
  const selected = $('research-strategy').value;
  $('research-strategy').innerHTML = completed.map(name => `<option value="${esc(name)}">${esc(researchNames[name] || name)}</option>`).join('');
  if (completed.includes(selected)) $('research-strategy').value = selected;
  renderResearchChart();
  researchFreshness();
}
function researchFreshness() {
  const timestamp = data.backtest_run?.synced_at;
  $('research-sync').textContent = timestamp ? `Azure snapshot synced ${new Date(timestamp).toLocaleString('en-IN',{timeZone:'Asia/Kolkata'})} IST \u00b7 ${ageText(timestamp)}. ${data.backtest_run.complete ? 'All strategy reports complete.' : 'Research sync checks every 60s while the local helper is running.'}` : 'Azure results have not been synced.';
  $('research-sync').classList.toggle('negative', Boolean(timestamp) && !data.backtest_run?.complete && age(timestamp) > 180);
}
function renderResearchChart() {
  const name = $('research-strategy').value, report = data.strategies?.reports?.[name];
  researchPoints = (data.equity_curves?.[name] || []).filter(p => finite(p.equity));
  const box = $('research-chart');
  if (!researchPoints.length) { box.innerHTML = '<p class="empty">No completed equity curve available.</p>'; return; }
  const W=1000,H=245,left=68,right=18,top=16,bottom=28;
  const max = Math.max(report?.starting_equity || 0, ...researchPoints.map(p=>p.equity))*1.05;
  const x = i => left + i / Math.max(1,researchPoints.length-1)*(W-left-right);
  const y = v => H-bottom-v/Math.max(1,max)*(H-top-bottom);
  const path = researchPoints.map((p,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(p.equity).toFixed(2)}`).join(' ');
  const grid = [0,.25,.5,.75,1].map(r=>`<line class="chart-grid" x1="${left}" x2="${W-right}" y1="${y(max*r)}" y2="${y(max*r)}"/><text class="chart-text" x="${left-8}" y="${y(max*r)+4}" text-anchor="end">${num(max*r/1000,0)}k</text>`).join('');
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(researchNames[name])} equity in rupees"><title>${esc(researchNames[name])}: ${esc(money(report?.starting_equity))} initial capital to ${esc(money(report?.ending_equity))}</title>${grid}<path class="chart-line" style="stroke:${report?.return_from_capital>=0?'var(--green)':'var(--red)'}" d="${path}"/><line id="research-crosshair" y1="${top}" y2="${H-bottom}" stroke="var(--amber)" stroke-dasharray="3 3"/><circle id="research-dot" r="4" fill="var(--amber)"/><text class="chart-text" x="${left}" y="${H-5}">${esc(researchPoints[0].date)}</text><text class="chart-text" x="${W-right}" y="${H-5}" text-anchor="end">${esc(researchPoints.at(-1).date)}</text></svg>`;
  $('research-range').textContent = `${report.first_execution} \u00b7 ${report.last_execution} \u00b7 ${researchPoints.length.toLocaleString()} simulated sessions`;
  researchIndex = researchPoints.length-1;
  const inspect = i => {
    researchIndex = Math.max(0,Math.min(researchPoints.length-1,i));
    const p=researchPoints[researchIndex], px=x(researchIndex);
    $('research-value').textContent=money(p.equity);
    $('research-value').className=sign(p.equity-report.starting_equity);
    $('research-point').textContent=`${p.date} \u00b7 ${pct(p.equity/report.starting_equity-1)} from starting capital`;
    $('research-crosshair').setAttribute('x1',px); $('research-crosshair').setAttribute('x2',px);
    $('research-dot').setAttribute('cx',px); $('research-dot').setAttribute('cy',y(p.equity));
  };
  box.onpointermove = event => { const rect=box.getBoundingClientRect(); inspect(Math.round(((event.clientX-rect.left)/rect.width*W-left)/(W-left-right)*(researchPoints.length-1))); };
  box.onpointerleave = () => inspect(researchPoints.length-1);
  box.onkeydown = event => { if (['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) { event.preventDefault(); inspect(event.key==='Home'?0:event.key==='End'?researchPoints.length-1:researchIndex+(event.key==='ArrowRight'?1:-1)); } };
  inspect(researchIndex);
}

const sellReasons = {hard_stop:'Hard stop', trailing_profit_stop:'Trailing profit stop', portfolio_rebalance:'Portfolio rebalance', fold_end_liquidation:'Fold-end liquidation', take_profit:'Take profit'};
let historyPage = 0, selectedHistoryId = null;
const historyPageSize = 50;
function filteredHistory() {
  const search=$('history-search').value.trim().toUpperCase(), reason=$('history-reason').value;
  const outcome=$('history-outcome').value, from=$('history-from').value, to=$('history-to').value;
  const rows=(data.hmm_history?.rows || []).filter(r => (!search || r.symbol.includes(search)) && (!reason || r.reason===reason) && (!from || r.exit_date>=from) && (!to || r.exit_date<=to) && (!outcome || (outcome==='loss' ? r.net_pnl<0 : r.net_pnl>0)));
  const sort=$('history-sort').value;
  rows.sort((a,b)=>sort==='loss'?a.net_pnl-b.net_pnl:sort==='profit'?b.net_pnl-a.net_pnl:sort==='fees'?b.total_fees-a.total_fees:sort==='earliest'?a.id-b.id:b.id-a.id);
  return rows;
}
function renderHMMHistory() {
  const history=data.hmm_history || {}, rows=filteredHistory(), prices=history.current_quotes?.prices || {};
  $('history-total').textContent=history.available ? `(${num(history.rows.length,0)} sell fills)` : '';
  const quoteReason=history.current_quotes?.reason;
  $('history-quotes').textContent=quoteReason ? `Current market prices unavailable: ${quoteReason}. ${/session|expired/i.test(quoteReason)?'A fresh Kite login is needed.':'The quote helper will retry.'} Dated historical closes are shown separately.` : history.available ? `Current quote check: ${history.current_quotes?.checked_at || 'not available'}. Quote timestamps are shown per stock. Backtest positions are as of ${history.as_of}.` : 'HMM trade history has not been imported.';
  const totalPages=Math.max(1,Math.ceil(rows.length/historyPageSize));
  historyPage=Math.min(historyPage,totalPages-1);
  const start=historyPage*historyPageSize, page=rows.slice(start,start+historyPageSize);
  const pnl=rows.reduce((s,r)=>s+r.net_pnl,0), fees=rows.reduce((s,r)=>s+r.total_fees,0);
  $('history-summary').textContent=`${num(rows.length,0)} matching sales · Net P/L ${money(pnl)} · Buy + sell costs ${money(fees)}`;
  $('history-rows').innerHTML=page.map(r=>{
    const quote=prices[r.instrument_id], historical=history.historical_prices?.[r.instrument_id];
    return `<tr><td>${esc(r.symbol)}<small>Fold ${r.fold} · ${r.holding_days} calendar days</small></td><td>${esc(r.entry_date)}<small>→ ${esc(r.exit_date)}</small></td><td>${num(r.quantity,0)}</td><td>${money(r.entry_price)}</td><td>${quote ? `${money(quote.price)}<small>${esc(quote.as_of || 'Timestamp unavailable')}</small>` : '<span class="muted">Unavailable</span>'}</td><td>${historical ? `${money(historical.price)}<small>${esc(historical.as_of)}</small>` : '—'}</td><td>${money(r.exit_price)}</td><td class="${sign(r.net_pnl)}">${money(r.net_pnl)}</td><td class="${sign(r.return_pct)}">${pct(r.return_pct)}</td><td>${num(r.shares_after_exit,0)}</td><td>${num(r.shares_at_end,0)}</td><td>${esc(sellReasons[r.reason] || r.reason)}</td><td><button data-history-id="${r.id}" aria-label="Details for ${esc(r.symbol)} sale ${r.id}">Details</button></td></tr>`;
  }).join('') || '<tr><td colspan="13" class="empty">No matching HMM sales.</td></tr>';
  document.querySelectorAll('[data-history-id]').forEach(button=>{button.onclick=()=>{selectedHistoryId=Number(button.dataset.historyId);renderHistoryDetail();};});
  $('history-page').textContent=`${rows.length ? start+1 : 0}–${Math.min(start+historyPageSize,rows.length)} of ${num(rows.length,0)} · Page ${historyPage+1}/${totalPages}`;
  $('history-prev').disabled=historyPage===0; $('history-next').disabled=historyPage+1>=totalPages;
  const summary=history.summary, report=data.strategies?.reports?.hmm;
  $('history-reconcile').textContent=summary && report ? `Whole-run reconciliation: starting ${money(report.starting_equity)} + realised trade P/L ${money(summary.realized_net_pnl)} + other cash movements ${money(summary.other_cash_movements)} = ending ${money(report.ending_equity)}. Other cash movements include dividends and are not allocated to individual sale rows.` : '';
  renderHistoryDetail();
}
function renderHistoryDetail() {
  const row=(data.hmm_history?.rows || []).find(r=>r.id===selectedHistoryId), box=$('history-detail');
  box.hidden=!row;
  if (!row) return;
  box.innerHTML=`<div class="section-head"><strong>${esc(row.symbol)} · Sale #${row.id} · ${esc(row.exit_date)}</strong><button id="history-close">Close</button></div><div class="history-detail-grid"><div><b>Entry and position</b><p>First entry ${esc(row.entry_date)}; last buy ${esc(row.last_entry_date)}<br>${row.entry_fill_count} acquisition fills in this position cycle<br>Weighted entry ${money(row.entry_price)} · sold ${num(row.quantity,0)} shares<br>${num(row.shares_after_exit,0)} shares remained after this sale; ${num(row.shares_at_end,0)} at run end.</p></div><div><b>P/L after costs</b><p>Allocated acquisition cost ${money(row.entry_cost)}<br>Net sale proceeds ${money(row.net_proceeds)}<br>Gross P/L ${money(row.gross_pnl)}<br>Buy costs ${money(row.entry_fees)} + sell costs ${money(row.exit_fees)}<br><strong class="${sign(row.net_pnl)}">Net P/L ${money(row.net_pnl)} (${pct(row.return_pct)})</strong></p></div><div><b>Why it sold</b><p>${esc(sellReasons[row.reason] || row.reason)}<br>Evidence: ${esc(row.reason_source)}<br>${finite(row.stop_level)?`Recorded stop level ${money(row.stop_level)}<br>`:''}Signal ${esc(row.signal_date)} · execution ${esc(row.exit_date)}<br>HMM ${esc(row.regime || 'unavailable')} · confidence ${pct(row.confidence)}<br>Target weight ${pct(row.target_weight_before)} → ${pct(row.target_weight_after)}</p></div></div>`;
  $('history-close').onclick=()=>{selectedHistoryId=null;renderHistoryDetail();};
}
function exportHistoryCSV() {
  const rows=filteredHistory(), history=data.hmm_history || {};
  const fields=['id','symbol','fold','entry_date','last_entry_date','signal_date','exit_date','quantity','entry_price','exit_price','net_pnl','return_pct','gross_pnl','entry_fees','exit_fees','entry_cost','net_proceeds','shares_after_exit','shares_at_end','holding_days','reason','reason_source','stop_level','regime','confidence'];
  const columns=[...fields,'current_quote','current_quote_time','last_dataset_price','last_dataset_price_date'];
  const cell=value=>{let text=String(value??'');if(typeof value==='string' && /^[=+@-]/.test(text))text="'"+text;return '"'+text.replaceAll('"','""')+'"';};
  const lines=[columns.map(cell).join(','),...rows.map(r=>{const q=history.current_quotes?.prices?.[r.instrument_id],p=history.historical_prices?.[r.instrument_id];return [...fields.map(k=>r[k]),q?.price,q?.as_of,p?.price,p?.as_of].map(cell).join(',');})];
  const url=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'})), link=document.createElement('a');
  link.href=url;link.download='hmm-backtest-sales.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
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
$('research-strategy').onchange = renderResearchChart;
['history-search','history-reason','history-outcome','history-from','history-to','history-sort'].forEach(id=>{$(id).oninput=()=>{historyPage=0;selectedHistoryId=null;renderHMMHistory();};});
$('history-prev').onclick=()=>{historyPage=Math.max(0,historyPage-1);renderHMMHistory();};
$('history-next').onclick=()=>{historyPage++;renderHMMHistory();};
$('history-export').onclick=exportHistoryCSV;
$('symbol').onchange = renderChart;
$('range').onclick = () => { shortRange = !shortRange; $('range').setAttribute('aria-pressed', shortRange); $('range').textContent = shortRange ? 'Full session' : 'Last 30'; renderChart(); };
$('search').oninput = renderPositions; $('sort').onchange = renderPositions;
function motionState() { document.documentElement.classList.toggle('motion-off', paused); $('motion').setAttribute('aria-pressed', paused); $('motion').textContent = paused ? 'Enable motion' : 'Pause motion'; }
$('motion').onclick = () => { paused = !paused; motionState(); };
motionState(); setInterval(() => { freshness(); researchFreshness(); }, 1000); poll();
