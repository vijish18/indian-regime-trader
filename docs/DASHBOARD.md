# Regime Desk local dashboard

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m monitoring.web_server
```

Open http://127.0.0.1:8765. Change the port with `--port 8766` and the input
directory with `--state-dir PATH`. Stop the server with Ctrl+C.

## Azure research results

Open http://127.0.0.1:8765/#research for the current backtest comparison,
ending equity, returns, drawdowns, costs, and selectable equity curves.
Hover over a chart or use the arrow keys to inspect individual sessions.
Only strategies with a completed manifest, final report, curve, and completion
marker are shown as final results. Pending strategies show saved fold counts.

Run the separate read-only sync helper locally:

```powershell
.\.venv\Scripts\python.exe -u -m monitoring.backtest_results --run-root /home/irt/backtest_runs/fresh_20260921_series_v2 --ssh-host irt@20.219.11.26 --ssh-key C:/Users/Vijish/.ssh/irt_azure --watch 60
```

It polls over SSH every 60 seconds and exits after all five reports are
complete. The local computer must remain running for synchronization; the
Azure backtest itself is independent. Last-sync time is displayed separately
from paper-account timestamps. The helper atomically writes
`state/backtest_dashboard.json`; the server overlays only research fields so
legacy paper refreshes cannot overwrite this run. Mixed-run fingerprints are
rejected, and undefined report metrics display as unavailable. A failed sync
retains the previous snapshot and its original timestamp.

The dashboard reads `state/dashboard_data.json` every two seconds while the
tab is visible. It also reads newer observations from `state/live_ticks.json`.
It does not call Zerodha, start quote collectors, change a paper book, or place
orders. Existing producers must continue writing those files. Polling the
dashboard faster does not increase the source feed's frequency.

Features: animated metric changes, confidence gauge and state probabilities;
intraday price explorer with pointer and keyboard inspection; instrument and
window selection; searchable/sortable holdings; position stop details; closed
trade journal; and available backtest summaries. Reduced-motion preferences
are respected, with a manual motion toggle. Missing data stays unavailable;
there are no generated market prices or simulated heartbeat claims.

Book and tick timestamps are shown separately. A book older than 120 seconds
is marked stale, even if the HTTP connection or tick recorder is healthy.
This conservative threshold also applies outside market hours; it is a data
age indicator, not an exchange-calendar status. HMM observations remain daily.

The server binds to loopback and exposes four fixed routes only. It does not
serve the repository or state directory. It omits broker metadata from the
JSON response. This is a local development monitor, not an authenticated
internet service. All frontend assets live in version control under
`monitoring/web/`, unlike the older ignored `state/control_room.html` template.

The current paper scripts do not run through production allocation/risk
orchestration. The dashboard states this explicitly and does not label its
displayed market-value/equity ratio as an approved allocation target.

For true event-driven updates, the next step is a read-only quote/event
publisher fed by the existing Kite ticker adapter, followed by SSE or WebSocket
delivery with exchange timestamps and reconnect/gap handling. Keep the daily
strategy clock separate from the quote clock. That integration is not part of
this first dashboard implementation.
