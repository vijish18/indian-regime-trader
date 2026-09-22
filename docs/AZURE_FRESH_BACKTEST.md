# Fresh Azure backtest: 21 September 2026 data

The batch uses the existing `irt-bt-01` VM, four concurrent workers, and a
separate immutable checkout and dataset. It creates no Azure resources and
does not touch broker sessions or place orders. Running the existing VM can
consume Azure credits or incur charges; free-account status is not a billing
guarantee.

Inputs cover 2015-01-01 through 2026-09-21. Each strategy starts with INR
100,000. Training windows contain 756 sessions and test windows up to 63
signal sessions. The final partial window is included: 34 folds, with the
last 54 signals from 6 July through 18 September, executed through 21 September.
The cutoff bounds execution/valuation dates; no signal requires future data.
The five
policies are HMM, shuffled HMM control, rolling volatility, moving-average
trend, and the existing `buy_and_hold` policy. That last name denotes the
existing always-invested stock-selection baseline, not passive NIFTY returns.

Session state is atomically replaced after each simulated session. Each
strategy has separate manifests, audit records, cash curves, and trade logs.
The batch retries a failed worker at most twice using its saved state; it
survives SSH disconnection when run as the systemd service below. A restart
still refits the interrupted fold's HMM before restoring execution state.
Code/config/data fingerprint changes reject resumption. Folds liquidate at
their terminal close with execution costs, then carry cash into the next fold.

Remote paths:

- Checkout: `/home/irt/backtest_20260921_series_v2`
- Dataset: `/home/irt/data_runs/2026-09-21-series-v2`
- Results: `/home/irt/backtest_runs/fresh_20260921_series_v2`
- Service: `irt-backtest-20260921`

The initial 33-fold run was stopped and retained at
`/home/irt/backtest_runs/fresh_20260921`. The corrected run starts independently:
its changed coverage/code fingerprint intentionally cannot resume old state.

The subsequent full-window run stopped during window 14 because the EQ-only
price export omitted ADANITRANS's BE-series bar on 2021-08-23. Its results remain
at `/home/irt/backtest_runs/fresh_20260921_full`. Series-v2 rebuilds all 2,081
equity histories with EQ/BE/BZ observations (one price bar per symbol/day,
preferring EQ when multiple series print). The entry universe remains EQ-only;
execution also rejects new buys on BE/BZ days. Existing holdings can be valued
and sold using the observed prices. No missing price is fabricated. Genuinely
missing fold-end liquidation prices still stop the run for investigation.

The engine also no longer records a future session's open as a fill on a day
with no bar. These data and execution corrections can affect prior windows,
so series-v2 starts fresh rather than importing the earlier equity/checkpoints.
Preflight verifies series metadata and uniqueness across the entire bar store.
The repaired dataset has 3,755,854 bars: 3,561,770 EQ, 166,726 BE, and 27,358 BZ.
NSE classifies BE/BZ as trade-for-trade equity series:
https://www.nseindia.com/static/market-data/legend-of-series

Inspect with `systemctl status irt-backtest-20260921`, and read each strategy's
`run.log` and `series/run.manifest.json` under the results directory. The
`series/sessions` files contain progress within an unfinished fold. After a
failure, correct the underlying operational issue and run
`sudo systemctl start irt-backtest-20260921`; do not change the frozen code or
dataset beneath existing checkpoints. A strategy writes `completed.txt` only
after its final report has been saved. Stop with
`sudo systemctl stop irt-backtest-20260921`.

The dataset retains the project's weekday/exchange-holiday calendar and
corporate-action exclusion/fallback assumptions. Inspect their impact before
using results to make investment decisions. This research run does not
activate the unfinished paper-trading runtime.
