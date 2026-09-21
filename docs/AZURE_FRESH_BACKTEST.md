# Fresh Azure backtest: 21 September 2026 data

The batch uses the existing `irt-bt-01` VM, four concurrent workers, and a
separate immutable checkout and dataset. It creates no Azure resources and
does not touch broker sessions or place orders. Running the existing VM can
consume Azure credits or incur charges; free-account status is not a billing
guarantee.

Inputs cover 2015-01-01 through 2026-09-21. Each strategy starts with INR
100,000. Training windows contain 756 sessions and test windows 63 sessions;
the latest incomplete window is excluded by the fold generator. The five
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

- Checkout: `/home/irt/backtest_20260921`
- Dataset: `/home/irt/data_runs/2026-09-21`
- Results: `/home/irt/backtest_runs/fresh_20260921`
- Service: `irt-backtest-20260921`

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
