# Large daily loss investigation

Before validating a backtest, investigate raw-price falls strictly greater
than 10%: previous available close to today's close or low, and today's open
to low. Record the previous observation date explicitly; a gap in history
is not a one-session return. Do not infer high-to-low ordering from daily bars.

Run `scripts/audit_price_losses.py` with `--data-root`, `--from`, `--to` and
`--output`. Its atomic JSON report contains observed prices, decline measures,
matching recorded actions and a dated internet research query. Exit code 1
means unresolved findings exist. A matching action never automatically clears
a finding. This is an investigation queue, not a completed repair or an
automatic internet researcher, and is not yet wired into the run launcher.

For each case, consult NSE PR corporate-action records and dated exchange or
company announcements. Record source URL, event terms, effective dates,
instrument identity, original observation and any correction. Classify real
market falls separately from corporate actions and bad data. Never repair a
price just because its fall looks large. Retain original bhavcopy as evidence;
use versioned action records or corrections to derive a new dataset. Update
share/cash entitlements and stop references consistently to avoid adjusting
prices twice. Recheck portfolio accounting before marking a case resolved.

The current report deliberately leaves every finding unresolved. Source
research, a validated resolution registry and enforcement in the backtest
launcher remain required before this becomes an end-to-end validation gate.
This screen supplements the complete corporate-action audit: events below
10%, missing observations and vanished securities still need separate checks.
