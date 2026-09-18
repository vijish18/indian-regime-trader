# Walk-forward backtest

- Generated: 2026-09-17T08:21:39.448676+00:00
- Period: 2016-01-01 to 2019-12-31
- Data: NSE bhavcopy (point-in-time universe), Kite index history,
  NSE corporate actions applied at read time.

| Strategy | CAGR | Vol | Sharpe | Max DD | Invested | Trades | Costs (Rs) |
|---|---|---|---|---|---|---|---|
| buy_and_hold | 0.00% | 0.00% | 0.00 | -0.00% | nan% | 0 | 0 |
| rolling_volatility | 2.30% | 2.09% | 1.10 | 1.21% | nan% | 265 | 76,337 |
| moving_average_trend | 2.99% | 3.43% | 0.88 | 1.56% | nan% | 560 | 100,998 |
| hmm | -0.74% | 1.21% | -0.61 | 1.46% | nan% | 54 | 29,417 |
| shuffled_regime_control | 0.00% | 0.00% | 0.00 | -0.00% | nan% | 0 | 0 |

## How to read this

docs/SPECIFICATION.md section 10.3 requires the HMM strategy to beat the
simple baseline **after costs**. The shuffled-regime control is the second
test: it keeps the same exposure *distribution* but destroys the timing, so
if the HMM does not beat it, the regime layer is only reducing average
exposure and a coin flip would do as well.

All figures are net of the Indian statutory cost model
(`backtest/costs.py`), not gross.
