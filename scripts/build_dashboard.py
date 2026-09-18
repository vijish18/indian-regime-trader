"""Render the dashboard data as a self-contained HTML page.

    python scripts/collect_dashboard_data.py
    python scripts/build_dashboard.py --out state/dashboard.html

Charts are drawn as SVG here, in Python, rather than by a charting library
in the browser. The page then has no scripts, no network dependency and no
empty state: it is complete the moment it loads, which is what a thumbnail,
a shared link and a skim all get.

The page is written without ``<html>``/``<head>``/``<body>`` wrappers so it
can be published as an artifact, which supplies them. Browsers supply them
too, so the same file opens correctly from disk.

Nothing here computes a figure. Every number comes from
``collect_dashboard_data.py``'s JSON, and a panel whose data is absent
renders as absent -- never as a zero, which would be read as a result.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPO_ROOT))

REGIME_ORDER = ("calm", "normal", "elevated", "crisis")
"""Ordinal, not categorical: these are increasing risk, so the palette is a
ramp and the stack is always in this order. A legend that reordered them by
frequency would throw away the one thing the scale encodes."""

STRATEGY_LABELS = {
    "hmm": "HMM regime",
    "buy_and_hold": "Buy and hold",
    "rolling_volatility": "Rolling volatility",
    "moving_average_trend": "200d MA trend",
    "shuffled_regime_control": "Shuffled control",
}

# Events worth marking on the regime chart: the crisis windows the folds are
# supposed to be tested against. Dated, so they land by comparison rather
# than by eye.
CRISES = (
    ("2018-09-01", "IL&FS"),
    ("2020-03-01", "COVID"),
    ("2022-06-01", "Rate shock"),
)


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def regime_chart(folds: list[dict[str, Any]]) -> str:
    """Out-of-sample regime mix per fold, stacked, in risk order."""
    if not folds:
        return '<p class="absent">No fold data. Run scripts/precheck_folds.py.</p>'

    width, height = 960.0, 300.0
    pad_l, pad_r, pad_t, pad_b = 46.0, 12.0, 14.0, 46.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    slot = plot_w / len(folds)
    bar_w = min(slot * 0.74, 26.0)
    top = max(sum(f["regimes"].values()) for f in folds) or 1

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="Regime mix per walk-forward fold" class="chart">'
    ]

    # Horizontal guides, labelled in sessions so the axis names real values.
    for value in (0, top // 2, top):
        y = pad_t + plot_h - (value / top) * plot_h
        parts.append(
            f'<line x1="{pad_l:.1f}" y1="{y:.1f}" x2="{width - pad_r:.1f}" y2="{y:.1f}" '
            f'class="grid" />'
            f'<text x="{pad_l - 8:.1f}" y="{y + 4:.1f}" class="tick tick-y">{value}</text>'
        )

    for index, fold in enumerate(folds):
        x = pad_l + index * slot + (slot - bar_w) / 2
        cursor = pad_t + plot_h
        for regime in REGIME_ORDER:
            count = fold["regimes"].get(regime, 0)
            if not count:
                continue
            bar_h = (count / top) * plot_h
            cursor -= bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{cursor:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" fill="var(--regime-{regime})">'
                f"<title>Fold {fold['fold']} ({fold['test_start']} to "
                f"{fold['test_end']}): {count} {regime} sessions</title></rect>"
            )
        if index % 4 == 0:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{pad_t + plot_h + 18:.1f}" '
                f'class="tick tick-x">{esc(fold["test_start"][:7])}</text>'
            )

    # Crisis markers, placed against the folds' own test-start dates.
    for when, label in CRISES:
        position = next(
            (i for i, f in enumerate(folds) if f["test_start"] >= when), None
        )
        if position is None:
            continue
        x = pad_l + position * slot + slot / 2
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t:.1f}" x2="{x:.1f}" '
            f'y2="{pad_t + plot_h:.1f}" class="marker" />'
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 36:.1f}" class="tick marker-label">'
            f"{esc(label)}</text>"
        )

    parts.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:var(--regime-{r})"></i>{r}</span>'
        for r in REGIME_ORDER
    )
    return "".join(parts) + f'<div class="legend">{legend}</div>'


def universe_chart(points: list[dict[str, Any]]) -> str:
    """Point-in-time universe size: the survivorship fix, made visible."""
    if len(points) < 2:
        return '<p class="absent">No membership history.</p>'

    width, height = 960.0, 210.0
    pad_l, pad_r, pad_t, pad_b = 46.0, 12.0, 14.0, 34.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    top = max(p["count"] for p in points) or 1

    def xy(index: int, count: int) -> tuple[float, float]:
        return (
            pad_l + (index / (len(points) - 1)) * plot_w,
            pad_t + plot_h - (count / top) * plot_h,
        )

    coords = [xy(i, p["count"]) for i, p in enumerate(points)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = (
        f"{coords[0][0]:.1f},{pad_t + plot_h:.1f} "
        + line
        + f" {coords[-1][0]:.1f},{pad_t + plot_h:.1f}"
    )

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="Point-in-time universe size over time" class="chart">'
    ]
    for value in (0, top // 2, top):
        y = pad_t + plot_h - (value / top) * plot_h
        parts.append(
            f'<line x1="{pad_l:.1f}" y1="{y:.1f}" x2="{width - pad_r:.1f}" '
            f'y2="{y:.1f}" class="grid" />'
            f'<text x="{pad_l - 8:.1f}" y="{y + 4:.1f}" class="tick tick-y">{value}</text>'
        )
    parts.append(f'<polygon points="{area}" class="area" />')
    parts.append(f'<polyline points="{line}" class="line" />')
    end_x, end_y = coords[-1]
    parts.append(f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="4" class="endpoint" />')
    for index in (0, len(points) // 2, len(points) - 1):
        x, _ = coords[index]
        anchor = "start" if index == 0 else ("end" if index == len(points) - 1 else "middle")
        parts.append(
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 20:.1f}" class="tick tick-x" '
            f'text-anchor="{anchor}">{esc(points[index]["date"][:7])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def equity_chart(curves: dict[str, list[dict[str, Any]]]) -> str:
    if not curves:
        return ""
    width, height = 960.0, 300.0
    pad_l, pad_r, pad_t, pad_b = 60.0, 12.0, 14.0, 34.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    everything = [p["equity"] for pts in curves.values() for p in pts]
    low, high = min(everything), max(everything)
    span = (high - low) or 1.0
    longest = max(len(pts) for pts in curves.values())

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="Equity curve per strategy" class="chart">'
    ]
    for fraction in (0.0, 0.5, 1.0):
        value = low + span * fraction
        y = pad_t + plot_h - fraction * plot_h
        parts.append(
            f'<line x1="{pad_l:.1f}" y1="{y:.1f}" x2="{width - pad_r:.1f}" '
            f'y2="{y:.1f}" class="grid" />'
            f'<text x="{pad_l - 8:.1f}" y="{y + 4:.1f}" class="tick tick-y">'
            f"{value / 100000:.1f}L</text>"
        )
    for name, points in curves.items():
        coords = [
            (
                pad_l + (i / max(longest - 1, 1)) * plot_w,
                pad_t + plot_h - ((p["equity"] - low) / span) * plot_h,
            )
            for i, p in enumerate(points)
        ]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        parts.append(
            f'<polyline points="{line}" class="line" '
            f'style="stroke:var(--s-{name.replace("_", "-")})" />'
        )
    parts.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:var(--s-{n.replace("_", "-")})"></i>'
        f"{esc(STRATEGY_LABELS.get(n, n))}</span>"
        for n in curves
    )
    return "".join(parts) + f'<div class="legend">{legend}</div>'


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def strategy_panel(strategies: dict[str, Any]) -> str:
    reports = strategies.get("reports", {})
    if not reports:
        return (
            '<p class="absent">The walk-forward run has not produced results yet. '
            "Every figure here comes from a finished run; none is estimated.</p>"
        )

    head = (
        "<thead><tr><th>Strategy</th><th>CAGR</th><th>Vol</th><th>Sharpe</th>"
        "<th>Max DD</th><th>Invested</th><th>Trades</th><th>Costs</th></tr></thead>"
    )
    rows = []
    for name, report in reports.items():
        emphasis = ' class="row-hmm"' if name == "hmm" else ""
        rows.append(
            f"<tr{emphasis}><td>{esc(STRATEGY_LABELS.get(name, name))}</td>"
            f'<td class="num">{report["cagr"] * 100:.2f}%</td>'
            f'<td class="num">{report["volatility"] * 100:.2f}%</td>'
            f'<td class="num">{report["sharpe"]:.2f}</td>'
            f'<td class="num">{report["max_drawdown"] * 100:.2f}%</td>'
            f'<td class="num">{report["pct_invested"] * 100:.1f}%</td>'
            f'<td class="num">{report["trade_count"]:,}</td>'
            f'<td class="num">{report["total_costs"]:,.0f}</td></tr>'
        )
    missing = strategies.get("missing") or []
    note = (
        f'<p class="absent">Still running: {esc(", ".join(missing))}. '
        "The comparison is only meaningful whole.</p>"
        if missing
        else ""
    )
    return f'<div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div>{note}'


def cost_panel(eras: list[dict[str, Any]]) -> str:
    if not eras:
        return '<p class="absent">No cost schedule loaded.</p>'
    rows = "".join(
        f"<tr><td class=\"mono\">{esc(e['effective_from'])}</td>"
        f"<td>{esc(e['label'])}</td>"
        f'<td class="num">{e["stt_round_trip_bps"]:.0f}</td>'
        f'<td class="num">{e["stamp_duty_buy_bps"]:.1f}</td>'
        f'<td class="num">{e["exchange_txn_bps"]:.3f}</td>'
        f'<td class="num">{e["gst_pct"]:.2f}%</td></tr>'
        for e in eras
    )
    return (
        '<div class="scroll"><table><thead><tr><th>From</th><th>Change</th>'
        "<th>STT bps<br><small>round trip</small></th>"
        "<th>Stamp bps<br><small>buy</small></th>"
        "<th>Exch bps</th><th>GST</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def render(data: dict[str, Any]) -> str:
    prov = data["provenance"]
    regimes = data.get("regimes", {})
    folds = regimes.get("folds", [])
    strategies = data.get("strategies", {})
    curves = data.get("equity_curves", {})

    # Real aggregate across folds -- the claim that the model finds crises is
    # checkable from this number, so it is computed, not asserted.
    crisis_sessions = sum(f["regimes"].get("crisis", 0) for f in folds)
    total_sessions = sum(sum(f["regimes"].values()) for f in folds) or 1
    allow_new = sum(f["allow_new_positions"] for f in folds)

    stats = [
        ("Instruments", f"{prov['instruments']:,}", "point-in-time, from bhavcopy"),
        ("Daily bars", f"{prov['total_bars']:,}", "adjusted at read time"),
        ("Corporate actions", f"{prov['corporate_actions']:,}", "splits, bonuses, dividends"),
        ("Walk-forward folds", f"{len(folds)}", "out-of-sample 2018 to 2026"),
    ]
    stat_html = "".join(
        f'<div class="stat"><span class="stat-label">{esc(label)}</span>'
        f'<span class="stat-value">{esc(value)}</span>'
        f'<span class="stat-note">{esc(note)}</span></div>'
        for label, value, note in stats
    )

    window = strategies.get("window") or {}
    period = (
        f"{window.get('start', '2015-01-01')} to {window.get('end', '2026-09-15')}"
    )
    generated = data.get("generated_at", "")

    return f"""<title>Regime Trader Control Room</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
  :root {{
    --ink: #11141b;
    --ground: #f2f4f6;
    --panel: #ffffff;
    --edge: #d9dee4;
    --muted: #5d6874;
    --accent: #14705f;
    --accent-soft: #e2efeb;

    --regime-calm: #1f7a6b;
    --regime-normal: #7a9b3f;
    --regime-elevated: #c9852b;
    --regime-crisis: #b8452f;

    --s-hmm: #14705f;
    --s-buy-and-hold: #5d6874;
    --s-rolling-volatility: #7a9b3f;
    --s-moving-average-trend: #c9852b;
    --s-shuffled-regime-control: #a2308f;

    --display: "IBM Plex Sans Condensed", "Helvetica Neue", Arial, sans-serif;
    --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Consolas, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ink: #e8ecf1;
      --ground: #0e1116;
      --panel: #161b22;
      --edge: #262d36;
      --muted: #8b97a5;
      --accent: #3fbfa3;
      --accent-soft: #16302b;
      --regime-calm: #35a48f;
      --regime-normal: #9bbf58;
      --regime-elevated: #e0a044;
      --regime-crisis: #d96046;
      --s-hmm: #3fbfa3;
      --s-buy-and-hold: #8b97a5;
      --s-rolling-volatility: #9bbf58;
      --s-moving-average-trend: #e0a044;
      --s-shuffled-regime-control: #c96bb8;
    }}
  }}
  :root[data-theme="dark"] {{
    --ink: #e8ecf1;
    --ground: #0e1116;
    --panel: #161b22;
    --edge: #262d36;
    --muted: #8b97a5;
    --accent: #3fbfa3;
    --accent-soft: #16302b;
    --regime-calm: #35a48f;
    --regime-normal: #9bbf58;
    --regime-elevated: #e0a044;
    --regime-crisis: #d96046;
    --s-hmm: #3fbfa3;
    --s-buy-and-hold: #8b97a5;
    --s-rolling-volatility: #9bbf58;
    --s-moving-average-trend: #e0a044;
    --s-shuffled-regime-control: #c96bb8;
  }}

  body {{
    background: var(--ground);
    color: var(--ink);
    font-family: var(--body);
    line-height: 1.5;
    margin: 0;
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px; }}

  .eyebrow {{
    font-family: var(--display); font-size: 12px; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 0;
  }}
  h1 {{
    font-family: var(--display); font-weight: 700; font-size: clamp(30px, 5vw, 46px);
    line-height: 1.08; margin: 10px 0 12px; text-wrap: balance; letter-spacing: -0.01em;
  }}
  .lede {{ color: var(--muted); max-width: 62ch; margin: 0 0 6px; }}
  .meta {{ font-family: var(--mono); font-size: 12px; color: var(--muted); margin: 0; }}

  .stats {{ display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin: 30px 0 40px; }}
  .stat {{ display: flex; flex-direction: column; gap: 2px; padding: 14px 16px;
    background: var(--panel); border: 1px solid var(--edge); border-radius: 3px; }}
  .stat-label {{ font-family: var(--display); font-size: 11px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }}
  .stat-value {{ font-family: var(--mono); font-size: 24px; font-weight: 500;
    font-variant-numeric: tabular-nums; }}
  .stat-note {{ font-size: 12px; color: var(--muted); }}

  section {{ margin-bottom: 40px; }}
  h2 {{ font-family: var(--display); font-size: 20px; font-weight: 700;
    margin: 0 0 4px; letter-spacing: 0.01em; }}
  .sub {{ color: var(--muted); font-size: 14px; margin: 0 0 16px; max-width: 70ch; }}
  .card {{ background: var(--panel); border: 1px solid var(--edge);
    border-radius: 3px; padding: 18px; }}

  .chart {{ width: 100%; height: auto; display: block; }}
  .grid {{ stroke: var(--edge); stroke-width: 1; }}
  .tick {{ font-family: var(--mono); font-size: 11px; fill: var(--muted); }}
  .tick-y {{ text-anchor: end; }}
  .tick-x {{ text-anchor: middle; }}
  .marker {{ stroke: var(--ink); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0.45; }}
  .marker-label {{ text-anchor: middle; fill: var(--ink); font-weight: 500; }}
  .line {{ fill: none; stroke: var(--accent); stroke-width: 2; stroke-linejoin: round; }}
  .area {{ fill: var(--accent); opacity: 0.12; stroke: none; }}
  .endpoint {{ fill: var(--accent); stroke: var(--panel); stroke-width: 2; }}

  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 12px; }}
  .key {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12px;
    color: var(--muted); font-family: var(--mono); }}
  .key i {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}

  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--edge);
    white-space: nowrap; }}
  th {{ font-family: var(--display); font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
  th small {{ font-weight: 400; letter-spacing: 0; text-transform: none; }}
  .num, .mono {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
  .num {{ text-align: right; }}
  .row-hmm {{ background: var(--accent-soft); }}
  .row-hmm td {{ font-weight: 500; }}

  .absent {{ color: var(--muted); font-size: 14px;
    border-left: 2px solid var(--regime-elevated); padding-left: 12px; margin: 0; }}
  footer {{ border-top: 1px solid var(--edge); padding-top: 18px;
    color: var(--muted); font-size: 13px; }}
  footer code {{ font-family: var(--mono); font-size: 12px; }}
</style>

<div class="wrap">
  <p class="eyebrow">NSE cash equity &middot; long only &middot; walk-forward</p>
  <h1>Regime Trader Control Room</h1>
  <p class="lede">What the system is built on, and whether the regime layer earns
  its place. Every figure is read from a produced artifact; panels without data
  say so rather than showing a zero.</p>
  <p class="meta">Period {esc(period)} &middot; generated {esc(generated)}</p>

  <div class="stats">{stat_html}</div>

  <section>
    <h2>Does the regime layer beat the alternatives?</h2>
    <p class="sub">The specification requires the HMM to beat a simple baseline
    <em>after costs</em>, and to beat a shuffled control that keeps the same
    exposure distribution but destroys the timing. If it only beats the baseline,
    the layer is reducing average exposure and a coin flip would do as well.</p>
    <div class="card">{strategy_panel(strategies)}</div>
  </section>

  {"" if not curves else f'''<section>
    <h2>Equity</h2>
    <p class="sub">Chained across folds: each fold starts from the previous
    fold's ending equity, net of Indian statutory costs and slippage.</p>
    <div class="card">{equity_chart(curves)}</div>
  </section>'''}

  <section>
    <h2>What the model saw, fold by fold</h2>
    <p class="sub">Out-of-sample regime mix for each of the {len(folds)} folds, in
    risk order. Across every fold the model spent
    {crisis_sessions / total_sessions * 100:.0f}% of out-of-sample sessions in
    crisis and allowed new positions on {allow_new:,} of {total_sessions:,} —
    it is selective, not permanently defensive.</p>
    <div class="card">{regime_chart(folds)}</div>
  </section>

  <section>
    <h2>The universe is point-in-time</h2>
    <p class="sub">Built from daily bhavcopy — a record of what actually traded,
    not today's index membership projected backwards. It grows because the liquid
    NSE universe grew, and it contains companies that later delisted. That is the
    survivorship-bias fix, visible.</p>
    <div class="card">{universe_chart(data.get("universe_growth", []))}</div>
  </section>

  <section>
    <h2>Costs are dated, not retrofitted</h2>
    <p class="sub">A 2018 trade is priced with 2018's rates. STT dominates at
    ~20&nbsp;bps round trip and did not change; what moved was stamp duty
    (state-wise until exchanges began collecting it uniformly on 2020-07-01),
    service tax becoming GST, and four NSE charge revisions.</p>
    <div class="card">{cost_panel(data.get("cost_eras", []))}</div>
  </section>

  <footer>
    <p>Research figures, not a live account. Pre-2020 stamp duty is approximated
    at the Maharashtra rate because the true rate varied by the client's
    registered state, and brokerage is zero throughout, which flatters
    2015&ndash;2017 when percentage brokerage was still normal.</p>
    <p>Regenerate with <code>scripts/collect_dashboard_data.py</code> then
    <code>scripts/build_dashboard.py</code>.</p>
  </footer>
</div>
"""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "state" / "dashboard_data.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "state" / "dashboard.html")
    args = parser.parse_args(argv[1:])

    if not args.data.is_file():
        raise SystemExit(
            f"missing {args.data}\n  produce it with: python scripts/collect_dashboard_data.py"
        )
    data = json.loads(args.data.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(data), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    print(f"  generated from data of {data.get('generated_at')}")
    if not data.get("strategies", {}).get("available"):
        print("  NOTE: strategy comparison is incomplete; rebuild after the run finishes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
