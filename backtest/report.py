"""CSV and human-readable (Markdown/HTML) writers for performance
comparisons and robustness diagnostics.

CSV is the machine-readable form -- one row per strategy or per robustness
variant, one column per metric, meant for a spreadsheet or a further
analysis script, not for reading top to bottom. Markdown and HTML carry
the same numbers but are meant to be *read*: every comparison keeps its
caveats attached next to the numbers they qualify, and the robustness
section is never omitted when robustness results are supplied -- there is
no code path that produces a comparison table without also being able to
show whether it held up under perturbation.

This module performs no analysis of its own; it only formats what
``backtest/performance.py``, ``backtest/comparison.py``, and
``backtest/robustness.py`` already computed.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtest.comparison import BaselineComparison
from backtest.performance import PerformanceReport
from backtest.robustness import RobustnessReport

# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def write_performance_csv(reports: dict[str, PerformanceReport], path: Path) -> None:
    """One row per strategy, one column per :class:`PerformanceReport`
    field."""
    if not reports:
        raise ValueError("reports must not be empty")
    rows = [{"strategy": name, **asdict(report)} for name, report in reports.items()]
    _write_csv(pd.DataFrame(rows), path)


def write_robustness_csv(report: RobustnessReport, path: Path) -> None:
    """One row per variant, one column per :class:`PerformanceReport`
    field, plus the dimension and variant label."""
    rows = [
        {
            "dimension": report.dimension.value,
            "variant": variant.variant_label,
            **asdict(variant.performance),
        }
        for variant in report.variants
    ]
    _write_csv(pd.DataFrame(rows), path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

_STANDING_PREAMBLE = (
    "This report never declares a strategy \"successful\" on the basis of a single "
    "metric -- a high Sharpe ratio included. Read the caveats under each comparison, "
    "and the robustness section, before drawing a conclusion from any number below."
)


def write_markdown_report(
    comparisons: dict[str, BaselineComparison],
    robustness: dict[str, RobustnessReport] | None,
    path: Path,
    title: str = "Strategy performance report",
) -> None:
    if not comparisons:
        raise ValueError("comparisons must not be empty")

    lines: list[str] = [f"# {title}", "", _STANDING_PREAMBLE, ""]

    for baseline_name, comparison in comparisons.items():
        lines.extend(_comparison_markdown(baseline_name, comparison))

    if robustness:
        lines.append("## Robustness diagnostics")
        lines.append("")
        for dimension_name, report in robustness.items():
            lines.extend(_robustness_markdown(dimension_name, report))

    _write_text(path, "\n".join(lines) + "\n")


def _comparison_markdown(baseline_name: str, comparison: BaselineComparison) -> list[str]:
    lines = [f"## HMM vs. {baseline_name}", ""]
    lines.append("| Metric | HMM | " + baseline_name + " | Delta (HMM - baseline) |")
    lines.append("|---|---:|---:|---:|")
    for delta in comparison.deltas:
        direction = " (lower is better)" if delta.lower_is_better else ""
        lines.append(
            f"| {delta.metric}{direction} | {delta.hmm_value:.4f} | "
            f"{delta.baseline_value:.4f} | {delta.delta:+.4f} |"
        )
    lines.append("")
    lines.append("**Caveats:**")
    lines.append("")
    for caveat in comparison.caveats:
        lines.append(f"- {caveat}")
    lines.append("")
    return lines


def _robustness_markdown(dimension_name: str, report: RobustnessReport) -> list[str]:
    lines = [f"### {dimension_name}", ""]
    lines.append(
        "| Variant | "
        + " | ".join(report.dispersion.keys())
        + " |"
    )
    lines.append("|---|" + "---:|" * len(report.dispersion))
    for variant in report.variants:
        values = " | ".join(
            f"{getattr(variant.performance, metric):.4f}" for metric in report.dispersion
        )
        lines.append(f"| {variant.variant_label} | {values} |")
    lines.append("")

    lines.append("Dispersion across variants (relative range = (max - min) / |mean|):")
    lines.append("")
    lines.append("| Metric | Min | Max | Mean | Stdev | Relative range |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for metric, dispersion in report.dispersion.items():
        lines.append(
            f"| {metric} | {dispersion.minimum:.4f} | {dispersion.maximum:.4f} | "
            f"{dispersion.mean:.4f} | {dispersion.stdev:.4f} | {dispersion.relative_range:.2f} |"
        )
    lines.append("")
    return lines


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def write_html_report(
    comparisons: dict[str, BaselineComparison],
    robustness: dict[str, RobustnessReport] | None,
    path: Path,
    title: str = "Strategy performance report",
) -> None:
    if not comparisons:
        raise ValueError("comparisons must not be empty")

    body: list[str] = [f"<h1>{_escape(title)}</h1>", f"<p>{_escape(_STANDING_PREAMBLE)}</p>"]
    for baseline_name, comparison in comparisons.items():
        body.extend(_comparison_html(baseline_name, comparison))
    if robustness:
        body.append("<h2>Robustness diagnostics</h2>")
        for dimension_name, report in robustness.items():
            body.extend(_robustness_html(dimension_name, report))

    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{_escape(title)}</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;color:#1a1a1a;background:#fff;}"
        "table{border-collapse:collapse;margin-bottom:1rem;width:100%;}"
        "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:right;font-size:0.9rem;}"
        "th:first-child,td:first-child{text-align:left;}"
        "th{background:#f2f2f2;}"
        "ul{margin-top:0;}"
        "</style></head><body>" + "".join(body) + "</body></html>"
    )
    _write_text(path, document)


def _comparison_html(baseline_name: str, comparison: BaselineComparison) -> list[str]:
    rows = "".join(
        f"<tr><td>{_escape(delta.metric)}"
        f"{' (lower is better)' if delta.lower_is_better else ''}</td>"
        f"<td>{delta.hmm_value:.4f}</td><td>{delta.baseline_value:.4f}</td>"
        f"<td>{delta.delta:+.4f}</td></tr>"
        for delta in comparison.deltas
    )
    caveats = "".join(f"<li>{_escape(caveat)}</li>" for caveat in comparison.caveats)
    return [
        f"<h2>HMM vs. {_escape(baseline_name)}</h2>",
        "<table><thead><tr><th>Metric</th><th>HMM</th>"
        f"<th>{_escape(baseline_name)}</th><th>Delta (HMM - baseline)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        f"<p><strong>Caveats:</strong></p><ul>{caveats}</ul>",
    ]


def _robustness_html(dimension_name: str, report: RobustnessReport) -> list[str]:
    metrics = list(report.dispersion.keys())
    header = "".join(f"<th>{_escape(metric)}</th>" for metric in metrics)
    variant_rows = "".join(
        "<tr><td>"
        + _escape(variant.variant_label)
        + "</td>"
        + "".join(f"<td>{getattr(variant.performance, metric):.4f}</td>" for metric in metrics)
        + "</tr>"
        for variant in report.variants
    )
    dispersion_rows = "".join(
        f"<tr><td>{_escape(metric)}</td><td>{d.minimum:.4f}</td><td>{d.maximum:.4f}</td>"
        f"<td>{d.mean:.4f}</td><td>{d.stdev:.4f}</td><td>{d.relative_range:.2f}</td></tr>"
        for metric, d in report.dispersion.items()
    )
    return [
        f"<h3>{_escape(dimension_name)}</h3>",
        f"<table><thead><tr><th>Variant</th>{header}</tr></thead>"
        f"<tbody>{variant_rows}</tbody></table>",
        "<p>Dispersion across variants "
        "(relative range = (max - min) / |mean|):</p>",
        "<table><thead><tr><th>Metric</th><th>Min</th><th>Max</th><th>Mean</th>"
        "<th>Stdev</th><th>Relative range</th></tr></thead>"
        f"<tbody>{dispersion_rows}</tbody></table>",
    ]


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
