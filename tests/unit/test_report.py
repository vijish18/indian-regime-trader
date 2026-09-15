"""CSV, Markdown, and HTML report writers: the numbers ``comparison.py``
and ``robustness.py`` computed, formatted for a spreadsheet or a reader
-- without adding, dropping, or reinterpreting any of them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest.comparison import compare_all
from backtest.performance import PerformanceReport
from backtest.report import (
    write_html_report,
    write_markdown_report,
    write_performance_csv,
    write_robustness_csv,
)
from backtest.robustness import RobustnessDimension, RobustnessSuite


def performance_report(**overrides: object) -> PerformanceReport:
    defaults: dict[str, object] = dict(
        cagr=0.10,
        total_return=0.10,
        max_drawdown=0.08,
        drawdown_duration_days=10,
        recovery_duration_days=5,
        volatility=0.15,
        downside_deviation=0.10,
        sharpe=0.60,
        sortino=0.80,
        calmar=1.25,
        turnover=1.5,
        average_holding_period_days=12.0,
        pct_invested=0.70,
        pct_cash=0.30,
        trade_count=120,
        win_rate=0.52,
        profit_factor=1.20,
        gross_pnl=120_000.0,
        total_costs=20_000.0,
        net_pnl=100_000.0,
        gross_return=0.12,
        net_return=0.10,
        cost_pct_of_turnover=0.002,
    )
    defaults.update(overrides)
    return PerformanceReport(**defaults)  # type: ignore[arg-type]


def sample_reports() -> dict[str, PerformanceReport]:
    return {
        "hmm": performance_report(cagr=0.15, sharpe=1.1),
        "buy_and_hold": performance_report(cagr=0.05, sharpe=0.3),
        "rolling_volatility": performance_report(cagr=0.08, sharpe=0.5),
    }


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def test_write_performance_csv_has_one_row_per_strategy(tmp_path: Path) -> None:
    path = tmp_path / "performance.csv"
    write_performance_csv(sample_reports(), path)

    frame = pd.read_csv(path)
    assert set(frame["strategy"]) == {"hmm", "buy_and_hold", "rolling_volatility"}
    assert "cagr" in frame.columns
    assert "sharpe" in frame.columns


def test_write_performance_csv_values_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "performance.csv"
    write_performance_csv(sample_reports(), path)

    frame = pd.read_csv(path, index_col="strategy")
    assert frame.loc["hmm", "cagr"] == pytest.approx(0.15)
    assert frame.loc["buy_and_hold", "sharpe"] == pytest.approx(0.3)


def test_write_performance_csv_rejects_empty_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_performance_csv({}, tmp_path / "performance.csv")


def test_write_performance_csv_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "performance.csv"
    write_performance_csv(sample_reports(), path)
    assert path.is_file()


def test_write_robustness_csv_has_one_row_per_variant(tmp_path: Path) -> None:
    suite = RobustnessSuite()
    report = suite.run(
        RobustnessDimension.TRAINING_WINDOW,
        {
            "60d": lambda: performance_report(cagr=0.08),
            "90d": lambda: performance_report(cagr=0.12),
        },
    )
    path = tmp_path / "robustness.csv"
    write_robustness_csv(report, path)

    frame = pd.read_csv(path)
    assert set(frame["variant"]) == {"60d", "90d"}
    assert (frame["dimension"] == "training_window").all()
    assert "cagr" in frame.columns


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def test_write_markdown_report_includes_every_baseline(tmp_path: Path) -> None:
    comparisons = compare_all(sample_reports(), hmm_key="hmm")
    path = tmp_path / "report.md"
    write_markdown_report(comparisons, None, path)

    text = path.read_text(encoding="utf-8")
    assert "## HMM vs. buy_and_hold" in text
    assert "## HMM vs. rolling_volatility" in text


def test_write_markdown_report_includes_the_standing_preamble(tmp_path: Path) -> None:
    comparisons = compare_all(sample_reports(), hmm_key="hmm")
    path = tmp_path / "report.md"
    write_markdown_report(comparisons, None, path)

    text = path.read_text(encoding="utf-8")
    assert "never declares a strategy" in text


def test_write_markdown_report_includes_every_caveat(tmp_path: Path) -> None:
    comparisons = compare_all(sample_reports(), hmm_key="hmm")
    path = tmp_path / "report.md"
    write_markdown_report(comparisons, None, path)

    text = path.read_text(encoding="utf-8")
    for comparison in comparisons.values():
        for caveat in comparison.caveats:
            assert caveat in text


def test_write_markdown_report_includes_robustness_section_when_supplied(
    tmp_path: Path,
) -> None:
    comparisons = compare_all(sample_reports(), hmm_key="hmm")
    suite = RobustnessSuite()
    robustness_report = suite.run(
        RobustnessDimension.SLIPPAGE,
        {
            "low": lambda: performance_report(cagr=0.08),
            "high": lambda: performance_report(cagr=0.12),
        },
    )
    path = tmp_path / "report.md"
    write_markdown_report(comparisons, {"slippage": robustness_report}, path)

    text = path.read_text(encoding="utf-8")
    assert "## Robustness diagnostics" in text
    assert "### slippage" in text
    assert "low" in text and "high" in text


def test_write_markdown_report_omits_robustness_section_when_not_supplied(
    tmp_path: Path,
) -> None:
    comparisons = compare_all(sample_reports(), hmm_key="hmm")
    path = tmp_path / "report.md"
    write_markdown_report(comparisons, None, path)

    text = path.read_text(encoding="utf-8")
    assert "Robustness diagnostics" not in text


def test_write_markdown_report_rejects_empty_comparisons(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_markdown_report({}, None, tmp_path / "report.md")


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def test_write_html_report_is_well_formed_and_includes_every_baseline(
    tmp_path: Path,
) -> None:
    comparisons = compare_all(sample_reports(), hmm_key="hmm")
    path = tmp_path / "report.html"
    write_html_report(comparisons, None, path)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "HMM vs. buy_and_hold" in text
    assert "HMM vs. rolling_volatility" in text
    assert "<table>" in text


def test_write_html_report_escapes_strategy_names() -> None:
    from backtest.comparison import compare_to_baseline
    from backtest.report import _escape

    assert _escape("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    # sanity: comparison itself doesn't choke on an unusual name either
    comparison = compare_to_baseline(performance_report(), performance_report(), "a & b")
    assert comparison.baseline_name == "a & b"


def test_write_html_report_includes_robustness_section_when_supplied(tmp_path: Path) -> None:
    comparisons = compare_all(sample_reports(), hmm_key="hmm")
    suite = RobustnessSuite()
    robustness_report = suite.run(
        RobustnessDimension.UNIVERSE_SIZE,
        {
            "small": lambda: performance_report(cagr=0.08),
            "large": lambda: performance_report(cagr=0.12),
        },
    )
    path = tmp_path / "report.html"
    write_html_report(comparisons, {"universe_size": robustness_report}, path)

    text = path.read_text(encoding="utf-8")
    assert "Robustness diagnostics" in text
    assert "universe_size" in text


def test_write_html_report_rejects_empty_comparisons(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_html_report({}, None, tmp_path / "report.html")
