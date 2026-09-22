"""Dashboard boundaries: no credentials/files, no invented freshness, no partial JSON."""
from __future__ import annotations

import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from monitoring.web_server import DashboardHandler, read_snapshot


def test_snapshot_excludes_broker_session_and_unknown_fields(tmp_path: Path) -> None:
    (tmp_path / "dashboard_data.json").write_text(json.dumps({
        "broker": {"access_token": "secret"}, "api_key": "secret",
        "live_book": {"equity": 100}, "generated_at": "2026-09-21",
    }))
    payload = json.loads(read_snapshot(tmp_path))
    assert payload == {"live_book": {"equity": 100}, "generated_at": "2026-09-21"}


@pytest.mark.parametrize("body", ['{"live_book":', '{"live_book":NaN}', '[]'])
def test_partial_or_invalid_snapshot_is_rejected(tmp_path: Path, body: str) -> None:
    (tmp_path / "dashboard_data.json").write_text(body)
    with pytest.raises(ValueError):
        read_snapshot(tmp_path)


def test_ticks_keep_book_timestamp_and_compare_absolute_times(tmp_path: Path) -> None:
    book_time = "2026-09-21T10:00:00+00:00"
    (tmp_path / "dashboard_data.json").write_text(json.dumps({
        "live_book": {"fetched_at": book_time},
        "ticks": {"updated_at": "2026-09-21T11:00:00+00:00", "points": []},
    }))
    ticks_file = tmp_path / "live_ticks.json"
    ticks_file.write_text(json.dumps({"ticks": [{"t": "2026-09-21T16:00:00+05:30"}]}))
    assert json.loads(read_snapshot(tmp_path))["ticks"]["points"] == []
    ticks_file.write_text(json.dumps({"ticks": [{"t": "2026-09-21T17:00:00+05:30"}]}))
    result = json.loads(read_snapshot(tmp_path))
    assert len(result["ticks"]["points"]) == 1
    assert result["live_book"]["fetched_at"] == book_time


def test_http_routes_do_not_expose_state_or_accept_writes(tmp_path: Path) -> None:
    (tmp_path / "dashboard_data.json").write_text('{"generated_at":"test"}')
    (tmp_path / "kite_session.json").write_text('{"access_token":"secret"}')
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(DashboardHandler, state_dir=tmp_path)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for route in ["/", "/app.js", "/style.css", "/api/snapshot"]:
            with urlopen(base + route, timeout=3) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert b"access_token" not in response.read()
        for route in ["/state/kite_session.json", "/../state/kite_session.json", "/.env"]:
            with pytest.raises(HTTPError) as error:
                urlopen(base + route, timeout=3)
            assert error.value.code == 404
            error.value.close()
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/snapshot", data=b"{}", timeout=3)
        assert error.value.code == 501
        error.value.close()
        (tmp_path / "dashboard_data.json").write_text('{"broken":')
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/snapshot", timeout=3)
        assert error.value.code == 503
        error.value.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_azure_research_overrides_legacy_results_without_refreshing_paper_book(
    tmp_path: Path,
) -> None:
    (tmp_path / "dashboard_data.json").write_text(json.dumps({
        "live_book": {"fetched_at": "old"}, "strategies": {"reports": {"stale": {}}},
    }))
    (tmp_path / "backtest_dashboard.json").write_text(json.dumps({
        "strategies": {"reports": {"hmm": {"ending_equity": 4334.62}}},
        "equity_curves": {}, "backtest_run": {"complete": False}, "api_key": "secret",
    }))
    result = json.loads(read_snapshot(tmp_path))
    assert set(result["strategies"]["reports"]) == {"hmm"}
    assert result["live_book"]["fetched_at"] == "old"
    assert result["backtest_run"]["complete"] is False
    assert "api_key" not in result


def test_trade_history_is_only_shown_for_the_selected_run(tmp_path: Path) -> None:
    (tmp_path / "dashboard_data.json").write_text("{}")
    (tmp_path / "backtest_dashboard.json").write_text(json.dumps({
        "strategies": {}, "equity_curves": {}, "backtest_run": {"run_id": "current"},
    }))
    path = tmp_path / "hmm_backtest_trades.json"
    path.write_text(json.dumps({"run_id": "old", "rows": []}))
    assert "hmm_history" not in json.loads(read_snapshot(tmp_path))
    path.write_text(json.dumps({"run_id": "current", "rows": [{"id": 1}]}))
    assert json.loads(read_snapshot(tmp_path))["hmm_history"]["rows"] == [{"id": 1}]
