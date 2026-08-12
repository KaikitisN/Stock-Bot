import json

import pytest

import background_runner


@pytest.fixture
def status_path(tmp_path, monkeypatch):
    path = tmp_path / "runner_status.json"
    monkeypatch.setattr(background_runner, "STATUS_PATH", str(path))
    monkeypatch.setattr(background_runner.config, "LOG_DIR", str(tmp_path))
    return path


def test_status_keeps_the_keys_the_dashboard_reads(status_path):
    background_runner.write_status(
        background_runner._base_running_status(5, "2026-08-12T00:00:00")
    )
    status = json.loads(status_path.read_text())
    for key in (
        "state", "cycle_started_at", "current_symbol", "current_index",
        "total_symbols", "last_completed_symbol", "last_result",
        "next_run_at", "interval_minutes", "progress_pct",
    ):
        assert key in status


def test_base_status_reports_the_evaluating_phase(status_path):
    status = background_runner._base_running_status(5, "2026-08-12T00:00:00")
    assert status["phase"] == "evaluating"


def test_allocating_phase_status_is_written(status_path):
    background_runner.write_status({
        **background_runner._base_running_status(5, "2026-08-12T00:00:00"),
        "phase": "allocating",
        "current_symbol": "placing orders…",
    })
    status = json.loads(status_path.read_text())
    assert status["phase"] == "allocating"
    assert status["current_symbol"] == "placing orders…"
