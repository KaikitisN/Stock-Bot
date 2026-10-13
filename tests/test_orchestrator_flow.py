import csv

import pytest

import orchestrator


class FakeClient:
    def __init__(self, positions=(), pending=()):
        self._positions = dict(positions)
        self._pending = set(pending)
        self.submitted = []

    # risk_manager / executor surface used by the orchestrator
    def get_all_positions(self):
        return [object()] * len(self._positions)


@pytest.fixture(autouse=True)
def isolate_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.config, "DECISIONS_LOG", str(tmp_path / "decisions.csv"))
    monkeypatch.setattr(orchestrator.config, "TRADES_LOG", str(tmp_path / "trades.csv"))
    return tmp_path


def _market_data(price=100.0, atr=2.0, sma_30=90.0, rsi=55.0):
    return {"close": price, "sma_10": price, "sma_30": sma_30, "rsi_14": rsi, "atr_14": atr}


def _decision(action="BUY", confidence=90, conviction=0.6, mu_pct=3.0):
    return {
        "symbol": "TEST", "action": action, "confidence": confidence,
        "reason": "stub", "provider": "stub",
        "mu_pct": mu_pct, "sigma_pct": 5.0, "conviction": conviction, "p_up": 0.9,
    }


def _patch_decision(monkeypatch, decision):
    monkeypatch.setattr(orchestrator, "get_decision", lambda *a, **k: dict(decision))


def _patch_position_checks(monkeypatch, side=None, pending=False):
    monkeypatch.setattr(orchestrator, "get_position_side", lambda c, s: side)
    monkeypatch.setattr(orchestrator, "has_pending_order", lambda c, s: pending)


# --- evaluate_symbol ---

def test_strong_signal_becomes_a_candidate(monkeypatch):
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch)
    decision, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
    )
    assert candidate is not None
    assert candidate["conviction"] == 0.6
    assert candidate["price"] == 100.0
    assert candidate["atr"] == 2.0


def test_evaluate_never_submits_an_order(monkeypatch):
    """Phase separation: evaluation must be side-effect free at the broker."""
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch)
    called = []
    monkeypatch.setattr(orchestrator, "submit_bracket_order",
                        lambda *a, **k: called.append(a))
    orchestrator.evaluate_symbol(FakeClient(), "TEST", _market_data(), "stub", False)
    assert called == []


def test_low_confidence_is_vetoed(monkeypatch):
    _patch_decision(monkeypatch, _decision(confidence=50))
    _patch_position_checks(monkeypatch)
    _, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
    )
    assert candidate is None


def test_news_origin_uses_a_lower_confidence_floor(monkeypatch):
    monkeypatch.setattr(orchestrator.config, "MIN_TRADE_CONFIDENCE", 80)
    monkeypatch.setattr(orchestrator.config, "NEWS_MIN_TRADE_CONFIDENCE", 70)
    _patch_decision(monkeypatch, _decision(confidence=75))
    _patch_position_checks(monkeypatch)

    _, watchlist = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
    )
    _, news = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
        min_confidence=70, position_scale=0.5,
    )
    assert watchlist is None
    assert news is not None
    assert news["position_scale"] == 0.5


def test_watchlist_candidate_keeps_a_full_position_scale(monkeypatch):
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch)
    _, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
    )
    assert candidate["position_scale"] == 1.0


def test_trend_filter_veto_is_recorded(monkeypatch):
    monkeypatch.setattr(orchestrator.config, "USE_TREND_FILTER", True)
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch)
    decision, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(sma_30=200.0), "stub", False,
    )
    assert candidate is None
    assert "Trend filter" in decision["error"]


def test_existing_long_blocks_another_buy(monkeypatch):
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch, side="long")
    _, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
    )
    assert candidate is None


def test_pending_buy_blocks_another_buy(monkeypatch):
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch, pending=True)
    _, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
    )
    assert candidate is None


def test_closing_sell_is_allowed_while_a_stop_is_resting(monkeypatch):
    monkeypatch.setattr(orchestrator.config, "USE_TREND_FILTER", False)
    _patch_decision(monkeypatch, _decision(action="SELL", mu_pct=-3.0))
    _patch_position_checks(monkeypatch, side="long", pending=True)
    _, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(sma_30=200.0, rsi=45.0), "stub", False,
    )
    assert candidate is not None
    assert candidate["closing"] is True


def test_halt_blocks_candidates(monkeypatch):
    _patch_decision(monkeypatch, _decision())
    _patch_position_checks(monkeypatch)
    decision, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(), "stub", False,
        trading_halted=True, halt_reason="daily loss",
    )
    assert candidate is None
    assert decision["error"] == "daily loss"


def test_sell_without_position_is_blocked_when_shorting_is_off(monkeypatch):
    monkeypatch.setattr(orchestrator.config, "ALLOW_SHORT_SELLING", False)
    _patch_decision(monkeypatch, _decision(action="SELL", mu_pct=-3.0))
    _patch_position_checks(monkeypatch, side=None)
    _, candidate = orchestrator.evaluate_symbol(
        FakeClient(), "TEST", _market_data(sma_30=200.0, rsi=45.0), "stub", False,
    )
    assert candidate is None


# --- decision logging ---

def test_decision_log_has_a_fixed_schema(isolate_logs):
    orchestrator.log_decision({"symbol": "A", "action": "BUY", "extra": "ignored"})
    with open(orchestrator.config.DECISIONS_LOG, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == orchestrator.DECISION_FIELDS


def test_legacy_log_with_a_different_header_is_rotated(isolate_logs):
    path = orchestrator.config.DECISIONS_LOG
    with open(path, "w", encoding="utf-8") as f:
        f.write("symbol,action,confidence\nA,BUY,80\n")
    orchestrator.log_decision({"symbol": "B", "action": "SELL"})
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == orchestrator.DECISION_FIELDS
    rotated = list(isolate_logs.glob("decisions_legacy_*.csv"))
    assert len(rotated) == 1


def test_missing_fields_are_written_as_blanks(isolate_logs):
    orchestrator.log_decision({"symbol": "A"})
    with open(orchestrator.config.DECISIONS_LOG, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    assert row["symbol"] == "A"
    assert row["qty"] == ""


def test_a_rejected_stop_is_written_on_the_decision(monkeypatch, isolate_logs):
    from types import SimpleNamespace

    import executor

    def _submit(*_a, **_k):
        executor.LAST_PROTECTION_ERROR = "stop rejected"
        return SimpleNamespace(id="entry-1")

    monkeypatch.setattr(orchestrator, "submit_bracket_order", _submit)
    plan = {
        "symbol": "UNI/USD",
        "action": "BUY",
        "price": 7.8,
        "atr": 0.1,
        "qty": 7,
        "weight_pct": 3.0,
        "decision": _decision(),
    }
    decision = orchestrator.execute_plan(
        FakeClient(), plan, orchestrator.config.DEFAULT_RISK, orchestrator.config.SIZING,
    )
    assert decision["trade_submitted"] is True
    assert "stop rejected" in decision["error"]


def test_run_once_protects_open_positions_before_new_entries(monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrator, "get_trading_client", lambda: FakeClient())
    monkeypatch.setattr(orchestrator, "is_stock_market_open", lambda _c: False)
    monkeypatch.setattr(orchestrator, "is_trading_halted", lambda *_a, **_k: (False, ""))
    monkeypatch.setattr(orchestrator, "get_market_snapshot", lambda _symbols: {})
    monkeypatch.setattr(orchestrator, "get_account_equity", lambda _c: (1000.0, 500.0))
    monkeypatch.setattr(
        orchestrator,
        "protect_open_positions",
        lambda *_a, **_k: calls.append("protect") or [],
    )
    orchestrator.run_once(["UNI/USD"], "stub", False, orchestrator.config.DEFAULT_RISK)
    assert calls == ["protect"]


def test_run_once_scores_news_names_on_the_lower_floor(monkeypatch):
    seen = {}

    def fake_eval(_client, symbol, _market_data, _provider, _use_news, **kwargs):
        seen[symbol] = kwargs.get("min_confidence")
        return {"symbol": symbol, "action": "HOLD", "confidence": 0}, None

    monkeypatch.setattr(orchestrator, "evaluate_symbol", fake_eval)
    monkeypatch.setattr(orchestrator, "get_trading_client", lambda: FakeClient())
    monkeypatch.setattr(orchestrator, "is_stock_market_open", lambda _c: True)
    monkeypatch.setattr(orchestrator, "is_trading_halted", lambda *_a, **_k: (False, ""))
    monkeypatch.setattr(
        orchestrator, "get_market_snapshot",
        lambda symbols: {s: {"close": 1} for s in symbols},
    )
    monkeypatch.setattr(orchestrator, "protect_open_positions", lambda *_a, **_k: [])
    monkeypatch.setattr(orchestrator, "get_account_equity", lambda _c: (1000.0, 500.0))
    monkeypatch.setattr(orchestrator.config, "NEWS_MIN_TRADE_CONFIDENCE", 70)
    monkeypatch.setattr(
        orchestrator, "discover_news_symbols", lambda _watchlist, _open: ["NVDA"],
    )

    orchestrator.run_once(["AAPL"], "stub", True, orchestrator.config.DEFAULT_RISK)
    assert seen["AAPL"] is None
    assert seen["NVDA"] == 70
