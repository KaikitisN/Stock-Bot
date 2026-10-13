"""Crypto exits: one position per coin, a real stop-limit, and a bot-managed target.

Alpaca rejects bracket orders and plain stop orders on crypto. These tests
describe the replacement: a market entry, a stop-limit underneath it, and a
market sell when price reaches the target.
"""
import json
from types import SimpleNamespace

import pytest
from alpaca.trading.enums import OrderClass, OrderSide, OrderType

import config
import executor
from executor import (
    get_position_side,
    liquidate_position,
    submit_bracket_order,
    symbol_key,
)
from exits import manage_open_exits


class Pos:
    def __init__(self, symbol, qty, avg_entry_price=8.13, current_price=8.0, market_value=None):
        self.symbol = symbol
        self.qty = qty
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price
        self.market_value = market_value if market_value is not None else float(qty) * float(current_price)


class FakeBroker:
    def __init__(self, positions=(), orders=()):
        self.positions = list(positions)
        self.orders = list(orders)
        self.submitted = []
        self.cancelled = []
        self.events = []

    def get_all_positions(self):
        return list(self.positions)

    def get_open_position(self, symbol):
        raise AssertionError(f"slash lookup must not be used for {symbol}")

    def get_orders(self, filter=None):
        return list(self.orders)

    def submit_order(self, req):
        self.submitted.append(req)
        self.events.append(("submit", type(req).__name__))
        order_id = f"order-{len(self.submitted)}"
        return SimpleNamespace(id=order_id)

    def cancel_order_by_id(self, order_id):
        self.cancelled.append(str(order_id))
        self.events.append(("cancel", str(order_id)))

    def close_position(self, symbol):
        self.events.append(("close", symbol))
        return SimpleNamespace(id="close-1")


def _stop_order(qty=40, limit_price=7.54, order_id="stop-1"):
    return SimpleNamespace(
        id=order_id,
        symbol="UNIUSD",
        qty=qty,
        side=OrderSide.SELL,
        type=OrderType.STOP_LIMIT,
        stop_price=7.62,
        limit_price=limit_price,
    )


@pytest.fixture
def plans_path(tmp_path, monkeypatch):
    path = tmp_path / "exit_plans.json"
    monkeypatch.setattr(config, "EXIT_PLANS_FILE", str(path))
    monkeypatch.setattr(config, "CRYPTO_STOP_LIMIT_BUFFER_PCT", 1.0)
    return path


def _write_plan(path, **overrides):
    plan = {
        "symbol": "UNI/USD",
        "qty": 40,
        "entry_price": 8.13,
        "stop_price": 7.62,
        "target_price": 8.16,
        "stop_order_id": "stop-1",
    }
    plan.update(overrides)
    path.write_text(json.dumps({"UNIUSD": plan}), encoding="utf-8")


def _saved(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def test_symbol_key_treats_slash_and_compact_as_the_same_coin():
    assert symbol_key("UNI/USD") == symbol_key("uniusd") == "UNIUSD"


def test_uniusd_position_counts_as_a_long_uni_usd():
    client = FakeBroker(positions=[Pos("UNIUSD", "40")])
    assert get_position_side(client, "UNI/USD") == "long"


def test_crypto_buy_is_a_market_order_plus_a_stop_limit(plans_path):
    client = FakeBroker()
    order = submit_bracket_order(
        client, "UNI/USD", 7, "BUY", 7.62, 8.16, entry_price=7.8,
    )
    assert order is not None
    assert [type(req).__name__ for req in client.submitted] == [
        "MarketOrderRequest",
        "StopLimitOrderRequest",
    ]
    buy, stop = client.submitted
    assert buy.order_class != OrderClass.BRACKET
    assert buy.side == OrderSide.BUY
    assert stop.side == OrderSide.SELL
    assert stop.qty == 7
    assert stop.stop_price == 7.62
    assert stop.limit_price == pytest.approx(7.54)
    assert stop.limit_price < stop.stop_price
    saved = _saved(plans_path)["UNIUSD"]
    assert saved["target_price"] == 8.16
    assert saved["stop_price"] == 7.62


def test_a_rejected_stop_is_reported_and_remembered(plans_path):
    class Boom(FakeBroker):
        def submit_order(self, req):
            if type(req).__name__ == "StopLimitOrderRequest":
                raise RuntimeError("crypto stop rejected")
            return super().submit_order(req)

    order = submit_bracket_order(Boom(), "UNI/USD", 7, "BUY", 7.62, 8.16, entry_price=7.8)
    assert order is not None
    assert "crypto stop rejected" in executor.LAST_PROTECTION_ERROR
    assert "UNIUSD" in _saved(plans_path)


def test_stock_buy_still_uses_a_bracket(plans_path):
    client = FakeBroker()
    submit_bracket_order(client, "AAPL", 1, "BUY", 90.0, 110.0, entry_price=100.0)
    assert len(client.submitted) == 1
    assert client.submitted[0].order_class == OrderClass.BRACKET
    assert _saved(plans_path) == {}


def test_liquidate_cancels_the_resting_stop_before_selling(plans_path):
    client = FakeBroker(
        positions=[Pos("UNIUSD", "40")],
        orders=[_stop_order()],
    )
    order = liquidate_position(client, "UNI/USD")
    assert order is not None
    assert client.events[0] == ("cancel", "stop-1")
    assert client.events[1][0] == "submit"
    sell = client.submitted[0]
    assert sell.side == OrderSide.SELL
    assert sell.qty == 40


def test_dip_inside_the_band_holds_and_arms_one_stop(plans_path):
    client = FakeBroker(positions=[Pos("UNIUSD", "40", avg_entry_price=8.13, current_price=8.0)])
    actions = manage_open_exits(
        client,
        {"UNI/USD": {"close": 8.0, "atr_14": 0.10}},
        config.SIZING,
        config.DEFAULT_RISK,
    )
    assert [type(req).__name__ for req in client.submitted] == ["StopLimitOrderRequest"]
    assert client.submitted[0].qty == 40
    assert client.submitted[0].side == OrderSide.SELL
    assert not any(item["action"] == "SELL" for item in actions)
    saved = _saved(plans_path)["UNIUSD"]
    assert saved["stop_price"] < 8.0 < saved["target_price"]


def test_an_existing_stop_for_the_full_size_is_left_alone(plans_path):
    _write_plan(plans_path)
    client = FakeBroker(
        positions=[Pos("UNIUSD", "40", current_price=8.0)],
        orders=[_stop_order(qty=40)],
    )
    manage_open_exits(
        client,
        {"UNI/USD": {"close": 8.0, "atr_14": 0.10}},
        config.SIZING,
        config.DEFAULT_RISK,
    )
    assert client.submitted == []
    assert client.cancelled == []


def test_an_undersized_stop_is_replaced_with_one_for_the_whole_position(plans_path):
    _write_plan(plans_path, qty=7)
    client = FakeBroker(
        positions=[Pos("UNIUSD", "40", current_price=8.0)],
        orders=[_stop_order(qty=7)],
    )
    manage_open_exits(
        client,
        {"UNI/USD": {"close": 8.0, "atr_14": 0.10}},
        config.SIZING,
        config.DEFAULT_RISK,
    )
    assert client.cancelled == ["stop-1"]
    assert [type(req).__name__ for req in client.submitted] == ["StopLimitOrderRequest"]
    assert client.submitted[0].qty == 40


def test_target_cancels_the_stop_and_sells(plans_path):
    _write_plan(plans_path)
    client = FakeBroker(
        positions=[Pos("UNIUSD", "40", current_price=8.20)],
        orders=[_stop_order()],
    )
    actions = manage_open_exits(
        client,
        {"UNI/USD": {"close": 8.20, "atr_14": 0.10}},
        config.SIZING,
        config.DEFAULT_RISK,
    )
    assert client.events[0] == ("cancel", "stop-1")
    assert client.submitted[0].side == OrderSide.SELL
    assert client.submitted[0].qty == 40
    assert actions[0]["action"] == "SELL"
    assert "target" in actions[0]["reason"].lower()
    assert "UNIUSD" not in _saved(plans_path)


def test_a_stop_breach_without_a_broker_stop_sells(plans_path):
    _write_plan(plans_path, stop_order_id="")
    client = FakeBroker(positions=[Pos("UNIUSD", "40", current_price=7.50)])
    actions = manage_open_exits(
        client,
        {"UNI/USD": {"close": 7.50, "atr_14": 0.10}},
        config.SIZING,
        config.DEFAULT_RISK,
    )
    assert [type(req).__name__ for req in client.submitted] == ["MarketOrderRequest"]
    assert client.submitted[0].side == OrderSide.SELL
    assert actions[0]["action"] == "SELL"
    assert "stop" in actions[0]["reason"].lower()
    assert "UNIUSD" not in _saved(plans_path)


def test_price_between_the_stop_and_its_limit_lets_the_broker_stop_work(plans_path):
    _write_plan(plans_path)
    client = FakeBroker(
        positions=[Pos("UNIUSD", "40", current_price=7.58)],
        orders=[_stop_order(limit_price=7.54)],
    )
    manage_open_exits(
        client,
        {"UNI/USD": {"close": 7.58, "atr_14": 0.10}},
        config.SIZING,
        config.DEFAULT_RISK,
    )
    assert client.submitted == []
    assert client.cancelled == []


def test_price_through_the_stop_limit_cancels_it_and_sells(plans_path):
    _write_plan(plans_path)
    client = FakeBroker(
        positions=[Pos("UNIUSD", "40", current_price=7.40)],
        orders=[_stop_order(limit_price=7.54)],
    )
    actions = manage_open_exits(
        client,
        {"UNI/USD": {"close": 7.40, "atr_14": 0.10}},
        config.SIZING,
        config.DEFAULT_RISK,
    )
    assert client.cancelled == ["stop-1"]
    assert client.submitted[0].side == OrderSide.SELL
    assert "stop" in actions[0]["reason"].lower()


def test_a_closed_position_drops_its_plan_without_a_new_order(plans_path):
    _write_plan(plans_path)
    client = FakeBroker()
    manage_open_exits(client, {}, config.SIZING, config.DEFAULT_RISK)
    assert client.submitted == []
    assert _saved(plans_path) == {}
