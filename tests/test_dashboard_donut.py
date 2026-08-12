"""The holdings donut is hand-rolled SVG, so its geometry needs guarding.

Plotly computed the arcs; now this module does, and a wrong dasharray produces
a silently misleading chart rather than an exception.
"""
import xml.etree.ElementTree as ET

import pytest

from dashboard_helpers import (
    _DONUT_CIRCUMFERENCE,
    build_donut_svg,
    portfolio_allocation,
)


def arcs_of(html: str):
    """Parse the markup and return the slice <circle> elements."""
    inner = html[html.index("<svg") : html.rindex("</div>")]
    return ET.fromstring(inner).find("g").findall("circle")


def visible_length(circle) -> float:
    return float(circle.get("stroke-dasharray").split()[0])


def offset_of(circle) -> float:
    return -float(circle.get("stroke-dashoffset"))


@pytest.fixture
def holdings():
    return [
        {"label": "AAPL", "value": 2000.0, "color": "#f97316"},
        {"label": "NVDA", "value": 1000.0, "color": "#22c55e"},
        {"label": "TSLA", "value": 1000.0, "color": "#a855f7"},
    ]


def test_markup_is_well_formed_xml(holdings):
    assert len(arcs_of(build_donut_svg(holdings, 4000.0))) == 3


def test_arcs_are_proportional_to_value(holdings):
    first, second, third = arcs_of(build_donut_svg(holdings, 4000.0))
    # AAPL is half the book, so its arc is twice either of the others.
    assert visible_length(first) == pytest.approx(
        visible_length(second) * 2, rel=0.02
    )
    assert visible_length(second) == pytest.approx(visible_length(third))


def test_arcs_are_laid_end_to_end_without_overlap(holdings):
    circles = arcs_of(build_donut_svg(holdings, 4000.0))
    offsets = [offset_of(c) for c in circles]

    assert offsets[0] == pytest.approx(0.0)
    for previous, current in zip(circles, offsets[1:]):
        # Each slice starts where the last one ended, gap included.
        assert current > offset_of(previous) + visible_length(previous)
    assert offsets[-1] + visible_length(circles[-1]) <= _DONUT_CIRCUMFERENCE


def test_single_holding_draws_a_continuous_ring():
    circle = arcs_of(build_donut_svg([{"label": "AAPL", "value": 5.0, "color": "#fff"}], 5.0))[0]
    assert visible_length(circle) == pytest.approx(_DONUT_CIRCUMFERENCE)


def test_empty_holdings_still_render_a_placeholder_ring():
    assert len(arcs_of(build_donut_svg([], 0.0))) == 1


def test_short_positions_are_drawn_by_absolute_value():
    """A negative market value is a real slice, not a negative arc."""
    segments, _ = portfolio_allocation(
        [{"symbol": "TSLA", "qty": -3, "current_price": 300, "market_value": -900}],
        equity=10_000,
        cash=0,
    )
    circle = arcs_of(build_donut_svg(segments, 900.0))[0]
    assert visible_length(circle) > 0


def test_labels_are_escaped_into_hover_titles(holdings):
    html = build_donut_svg([{"label": "A&B", "value": 1.0, "color": "#fff"}], 1.0)
    assert "A&amp;B" in html
    assert arcs_of(html)[0].find("title").text.startswith("A&B")


def test_total_is_rendered_in_the_centre(holdings):
    assert "$ 4,000.00" in build_donut_svg(holdings, 4000.0)


def test_no_plotly_chart_is_used_by_the_dashboard():
    """The 4.5 MB PlotlyChart JS chunk must stay unrequested on page load."""
    with open("dashboard.py", encoding="utf-8") as f:
        assert "plotly_chart" not in f.read()
