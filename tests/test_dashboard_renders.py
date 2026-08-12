"""Smoke test: the dashboard script runs top to bottom without raising.

Catches import errors, bad f-strings and fragment misuse that would otherwise
only surface as a red screen in the browser.
"""
import pytest

from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file("dashboard.py", default_timeout=60)
    at.run()
    return at


def test_script_runs_without_exception(app):
    assert not app.exception, [e.value for e in app.exception]


def test_the_holdings_donut_is_inline_svg(app):
    markup = " ".join(m.value for m in app.markdown)
    assert "cc-donut" in markup
    assert "<svg" in markup
