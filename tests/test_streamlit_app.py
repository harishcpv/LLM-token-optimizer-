"""Smoke-test the real Streamlit submission flow when the app extra is installed."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_default_streamlit_audit_reaches_report_card() -> None:
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)

    assert not app.exception
    assert any("Token footprint" in item.value for item in app.markdown)

    submit = next(button for button in app.button if button.label == "Run token audit")
    app = submit.click().run(timeout=30)

    assert not app.exception
    assert any(metric.label == "Prompt tokens" for metric in app.metric)
