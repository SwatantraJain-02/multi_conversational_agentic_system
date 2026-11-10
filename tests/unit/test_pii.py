"""Redaction of secrets and PII (the Langfuse mask)."""

import pytest

from multi_agenti_conversation_system.guardrails import pii
from multi_agenti_conversation_system.guardrails.pii import mask, redact, redact_secrets


def test_northwind_row_keeps_dates_ids_prices_and_loses_phones():
    row = (
        "Order 10248 on 2024-07-04 10:30:00 for customer VINET, freight 32.38, "
        "total 1234567.89, phones (171) 555-7788 and 030-0074321, ip 192.168.0.1, "
        "ref ORD-2024-0001"
    )
    out = redact(row)
    for keep in (
        "10248",
        "2024-07-04",
        "VINET",
        "32.38",
        "1234567.89",
        "192.168.0.1",
        "ORD-2024-0001",
    ):
        assert keep in out, keep
    assert "555-7788" not in out and "0074321" not in out
    assert out.count("[REDACTED:phone]") == 2


@pytest.mark.parametrize(
    "text,gone",
    [
        ("mail ana@example.com now", "ana@example.com"),
        ("ssn 123-45-6789", "123-45-6789"),
        ("card 4111 1111 1111 1111 ok", "4111"),
        ("call +49 30 1234567", "1234567"),
    ],
)
def test_personal_data_is_redacted(text, gone):
    assert gone not in redact(text)


def test_a_non_card_digit_run_is_not_redacted():
    # Fails the Luhn check, so it is probably an id, not a card.
    assert "1234 5678 9012 3456" in redact("ref 1234 5678 9012 3456")


@pytest.mark.parametrize(
    "secret",
    [
        "Authorization: Bearer abcdef123456789",
        "key sk-or-v1-0123456789abcdef0123",
        "pk-lf-0123456789abcdef0123",
        "tvly-dev-0123456789",
        "pcsk_0123456789abcdef",
    ],
)
def test_credentials_are_redacted_by_shape(secret):
    assert "0123456789" not in redact(secret) and "abcdef123456789" not in redact(
        secret
    )


@pytest.fixture(autouse=True)
def _fresh_secrets():
    pii.configured_secrets.cache_clear()
    yield
    pii.configured_secrets.cache_clear()


def test_configured_secret_values_are_replaced(monkeypatch):
    monkeypatch.setattr(pii.settings, "A2A_TOKEN", "super-secret-token-value")
    assert "super-secret-token-value" not in redact_secrets(
        "x super-secret-token-value y"
    )


def test_empty_and_short_secrets_never_corrupt_text(monkeypatch):
    monkeypatch.setattr(pii.settings, "A2A_TOKEN", "")
    monkeypatch.setattr(pii.settings, "WEB_MCP_TOKEN", "abc")
    assert redact_secrets("an ordinary sentence with abc in it") == (
        "an ordinary sentence with abc in it"
    )


def test_mask_walks_nested_data_and_leaves_other_types():
    data = {"a": ["mail ana@example.com", {"b": 5, "c": None}], "d": ("x@y.io",)}
    out = mask(data=data)
    assert out["a"][0] == "mail [REDACTED:email]"
    assert out["a"][1] == {"b": 5, "c": None}
    assert out["d"] == ["[REDACTED:email]"]


def test_mask_never_raises(monkeypatch):
    def boom(_):
        raise RuntimeError("regex exploded")

    monkeypatch.setattr(pii, "redact", boom)
    assert mask(data="anything") == "[REDACTED:mask-error]"
