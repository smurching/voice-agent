"""
Tests for the WebRTC test page and ephemeral session endpoint.

Unit tests run without any network access (mock or offline).
Integration tests require a live server and a real OPENAI_API_KEY; they are
skipped automatically when the key is absent.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_server.start_server import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HTML_PATH = Path(__file__).parent.parent / "agent_server" / "templates" / "webrtc_test.html"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def html_source():
    return HTML_PATH.read_text()


# ---------------------------------------------------------------------------
# Unit tests — no network
# ---------------------------------------------------------------------------


def test_webrtc_test_page_returns_200(client):
    resp = client.get("/webrtc-test")
    assert resp.status_code == 200


def test_webrtc_test_page_is_html(client):
    resp = client.get("/webrtc-test")
    assert "text/html" in resp.headers["content-type"]


def test_html_has_connect_button(html_source):
    assert "Connect" in html_source


def test_html_references_rtc_session_endpoint(html_source):
    assert "/api/rtc-session" in html_source


def test_html_creates_peer_connection(html_source):
    assert "RTCPeerConnection" in html_source


def test_html_shows_ice_state(html_source):
    assert "iceConnectionState" in html_source


def test_html_no_api_key(html_source):
    assert "sk-" not in html_source


# ---------------------------------------------------------------------------
# Integration tests — require live server + real OPENAI_API_KEY
# ---------------------------------------------------------------------------

SKIP_INTEGRATION = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live integration tests",
)


@SKIP_INTEGRATION
def test_rtc_session_endpoint_returns_token(client):
    resp = client.post("/api/rtc-session")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    token = data.get("client_secret", {}).get("value", "")
    assert token.startswith("ek_"), f"Unexpected token format: {token!r}"


@SKIP_INTEGRATION
def test_rtc_session_token_is_short_lived(client):
    resp = client.post("/api/rtc-session")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # expires_at should be present and numeric (0 means no expiry; positive = epoch seconds)
    assert "expires_at" in data, "Response missing expires_at field"
    assert isinstance(data["expires_at"], (int, float)), "expires_at should be numeric"
