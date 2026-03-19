"""
Integration tests for voice endpoints against a live server.

Run with the server already started, or via the fixture that spawns it:
    uv run pytest tests/test_voice_integration.py -v

The server is started on TEST_PORT (default 8100) for test isolation.
"""

import json
import os
import struct
import subprocess
import sys
import time
import base64
from pathlib import Path

import httpx
import pytest
import websockets.sync.client as ws_sync

TEST_PORT = int(os.environ.get("TEST_PORT", "8100"))
BASE_URL = f"http://localhost:{TEST_PORT}"
WS_URL = f"ws://localhost:{TEST_PORT}/ws/voice"


# ---------------------------------------------------------------------------
# Session-scoped fixture: spin up the server for the duration of the test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def live_server():
    """Start the FastAPI server in a subprocess and wait until it's ready."""
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "sk-test-dummy-key-for-tests")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "agent_server.start_server:app",
            "--host", "127.0.0.1",
            "--port", str(TEST_PORT),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent.parent),
    )

    # Wait up to 30 s for the server to respond
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            httpx.get(f"{BASE_URL}/health", timeout=1)
            break
        except Exception:
            # /health may not exist; try /voice as a proxy for "server is up"
            try:
                r = httpx.get(f"{BASE_URL}/voice", timeout=1)
                if r.status_code < 500:
                    break
            except Exception:
                pass
        if proc.poll() is not None:
            out = proc.stdout.read().decode()
            pytest.fail(f"Server process exited early:\n{out}")
        time.sleep(0.5)
    else:
        proc.kill()
        out = proc.stdout.read().decode()
        pytest.fail(f"Server did not start in time. Output:\n{out}")

    yield BASE_URL

    proc.kill()
    proc.wait()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def silent_pcm16_b64(num_samples: int = 512) -> str:
    samples = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    return base64.b64encode(samples).decode()


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------

class TestVoiceUIEndpoint:
    def test_get_voice_status_200(self, live_server):
        r = httpx.get(f"{live_server}/voice")
        assert r.status_code == 200

    def test_get_voice_content_type_html(self, live_server):
        r = httpx.get(f"{live_server}/voice")
        assert "text/html" in r.headers["content-type"]

    def test_html_has_connect_button(self, live_server):
        r = httpx.get(f"{live_server}/voice")
        assert 'id="connectBtn"' in r.text

    def test_html_has_disconnect_button(self, live_server):
        r = httpx.get(f"{live_server}/voice")
        assert 'id="disconnectBtn"' in r.text

    def test_html_has_transcript_div(self, live_server):
        r = httpx.get(f"{live_server}/voice")
        assert 'id="transcript"' in r.text

    def test_html_references_ws_voice(self, live_server):
        r = httpx.get(f"{live_server}/voice")
        assert "ws/voice" in r.text

    def test_html_contains_audio_context(self, live_server):
        r = httpx.get(f"{live_server}/voice")
        assert "AudioContext" in r.text

    def test_html_contains_get_user_media(self, live_server):
        r = httpx.get(f"{live_server}/voice")
        assert "getUserMedia" in r.text

    def test_html_contains_pcm_conversion(self, live_server):
        """Browser must convert Float32 mic samples to PCM16 before sending."""
        r = httpx.get(f"{live_server}/voice")
        assert "Int16Array" in r.text

    def test_html_contains_input_audio_buffer_append(self, live_server):
        """The JS must send input_audio_buffer.append events to the server."""
        r = httpx.get(f"{live_server}/voice")
        assert "input_audio_buffer.append" in r.text

    def test_html_handles_audio_delta(self, live_server):
        """The JS must handle response.audio.delta events for playback."""
        r = httpx.get(f"{live_server}/voice")
        assert "response.audio.delta" in r.text

    def test_html_handles_transcript_delta(self, live_server):
        """The JS must handle transcript delta events."""
        r = httpx.get(f"{live_server}/voice")
        assert "response.audio_transcript.delta" in r.text

    def test_html_no_api_key_in_source(self, live_server):
        """API key must never appear in the served HTML."""
        r = httpx.get(f"{live_server}/voice")
        assert "sk-" not in r.text


# ---------------------------------------------------------------------------
# WebSocket endpoint tests
# ---------------------------------------------------------------------------

class TestVoiceWebSocketEndpoint:
    def test_websocket_accepts_connection(self, live_server):
        """
        The /ws/voice endpoint should accept the connection before failing
        (it will error quickly with a dummy API key, but it must accept first).
        """
        accepted = False
        try:
            with ws_sync.connect(WS_URL, open_timeout=5) as ws:
                accepted = True
                # Read whatever comes back (likely an error event from OpenAI)
                try:
                    ws.recv(timeout=5)
                except Exception:
                    pass
        except Exception:
            # Connection refused or immediate close is a failure
            pass
        assert accepted, "Server should accept the WebSocket connection"

    def test_websocket_sends_json_events(self, live_server):
        """
        Any message the server relays to the browser must be valid JSON
        with a 'type' field.
        """
        messages = []
        try:
            with ws_sync.connect(WS_URL, open_timeout=5) as ws:
                # Send a dummy audio buffer event
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": silent_pcm16_b64(512),
                }))
                # Collect up to 3 messages or until timeout
                for _ in range(3):
                    try:
                        msg = ws.recv(timeout=3)
                        messages.append(msg)
                    except Exception:
                        break
        except Exception:
            pass  # Connection may drop; we only check what was received

        for msg in messages:
            parsed = json.loads(msg)
            assert "type" in parsed, f"Event missing 'type': {parsed}"

    def test_websocket_rejects_non_ws_request(self, live_server):
        """Plain HTTP GET to /ws/voice should get a non-101 response."""
        r = httpx.get(f"{live_server}/ws/voice")
        # FastAPI returns 403 or 400 for non-upgrade requests to a WS route
        assert r.status_code != 200
